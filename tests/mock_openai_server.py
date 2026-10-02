"""OpenAI-compatible mock endpoint for end-to-end runs inside a real Hermes, at zero API cost.

    python tests/mock_openai_server.py --port 18765

* Probe requests (no tools): answered by tests/mock_model.MockModel from the plugin's own suite.
* Agent requests (tools offered, e.g. a cron turn): first a tool call to model_drift(action=run),
  then, once the tool result is in the conversation, its "message" field as the final reply.
* POST /control {"degrade": 0.2, "rate_limit": 0, "network": 0, "served": "...", "out_tokens": 120}
  switches behaviour at runtime; "down": true makes chat requests fail with a dropped connection.
* GET /stats returns request counts and the size of every agent request (to measure the overhead
  of the scheduled agent turn).
"""

import argparse
import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("model_drift_plugin", ROOT / "__init__.py",
                                              submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec)
sys.modules["model_drift_plugin"] = pkg
spec.loader.exec_module(pkg)
from model_drift_plugin.suite import load_suite  # noqa: E402
from mock_model import MockModel  # noqa: E402

ITEMS = load_suite([]).items
STATE = {"degrade": 0.0, "rate_limit": 0.0, "network": 0.0, "served": "mock-model-1", "out_tokens": 120,
         "down": False, "seed": 0}
LOCK = threading.Lock()
MODEL = {"m": MockModel(ITEMS, seed=0)}
STATS = {"probe": 0, "agent": 0, "rate_limited": 0, "agent_request_chars": [], "agent_prompt_tokens": [],
         "agent_tools_offered": [], "probe_models": {}, "agent_models": {}}
LAST_AGENT_REQUEST = {}


def rebuild():
    MODEL["m"] = MockModel(ITEMS, seed=STATE["seed"], degrade=STATE["degrade"], rate_limit=STATE["rate_limit"],
                           network=STATE["network"], served=STATE["served"], out_tokens=STATE["out_tokens"])


def _text(content):
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def find_message(text):
    """The "message" field of the model_drift result, wherever the tool bridge nested it."""
    stack = [text]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            try:
                cur = json.loads(cur)
            except ValueError:
                continue
        if isinstance(cur, dict):
            if isinstance(cur.get("message"), str) and "status" in cur:
                return cur["message"]
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return text


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list", "data": [{"id": STATE["served"], "object": "model"}]})
        if self.path.startswith("/stats"):
            return self._json(200, STATS)
        if self.path.startswith("/last-agent-request"):
            return self._json(200, LAST_AGENT_REQUEST.get("body", {}))
        self._json(404, {"error": "not found"})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw or b"{}")
        if self.path.startswith("/control"):
            with LOCK:
                STATE.update(body)
                rebuild()
            return self._json(200, STATE)
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": "not found"})
        if STATE["down"]:
            self.close_connection = True
            self.connection.shutdown(2)
            return
        messages = body.get("messages", [])
        if body.get("tools"):
            return self._agent(body, messages, len(raw))
        STATS["probe"] += 1
        STATS["probe_models"][body.get("model", "")] = STATS["probe_models"].get(body.get("model", ""), 0) + 1
        try:
            text, served, tin, tout = MODEL["m"].answer(_text(messages[-1].get("content")))
        except KeyError:
            text, served, tin, tout = "I don't know.", STATE["served"], 50, 10
        except Exception as exc:  # the mock raises provider-shaped errors
            if getattr(exc, "status_code", None) == 429:
                STATS["rate_limited"] += 1
                return self._json(429, {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
                                  {"retry-after": "1"})
            self.close_connection = True
            self.connection.shutdown(2)
            return
        tout = min(tout, int(body.get("max_tokens") or body.get("max_completion_tokens") or tout))
        self._reply(body, served, {"role": "assistant", "content": text}, tin, tout)

    def _agent(self, body, messages, size):
        STATS["agent"] += 1
        STATS["agent_models"][body.get("model", "")] = STATS["agent_models"].get(body.get("model", ""), 0) + 1
        STATS["agent_request_chars"].append(size)
        STATS["agent_tools_offered"].append(sorted(t.get("function", {}).get("name", "") for t in body["tools"]))
        LAST_AGENT_REQUEST["body"] = body
        prompt_tokens = size // 4
        STATS["agent_prompt_tokens"].append(prompt_tokens)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        names = {t.get("function", {}).get("name") for t in body["tools"]}
        if not tool_msgs:  # behave like a model that follows the cron prompt
            if "model_drift" in names:
                fn = {"name": "model_drift", "arguments": json.dumps({"action": "run"})}
            else:
                fn = {"name": "tool_call", "arguments": json.dumps(
                    {"calls": [{"name": "model_drift", "arguments": {"action": "run"}}]})}
            call = {"id": f"call_{int(time.time() * 1000)}", "type": "function", "function": fn}
            return self._reply(body, STATE["served"], {"role": "assistant", "content": None, "tool_calls": [call]},
                               prompt_tokens, 20, finish="tool_calls")
        message = find_message(_text(tool_msgs[-1].get("content")))
        self._reply(body, STATE["served"], {"role": "assistant", "content": message}, prompt_tokens,
                    max(1, len(message) // 4))

    def _reply(self, body, model, msg, tin, tout, finish="stop"):
        usage = {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout}
        cid = f"chatcmpl-{int(time.time() * 1000)}"
        if not body.get("stream"):
            return self._json(200, {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            data = f"data: {json.dumps(obj) if not isinstance(obj, str) else obj}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")

        base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
        send({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, **tc}]}, "finish_reason": None}]})
        elif msg.get("content"):
            send({**base, "choices": [{"index": 0, "delta": {"content": msg["content"]}, "finish_reason": None}]})
        send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        send({**base, "choices": [], "usage": usage})
        send("[DONE]")
        self.wfile.write(b"0\r\n\r\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18765)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock OpenAI endpoint on http://127.0.0.1:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
