"""Argument shapes a model is likely to send. No network."""
import json


class _Ctx:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, toolset, schema, handler, **_kwargs):
        self.tools[name] = handler

    def register_command(self, *args, **kwargs):
        return None

    def register_cli_command(self, *args, **kwargs):
        return None

    def get_config(self, key, default=None):
        return default


def _handler(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ctx = _Ctx()
    plugin.register(ctx)
    return ctx.tools["model_drift"]


def test_action_case_and_target_separator_reach_status(plugin, tmp_path, monkeypatch):
    seen = {}

    def fake_status(_env, target=""):
        seen["target"] = target
        return {"status": "ok", "message": "idle"}

    monkeypatch.setattr(plugin.service, "status", fake_status)
    out = json.loads(_handler(plugin, tmp_path, monkeypatch)({
        "action": "Ｓｔａｔｕｓ",
        "target": "custom：demo-１",
    }))
    assert out["status"] == "ok"
    assert seen["target"] == "custom:demo-1"


def test_missing_action_defaults_to_status(plugin, tmp_path, monkeypatch):
    seen = {}

    def fake_status(_env, target=""):
        seen["action"] = "status"
        seen["target"] = target
        return {"status": "ok", "message": "idle"}

    monkeypatch.setattr(plugin.service, "status", fake_status)
    out = json.loads(_handler(plugin, tmp_path, monkeypatch)({}))
    assert out["status"] == "ok"
    assert seen == {"action": "status", "target": ""}


def test_extra_argument_is_refused_without_the_value(plugin, tmp_path, monkeypatch):
    out = json.loads(_handler(plugin, tmp_path, monkeypatch)({"action": "status", "limit": 10.0}))
    text = json.dumps(out)
    assert out["ok"] is False
    assert out["error_type"] == "bad_args"
    assert "for example action status" in out["message"]
    assert "10.0" not in text
    assert "limit" not in text


def test_float_action_is_refused_without_the_number(plugin, tmp_path, monkeypatch):
    out = json.loads(_handler(plugin, tmp_path, monkeypatch)({"action": 10.0}))
    text = json.dumps(out)
    assert out["ok"] is False
    assert out["error_type"] == "bad_args"
    assert "action must be status, estimate, or run" in out["message"]
    assert "for example action status" in out["message"]
    assert "10.0" not in text


def test_japanese_status_and_estimate_do_not_spend(plugin, tmp_path, monkeypatch):
    seen = []

    def fake_status(_env, target=""):
        seen.append("status")
        return {"status": "ok", "message": "idle"}

    def fake_estimate(_env, target=""):
        seen.append("estimate")
        return {"status": "ok", "message": "0 tokens"}

    def fake_run(_env, target=""):
        raise AssertionError("a Japanese ask must not start a paid run")

    monkeypatch.setattr(plugin.service, "status", fake_status)
    monkeypatch.setattr(plugin.service, "estimate", fake_estimate)
    monkeypatch.setattr(plugin.service, "run", fake_run)
    handler = _handler(plugin, tmp_path, monkeypatch)
    assert json.loads(handler({"action": "状態"}))["status"] == "ok"
    assert json.loads(handler({"action": "見積もり"}))["status"] == "ok"
    assert json.loads(handler({"action": "check"}))["status"] == "ok"
    blocked = json.loads(handler({"action": "実行"}))
    assert blocked["ok"] is False
    assert "action=estimate" in blocked["message"]
    assert "Nothing was spent" in blocked["message"]
    assert seen == ["status", "estimate", "status"]


def test_fullwidth_digits_are_not_an_action(plugin, tmp_path, monkeypatch):
    out = json.loads(_handler(plugin, tmp_path, monkeypatch)({"action": "１０"}))
    text = json.dumps(out, ensure_ascii=False)
    assert out["ok"] is False
    assert out["error_type"] == "bad_args"
    assert "for example action status" in out["message"]
    assert "１０" not in text
