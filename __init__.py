"""model-drift-watch for Hermes Agent: notice when the model you use gets quietly worse.

Registers one tool (model_drift), one slash command (/model-drift-watch) and one CLI command
(hermes model-drift-watch). Model calls go through ctx.llm, so Hermes keeps the credentials and the
plugin never sees a key. State lives in Hermes's per-plugin data folder.
"""

# Hermes imports this directory as a package. pytest's collector also imports this file on its own
# (the repo root is the package), where relative imports cannot work.
import asyncio
import unicodedata

if __package__:
    from . import service
    from .store import Store, default_data_dir

TOOL_SCHEMA = {
    "name": "model_drift",
    "description": (
        "Check whether your Hermes model has got worse. action=status reads the last result (free). "
        "action=estimate shows tokens and dollars before running (free). action=run asks a fixed set of "
        "exactly graded questions and tests the answers against the same model's own baseline; it spends "
        "API tokens up to the user's cap. Show the user the 'report' field as written. "
        "Use it when someone asks \"has my model got worse?\" or whether their LLM has degraded, such as a silent quality regression or a behavior change behind the same model name. "
        "Each model-drift-watch run records accuracy, so quality can be compared over time with that baseline. "
        "The fixed questions are a small canary benchmark of that model. "
        "These words in Japanese, 状態 and 確認, mean status, and 見積もり means estimate. Both are free. 実行 does not start a paid check."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "estimate", "run"],
                       "description": "status (free, the default), estimate (free) or run (spends tokens)."},
            "target": {"type": "string",
                       "description": "Optional 'provider:model' of one watched model. Default: all watched models."},
        },
    },
}

_ACTIONS = ("status", "estimate", "run")
_TOOL_KEYS = {"action", "target"}
_STATUS_WORDS = {"status", "state", "check", "info", "状態", "確認", "様子", "ステータス"}
_ESTIMATE_WORDS = {"estimate", "cost", "price", "見積", "見積もり", "料金", "費用"}
# These ask for a paid run. They are not run: the next call is the free estimate.
_SPEND_ASKS = {"start", "go", "実行", "計測", "測定"}
_SPEND_MSG = (
    "That asks for a paid check. Call action=estimate first. It is free. "
    "Call action=run only after that, if you want to spend tokens up to the cap. Nothing was spent."
)
_ACTION_BAD = (
    "action must be status, estimate, or run, for example action status."
)
_TOOL_EXTRA = (
    "model_drift accepts only action and target, so drop other arguments and call again, for example action status."
)
_TARGET_BAD = (
    "target must be a provider:model string or be omitted, so pass one watched model or leave it out, "
    "for example target openrouter:deepseek/deepseek-chat."
)
_TARGET_FOLD = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "：": ":",
})

_ENV = {}


def _env():
    env = _ENV.get("env")
    if env is None:
        ctx = _ENV["ctx"]
        env = service.Env(get_config=ctx.get_config, llm=_LazyLlm(ctx), store=Store(default_data_dir()))
        _ENV["env"] = env
    return env


class _LazyLlm:
    """ctx.llm is resolved at call time, so config changes (model switch, grants) apply immediately."""

    def __init__(self, ctx):
        self._ctx = ctx

    def complete(self, **kw):
        return self._ctx.llm.complete(**kw)


def _args_error(message: str) -> str:
    return service.to_json({
        "ok": False, "status": "error", "error_type": "bad_args", "message": message,
    })


def _fold_target(value: str) -> str:
    return value.translate(_TARGET_FOLD).strip()


def _tool(args, **_kwargs):
    raw = args if isinstance(args, dict) else {}
    if not isinstance(args, dict) and args not in (None,):
        return _args_error(_ACTION_BAD)
    if set(raw) - _TOOL_KEYS:
        return _args_error(_TOOL_EXTRA)
    if "action" not in raw or raw.get("action") in (None, ""):
        action = "status"
    elif isinstance(raw.get("action"), str):
        action = unicodedata.normalize("NFKC", raw["action"]).strip().lower()
        if action in _SPEND_ASKS:
            return _args_error(_SPEND_MSG)
        if action in _STATUS_WORDS:
            action = "status"
        elif action in _ESTIMATE_WORDS:
            action = "estimate"
        elif action not in _ACTIONS:
            return _args_error(_ACTION_BAD)
    else:
        return _args_error(_ACTION_BAD)
    if "target" in raw and raw.get("target") not in (None, ""):
        if not isinstance(raw.get("target"), str):
            return _args_error(_TARGET_BAD)
        target = _fold_target(raw["target"])
    else:
        target = ""
    try:
        if action == "run":
            return service.to_json(service.run(_env(), target))
        if action == "estimate":
            return service.to_json(service.estimate(_env(), target))
        if action == "status":
            return service.to_json(service.status(_env(), target))
        return _args_error(_ACTION_BAD)
    except Exception as exc:  # a tool must answer, never crash the agent turn
        return service.to_json({"status": "error",
                                "message": f"[model-drift-watch] Internal error ({type(exc).__name__}: {exc}). "
                                           "Please report it with `hermes logs` output."})


SLASH_HELP = ("/model-drift-watch status | estimate | run | targets\n"
              "status: last result (free). estimate: cost before running (free). run: check now (spends tokens). "
              "targets: which models are watched.")


def _slash_work(raw_args: str):
    sub, _, rest = (raw_args or "").strip().partition(" ")
    sub = sub.lower() or "status"
    if sub in ("status", "estimate", "run"):
        result = getattr(service, sub)(_env(), rest.strip())
        return result.get("report") or result["message"]
    if sub == "targets":
        return service.targets_text(_env())
    return SLASH_HELP


async def _slash(raw_args: str):
    # A check can call the model. The worker thread keeps that off the gateway event loop.
    return await asyncio.to_thread(_slash_work, raw_args)


def _cli_setup(parser):
    subs = parser.add_subparsers(dest="model_drift_command")
    for name, text in (("status", "Show the last result per watched model (free)"),
                       ("estimate", "Show tokens and dollars per check before spending anything (free)"),
                       ("run", "Run a check now (spends tokens, up to your cap)")):
        p = subs.add_parser(name, help=text)
        p.add_argument("--target", default="", help="provider:model of one watched model")
        p.add_argument("--json", action="store_true", help="print the raw result as JSON")
    p = subs.add_parser("schedule", help="Run the check on a schedule. The cron job runs a script and does not call a model.")
    p.add_argument("--at", default=service.DEFAULT_SCHEDULE,
                   help='cron expression or phrase, default "0 9 * * *" (09:00 daily)')
    p.add_argument("--deliver", default="local",
                   help="where results go: telegram, slack, discord, ... (home channel), platform:chat_id, or local")
    subs.add_parser("unschedule", help="Remove the scheduled check and its script (history is kept)")
    subs.add_parser("targets", help="List models in your config and which ones are watched")
    p = subs.add_parser("reset-baseline", help="Start a new baseline (after you change models on purpose)")
    p.add_argument("--target", default="", help="provider:model of one watched model")
    subs.add_parser("validate-suite", help="Check the question files, including your own")
    parser.set_defaults(func=_cli)


def _cli(args):
    cmd = getattr(args, "model_drift_command", None) or "status"
    env = _env()
    if cmd in ("status", "estimate", "run"):
        result = getattr(service, cmd)(env, getattr(args, "target", ""))
        if getattr(args, "json", False):
            print(service.to_json(result))
        else:
            print(result.get("report") or result["message"])
        return 0 if result.get("status") not in ("refused", "error") else 1
    if cmd == "schedule":
        print(service.schedule(args.at, args.deliver))
    elif cmd == "unschedule":
        print(service.unschedule())
    elif cmd == "targets":
        print(service.targets_text(env))
    elif cmd == "reset-baseline":
        print(service.reset_baseline(env, args.target))
    elif cmd == "validate-suite":
        print(service.validate_suite(env))
    return 0


def register(ctx):
    _ENV["ctx"] = ctx
    _ENV.pop("env", None)
    ctx.register_tool(name="model_drift", toolset="model_drift", schema=TOOL_SCHEMA, handler=_tool, emoji="📉")
    ctx.register_command("model-drift-watch", handler=_slash, description="Check whether your model got quietly worse",
                         args_hint="[status|estimate|run|targets]")
    ctx.register_cli_command(name="model-drift-watch", help="Watch your models for quiet quality drops",
                             setup_fn=_cli_setup, handler_fn=_cli,
                             description="Fixed questions, exact grading, an exact statistical test against each model's own baseline.")
    try:
        from pathlib import Path
        ctx.register_skill(
            "model-quality",
            Path(__file__).resolve().parent / "skills" / "model-quality" / "SKILL.md",
            description=(
                "Check whether your Hermes model has got worse. "
                "Before web_search, use tool_search in English to find model_drift. "
                "The first call is action status, which is free."
            ),
        )
    except Exception:
        import logging
        logging.getLogger(__name__).warning("model-drift skill was not registered", exc_info=True)
