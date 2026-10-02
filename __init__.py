"""model-drift-watch for Hermes Agent: notice when the model you use gets quietly worse.

Registers one tool (model_drift), one slash command (/model-drift-watch) and one CLI command
(hermes model-drift-watch). Model calls go through ctx.llm, so Hermes keeps the credentials and the
plugin never sees a key. State lives in Hermes's per-plugin data folder.
"""

# Hermes imports this directory as a package. pytest's collector also imports this file on its own
# (the repo root is the package), where relative imports cannot work.
if __package__:
    from . import service
    from .store import Store, default_data_dir

TOOL_SCHEMA = {
    "name": "model_drift",
    "description": (
        "Check whether the user's LLM has quietly got worse. action=status reads the last result (free). "
        "action=estimate shows tokens and dollars before running (free). action=run asks a fixed set of "
        "exactly graded questions and tests the answers against the same model's own baseline; it spends "
        "API tokens up to the user's cap. Show the user the 'report' field as written."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "estimate", "run"],
                       "description": "status (free), estimate (free) or run (spends tokens)."},
            "target": {"type": "string",
                       "description": "Optional 'provider:model' of one watched model. Default: all watched models."},
        },
        "required": ["action"],
    },
}

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


def _tool(args, **_kwargs):
    action = str((args or {}).get("action") or "status").strip().lower()
    target = str((args or {}).get("target") or "").strip()
    try:
        if action == "run":
            return service.to_json(service.run(_env(), target))
        if action == "estimate":
            return service.to_json(service.estimate(_env(), target))
        if action == "status":
            return service.to_json(service.status(_env(), target))
        return service.to_json({"status": "error", "message": f"Unknown action {action!r}. Use status, estimate or run."})
    except Exception as exc:  # a tool must answer, never crash the agent turn
        return service.to_json({"status": "error",
                                "message": f"[model-drift-watch] Internal error ({type(exc).__name__}: {exc}). "
                                           "Please report it with `hermes logs` output."})


SLASH_HELP = ("/model-drift-watch status | estimate | run | targets\n"
              "status: last result (free). estimate: cost before running (free). run: check now (spends tokens). "
              "targets: which models are watched.")


def _slash(raw_args: str):
    sub, _, rest = (raw_args or "").strip().partition(" ")
    sub = sub.lower() or "status"
    if sub in ("status", "estimate", "run"):
        result = getattr(service, sub)(_env(), rest.strip())
        return result.get("report") or result["message"]
    if sub == "targets":
        return service.targets_text(_env())
    return SLASH_HELP


def _cli_setup(parser):
    subs = parser.add_subparsers(dest="model_drift_command")
    for name, text in (("status", "Show the last result per watched model (free)"),
                       ("estimate", "Show tokens and dollars per check before spending anything (free)"),
                       ("run", "Run a check now (spends tokens, up to your cap)")):
        p = subs.add_parser(name, help=text)
        p.add_argument("--target", default="", help="provider:model of one watched model")
        p.add_argument("--json", action="store_true", help="print the raw result as JSON")
    p = subs.add_parser("schedule", help="Run a check on a schedule through Hermes cron")
    p.add_argument("--at", default=service.DEFAULT_SCHEDULE,
                   help='cron expression or phrase, default "0 9 * * *" (09:00 daily)')
    p.add_argument("--deliver", default="local",
                   help="where results go: telegram, slack, discord, ... (home channel), platform:chat_id, or local")
    p.add_argument("--agent-model", default="",
                   help="cheap model for the scheduled agent turn that calls the tool (does not change what is tested)")
    p.add_argument("--agent-provider", default="", help="provider for --agent-model")
    subs.add_parser("unschedule", help="Remove the scheduled check (history is kept)")
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
        print(service.schedule(args.at, args.deliver, args.agent_model, args.agent_provider))
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
