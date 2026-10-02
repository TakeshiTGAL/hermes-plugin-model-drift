"""What the tool, the slash command and the CLI share."""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from . import cost, report, runner, sandbox, targets
from .store import Store, default_data_dir
from .suite import load_suite

JOB_NAME = "model-drift-watch"
DEFAULT_SCHEDULE = "0 9 * * *"
CRON_PROMPT = (
    'Run the model_drift tool once with arguments {"action": "run"}. If model_drift is not in your tool '
    'list, it is a deferred tool: invoke it directly with tool_call, calls=[{"name": "model_drift", '
    '"arguments": {"action": "run"}}], without searching first. Then reply with exactly the text of the '
    '"message" field of its result and nothing else: no summary, no comment, no formatting. If that text '
    "is [SILENT], reply with exactly [SILENT]."
)
SEVERITY = ["busy", "stable", "improved", "unconfirmed", "collecting", "baseline_ready", "suspect", "token_shift",
            "refused", "measurement_failed", "model_changed", "drift"]


@dataclass
class Env:
    """Everything a command needs, injectable for tests."""
    get_config: Callable[[str, Any], Any]
    llm: Any
    store: Store
    hermes_config: Callable[[], Dict[str, Any]] = targets.load_config
    price_lookup: Optional[Callable[[str, str], Any]] = None
    caller: Optional[runner.Caller] = None


def _settings(env: Env):
    return runner.Settings.from_getter(env.get_config)


def _suite(env: Env, s):
    suite = load_suite([env.store.suites_dir], s.categories)
    if s.max_tokens_scale != 1.0:  # part of each question's key: changing it starts a new baseline
        suite.items = [dataclasses.replace(i, max_tokens=max(16, int(i.max_tokens * s.max_tokens_scale)))
                       for i in suite.items]
    return suite


def _targets(env: Env, s, only: str = "") -> List[targets.Target]:
    ts = targets.monitored(env.hermes_config(), s.extra_targets)
    if only:
        ts = [t for t in ts if t.id == only or t.model == only]
    return ts


def _price(env: Env, s, t: targets.Target):
    return cost.lookup_price(t.provider, t.model, s.price_input_per_mtok, s.price_output_per_mtok, env.price_lookup)


def _no_target(only: str) -> Dict[str, Any]:
    if only:
        msg = (f"[model-drift-watch] {only} is not a watched model. Watched models are your main model plus "
               "extra_targets; see `hermes model-drift-watch targets`.")
    else:
        msg = "[model-drift-watch] No model is configured in Hermes yet. Pick one with `hermes model`, then try again."
    return {"status": "refused", "message": msg, "checks": []}


HERMES_DEFAULT_TOOL_LIMIT_S = 420.0  # agent/tool_executor.py _DEFAULT_CONCURRENT_TOOL_TIMEOUT_S


def hermes_tool_limit(cfg: Dict[str, Any]) -> Optional[float]:
    """Seconds Hermes allows one tool call, resolved in Hermes's own order: timeouts.tools.sequential_call,
    then timeouts.tools.concurrent_batch, then HERMES_CONCURRENT_TOOL_TIMEOUT_S, then Hermes's default
    (420 s). None = the user turned the limit off (0 or negative)."""
    import os
    tools = ((cfg or {}).get("timeouts") or {}).get("tools") or {}
    for raw in (tools.get("sequential_call"), tools.get("concurrent_batch"),
                os.environ.get("HERMES_CONCURRENT_TOOL_TIMEOUT_S")):
        if raw is None or isinstance(raw, bool) or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        return value if value > 0 else None  # 0 or negative disables the limit in Hermes
    return HERMES_DEFAULT_TOOL_LIMIT_S


def run(env: Env, only: str = "") -> Dict[str, Any]:
    s, problems = _settings(env)
    suite = _suite(env, s)
    ts = _targets(env, s, only)
    if not ts:
        return _no_target(only)
    if suite.errors and not suite.items:
        return {"status": "refused", "message": "[model-drift-watch] The question set has errors:\n" + "\n".join(suite.errors[:8]),
                "checks": []}
    run_code = sandbox.run_python if s.code_execution else None
    call = env.caller or runner.ctx_llm_caller(env.llm)
    outcomes = []
    # one deadline for the whole tool call, every watched model included: Hermes stops a tool call
    # after timeouts.tools.sequential_call (420 s by default), and a late answer would be lost
    budget_s = s.check_deadline
    limit = hermes_tool_limit(env.hermes_config())
    if limit:  # never outlast the user's own Hermes tool-call limit
        budget_s = min(budget_s, max(20.0, limit - 30))
    deadline = time.monotonic() + budget_s
    for t in ts:
        outcomes.append(runner.run_check(call, t, s, suite.items, env.store, _price(env, s, t), run_code,
                                         deadline=deadline))
    worst = max(outcomes, key=lambda o: SEVERITY.index(o.status) if o.status in SEVERITY else 0)
    notes = problems + [f"Question file problem: {e}" for e in suite.errors[:5]]
    full = "\n\n".join([o.message for o in outcomes] + notes)
    loud = [o.message for o in outcomes if not o.silent]
    # message: what a scheduled run delivers ([SILENT] = deliver nothing); report: always the full text
    message = "\n\n".join(loud + notes) if loud else "[SILENT]"
    return {"status": worst.status, "message": message, "report": full, "checks": [o.details for o in outcomes]}


def estimate(env: Env, only: str = "") -> Dict[str, Any]:
    s, problems = _settings(env)
    suite = _suite(env, s)
    ts = _targets(env, s, only)
    if not ts:
        return _no_target(only)
    items = [i for i in suite.items if s.code_execution or i.grader["type"] != "python"]
    parts, data = [], []
    for t in ts:
        est = cost.estimate(items, s.samples_per_item, _price(env, s, t), env.store.runs(t.id))
        parts.append(report.estimate_text(est, s, t.id, len(items)))
        data.append({"target": t.id, **est.to_dict()})
    msg = "\n\n".join(parts + problems + suite.errors[:5])
    return {"status": "ok", "message": msg, "estimates": data}


def status(env: Env, only: str = "") -> Dict[str, Any]:
    s, problems = _settings(env)
    ts = _targets(env, s, only)
    if not ts:
        return _no_target(only)
    lines, data = [], []
    for t in ts:
        last = env.store.last_check(t.id)
        if not last:
            lines.append(f"{t.id}: no checks yet. Run `hermes model-drift-watch run`, or schedule daily checks with "
                         "`hermes model-drift-watch schedule --deliver telegram`.")
            data.append({"target": t.id, "status": "never_run"})
            continue
        lines.append(f"{t.id}: {last['status']} at {last['ts']}\n{last.get('message', '')}")
        data.append({"target": t.id, **{k: last.get(k) for k in ("status", "ts", "accuracy", "spent", "estimate")}})
    job = find_job()
    if job:
        lines.append(f"Scheduled: {job.get('schedule_display') or job.get('schedule')} -> {job.get('deliver')} "
                     f"(cron job {job.get('id')}, next run {job.get('next_run_at') or 'unknown'}).")
    else:
        lines.append("Not scheduled. Schedule daily checks with `hermes model-drift-watch schedule --deliver telegram` "
                     "(or slack, discord, ... - any platform with a home channel).")
    return {"status": "ok", "message": "\n\n".join(lines + problems), "targets": data, "scheduled": bool(job)}


def targets_text(env: Env) -> str:
    s, _ = _settings(env)
    cfg = env.hermes_config()
    found = targets.discover(cfg)
    watched = {(t.provider, t.model) for t in targets.monitored(cfg, s.extra_targets)}
    if not found and not watched:
        return "No models found in your Hermes config. Pick one with `hermes model`."
    lines = ["Models in your Hermes config:"]
    for t in found:
        mark = "watched" if (t.provider, t.model) in watched else "not watched"
        lines.append(f"  {t.id}  ({t.role}, {mark})")
    extra = [t for t in targets.monitored(cfg, s.extra_targets) if t.override]
    for t in extra:
        if all((t.provider, t.model) != (f.provider, f.model) for f in found):
            lines.append(f"  {t.id}  (extra_targets, watched)")
    lines.append("")
    lines.append("Your main model is watched with no extra setup. To also watch another model, add it to "
                 "extra_targets, e.g.:")
    lines.append(f"  {report.SET}extra_targets '[\"openrouter:deepseek/deepseek-chat\"]'")
    snippet = targets.grant_snippet(extra)
    if snippet:
        lines.append("")
        lines.append("Hermes only lets a plugin pick a non-active model after you allow it. Add this to config.yaml:")
        lines.append(snippet)
    return "\n".join(lines)


def reset_baseline(env: Env, only: str = "", reason: str = "manual reset") -> str:
    s, _ = _settings(env)
    ts = _targets(env, s, only)
    if not ts:
        return _no_target(only)["message"]
    now = runner._now()
    for t in ts:
        env.store.reset_baseline(t.id, now, reason)
    names = ", ".join(t.id for t in ts)
    return (f"New baseline started for {names}. The next {s.baseline_runs} successful runs form it; "
            "earlier runs stay in the history but are no longer compared against.")


# --- scheduling through Hermes cron -------------------------------------------------------------------------
def find_job() -> Optional[Dict[str, Any]]:
    try:
        from cron import jobs
        return next((j for j in jobs.list_jobs(include_disabled=True) if j.get("name") == JOB_NAME), None)
    except Exception:
        return None


def schedule(when: str = DEFAULT_SCHEDULE, deliver: str = "local", model: str = "", provider: str = "") -> str:
    try:
        from cron import jobs
    except Exception as exc:
        return f"Hermes cron is not available here ({exc}). Use `hermes cron create` instead; see the README."
    old = find_job()
    if old:
        jobs.remove_job(old["id"])
    kw: Dict[str, Any] = dict(prompt=CRON_PROMPT, schedule=when, name=JOB_NAME, deliver=deliver,
                              enabled_toolsets=["model_drift", "no_mcp"])
    if model:
        kw["model"] = model
        if provider:
            kw["provider"] = provider
    try:
        job = jobs.create_job(**kw)
    except ValueError as exc:
        return f"Could not schedule: {exc}. Examples of valid schedules: \"0 9 * * *\" (09:00 daily), \"every 12h\"."
    where = ("saved locally only (see `hermes cron list`); add --deliver telegram (or slack, discord, ...) "
             "to get messages" if deliver == "local" else f"sent to {deliver}")
    return (f"{'Re-scheduled' if old else 'Scheduled'} model-drift-watch: {job.get('schedule_display') or when}, results {where}. "
            f"You only hear about drift, changed models, failed measurements and the baseline becoming ready. "
            f"Cron job id {job.get('id')}; it needs the Hermes gateway running (`hermes gateway`).")


def unschedule() -> str:
    old = find_job()
    if not old:
        return "model-drift-watch is not scheduled."
    from cron import jobs
    jobs.remove_job(old["id"])
    return f"Removed the scheduled model-drift-watch check (cron job {old['id']}). History is kept."


def validate_suite(env: Env) -> str:
    s, problems = _settings(env)
    suite = _suite(env, s)
    by_cat: Dict[str, int] = {}
    for i in suite.items:
        by_cat[i.category] = by_cat.get(i.category, 0) + 1
    lines = [f"{len(suite.items)} questions: " + ", ".join(f"{k} {v}" for k, v in sorted(by_cat.items())),
             f"Your question files go in: {env.store.suites_dir}"]
    if suite.files:
        lines.append("Loaded: " + ", ".join(suite.files))
    bad = []
    run_code = sandbox.run_python if s.code_execution else None
    from .graders import grade
    for item in suite.items:
        if item.reference:
            g = grade(item, item.reference, run_code)
            if not g.passed and not g.skipped:
                bad.append(f"{item.id}: its own reference answer fails ({g.reason}).")
    lines += suite.errors + bad + problems
    if not suite.errors and not bad:
        lines.append("All questions are valid" + (" and every reference answer passes its grader." if any(
            i.reference for i in suite.items) else "."))
    return "\n".join(lines)


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
