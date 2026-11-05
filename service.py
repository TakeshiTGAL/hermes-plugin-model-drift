"""What the tool, the slash command and the CLI share."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import cost, public_cron, report, runner, sandbox, targets
from .store import Store, default_data_dir
from .suite import load_suite

JOB_NAME = "model-drift-watch"
DEFAULT_SCHEDULE = "0 9 * * *"
SCRIPT_NAME = "model-drift-watch.py"
# Hermes runs this file with its own Python, from HERMES_HOME/scripts, and delivers
# stdout as the cron message. Empty stdout sends nothing. The check itself is the
# public CLI, which calls the model through ctx.llm. This process does not.
WATCH_SCRIPT = '''"""model-drift-watch. Hermes delivers stdout and sends nothing when it is empty."""
import json
import os
import shutil
import subprocess
import sys


def _hermes():
    sibling = os.path.join(os.path.dirname(sys.executable), "hermes")
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    return shutil.which("hermes")


def _payload(text):
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def main():
    binary = _hermes()
    if not binary:
        print("The hermes command was not found, so the check did not run.", file=sys.stderr)
        raise SystemExit(1)
    # copy() keeps PYTHONPATH. Hermes 0.21.6 and later pass dependency locations that way.
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    env["HERMES_DISABLE_UPDATE_CHECK"] = "1"
    try:
        done = subprocess.run(
            [binary, "model-drift-watch", "run", "--json"],
            capture_output=True,
            text=True,
            env=env,
            timeout=3300,
        )
    except subprocess.TimeoutExpired:
        print("The check timed out.", file=sys.stderr)
        raise SystemExit(1)
    if done.returncode not in (0, 1):
        err = (done.stderr or done.stdout or "the check did not finish").strip()
        print(err.splitlines()[0] if err else "the check did not finish", file=sys.stderr)
        raise SystemExit(done.returncode or 1)
    payload = _payload(done.stdout)
    if not isinstance(payload, dict):
        print("The check did not return JSON.", file=sys.stderr)
        raise SystemExit(1)
    message = payload.get("message")
    if not isinstance(message, str) or message == "" or message == "[SILENT]":
        return
    sys.stdout.write(message)
    if not message.endswith("\\n"):
        sys.stdout.write("\\n")


if __name__ == "__main__":
    main()
'''
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
        msg = ("[model-drift-watch] No model is configured in Hermes yet. "
               "Pick one with `hermes model`, then run `hermes model-drift-watch status` again.")
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


def status(env: Env, only: str = "", cron: Any = None) -> Dict[str, Any]:
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
    job, unread = find_job(cron)
    if unread:
        lines.append(f"The schedule could not be read ({unread}).")
    elif job:
        lines.append(f"Scheduled: {job.get('schedule') or 'unknown'} -> {job.get('deliver') or 'unknown'} "
                     f"(cron job {job.get('id')}).")
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


# --- scheduling through the public hermes cron command -------------------------------------------------------
def _named(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [j for j in jobs if j.get("name") == JOB_NAME and j.get("id")]


def find_job(cron: Any = None) -> tuple:
    """(job or None, unread detail or None). An unread list is not an empty list."""
    store = cron if cron is not None else public_cron.Store()
    try:
        listed = store.load_jobs()
    except public_cron.CronUnread as exc:
        return None, exc.detail
    return next((j for j in listed if j.get("name") == JOB_NAME), None), None


def watch_script_path() -> Optional[Path]:
    try:
        home = public_cron.resolved_hermes_home()
    except (OSError, RuntimeError, ImportError):
        return None
    if not home:
        return None
    return Path(home) / "scripts" / SCRIPT_NAME


def write_watch_script() -> Optional[str]:
    """Write the relative script Hermes cron will run. None means it is in place."""
    path = watch_script_path()
    if path is None:
        return "The Hermes home could not be resolved, so the watch script was not written."
    payload = WATCH_SCRIPT.encode("utf-8")
    fd = None
    tmp = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".watch-", suffix=".tmp")
        os.write(fd, payload)
        os.close(fd)
        fd = None
        os.replace(tmp, path)
        tmp = ""
    except OSError:
        return "The watch script could not be written, so the job was not changed."
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return None


def remove_watch_script() -> Optional[str]:
    """Delete the watch script. None means it is gone, including when it was already absent."""
    path = watch_script_path()
    if path is None:
        return "The Hermes home could not be resolved, so the watch script was not removed."
    if not path.is_file():
        return None
    try:
        path.unlink()
    except OSError:
        return "The watch script could not be removed."
    return None


def _where(deliver: str) -> str:
    if deliver == "local":
        return "saved locally only (see `hermes cron list`). Pass --deliver telegram (or slack, discord, ...) to get messages"
    return f"sent to {deliver}"


def schedule(when: str = DEFAULT_SCHEDULE, deliver: str = "local", cron: Any = None) -> str:
    store = cron if cron is not None else public_cron.Store()
    try:
        listed = store.load_jobs()
    except public_cron.CronUnread as exc:
        return f"The cron job list could not be read ({exc.detail}). Nothing was changed."
    named = _named(listed)
    if len(named) > 1:
        ids = ", ".join(str(j["id"]) for j in named)
        return (f"More than one cron job is named {JOB_NAME} ({ids}). "
                "Nothing was changed, so a second job was not added and none of them was removed.")
    script_err = write_watch_script()
    if script_err:
        return script_err if script_err.endswith("The job was not changed.") else script_err + " The job was not changed."
    where = _where(deliver)
    if named:
        job_id = str(named[0]["id"])
        updated = store.update_job(job_id, {
            "schedule": when,
            "deliver": deliver,
            "script": SCRIPT_NAME,
            "no_agent": True,
        })
        if not updated:
            return (f"The existing job {job_id} could not be switched to the watch script. "
                    "The job was not removed. "
                    f"Switch it with `hermes cron edit {job_id} --script {SCRIPT_NAME} --no-agent`.")
        return (f"Scheduled {JOB_NAME} on the existing job {job_id}: {when}, results {where}. "
                "The job runs a script and does not call a model. Empty script output sends nothing. "
                "You only hear about drift, changed models, failed measurements and the baseline becoming ready. "
                "Running schedule again updates this same job and does not add another. "
                "It needs the Hermes gateway running (`hermes gateway`). "
                f"Run `hermes model-drift-watch unschedule` before removing the plugin.")
    try:
        created = store.create_job(schedule=when, name=JOB_NAME, deliver=deliver, script=SCRIPT_NAME)
    except (RuntimeError, OSError) as exc:
        remove_watch_script()
        return (f"Could not schedule: {exc}. Examples of valid schedules: \"0 9 * * *\" (09:00 daily), \"every 12h\". "
                "No job was created.")
    return (f"Scheduled {JOB_NAME}: {created.get('schedule') or when}, results {where}. "
            "The job runs a script and does not call a model. Empty script output sends nothing. "
            "You only hear about drift, changed models, failed measurements and the baseline becoming ready. "
            f"Cron job id {created.get('id')}. Running schedule again updates this same job and does not add another. "
            "It needs the Hermes gateway running (`hermes gateway`). "
            f"Run `hermes model-drift-watch unschedule` before removing the plugin.")


def unschedule(cron: Any = None) -> str:
    store = cron if cron is not None else public_cron.Store()
    try:
        listed = store.load_jobs()
    except public_cron.CronUnread as exc:
        return f"The cron job list could not be read ({exc.detail}). Nothing was removed."
    named = _named(listed)
    if len(named) > 1:
        ids = ", ".join(str(j["id"]) for j in named)
        return (f"More than one cron job is named {JOB_NAME} ({ids}). "
                "Nothing was removed.")
    if not named:
        script_err = remove_watch_script()
        if script_err:
            return script_err + " History was left in place."
        return f"{JOB_NAME} is not scheduled. No cron job was found. The watch script is gone. History was left in place."
    job_id = str(named[0]["id"])
    if not store.remove_job(job_id):
        return (f"The cron job {job_id} could not be removed. The watch script was left in place. "
                "History was left in place.")
    script_err = remove_watch_script()
    if script_err:
        return script_err + f" The job {job_id} was removed. History was left in place."
    return f"Removed the scheduled {JOB_NAME} check (cron job {job_id}). The watch script was removed. History is kept."


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
