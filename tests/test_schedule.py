"""Schedule through the public hermes cron command. An existing job keeps its id."""

import argparse
import io
import json
import os
import runpy
import stat
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def home_resolver_without_hermes(monkeypatch):
    """The offline suite has no Hermes install. Schedule tests still need a home directory."""
    try:
        import hermes_constants  # noqa: F401
    except ImportError:
        import model_drift_plugin.public_cron as pc

        def fallback():
            chosen = os.environ.get("HERMES_HOME", "").strip()
            if chosen:
                return chosen
            return str(Path(os.environ.get("HOME", "")) / ".hermes")

        monkeypatch.setattr(pc, "resolved_hermes_home", fallback)


class Mem:
    """In-memory stand-in for hermes cron. It records every call and never starts a process."""

    def __init__(self, jobs=None, fail_list="", fail_edit=False, fail_create=False, fail_remove=False):
        self.jobs = [dict(j) for j in (jobs or [])]
        self.calls = []
        self.fail_list = fail_list
        self.fail_edit = fail_edit
        self.fail_create = fail_create
        self.fail_remove = fail_remove

    def load_jobs(self):
        self.calls.append(("list",))
        if self.fail_list:
            from model_drift_plugin import public_cron
            raise public_cron.CronUnread(self.fail_list)
        return [dict(j) for j in self.jobs]

    def create_job(self, *, schedule, name, deliver, script):
        self.calls.append(("create", schedule, name, deliver, script))
        if self.fail_create:
            raise RuntimeError("create failed")
        job = {"id": "new1", "name": name, "schedule": schedule, "deliver": deliver, "script": script}
        self.jobs.append(job)
        return dict(job)

    def update_job(self, job_id, updates):
        self.calls.append(("edit", job_id, dict(updates)))
        if self.fail_edit:
            return None
        for job in self.jobs:
            if job.get("id") == job_id:
                job.update(updates)
                return dict(job)
        return None

    def remove_job(self, job_id):
        self.calls.append(("remove", job_id))
        if self.fail_remove:
            return False
        self.jobs = [j for j in self.jobs if j.get("id") != job_id]
        return True


def _script(mods):
    return Path(os.environ["HERMES_HOME"]) / "scripts" / mods.service.SCRIPT_NAME


def _names(mem, name):
    return [j for j in mem.jobs if j.get("name") == name]


def test_public_create_and_edit_use_the_cli_flags(mods):
    seen = []

    def run(args):
        seen.append(list(args))
        if args[0] == "create":
            return mods.service.public_cron.Cmd(0, "Created job: abc123\n", "")
        if args[0] == "list":
            text = (
                "  old1 [active]\n"
                "    Name:      model-drift-watch\n"
                "    Schedule:  0 9 * * *\n"
                "    Deliver:   local\n"
            )
            return mods.service.public_cron.Cmd(0, "\x1b[33m" + text + "\x1b[0m", "")
        return mods.service.public_cron.Cmd(0, "", "")

    store = mods.service.public_cron.Store(run=run)
    created = store.create_job(
        schedule="0 9 * * *", name="model-drift-watch", deliver="local", script="model-drift-watch.py")
    assert created["id"] == "abc123"
    assert seen[0] == [
        "create", "0 9 * * *", "--name", "model-drift-watch", "--deliver", "local",
        "--no-agent", "--script", "model-drift-watch.py",
    ]
    updated = store.update_job("old1", {
        "schedule": "0 10 * * *", "deliver": "telegram", "script": "model-drift-watch.py", "no_agent": True,
    })
    assert updated["id"] == "old1"
    assert seen[1] == [
        "edit", "old1", "--schedule", "0 10 * * *", "--deliver", "telegram",
        "--script", "model-drift-watch.py", "--no-agent",
    ]
    listed = store.load_jobs()
    assert listed == [{"id": "old1", "name": "model-drift-watch", "schedule": "0 9 * * *", "deliver": "local"}]
    assert seen[2][:2] == ["list", "--all"]


def test_unread_list_is_not_an_empty_list(mods):
    def run(_args):
        return mods.service.public_cron.Cmd(1, "", "cron list failed")

    store = mods.service.public_cron.Store(run=run)
    with pytest.raises(mods.service.public_cron.CronUnread):
        store.load_jobs()


def test_existing_job_is_edited_and_not_duplicated(mods):
    mem = Mem(jobs=[{"id": "keep-me", "name": "model-drift-watch", "schedule": "0 8 * * *", "deliver": "local"}])
    text = mods.service.schedule("0 10 * * *", "telegram", cron=mem)
    assert "keep-me" in text
    assert "existing job" in text
    named = _names(mem, "model-drift-watch")
    assert [j["id"] for j in named] == ["keep-me"]
    assert named[0]["script"] == "model-drift-watch.py"
    assert named[0]["no_agent"] is True
    assert named[0]["schedule"] == "0 10 * * *"
    assert named[0]["deliver"] == "telegram"
    assert not any(call[0] == "create" for call in mem.calls)
    assert not any(call[0] == "remove" for call in mem.calls)
    script = _script(mods)
    assert script.is_file()
    body = script.read_text(encoding="utf-8")
    assert "from cron import" not in body and "import cron" not in body
    assert "model-drift-watch" in body and "run" in body and "--json" in body
    again = mods.service.schedule("0 11 * * *", "local", cron=mem)
    assert "keep-me" in again
    assert [j["id"] for j in _names(mem, "model-drift-watch")] == ["keep-me"]
    assert sum(1 for call in mem.calls if call[0] == "create") == 0


def test_list_failure_changes_nothing(mods):
    mem = Mem(jobs=[{"id": "keep-me", "name": "model-drift-watch"}], fail_list="list failed")
    text = mods.service.schedule(cron=mem)
    assert "could not be read" in text
    assert "Nothing was changed" in text
    assert mem.calls == [("list",)]
    assert not _script(mods).exists()
    gone = mods.service.unschedule(cron=mem)
    assert "Nothing was removed" in gone
    assert mem.jobs[0]["id"] == "keep-me"


def test_edit_failure_leaves_the_job(mods):
    mem = Mem(jobs=[{"id": "keep-me", "name": "model-drift-watch"}], fail_edit=True)
    text = mods.service.schedule(cron=mem)
    assert "could not be switched" in text
    assert "not removed" in text
    assert [j["id"] for j in mem.jobs] == ["keep-me"]
    assert not any(call[0] == "create" for call in mem.calls)
    assert _script(mods).is_file()


def test_create_failure_removes_the_script(mods):
    mem = Mem(fail_create=True)
    text = mods.service.schedule(cron=mem)
    assert "No job was created" in text
    assert mem.jobs == []
    assert not _script(mods).exists()


def test_unschedule_removes_job_and_script_and_keeps_history(mods):
    mem = Mem(jobs=[{"id": "keep-me", "name": "model-drift-watch"}])
    assert mods.service.write_watch_script() is None
    history = Path(os.environ["HERMES_HOME"]) / "plugin-data" / "model-drift-watch" / "runs.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text("keep\n", encoding="utf-8")
    text = mods.service.unschedule(cron=mem)
    assert "keep-me" in text
    assert "History is kept" in text
    assert mem.jobs == []
    assert not _script(mods).exists()
    assert history.read_text(encoding="utf-8") == "keep\n"


def test_missing_job_removes_a_leftover_script_and_keeps_history(mods):
    mem = Mem()
    assert mods.service.write_watch_script() is None
    history = Path(os.environ["HERMES_HOME"]) / "plugin-data" / "model-drift-watch" / "runs.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text("keep\n", encoding="utf-8")
    text = mods.service.unschedule(cron=mem)
    assert "not scheduled" in text
    assert "No cron job was found" in text
    assert not _script(mods).exists()
    assert history.read_text(encoding="utf-8") == "keep\n"


def test_duplicate_names_are_left_alone(mods):
    mem = Mem(jobs=[
        {"id": "one", "name": "model-drift-watch"},
        {"id": "two", "name": "model-drift-watch"},
    ])
    scheduled = mods.service.schedule(cron=mem)
    removed = mods.service.unschedule(cron=mem)
    assert "one" in scheduled and "two" in scheduled and "Nothing was changed" in scheduled
    assert "one" in removed and "two" in removed and "Nothing was removed" in removed
    assert [j["id"] for j in mem.jobs] == ["one", "two"]
    assert mem.calls == [("list",), ("list",)]
    assert not _script(mods).exists()


def test_status_distinguishes_unread_from_absent(mods, tmp_path):
    env = mods.service.Env(
        get_config=lambda k, d: d, llm=None, store=mods.store.Store(tmp_path),
        hermes_config=lambda: {"model": {"provider": "custom", "default": "mock-model"}},
    )
    absent = mods.service.status(env, cron=Mem())
    assert "Not scheduled" in absent["message"]
    assert absent["scheduled"] is False
    unread = mods.service.status(env, cron=Mem(fail_list="list failed"))
    assert "could not be read" in unread["message"]
    assert "Not scheduled" not in unread["message"]
    assert unread["scheduled"] is False


def test_cli_has_no_agent_model_flag(plugin):
    parser = argparse.ArgumentParser()
    plugin._cli_setup(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["schedule", "--agent-model", "other"])
    with pytest.raises(SystemExit):
        parser.parse_args(["schedule", "--agent-provider", "other"])
    args = parser.parse_args(["schedule", "--at", "0 10 * * *", "--deliver", "local"])
    assert args.at == "0 10 * * *" and args.deliver == "local"


def test_source_outside_tests_does_not_import_the_cron_store():
    hits = []
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if "from cron import" in line or "import cron" in line:
                hits.append(f"{path.name}:{i}:{line.strip()}")
    assert hits == []


def test_home_resolver_follows_the_env_and_the_platform_default(mods, tmp_path, monkeypatch):
    pytest.importorskip("hermes_constants")
    chosen = tmp_path / "chosen"
    monkeypatch.setenv("HERMES_HOME", str(chosen))
    assert mods.service.public_cron.resolved_hermes_home() == str(chosen)
    assert mods.service.write_watch_script() is None
    assert (chosen / "scripts" / "model-drift-watch.py").is_file()

    platform = tmp_path / "platform-user"
    platform.mkdir()
    monkeypatch.setenv("HOME", str(platform))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert mods.service.public_cron.resolved_hermes_home() == str(platform / ".hermes")


def _fake_hermes(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True)
    python = directory / "python3"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hermes = directory / "hermes"
    hermes.write_text(body, encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    hermes.chmod(hermes.stat().st_mode | stat.S_IEXEC)
    return python


def _run_watch_script(script: Path, python: Path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(python))
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with redirect_stdout(out), redirect_stderr(err):
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


def test_watch_script_prints_a_message_and_stays_quiet(mods, tmp_path, monkeypatch):
    assert mods.service.write_watch_script() is None
    script = _script(mods)
    hermes = """#!/bin/sh
printf '%s\\n' "$MDW_OUT"
exit "$MDW_CODE"
"""
    python = _fake_hermes(tmp_path / "pybin", hermes)
    monkeypatch.setenv("MDW_OUT", json.dumps({"message": "[SILENT]"}))
    monkeypatch.setenv("MDW_CODE", "0")
    code, out, _err = _run_watch_script(script, python, monkeypatch)
    assert code == 0 and out == ""

    monkeypatch.setenv("MDW_OUT", json.dumps({"message": ""}))
    code, out, _err = _run_watch_script(script, python, monkeypatch)
    assert code == 0 and out == ""

    monkeypatch.setenv("MDW_OUT", json.dumps({"message": "DRIFT happened"}))
    code, out, _err = _run_watch_script(script, python, monkeypatch)
    assert code == 0 and out == "DRIFT happened\n"

    monkeypatch.setenv("MDW_OUT", "not json")
    monkeypatch.setenv("MDW_CODE", "2")
    code, out, err = _run_watch_script(script, python, monkeypatch)
    assert code == 2 and out == "" and err.strip()


def test_real_hermes_keeps_one_job_id(mods):
    """Create, edit, and remove one job with the real hermes cron. Do not run the check."""
    binary = mods.service.public_cron.hermes_binary()
    if not binary:
        pytest.skip("hermes is not installed next to this Python or on PATH")
    home = os.environ["HERMES_HOME"]
    env = os.environ.copy()
    env.update({
        "HERMES_HOME": home,
        "HOME": os.environ["HOME"],
        "NO_COLOR": "1",
        "TERM": "dumb",
        "HERMES_DISABLE_UPDATE_CHECK": "1",
    })

    def cron(*args):
        done = subprocess.run(
            [binary, "cron", *args], capture_output=True, text=True, env=env, timeout=60,
        )
        return done.returncode, (done.stdout or "") + (done.stderr or "")

    code, out = cron(
        "create", "0 9 * * *", "old agent prompt",
        "--name", "model-drift-watch", "--deliver", "local",
    )
    if code != 0:
        pytest.skip(f"hermes cron create did not run here: {out.strip()[:180]}")
    created = mods.service.public_cron.parse_created_id(out)
    assert created
    text = mods.service.schedule("0 10 * * *", "local")
    assert created in text
    jobs_path = Path(home) / "cron" / "jobs.json"
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")).get("jobs") or []
    named = [j for j in jobs if j.get("name") == "model-drift-watch"]
    assert [j.get("id") for j in named] == [created]
    assert named[0].get("no_agent") is True
    assert named[0].get("script") == "model-drift-watch.py"
    again = mods.service.schedule("0 11 * * *", "telegram")
    assert created in again
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")).get("jobs") or []
    named = [j for j in jobs if j.get("name") == "model-drift-watch"]
    assert [j.get("id") for j in named] == [created]
    shown = str(named[0].get("schedule_display") or "")
    raw = named[0].get("schedule") or {}
    expr = str(raw.get("expr") or raw.get("expression") or raw.get("value") or "")
    assert "11" in shown or "11" in expr
    deliver = named[0].get("deliver")
    assert deliver == "telegram" or deliver == ["telegram"]
    history = Path(home) / "plugin-data" / "model-drift-watch" / "runs.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text("keep\n", encoding="utf-8")
    removed = mods.service.unschedule()
    assert created in removed
    jobs = json.loads(jobs_path.read_text(encoding="utf-8")).get("jobs") or [] if jobs_path.exists() else []
    assert [j for j in jobs if j.get("name") == "model-drift-watch"] == []
    assert not _script(mods).exists()
    assert history.read_text(encoding="utf-8") == "keep\n"
