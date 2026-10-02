"""Append-only history under Hermes's per-plugin data folder (survives plugin updates)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

LOCK_STALE_S = 15 * 60


def default_data_dir() -> Path:
    try:
        from plugins.plugin_storage import plugin_data_dir  # Hermes's sanctioned plugin data root
        return Path(plugin_data_dir("model-drift-watch"))
    except Exception:  # running outside Hermes (tests, plain python)
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        path = home / "plugin-data" / "model-drift-watch"
        path.mkdir(parents=True, exist_ok=True)
        return path


def target_slug(target_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", target_id)[:48].strip("_") or "target"
    return f"{readable}-{hashlib.sha256(target_id.encode('utf-8')).hexdigest()[:8]}"


class Busy(RuntimeError):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def suites_dir(self) -> Path:
        return self.root / "suites"

    def _dir(self, target_id: str) -> Path:
        d = self.root / "targets" / target_slug(target_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- runs (one line per pass) and checks (one line per check) ---------------------------------
    def append_run(self, target_id: str, record: Dict[str, Any]) -> None:
        self._append(self._dir(target_id) / "runs.jsonl", record)

    def append_check(self, target_id: str, record: Dict[str, Any]) -> None:
        self._append(self._dir(target_id) / "checks.jsonl", record)

    def runs(self, target_id: str) -> List[Dict[str, Any]]:
        return self._read(self._dir(target_id) / "runs.jsonl")

    def checks(self, target_id: str) -> List[Dict[str, Any]]:
        return self._read(self._dir(target_id) / "checks.jsonl")

    def last_check(self, target_id: str) -> Optional[Dict[str, Any]]:
        checks = self.checks(target_id)
        return checks[-1] if checks else None

    # --- baseline window ------------------------------------------------------------------------------
    def state(self, target_id: str) -> Dict[str, Any]:
        path = self._dir(target_id) / "state.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def reset_baseline(self, target_id: str, now_iso: str, reason: str) -> None:
        """Start a new baseline from the next run. Nothing is deleted: old runs stay in the log."""
        state = self.state(target_id)
        start = len(self.runs(target_id))  # runs are append-only, so an index is an exact boundary
        state.setdefault("resets", []).append({"at": now_iso, "reason": reason, "start_index": start})
        state["baseline_start_index"] = start
        path = self._dir(target_id) / "state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    # --- one check at a time per target ---------------------------------------------------------------
    @contextmanager
    def lock(self, target_id: str) -> Iterator[None]:
        path = self._dir(target_id) / "check.lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - path.stat().st_mtime
            if age < LOCK_STALE_S:
                raise Busy(f"another check of this model started {int(age)}s ago and is still running")
            path.unlink(missing_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _append(path: Path, record: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _read(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue  # a torn last line from a crash must not hide the rest of the history
        return out
