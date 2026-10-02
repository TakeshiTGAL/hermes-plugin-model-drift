"""Run a code answer against its tests in a separate, locked-down Python process.

Defence in depth, not a security boundary: the questions are fixed and benign, so the risk being
managed is a buggy answer (infinite loop, huge allocation, stray file write), not an attacker.
  * a fresh interpreter in isolated mode (-I -S -B): no site-packages, no user env vars
  * an empty temporary working directory and an environment with no secrets in it
  * CPU, memory and file-size limits where the OS supports them, plus a wall-clock timeout
  * an audit hook refuses ordinary file writes, network, further processes and ctypes, and reads
    outside the Python installation and the temporary folder. Code written to get around it can
    (for example by replacing built-ins), so this is not a sandbox for hostile code. The names the
    hook uses are bound when it is installed, which stops the simplest way of doing that.
  * a pass is only counted when the process exits 0 AND prints a per-run random token last, so
    an answer that simply exits early is not a pass (code written to defeat this could still find
    the token: this is not a sandbox for hostile code)
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile

from .graders import Grade

TIMEOUT_S = 10

_RUNNER = r'''
import json, os, sys


def _harness():
    token = sys.stdin.readline().strip()
    payload = json.loads(sys.stdin.readline())
    try:
        import resource
        for lim, val in ((resource.RLIMIT_CPU, %(cpu)d), (resource.RLIMIT_FSIZE, 1 << 20),
                         (resource.RLIMIT_AS, 1 << 30)):
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass
    except ImportError:
        pass
    roots = tuple(sorted({os.path.realpath(p) + os.sep for p in (sys.prefix, sys.base_prefix, sys.exec_prefix,
                                                                 sys.base_exec_prefix, os.getcwd())}))
    blocked_events = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
                      "os.fork", "os.forkpty", "os.kill", "os.killpg", "os.remove", "os.unlink",
                      "os.rmdir", "os.rename", "os.replace", "os.truncate", "os.chmod", "os.chown",
                      "os.link", "os.symlink", "os.mkdir", "os.chdir", "shutil.", "ctypes.",
                      "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.", "pty.", "winreg.")
    blocked_modules = {"ctypes", "_ctypes", "socket", "_socket", "ssl", "_ssl", "subprocess",
                       "_posixsubprocess", "multiprocessing", "pty", "urllib", "http", "ftplib",
                       "smtplib", "webbrowser", "asyncio", "sqlite3", "_sqlite3", "dbm", "_dbm", "_gdbm"}

    # everything the hook calls is bound now, so an answer that replaces a built-in or os.path
    # later cannot change what the hook does with it
    def guard(event, args, isinstance=isinstance, any=any, str=str, bytes=bytes, len=len, realpath=os.path.realpath,
              fsdecode=os.fsdecode, sep=os.sep, roots=roots, blocked_events=blocked_events,
              blocked_modules=blocked_modules, PermissionError=PermissionError, ImportError=ImportError):
        if event == "open":
            path, mode = args[0], (args[1] if len(args) > 1 else "r")
            flags = args[2] if len(args) > 2 else 0
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & 3):
                raise PermissionError("file writes are not allowed in model-drift-watch code checks")
            if isinstance(path, (str, bytes)):
                real = realpath(fsdecode(path))
                if not (real + sep).startswith(roots) and not real.startswith(roots):
                    raise PermissionError("reading files outside Python is not allowed in model-drift-watch code checks")
        elif event in ("os.listdir", "os.scandir"):
            path = args[0] if args else None
            if isinstance(path, (str, bytes)):
                real = realpath(fsdecode(path))
                if not (real + sep).startswith(roots):
                    raise PermissionError("listing folders outside Python is not allowed in model-drift-watch code checks")
        elif event == "import":
            if str(args[0]).split(".")[0] in blocked_modules:
                raise ImportError("module %%r is not available in model-drift-watch code checks" %% args[0])
        elif event.startswith(blocked_events):
            raise PermissionError(event + " is not allowed in model-drift-watch code checks")

    sys.addaudithook(guard)
    ns = {"__name__": "solution"}
    try:
        exec(compile(payload["code"], "<answer>", "exec"), ns)
    except BaseException as exc:
        print("answer did not load: " + type(exc).__name__)
        return
    if not callable(ns.get(payload["entry"])):
        print("function " + payload["entry"] + " is not defined")
        return
    for n, test in enumerate(payload["tests"], 1):
        try:
            exec(compile(test, "<test %%d>" %% n, "exec"), ns)
        except AssertionError:
            print("test %%d failed" %% n)
            return
        except BaseException as exc:
            print("test %%d raised " %% n + type(exc).__name__)
            return
    sys.stdout.write("\n" + token + "\n")


_harness()
sys.stdout.flush()
'''


def _clean_env() -> dict:
    env = {"PATH": os.defpath, "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", "LC_ALL": "C.UTF-8"}
    for key in ("SYSTEMROOT", "WINDIR"):  # the Windows interpreter cannot start without them
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def run_python(code: str, tests: list, entry: str, timeout: float = TIMEOUT_S) -> Grade:
    token = secrets.token_hex(16)
    payload = json.dumps({"code": code, "tests": tests, "entry": entry})
    runner = _RUNNER % {"cpu": int(timeout) + 1}
    with tempfile.TemporaryDirectory(prefix="model-drift-watch-") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", runner],
                input=token + "\n" + payload + "\n", capture_output=True, text=True,
                encoding="utf-8", errors="replace", cwd=workdir, env=_clean_env(), timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Grade(False, f"timed out after {timeout:g}s")
        except OSError as exc:
            return Grade(False, f"could not start the checker ({exc.__class__.__name__})", skipped=True)
    out = proc.stdout.rstrip().splitlines()
    if proc.returncode == 0 and out and out[-1] == token:
        return Grade(True)
    detail = next((line for line in reversed(out) if line.strip()), "") or f"exit code {proc.returncode}"
    return Grade(False, detail[:120])
