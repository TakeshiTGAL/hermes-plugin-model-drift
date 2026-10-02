"""Deterministic graders. Same text in, same verdict out: no model ever judges another model."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional

ANSWER_LINE_RE = re.compile(
    r"^[\s>*_#-]*(?:final\s+)?(?:answer|答え|解答)[\s*_]*[:：][\s*_]*(?P<a>.*?)[\s*_]*$",
    re.IGNORECASE | re.MULTILINE,
)
CODE_BLOCK_RE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(?P<c>.*?)```", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(r"[-−]?\d+(?:\.\d+)?")
_WRAPPERS = "\"'`「」『』“”‘’()[]（）【】*_ "
_TRAILING = ".。!！"


@dataclass(frozen=True)
class Grade:
    passed: bool
    reason: str = ""
    skipped: bool = False  # not gradable here (e.g. code execution turned off): excluded, not failed


def normalize(text: str, case_sensitive: bool = False) -> str:
    t = unicodedata.normalize("NFKC", text).strip()
    for _ in range(3):  # peel a few layers of quotes, bold markers and a closing full stop
        t = t.strip(_WRAPPERS).rstrip(_TRAILING).strip(_WRAPPERS)
    t = re.sub(r"\s+", " ", t)
    return t if case_sensitive else t.casefold()


def extract_answer_line(text: str) -> Optional[str]:
    matches = [m.group("a") for m in ANSWER_LINE_RE.finditer(text or "") if m.group("a").strip()]
    return matches[-1] if matches else None


def extract_code(text: str) -> Optional[str]:
    blocks = [m.group("c") for m in CODE_BLOCK_RE.finditer(text or "")]
    if blocks:
        with_def = [b for b in blocks if "def " in b]
        return (with_def or blocks)[-1]
    if text and re.search(r"^\s*def \w+\(", text, re.MULTILINE):
        return text
    return None


def _to_number(text: str) -> Optional[float]:
    t = unicodedata.normalize("NFKC", text).replace(",", "").replace("_", "")
    m = NUMBER_RE.search(t)
    if not m:
        return None
    return float(m.group(0).replace("−", "-"))


def _json_equal(a: Any, b: Any, unordered: bool) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k], unordered) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        if not unordered:
            return all(_json_equal(x, y, unordered) for x, y in zip(a, b))
        key = lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False)  # noqa: E731
        return sorted(map(key, a)) == sorted(map(key, b))
    return a == b


def grade(item, text: str, run_code: Optional[Callable[[str, list, str], Grade]] = None) -> Grade:
    """Grade one response. ``run_code(code, tests, entry)`` executes code answers; None skips them."""
    g = item.grader
    if item.extract == "answer_line":
        answer = extract_answer_line(text)
        if answer is None:
            return Grade(False, "no 'ANSWER:' line")
    elif item.extract == "code":
        answer = extract_code(text)
        if answer is None:
            return Grade(False, "no Python code block")
    else:
        answer = (text or "").strip()

    t = g["type"]
    if t == "exact":
        cs = bool(g.get("case_sensitive", False))
        if g.get("strict"):  # instruction following: only surrounding whitespace is forgiven
            ok = answer.strip() in g["answers"]
            return Grade(ok, "" if ok else "output is not exactly the expected text")
        got = normalize(answer, cs)
        ok = any(got == normalize(a, cs) for a in g["answers"])
        return Grade(ok, "" if ok else "answer differs")
    if t == "number":
        got = _to_number(answer)
        if got is None:
            return Grade(False, "no number in the answer")
        ok = abs(got - float(g["answer"])) <= float(g.get("tolerance", 0)) + 1e-9
        return Grade(ok, "" if ok else "wrong number")
    if t == "regex":
        missing = [p for p in g["patterns"] if not re.search(p, answer)]
        present = [p for p in g.get("forbid", []) if re.search(p, answer)]
        ok = not missing and not present
        return Grade(ok, "" if ok else "format check failed")
    if t == "json":
        try:
            value = json.loads(answer)
        except (json.JSONDecodeError, TypeError):
            return Grade(False, "not valid JSON on its own")
        ok = _json_equal(value, g["equals"], bool(g.get("unordered", False)))
        return Grade(ok, "" if ok else "JSON differs")
    if t == "python":
        if run_code is None:
            return Grade(False, "code execution is turned off", skipped=True)
        return run_code(answer, list(g["tests"]), g["entry"])
    return Grade(False, f"unknown grader {t!r}")
