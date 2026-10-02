import datetime as dt
import json
import math

import pytest


def test_default_suite_shape(suite_items):
    cats = {}
    for i in suite_items:
        cats[i.category] = cats.get(i.category, 0) + 1
    assert 30 <= len(suite_items) <= 50
    assert set(cats) == {"reasoning", "code", "instruction", "japanese"}
    assert all(n >= 10 for n in cats.values())
    assert len({i.key for i in suite_items}) == len(suite_items)


def test_every_reference_answer_passes_its_grader(mods, suite_items):
    for item in suite_items:
        g = mods.graders.grade(item, item.reference, mods.sandbox.run_python)
        assert g.passed, (item.id, g.reason)


def test_wrong_answers_fail(mods, suite_items):
    from mock_model import wrong_answer
    for item in suite_items:
        g = mods.graders.grade(item, wrong_answer(item), mods.sandbox.run_python)
        assert not g.passed, item.id


def test_answer_key_audit(suite_items):
    """Recompute the answers that can be computed, so a typo in the key cannot hide."""
    by_id = {i.id: i for i in suite_items}

    def num(qid):
        return by_id[qid].grader["answer"]

    assert num("r01-arith") == 4837 * 29 - 6251
    assert num("r02-dates") == (dt.date(2024, 3, 15) - dt.date(2024, 2, 15)).days == 29
    assert num("r03-multiply") == 987654 * 3219
    assert num("r04-letters") == "strawberry raspberry recurring".count("r")
    assert by_id["r05-weekday"].grader["answers"] == ["Friday"]  # 100 % 7 == 2 days after Wednesday
    assert num("r07-primes") == sum(p for p in range(2, 100) if all(p % d for d in range(2, p)))
    assert num("r08-last-digit") == pow(7, 222, 10)
    assert num("r09-divisible") == len([k for k in range(1, 1000) if (k % 3 == 0 or k % 5 == 0) and k % 15])
    assert num("r10-zeros") == len(str(math.factorial(125))) - len(str(math.factorial(125)).rstrip("0"))
    assert by_id["r11-reverse"].grader["answers"] == ["drift-monitor-2026"[::-1]]
    ways = [1, 1]
    for _ in range(11):
        ways.append(ways[-1] + ways[-2])
    assert num("r12-stairs") == ways[12]
    assert num("j06-wareki") == 2026  # Reiwa 1 = 2019
    assert num("j08-moji-count") == len("とうきょうとっきょきょかきょく") == 15
    assert num("r13-vowels") == sum("hermes plugin catalog admission rules".count(c) for c in "aeiou")
    assert by_id["i08-sorted-letters"].grader["answers"] == ["".join(sorted("consequential"))]


@pytest.mark.parametrize("text,ok", [
    ("Thinking...\nANSWER: 134022", True),
    ("**Answer:** 134,022", True),
    ("ANSWER: 134022.0", True),
    ("final answer: 134022", True),
    ("ANSWER: 134023", False),
    ("134022", False),  # the format was requested; no ANSWER line is a miss
])
def test_number_grader(mods, suite_items, text, ok):
    item = next(i for i in suite_items if i.id == "r01-arith")
    assert mods.graders.grade(item, text).passed is ok


@pytest.mark.parametrize("text,ok", [
    ("ANSWER: が、で、を", True), ("ANSWER: が,で,を", True), ("答え：が、で、を", True),
    ("ANSWER: 「が、で、を」。", True), ("ANSWER: が、を、で", False),
])
def test_japanese_exact(mods, suite_items, text, ok):
    item = next(i for i in suite_items if i.id == "j04-joshi")
    assert mods.graders.grade(item, text).passed is ok


@pytest.mark.parametrize("qid,text,ok", [
    ("i01-only-ok", "OK", True), ("i01-only-ok", "OK.", False), ("i01-only-ok", "ok", False),
    ("i02-json-array", '["blue","red","green"]', True), ("i02-json-array", '```json\n["red","green","blue"]\n```', False),
    ("i04-three-words", "Paris, of course.", True), ("i04-three-words", "Paris", False),
    ("i07-json-object", '{"version": 3, "name": "Hermes"}', True), ("i07-json-object", '{"name": "Hermes", "version": "3"}', False),
    ("i10-date-line", "DATE: 01/03/2025", True), ("i10-date-line", "DATE: 03/01/2025", False),
])
def test_instruction_graders(mods, suite_items, qid, text, ok):
    item = next(i for i in suite_items if i.id == qid)
    assert mods.graders.grade(item, text).passed is ok


def test_code_without_execution_is_skipped_not_failed(mods, suite_items):
    item = next(i for i in suite_items if i.grader["type"] == "python")
    g = mods.graders.grade(item, item.reference, None)
    assert g.skipped and not g.passed


@pytest.mark.parametrize("code,reason", [
    ("import socket\ndef f(x): return 1", "ImportError"),
    ("def f(x):\n    open('pwned.txt', 'w').write('x')\n    return 1", "PermissionError"),
    ("import os\ndef f(x):\n    os.system('echo hi')\n    return 1", "PermissionError"),
    ("import os\nos._exit(0)\ndef f(x): return 1", "exit code 0"),
    ("def f(x):\n    return open('/etc/hosts').read()", "PermissionError"),
    ("import os\ndef f(x):\n    return os.listdir('/')", "PermissionError"),
    ("def f(x):\n    while True:\n        pass", "timed out"),
])
def test_sandbox_refuses_side_effects(mods, code, reason):
    g = mods.sandbox.run_python(code, ["assert f(1) == 1"], "f", timeout=3)
    assert not g.passed and reason in g.reason


@pytest.mark.parametrize("code,reason", [
    # replacing a built-in or os.path.realpath after the hook is installed must not open it up
    ("import builtins\nbuiltins.isinstance = lambda *a: False\ndef f(x):\n    return open('/etc/hosts').read()",
     "PermissionError"),
    ("import builtins\nbuiltins.any = lambda *a: False\ndef f(x):\n    open('pwned.txt', 'w').write('x')\n    return 1",
     "PermissionError"),
    ("import os, sys\nos.path.realpath = lambda p: sys.prefix\ndef f(x):\n    return open('/etc/hosts').read()",
     "PermissionError"),
    ("import builtins\nbuiltins.str = lambda o='': 'math'\ndef f(x):\n    import socket\n    return 1",
     "ImportError"),
    # modules that write files from C code, outside the audited open()
    ("import sqlite3\ndef f(x): return 1", "ImportError"),
    ("import dbm\ndef f(x): return 1", "ImportError"),
])
def test_sandbox_resists_simple_ways_around_the_hook(mods, code, reason, tmp_path, monkeypatch):
    g = mods.sandbox.run_python(code, ["assert f(1) == 1"], "f", timeout=3)
    assert not g.passed and reason in g.reason, g.reason


def test_user_suite_files(mods, tmp_path):
    d = tmp_path / "suites"
    d.mkdir()
    (d / "mine.json").write_text(json.dumps({"items": [
        {"id": "my-1", "category": "domain", "prompt": "2+2?", "grader": {"type": "number", "answer": 4},
         "reference": "ANSWER: 4"}]}), encoding="utf-8")
    (d / "more.jsonl").write_text(json.dumps(
        {"id": "my-2", "category": "domain", "prompt": "Say hi", "extract": "full",
         "grader": {"type": "regex", "patterns": ["(?i)hi"]}}) + "\n", encoding="utf-8")
    (d / "more.yaml").write_text("items:\n  - id: my-3\n    category: domain\n    prompt: Capital of Japan?\n"
                                 "    grader: {type: exact, answers: [Tokyo]}\n", encoding="utf-8")
    s = mods.suite.load_suite([d])
    assert not s.errors
    assert {"my-1", "my-2", "my-3"} <= {i.id for i in s.items}
    only = mods.suite.load_suite([d], categories=["domain"])
    assert {i.id for i in only.items} == {"my-1", "my-2", "my-3"}


def test_bad_user_questions_get_fixable_messages(mods, tmp_path):
    d = tmp_path / "suites"
    d.mkdir()
    (d / "bad.json").write_text(json.dumps([
        {"id": "r01-arith", "category": "x", "prompt": "dup", "grader": {"type": "number", "answer": 1}},
        {"id": "b-2", "category": "x", "prompt": "", "grader": {"type": "exact", "answers": []}},
        {"id": "b-3", "category": "x", "prompt": "code?", "grader": {"type": "python", "entry": "f", "tests": ["assert f()"]}},
        {"id": "b-4", "category": "x", "prompt": "re", "grader": {"type": "regex", "patterns": ["("]}},
    ]), encoding="utf-8")
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    s = mods.suite.load_suite([d])
    text = "\n".join(s.errors)
    assert "already used" in text
    assert "prompt must be non-empty" in text and "at least one non-empty string" in text
    assert "needs extract: code" in text
    assert "does not compile" in text
    assert "broken.json: could not be read" in text
    assert len([i for i in s.items if i.source != "builtin"]) == 0


def test_editing_a_question_changes_its_key(mods, suite_items):
    item = suite_items[0]
    edited = mods.suite.Item(**{**item.__dict__, "prompt": item.prompt + " "})
    assert edited.key != item.key and edited.key.split("@")[0] == item.id
