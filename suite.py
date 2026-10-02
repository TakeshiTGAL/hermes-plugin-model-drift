"""Question sets: the bundled default plus the user's own files, validated with fixable messages."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BUILTIN_SUITE = Path(__file__).resolve().parent / "data" / "default_suite.json"

# Bump when the wording below changes: every item key changes with it, so old answers are never
# compared with answers to a differently worded request.
PROMPT_VERSION = "p1"

SYSTEM_PROMPT = (
    "You are answering a fixed test question that is checked automatically. "
    "Follow the requested output format exactly."
)
ANSWER_LINE_SUFFIX = {
    "en": "\n\nWork it out, then end your reply with one final line in the form:\nANSWER: <your answer>",
    "ja": "\n\n考えてから、最後の行に次の形で答えだけを書いてください。\nANSWER: <答え>",
}
CODE_SUFFIX = (
    "\n\nReply with one Python code block (```python ... ```) that defines the function. "
    "Use only the standard library. Do not include tests, prints or example usage."
)

EXTRACT_MODES = ("answer_line", "full", "code")
GRADER_TYPES = ("exact", "number", "regex", "json", "python")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
DEFAULT_MAX_TOKENS = {"answer_line": 1024, "full": 512, "code": 1536}


@dataclass(frozen=True)
class Item:
    id: str
    category: str
    prompt: str
    grader: Dict[str, Any]
    extract: str = "answer_line"
    lang: str = "en"
    max_tokens: int = 1024
    reference: str = ""
    source: str = "builtin"

    @property
    def key(self) -> str:
        """Stable identity of what is asked and how it is graded. History is joined on this key,
        so editing a question starts that question's baseline over instead of mixing versions."""
        payload = json.dumps(
            [PROMPT_VERSION, self.prompt, self.extract, self.lang, self.grader, self.max_tokens],
            sort_keys=True, ensure_ascii=False,
        )
        return f"{self.id}@{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:10]}"

    def messages(self) -> List[Dict[str, str]]:
        if self.extract == "answer_line":
            body = self.prompt + ANSWER_LINE_SUFFIX.get(self.lang, ANSWER_LINE_SUFFIX["en"])
        elif self.extract == "code":
            body = self.prompt + CODE_SUFFIX
        else:
            body = self.prompt
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}]


@dataclass
class Suite:
    items: List[Item] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)

    def categories(self) -> List[str]:
        return sorted({i.category for i in self.items})


def _validate(raw: Any, where: str) -> Tuple[Optional[Item], List[str]]:
    errs: List[str] = []
    if not isinstance(raw, dict):
        return None, [f"{where}: each question must be an object with id, category, prompt and grader."]
    qid = str(raw.get("id", ""))
    where = f"{where} ({qid or 'no id'})"
    if not ID_RE.match(qid):
        errs.append(f"{where}: id must be 1-64 letters, digits, '.', '_' or '-'.")
    category = str(raw.get("category", "")).strip().lower()
    if not CATEGORY_RE.match(category):
        errs.append(f"{where}: category must be a short lowercase word such as 'reasoning' or 'my-domain'.")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errs.append(f"{where}: prompt must be non-empty text.")
    extract = raw.get("extract", "answer_line")
    if extract not in EXTRACT_MODES:
        errs.append(f"{where}: extract must be one of {', '.join(EXTRACT_MODES)} (got {extract!r}).")
    lang = str(raw.get("lang", "en"))
    grader = raw.get("grader")
    if not isinstance(grader, dict) or grader.get("type") not in GRADER_TYPES:
        errs.append(f"{where}: grader.type must be one of {', '.join(GRADER_TYPES)}.")
    else:
        errs.extend(_validate_grader(grader, extract, where))
    try:
        max_tokens = int(raw.get("max_tokens", DEFAULT_MAX_TOKENS.get(extract, 1024)))
        if not 16 <= max_tokens <= 32000:
            raise ValueError
    except (TypeError, ValueError):
        errs.append(f"{where}: max_tokens must be a whole number from 16 to 32000.")
        max_tokens = 1024
    if errs:
        return None, errs
    return Item(id=qid, category=category, prompt=prompt.strip(), grader=grader, extract=extract,
                lang=lang, max_tokens=max_tokens, reference=str(raw.get("reference", "")),
                source=str(raw.get("source", "builtin"))), []


def _validate_grader(g: Dict[str, Any], extract: str, where: str) -> List[str]:
    t = g["type"]
    errs = []
    if t == "exact":
        answers = g.get("answers")
        if not isinstance(answers, list) or not answers or not all(isinstance(a, str) and a for a in answers):
            errs.append(f"{where}: an exact grader needs answers: [\"...\"] with at least one non-empty string.")
    elif t == "number":
        if not isinstance(g.get("answer"), (int, float)) or isinstance(g.get("answer"), bool):
            errs.append(f"{where}: a number grader needs answer: <number>.")
        tol = g.get("tolerance", 0)
        if not isinstance(tol, (int, float)) or tol < 0:
            errs.append(f"{where}: tolerance must be a number >= 0.")
    elif t == "regex":
        pats = g.get("patterns")
        if not isinstance(pats, list) or not pats:
            errs.append(f"{where}: a regex grader needs patterns: [\"...\"] (all must match).")
        for p in list(pats or []) + list(g.get("forbid") or []):
            try:
                re.compile(p)
            except (re.error, TypeError) as exc:
                errs.append(f"{where}: pattern {p!r} does not compile: {exc}.")
    elif t == "json":
        if "equals" not in g:
            errs.append(f"{where}: a json grader needs equals: <the expected JSON value>.")
    elif t == "python":
        if extract != "code":
            errs.append(f"{where}: a python grader needs extract: code.")
        if not isinstance(g.get("entry"), str) or not g["entry"].isidentifier():
            errs.append(f"{where}: a python grader needs entry: <function name>.")
        tests = g.get("tests")
        if not isinstance(tests, list) or not tests or not all(isinstance(x, str) for x in tests):
            errs.append(f"{where}: a python grader needs tests: [\"assert ...\"].")
    return errs


def _read_items(path: Path) -> List[Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix in (".yaml", ".yml"):
        doc = _load_yaml(text)
    else:
        doc = json.loads(text)
    if isinstance(doc, dict):
        doc = doc.get("items", [])
    if not isinstance(doc, list):
        raise ValueError("the file must hold a list of questions or an object with an 'items' list")
    return doc


def _load_yaml(text: str) -> Any:
    try:
        from ruamel.yaml import YAML  # ships with Hermes
        return YAML(typ="safe", pure=True).load(text)
    except ImportError:
        import yaml  # type: ignore[import-not-found]
        return yaml.safe_load(text)


def user_suite_files(dirs: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for d in dirs:
        if d.is_dir():
            out.extend(sorted(p for p in d.iterdir()
                              if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".yaml", ".yml")))
    return out


def load_suite(user_dirs: Iterable[Path] = (), categories: Iterable[str] = (),
               include_builtin: bool = True, builtin_path: Path = BUILTIN_SUITE) -> Suite:
    suite = Suite()
    sources: List[Tuple[str, List[Any]]] = []
    if include_builtin:
        sources.append(("builtin", _read_items(builtin_path)))
    for path in user_suite_files(user_dirs):
        try:
            sources.append((str(path), _read_items(path)))
            suite.files.append(str(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            suite.errors.append(f"{path}: could not be read ({exc}). Fix the file or move it out of the folder.")
        except Exception as exc:  # YAML parser errors have no shared base class
            suite.errors.append(f"{path}: could not be parsed ({exc}). Fix the file or move it out of the folder.")
    seen: Dict[str, str] = {}
    for source, raws in sources:
        for n, raw in enumerate(raws, 1):
            where = "default suite" if source == "builtin" else f"{Path(source).name} #{n}"
            if isinstance(raw, dict):
                raw = {**raw, "source": source}
            item, errs = _validate(raw, where)
            if errs:
                suite.errors.extend(errs)
                continue
            assert item is not None
            if item.id in seen:
                suite.errors.append(f"{where}: id {item.id!r} is already used in {seen[item.id]}; ids must be unique.")
                continue
            seen[item.id] = where
            suite.items.append(item)
    wanted = {c.strip().lower() for c in categories if str(c).strip()}
    if wanted:
        unknown = wanted - {i.category for i in suite.items}
        if unknown:
            suite.errors.append(f"categories setting names {sorted(unknown)}, which no question uses; "
                                f"available: {sorted({i.category for i in suite.items})}.")
        suite.items = [i for i in suite.items if i.category in wanted]
    return suite
