"""One check: estimate, guard the budget, ask every question, grade, compare, maybe confirm."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import cost, report, stats
from .graders import Grade, grade
from .store import Busy
from .suite import PROMPT_VERSION, Item
from .targets import Target

FATAL = {"not_permitted", "auth", "no_provider"}  # same answer for every question: stop asking
RETRYABLE = {"rate_limited", "server_error", "timeout", "network"}
BACKOFF_S = (2.0, 8.0)


# --- settings ------------------------------------------------------------------------------------------
@dataclass
class Settings:
    max_tokens_per_check: int = 120_000
    max_usd_per_check: float = 0.5
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    baseline_runs: int = 5
    baseline_max_runs: int = 15
    samples_per_item: int = 1
    alpha: float = 0.01
    min_drop_points: float = 5.0
    confirm: bool = True
    max_error_rate: float = 0.2
    concurrency: int = 4
    request_timeout: int = 90
    check_deadline: int = 300
    baseline_min_gap_hours: float = 12.0
    max_tokens_scale: float = 1.0
    temperature: float = 0.0
    send_temperature: bool = True
    code_execution: bool = False  # off by default: see README (Hermes denies execute_code in cron by default)
    notify: str = "problems"
    categories: List[str] = field(default_factory=list)
    extra_targets: List[Any] = field(default_factory=list)

    @classmethod
    def from_getter(cls, get: Callable[[str, Any], Any]) -> Tuple["Settings", List[str]]:
        s, problems = cls(), []
        for name, default in list(vars(s).items()):
            raw = get(name, default)
            try:
                if isinstance(default, bool):
                    val = raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(default, int):
                    val = int(raw)
                elif isinstance(default, float):
                    val = float(raw)
                elif isinstance(default, list):
                    val = list(raw or [])
                else:
                    val = str(raw)
            except (TypeError, ValueError):
                problems.append(f"setting {name}={raw!r} is not valid; using {default!r}.")
                val = default
            setattr(s, name, val)
        bounds = {"baseline_runs": (2, 30), "samples_per_item": (1, 10), "concurrency": (1, 16),
                  "request_timeout": (5, 600), "check_deadline": (30, 400),
                  "baseline_max_runs": (2, 60)}
        for name, (lo, hi) in bounds.items():
            v = getattr(s, name)
            if not lo <= v <= hi:
                problems.append(f"setting {name}={v} is outside {lo}-{hi}; using {min(max(v, lo), hi)}.")
                setattr(s, name, min(max(v, lo), hi))
        if s.baseline_max_runs < s.baseline_runs:
            problems.append(f"setting baseline_max_runs={s.baseline_max_runs} is below baseline_runs; using {s.baseline_runs}.")
            s.baseline_max_runs = s.baseline_runs
        if not 0 <= s.baseline_min_gap_hours <= 168:
            problems.append(f"setting baseline_min_gap_hours={s.baseline_min_gap_hours} must be 0-168; using 12.")
            s.baseline_min_gap_hours = 12.0
        if not 0.25 <= s.max_tokens_scale <= 8:
            problems.append(f"setting max_tokens_scale={s.max_tokens_scale} must be 0.25-8; using 1.")
            s.max_tokens_scale = 1.0
        if not 0 < s.alpha < 0.5:
            problems.append(f"setting alpha={s.alpha} must be between 0 and 0.5; using 0.01.")
            s.alpha = 0.01
        if not 0 <= s.max_error_rate < 1:
            problems.append(f"setting max_error_rate={s.max_error_rate} must be from 0 to 1; using 0.2.")
            s.max_error_rate = 0.2
        if s.notify not in ("problems", "always"):
            problems.append(f"setting notify={s.notify!r} must be 'problems' or 'always'; using 'problems'.")
            s.notify = "problems"
        return s, problems


# --- talking to the model --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


Caller = Callable[..., Reply]


def ctx_llm_caller(llm) -> Caller:
    """Adapter over Hermes's ctx.llm: host-owned auth and routing; the plugin never sees a key."""
    def call(messages, *, max_tokens: int, temperature: Optional[float], timeout: float, target: Target) -> Reply:
        kw: Dict[str, Any] = dict(messages=messages, max_tokens=max_tokens, timeout=timeout,
                                  purpose="model-drift-watch.probe")
        if temperature is not None:
            kw["temperature"] = temperature
        if target.override:
            kw.update(provider=target.provider, model=target.model)
        r = llm.complete(**kw)
        usage = getattr(r, "usage", None)
        return Reply(text=r.text or "", model=(r.model or "").strip(),
                     input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                     output_tokens=int(getattr(usage, "output_tokens", 0) or 0))
    return call


def classify_error(exc: BaseException) -> Tuple[str, Optional[float], str]:
    """(code, retry_after_seconds, short detail). Codes are stable; they appear in reports."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    retry_after = None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            retry_after = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            retry_after = None
    detail = f"{name}{f' {status}' if status else ''}: {str(exc)[:160]}"
    if name == "PluginLlmTrustError":
        return "not_permitted", None, detail
    if status == 429 or "RateLimit" in name:
        return "rate_limited", retry_after, detail
    if status in (401, 403) or "Authentication" in name or "PermissionDenied" in name:
        return "auth", None, detail
    if isinstance(status, int) and status >= 500 or "InternalServer" in name or "ServiceUnavailable" in name:
        return "server_error", retry_after, detail
    if "Timeout" in name or isinstance(exc, TimeoutError):
        return "timeout", None, detail
    if "Connection" in name or isinstance(exc, (ConnectionError, OSError)):
        return "network", None, detail
    if status in (400, 404, 422) or "BadRequest" in name or "NotFound" in name:
        return "bad_request", None, detail
    if "provider" in str(exc).lower() and ("no " in str(exc).lower() or "not configured" in str(exc).lower()):
        return "no_provider", None, detail
    return "error", None, detail


# --- budget: a request is sent only while its longest possible answer still fits ---------------------------
class Budget:
    def __init__(self, cap_tokens: int, cap_usd: Optional[float], price: Optional[cost.Price]):
        self.cap_tokens, self.cap_usd, self.price = cap_tokens, cap_usd, price
        self.spent_tokens = 0
        self.spent_in = self.spent_out = 0
        self._reserved_tokens = 0
        self._reserved_usd = 0.0
        self._lock = threading.Lock()

    def _usd(self, tin: float, tout: float) -> float:
        return self.price.usd(tin, tout) if self.price else 0.0

    def try_reserve(self, est_input: int, max_output: int) -> Optional[Tuple[int, float]]:
        tokens = math.ceil(est_input * cost.INPUT_SAFETY) + max_output
        usd = self._usd(est_input * cost.INPUT_SAFETY, max_output)
        with self._lock:
            if self.spent_tokens + self._reserved_tokens + tokens > self.cap_tokens:
                return None
            if self.price and self.cap_usd is not None and \
                    self._usd(self.spent_in, self.spent_out) + self._reserved_usd + usd > self.cap_usd:
                return None
            self._reserved_tokens += tokens
            self._reserved_usd += usd
            return tokens, usd

    def settle(self, reservation: Tuple[int, float], actual_in: int, actual_out: int) -> None:
        with self._lock:
            self._reserved_tokens -= reservation[0]
            self._reserved_usd -= reservation[1]
            self.spent_in += actual_in
            self.spent_out += actual_out
            self.spent_tokens += actual_in + actual_out

    @property
    def spent_usd(self) -> Optional[float]:
        return self._usd(self.spent_in, self.spent_out) if self.price else None


# --- one pass over the questions ---------------------------------------------------------------------------
@dataclass
class Sample:
    key: str
    outcome: Optional[int] = None  # 1 right, 0 wrong, None = not measured (see error)
    error: Optional[str] = None
    detail: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_input: int = 0
    served: str = ""


def probe(call: Caller, items: Sequence[Item], target: Target, s: Settings, budget: Budget,
          run_code: Optional[Callable[[str, list, str], Grade]], deadline: float,
          clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> List[Sample]:
    tasks = [item for item in items for _ in range(s.samples_per_item)]
    samples: List[Sample] = []
    lock = threading.Lock()
    slots = threading.Semaphore(s.concurrency)
    fatal: List[str] = []
    # circuit breaker: when the first requests ALL fail to reach the model, the provider is down or
    # throttling hard; asking the remaining questions would only add minutes of retries
    reached = {"ok": 0, "failed": 0, "code": ""}
    breaker_after = max(6, 2 * s.concurrency)
    temperature = s.temperature if s.send_temperature else None

    def work(item: Item, sample: Sample, reservation: Tuple[int, float]) -> None:
        charged_in = 0  # input tokens a failed attempt may still have been billed for
        try:
            for attempt in range(len(BACKOFF_S) + 1):
                remaining = deadline - clock()
                if remaining < 5:
                    sample.error, sample.detail = "deadline", "check_deadline reached while retrying"
                    return
                try:
                    reply = call(item.messages(), max_tokens=item.max_tokens, temperature=temperature,
                                 timeout=float(min(s.request_timeout, remaining - 2)), target=target)
                except Exception as exc:  # every provider failure becomes a labelled, excluded sample
                    code, retry_after, detail = classify_error(exc)
                    sample.error, sample.detail = code, detail
                    if code in ("timeout", "server_error"):
                        charged_in += sample.estimated_input
                    if code in FATAL:
                        fatal.append(code)
                        return
                    if code not in RETRYABLE or attempt == len(BACKOFF_S):
                        return
                    wait = min(30.0, retry_after if retry_after is not None else BACKOFF_S[attempt])
                    if clock() + wait > deadline:
                        return
                    sleep(wait)
                    continue
                sample.error, sample.detail = None, ""
                sample.input_tokens, sample.output_tokens = reply.input_tokens, reply.output_tokens
                sample.served = reply.model
                budget.settle(reservation, reply.input_tokens or sample.estimated_input, reply.output_tokens)
                reservation = (0, 0.0)
                if not reply.text.strip():
                    sample.error, sample.detail = "empty_response", "the model returned no text"
                    return
                g = grade(item, reply.text, run_code and (
                    lambda code, tests, entry: run_code(code, tests, entry, max(2.0, min(10.0, deadline - clock())))))
                if g.skipped:
                    sample.error, sample.detail = "checker_unavailable", g.reason
                    return
                sample.outcome, sample.detail = (1 if g.passed else 0), g.reason
                return
        finally:
            if reservation != (0, 0.0):
                budget.settle(reservation, charged_in, 0)
            with lock:
                if sample.outcome is not None or sample.served:
                    reached["ok"] += 1
                else:
                    reached["failed"] += 1
                    reached["code"] = sample.error or "error"
            slots.release()

    with ThreadPoolExecutor(max_workers=s.concurrency, thread_name_prefix="model-drift-watch") as pool:
        for item in tasks:
            sample = Sample(key=item.key, estimated_input=cost.estimate_input_tokens(item.messages()))
            with lock:
                samples.append(sample)
            slots.acquire()
            if fatal:
                sample.error, sample.detail = fatal[0], "stopped after the same error on an earlier question"
                slots.release()
                continue
            with lock:
                tripped = reached["ok"] == 0 and reached["failed"] >= breaker_after
            if tripped:
                sample.error = reached["code"]
                sample.detail = f"not asked: the first {reached['failed']} requests all failed"
                slots.release()
                continue
            if clock() > deadline:
                sample.error, sample.detail = "deadline", "check_deadline reached before this question"
                slots.release()
                continue
            reservation = budget.try_reserve(sample.estimated_input, item.max_tokens)
            if reservation is None:
                sample.error, sample.detail = "budget", "cap reached before this question"
                slots.release()
                continue
            pool.submit(work, item, sample, reservation)
    return samples


def mark_rerouted(samples: List[Sample]) -> str:
    """Answers served by a model other than this run's majority are excluded (Hermes fallback)."""
    served = Counter(x.served for x in samples if x.outcome is not None and x.served)
    if not served:
        return ""
    major = served.most_common(1)[0][0]
    for x in samples:
        if x.outcome is not None and x.served and x.served != major:
            x.outcome, x.error, x.detail = None, "rerouted", f"answered by {x.served}"
    return major


def pass_record(samples: List[Sample], target: Target, s: Settings, role: str, check_id: str, now_iso: str,
                estimate: cost.Estimate) -> Dict[str, Any]:
    served_major = mark_rerouted(samples)
    items: Dict[str, Dict[str, list]] = {}
    for x in samples:
        rec = items.setdefault(x.key, {"o": [], "t": [], "e": []})
        if x.outcome is None:
            rec["e"].append(x.error or "error")
        else:
            rec["o"].append(x.outcome)
            rec["t"].append(x.output_tokens)
    errors = Counter(x.error for x in samples if x.outcome is None)
    total = len(samples)
    error_rate = (sum(errors.values()) / total) if total else 1.0
    measured = [x for x in samples if x.input_tokens or x.output_tokens]
    return {
        "id": f"{now_iso.replace(':', '').replace('-', '')[:15]}-{uuid.uuid4().hex[:6]}",
        "check": check_id, "ts": now_iso, "target": target.id, "role": role, "prompt_version": PROMPT_VERSION,
        "valid": total > 0 and error_rate <= s.max_error_rate,
        "error_rate": round(error_rate, 4), "errors": dict(errors),
        "error_details": sorted({x.detail for x in samples if x.outcome is None and x.detail})[:5],
        "served": dict(Counter(x.served for x in samples if x.served)), "served_major": served_major,
        "samples": total, "answered": sum(1 for x in samples if x.outcome is not None),
        "correct": sum(1 for x in samples if x.outcome == 1),
        "usage": {"input": sum(x.input_tokens for x in samples), "output": sum(x.output_tokens for x in samples),
                  "estimated_input": sum(x.estimated_input for x in measured), "calls": len(measured)},
        "estimate": estimate.to_dict(),
        "items": items,
    }


# --- baseline --------------------------------------------------------------------------------------------
@dataclass
class Baseline:
    counts: Dict[str, stats.ItemCounts]
    runs: int
    ready: bool
    served: str
    last_ts: Optional[str] = None
    progress: Dict[str, int] = field(default_factory=dict)  # questions still short of k answers


def _hours(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp() / 3600
    except (TypeError, ValueError):
        return 0.0


def eligible_runs(history: List[Dict[str, Any]], start_index: int, gap_hours: float) -> List[Dict[str, Any]]:
    """Valid first-pass runs, at most one per gap_hours, so the baseline spans several days."""
    out: List[Dict[str, Any]] = []
    for r in history[start_index:]:
        if not (r.get("valid") and r.get("role") == "probe"):
            continue
        if out and gap_hours > 0 and _hours(r["ts"]) - _hours(out[-1]["ts"]) < gap_hours:
            continue
        out.append(r)
    return out


CLEAN = {"collecting", "baseline_ready", "stable", "improved"}  # checks whose run may join the baseline


def baseline_from(history: List[Dict[str, Any]], start_index: int, k: int, gap_hours: float = 0.0,
                  max_runs: int = 0, clean_checks: Optional[set] = None) -> Baseline:
    """The first k eligible runs form the baseline; later runs of checks that ended clean keep joining
    it until max_runs, so one lucky or unlucky early run cannot set the bar for good. After that it is
    frozen, so a slow decline is not absorbed."""
    max_runs = max(k, max_runs)
    eligible = eligible_runs(history, start_index, gap_hours)
    runs = []
    for r in eligible:
        if len(runs) >= max_runs:
            break
        if len(runs) < k or clean_checks is None or r.get("check") in clean_checks:
            runs.append(r)
    in_baseline = {id(r) for r in runs}
    # Per question: the baseline runs, plus, for a question that is new or was edited (its key is new),
    # later clean runs until it has its own k answers. This works at any time, also after the baseline
    # froze, so adding a question or changing max_tokens_scale never stops the monitor.
    per_key: Dict[str, List[Tuple[List[int], List[int]]]] = {}
    for run in eligible:
        member = id(run) in in_baseline
        clean = clean_checks is None or run.get("check") in clean_checks
        for key, rec in run.get("items", {}).items():
            got = per_key.setdefault(key, [])
            if rec.get("o") and len(got) < max_runs and (member or (clean and len(got) < k)):
                got.append((rec["o"], rec.get("t", [])))
    counts = stats.pool([{key: v[i] for key, v in per_key.items() if len(v) >= k and i < len(v)}
                         for i in range(max_runs)])
    served = Counter(r["served_major"] for r in runs if r.get("served_major"))
    last = runs[-1]["ts"] if runs else None
    return Baseline(counts=counts, runs=len(runs), ready=len(runs) >= k,
                    served=served.most_common(1)[0][0] if served else "", last_ts=last,
                    progress={key: len(v) for key, v in per_key.items() if len(v) < k})


def current_counts(record: Dict[str, Any]) -> Dict[str, stats.ItemCounts]:
    return stats.pool([{k: (v["o"], v["t"]) for k, v in record["items"].items() if v["o"]}])


# --- the check -------------------------------------------------------------------------------------------
@dataclass
class Outcome:
    status: str
    message: str
    silent: bool
    details: Dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_check(call: Caller, target: Target, s: Settings, suite_items: Sequence[Item], store,
              price: Optional[cost.Price], run_code: Optional[Callable[[str, list, str], Grade]],
              clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
              now: Callable[[], str] = _now, deadline: Optional[float] = None) -> Outcome:
    """deadline (clock() units) is shared by every target of one tool call; default: now + check_deadline."""
    if deadline is None:
        deadline = clock() + s.check_deadline
    items = [i for i in suite_items if s.code_execution or i.grader["type"] != "python"]
    check_id = uuid.uuid4().hex[:10]
    started = now()
    history = store.runs(target.id)
    est = cost.estimate(items, s.samples_per_item, price, history)
    cap_usd = s.max_usd_per_check if s.max_usd_per_check > 0 else None
    base_details = {"target": target.id, "check": check_id, "estimate": est.to_dict()}

    refusal = report.refusal(est, s, cap_usd, len(items))
    if refusal:
        out = Outcome("refused", refusal, False, {**base_details, "status": "refused"})
        store.append_check(target.id, {"id": check_id, "ts": started, "target": target.id, "status": "refused",
                                       "message": refusal})
        return out

    try:
        with store.lock(target.id):
            return _locked_check(call, target, s, items, store, price, run_code, clock, sleep, now,
                                 check_id, started, est, cap_usd, base_details, deadline)
    except Busy as exc:
        msg = (f"[model-drift-watch] {target.id}: skipped, {exc}. Nothing was spent. "
               "If no check is really running (for example after a crash), it unlocks itself after 15 minutes.")
        return Outcome("busy", msg, False, {**base_details, "status": "busy"})


def _locked_check(call, target, s, items, store, price, run_code, clock, sleep, now, check_id, started,
                  est, cap_usd, base_details, deadline) -> Outcome:
    history = store.runs(target.id)  # re-read under the lock
    budget = Budget(s.max_tokens_per_check, cap_usd, price)
    t0 = clock()
    start = int(store.state(target.id).get("baseline_start_index") or 0)
    clean = {c["id"] for c in store.checks(target.id) if c.get("status") in CLEAN}
    base = baseline_from(history, start, s.baseline_runs, s.baseline_min_gap_hours, s.baseline_max_runs, clean)

    samples = probe(call, items, target, s, budget, run_code, deadline, clock, sleep)
    rec1 = pass_record(samples, target, s, "probe", check_id, started, est)
    store.append_run(target.id, rec1)
    category_of = {i.key: i.category for i in items}
    seed = rec1["id"]

    cmp1 = cmp2 = pooled = None
    rec2 = None
    skipped_reason = ""
    counts_now = False
    areas: List[str] = []

    def compare(counts, seed_):
        return stats.compare(base.counts, counts, category_of, s.min_drop_points, seed_)

    if not rec1["valid"]:
        status = "measurement_failed"
    elif not base.ready:
        counts_now = not (base.last_ts and s.baseline_min_gap_hours > 0
                          and _hours(rec1["ts"]) - _hours(base.last_ts) < s.baseline_min_gap_hours)
        if not counts_now:
            skipped_reason = (f"This run is not part of the baseline: baseline runs must be at least "
                              f"{s.baseline_min_gap_hours:g} hours apart so the baseline sees day-to-day variation.")
        status = "baseline_ready" if counts_now and base.runs + 1 >= s.baseline_runs else "collecting"
    elif base.served and rec1["served_major"] and rec1["served_major"] != base.served:
        status = "model_changed"
        cmp1 = compare(current_counts(rec1), seed)
    else:
        cmp1 = compare(current_counts(rec1), seed)
        if cmp1 is None:  # every question is new or changed: they collect their own baseline first
            status = "collecting"
            done = min((base.progress.get(i.key, 0) for i in items), default=0) + 1
            skipped_reason = (f"All questions are new or changed (new wording, or a new max_tokens_scale), so "
                              f"they are collecting their own baseline: {min(done, s.baseline_runs)} of "
                              f"{s.baseline_runs} runs. Testing resumes after that.")
        elif not s.confirm:  # decide on one run at alpha
            areas = cmp1.drift_in(s.alpha)
            status = ("drift" if areas else "token_shift" if cmp1.token_shift()
                      else "improved" if cmp1.improved_in(s.alpha) else "stable")
        elif cmp1.drift_in(stats.SCREEN_ALPHA) or cmp1.token_shift(stats.SCREEN_ALPHA):
            # screen at 5%, then decide on both runs together at alpha (see stats.confirmed)
            first_duration = clock() - t0
            left_tokens = s.max_tokens_per_check - budget.spent_tokens
            left_usd = (cap_usd - (budget.spent_usd or 0)) if (cap_usd is not None and price) else None
            if left_tokens < est.expected_tokens or (left_usd is not None and left_usd < (est.expected_usd or 0)):
                skipped_reason = ("The confirmation run was skipped because it would not fit in what is left of "
                                  "the cap. Raise max_tokens_per_check / max_usd_per_check to about twice the "
                                  "estimate (see `hermes model-drift-watch estimate`).")
            elif deadline - clock() < first_duration * 1.2 + 5:
                skipped_reason = ("The confirmation run was skipped because it would not finish before "
                                  "check_deadline. Raise concurrency so a run takes less time.")
            else:
                samples2 = probe(call, items, target, s, budget, run_code, deadline, clock, sleep)
                rec2 = pass_record(samples2, target, s, "confirmation", check_id, now(), est)
                store.append_run(target.id, rec2)
            if rec2 and rec2["valid"]:
                cmp2 = compare(current_counts(rec2), None)
                both = stats.pool([{k: (v["o"], v["t"]) for k, v in r["items"].items() if v["o"]}
                                   for r in (rec1, rec2)])
                pooled = compare(both, rec2["id"])
            areas, shift_ok = stats.confirmed(s.alpha, pooled, cmp2)
            if cmp2 is None:
                status = "suspect" if (cmp1.drift_in(s.alpha) or cmp1.token_shift()) else "stable"
            elif areas:
                status = "drift"
            elif shift_ok:
                status = "token_shift"
            else:
                status = "unconfirmed" if cmp1.drift_in(s.alpha) else "stable"
        elif cmp1.improved_in(s.alpha):
            status = "improved"
        else:
            status = "stable"

    varied = None
    if status == "baseline_ready":
        merged: Dict[str, List[int]] = {}
        for run in eligible_runs(history, start, 0) + [rec1]:
            if run.get("role") != "probe" or not run.get("valid"):
                continue
            for key, rec in run.get("items", {}).items():
                merged.setdefault(key, []).extend(rec.get("o", []))
        varied = (sum(1 for v in merged.values() if v and 0 < sum(v) < len(v)), len(merged))
    spent = {"input": budget.spent_in, "output": budget.spent_out, "total": budget.spent_tokens,
             "usd": budget.spent_usd}
    message = report.render(status=status, target=target, s=s, rec1=rec1, rec2=rec2, cmp1=cmp1, cmp2=cmp2,
                            base=base, est=est, spent=spent, price=price, pooled=pooled, areas=areas,
                            note=skipped_reason, counted=counts_now, varied=varied)
    silent = s.notify == "problems" and status in ("stable", "collecting", "improved", "unconfirmed")
    collected = base.runs + (1 if counts_now and status in ("collecting", "baseline_ready") else 0)
    details = {**base_details, "status": status, "runs": [r["id"] for r in (rec1, rec2) if r],
               "spent": spent, "baseline_runs": f"{min(collected, s.baseline_runs)}/{s.baseline_runs}",
               "accuracy": report.accuracy_dict(s.alpha, cmp1, cmp2, pooled, rec1),
               "errors": rec1["errors"], "served_model": rec1["served_major"], "baseline_served_model": base.served}
    store.append_check(target.id, {"id": check_id, "ts": started, "finished": now(), "target": target.id,
                                   "status": status, "message": message, "spent": spent,
                                   "estimate": {"expected_tokens": est.expected_tokens,
                                                "worst_tokens": est.worst_tokens,
                                                "expected_usd": est.expected_usd},
                                   "runs": details["runs"], "accuracy": details["accuracy"]})
    return Outcome(status, message, silent, details)
