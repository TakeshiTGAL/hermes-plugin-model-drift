"""Statistics: an exact, item-paired test of "is this run worse than the baseline?"

Each question is its own stratum. Under the null hypothesis (nothing changed) the answers from the
baseline runs and from the current run are exchangeable within a question, so, given how many of
that question's answers were right in total, the number of right answers that land in the current
run follows a hypergeometric distribution. The run's total is the sum over questions; its exact
distribution is the convolution of those hypergeometrics. No normal approximation, no prior, no
random numbers: the p-value is a deterministic function of the counts.

Properties that matter for a monitor:
  * question difficulty drops out (each question is only compared with itself);
  * questions every run gets right, or every run gets wrong, carry no weight automatically;
  * the test is exact and conservative, so the false-alarm rate per run is at most alpha.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def hypergeom_pmf(total: int, successes: int, draws: int) -> List[float]:
    """P(X = k) for k = 0..draws, X = successes among ``draws`` taken without replacement."""
    denom = math.comb(total, draws)
    return [math.comb(successes, k) * math.comb(total - successes, draws - k) / denom
            for k in range(draws + 1)]


def convolve(a: Sequence[float], b: Sequence[float]) -> List[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


@dataclass(frozen=True)
class Stratum:
    base_pass: int
    base_n: int
    cur_pass: int
    cur_n: int

    @property
    def informative(self) -> bool:
        total = self.base_pass + self.cur_pass
        return 0 < total < self.base_n + self.cur_n

    @property
    def diff(self) -> float:
        return self.cur_pass / self.cur_n - self.base_pass / self.base_n


@dataclass(frozen=True)
class TestResult:
    items: int
    informative: int
    observed: int
    expected: float
    p_lower: float  # small = fewer right answers than the baseline explains (drift)
    p_upper: float  # small = more right answers than the baseline explains (improvement)
    base_rate: float  # mean over questions, in [0, 1]
    cur_rate: float

    @property
    def change_points(self) -> float:
        return 100.0 * (self.cur_rate - self.base_rate)


def stratified_exact_test(strata: Sequence[Stratum]) -> TestResult:
    dist = [1.0]
    observed = 0
    expected = 0.0
    for s in strata:
        total, succ, draws = s.base_n + s.cur_n, s.base_pass + s.cur_pass, s.cur_n
        dist = convolve(dist, hypergeom_pmf(total, succ, draws))
        observed += s.cur_pass
        expected += draws * succ / total
    p_lower = min(1.0, sum(dist[: observed + 1]))
    p_upper = min(1.0, sum(dist[observed:]))
    n = len(strata)
    base_rate = sum(s.base_pass / s.base_n for s in strata) / n if n else 0.0
    cur_rate = sum(s.cur_pass / s.cur_n for s in strata) / n if n else 0.0
    return TestResult(items=n, informative=sum(s.informative for s in strata), observed=observed,
                      expected=expected, p_lower=p_lower, p_upper=p_upper,
                      base_rate=base_rate, cur_rate=cur_rate)


def holm(pvalues: Dict[str, float], alpha: float) -> Dict[str, bool]:
    """Holm-Bonferroni: family-wise error <= alpha across all the tests passed in."""
    order = sorted(pvalues, key=pvalues.get)
    rejected: Dict[str, bool] = {k: False for k in pvalues}
    m = len(order)
    for rank, key in enumerate(order):
        if pvalues[key] <= alpha / (m - rank):
            rejected[key] = True
        else:
            break
    return rejected


def bootstrap_ci(values: Sequence[float], seed: str, iters: int = 2000, level: float = 0.95) -> Tuple[float, float]:
    """Percentile interval for the mean, resampling questions. Seeded, so it is reproducible."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    lo = means[int((1 - level) / 2 * iters)]
    hi = means[min(iters - 1, int((1 + level) / 2 * iters))]
    return (lo, hi)


def sign_test(down: int, up: int) -> float:
    """Exact two-sided binomial sign test (ties already removed)."""
    n = down + up
    if n == 0:
        return 1.0
    k = min(down, up)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@dataclass
class ItemCounts:
    passes: int = 0
    n: int = 0
    tokens: List[int] = field(default_factory=list)


OVERALL_SHARE = 0.6  # share of alpha for the whole-set test; the rest is Holm-split across categories
SCREEN_ALPHA = 0.05  # a first run this suspicious earns a same-day confirmation run
CONFIRM_ALPHA = 0.05  # the confirmation run must show the drop on its own at this level
TOKEN_ALPHA = 0.01
MIN_TOKEN_CHANGE = 0.25


@dataclass
class Comparison:
    overall: TestResult
    categories: Dict[str, TestResult]
    min_drop_points: float
    ci_points: Optional[Tuple[float, float]]
    token_ratio: Optional[float]  # current / baseline output tokens, median over questions
    token_p: Optional[float]
    pending_items: int  # questions still collecting their own baseline

    def _areas(self, alpha: float, upper: bool = False) -> List[str]:
        """Areas ("overall" or a category) that moved significantly AND by at least the floor.

        alpha is split: OVERALL_SHARE of it for the whole set, the rest Holm-corrected across the
        categories, so a collapse in one area cannot hide behind the average while the family-wise
        error stays <= alpha.
        """
        pick = (lambda t: t.p_upper) if upper else (lambda t: t.p_lower)
        sig = {"overall": pick(self.overall) <= alpha * OVERALL_SHARE,
               **holm({n: pick(t) for n, t in self.categories.items()}, alpha * (1 - OVERALL_SHARE))}
        family = {"overall": self.overall, **self.categories}
        sign = 1 if upper else -1
        return [n for n, t in family.items() if sig[n] and sign * t.change_points >= self.min_drop_points]

    def drift_in(self, alpha: float) -> List[str]:
        return self._areas(alpha)

    def improved_in(self, alpha: float) -> List[str]:
        return self._areas(alpha, upper=True)

    def token_shift(self, alpha: float = TOKEN_ALPHA) -> bool:
        return (self.token_p is not None and self.token_p < alpha and self.token_ratio is not None
                and abs(self.token_ratio - 1) >= MIN_TOKEN_CHANGE)

    def p_in(self, area: str) -> float:
        return (self.overall if area == "overall" else self.categories.get(area, self.overall)).p_lower


def confirmed(alpha: float, pooled: Optional[Comparison], second: Optional[Comparison]) -> Tuple[List[str], bool]:
    """(drifting areas, token_shift) after a same-day confirmation run.

    An alert needs (a) the first and second runs together to clear the full test at alpha, and
    (b) the second run, which was not selected for looking bad, to show the same movement by itself
    (one-sided p < CONFIRM_ALPHA). (a) alone keeps the false-alarm rate at or below alpha whatever
    triggered the confirmation; (b) is what stops one unlucky first run from carrying the pooled test.
    """
    if pooled is None or second is None:
        return [], False
    areas = [a for a in pooled.drift_in(alpha) if second.p_in(a) < CONFIRM_ALPHA]
    shift = bool(pooled.token_shift() and second.token_shift(CONFIRM_ALPHA)
                 and (second.token_ratio - 1) * (pooled.token_ratio - 1) > 0)
    return areas, shift


def compare(baseline: Dict[str, ItemCounts], current: Dict[str, ItemCounts], category_of: Dict[str, str],
            min_drop_points: float, seed: Optional[str] = None, min_category_items: int = 3) -> Optional[Comparison]:
    """Test the current counts against the baseline. seed=None skips the (slow) bootstrap interval."""
    keys = [k for k in current if current[k].n > 0 and k in baseline and baseline[k].n > 0]
    pending = sum(1 for k in current if current[k].n > 0 and k not in baseline)
    if not keys:
        return None
    strata = {k: Stratum(baseline[k].passes, baseline[k].n, current[k].passes, current[k].n) for k in keys}
    overall = stratified_exact_test([strata[k] for k in keys])
    cats: Dict[str, TestResult] = {}
    for cat in sorted({category_of.get(k, "other") for k in keys}):
        ck = [k for k in keys if category_of.get(k, "other") == cat]
        if len(ck) >= min_category_items:
            cats[cat] = stratified_exact_test([strata[k] for k in ck])
    ci = None
    if seed is not None:
        lo, hi = bootstrap_ci([strata[k].diff for k in keys], seed)
        ci = (100 * lo, 100 * hi)
    ratios = []
    for k in keys:
        base_tok = [t for t in baseline[k].tokens if t > 0]
        cur_tok = [t for t in current[k].tokens if t > 0]
        if base_tok and cur_tok:
            ratios.append(math.log((sum(cur_tok) / len(cur_tok)) / median(base_tok)))
    token_ratio = token_p = None
    if len(ratios) >= 5:
        token_ratio = math.exp(median(ratios))
        token_p = sign_test(sum(r < 0 for r in ratios), sum(r > 0 for r in ratios))
    return Comparison(overall=overall, categories=cats, min_drop_points=min_drop_points, ci_points=ci,
                      token_ratio=token_ratio, token_p=token_p, pending_items=pending)


def pool(runs: Iterable[Dict[str, Tuple[List[int], List[int]]]]) -> Dict[str, ItemCounts]:
    """Merge per-run {key: (outcomes, output_tokens)} into per-question counts."""
    out: Dict[str, ItemCounts] = {}
    for run in runs:
        for key, (outcomes, tokens) in run.items():
            c = out.setdefault(key, ItemCounts())
            c.passes += sum(outcomes)
            c.n += len(outcomes)
            c.tokens.extend(tokens)
    return out
