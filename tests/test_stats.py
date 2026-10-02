import itertools
import math
import random


def test_hypergeom_sums_to_one(mods):
    for total, succ, draws in [(4, 3, 1), (10, 4, 3), (6, 0, 2), (6, 6, 2)]:
        pmf = mods.stats.hypergeom_pmf(total, succ, draws)
        assert math.isclose(sum(pmf), 1.0)


def test_exact_test_matches_brute_force(mods):
    """Enumerate every way the right answers could fall between baseline and current run."""
    S = mods.stats.Stratum
    strata = [S(3, 3, 0, 1), S(2, 3, 1, 1), S(1, 3, 0, 1), S(3, 3, 1, 1)]
    res = mods.stats.stratified_exact_test(strata)
    # brute force: for each stratum, the current answer is one of N answers chosen uniformly
    probs = []
    for s in strata:
        total, succ = s.base_n + s.cur_n, s.base_pass + s.cur_pass
        probs.append(succ / total)
    observed = sum(s.cur_pass for s in strata)
    p_lower = 0.0
    for outcome in itertools.product([0, 1], repeat=len(strata)):
        p = math.prod(pr if o else 1 - pr for o, pr in zip(outcome, probs))
        if sum(outcome) <= observed:
            p_lower += p
    assert math.isclose(res.p_lower, p_lower)
    assert res.observed == 2 and res.informative == 3  # the all-right stratum carries no weight


def test_always_right_questions_carry_no_weight(mods):
    S = mods.stats.Stratum
    res = mods.stats.stratified_exact_test([S(3, 3, 1, 1)] * 10)
    assert res.p_lower == 1.0 and res.informative == 0


def test_holm(mods):
    rej = mods.stats.holm({"a": 0.001, "b": 0.02, "c": 0.04}, 0.05)
    assert rej == {"a": True, "b": True, "c": True}  # 0.001<=0.05/3, 0.02<=0.05/2, 0.04<=0.05/1
    rej = mods.stats.holm({"a": 0.03, "b": 0.04}, 0.05)
    assert rej == {"a": False, "b": False}


def test_sign_test(mods):
    assert mods.stats.sign_test(0, 0) == 1.0
    assert math.isclose(mods.stats.sign_test(10, 0), 2 / 2 ** 10)
    assert mods.stats.sign_test(5, 5) == 1.0


def _simulate(mods, p_items, degrade, runs_base=5, alpha=0.01, min_drop=5, trials=1500, seed=7):
    """Share of checks that end in an alert, with the default screen-then-confirm procedure."""
    st = mods.stats
    rng = random.Random(seed)
    n = len(p_items)
    cats = {f"q{i}": ("a", "b", "c", "d")[i % 4] for i in range(n)}
    alerts = 0
    for _ in range(trials):
        base = {f"q{i}": st.ItemCounts(sum(rng.random() < p for _ in range(runs_base)), runs_base)
                for i, p in enumerate(p_items)}
        wrong = set(rng.sample(range(n), round(degrade * n)))

        def run():
            return {f"q{i}": 0 if i in wrong else int(rng.random() < p) for i, p in enumerate(p_items)}

        r1 = run()
        c1 = st.compare(base, {k: st.ItemCounts(v, 1) for k, v in r1.items()}, cats, min_drop)
        if not c1.drift_in(st.SCREEN_ALPHA):
            continue
        r2 = run()
        c2 = st.compare(base, {k: st.ItemCounts(v, 1) for k, v in r2.items()}, cats, min_drop)
        pooled = st.compare(base, {k: st.ItemCounts(r1[k] + r2[k], 2) for k in r1}, cats, min_drop)
        alerts += bool(st.confirmed(alpha, pooled, c2)[0])
    return alerts / trials


def test_false_alarm_rate_is_at_most_alpha(mods):
    rng = random.Random(1)
    p_items = [rng.uniform(0.8, 1.0) for _ in range(43)]
    rate = _simulate(mods, p_items, degrade=0.0, trials=3000)
    assert rate <= 0.01


def test_power_against_twenty_percent_wrong(mods):
    rng = random.Random(1)
    p_items = [rng.uniform(0.8, 1.0) for _ in range(43)]
    rate = _simulate(mods, p_items, degrade=0.2, trials=600)
    assert rate >= 0.8
