"""Measure the decision rule's false-alarm rate and detection power by simulation.

    python scripts/simulate_error_rates.py [--trials 20000] [--users 200 --days 300]

Simulates the whole per-check procedure (first run, screen, confirmation run, pooled test) on the
default question set's size, with per-question success rates spread like a strong model's and
answer lengths that vary from answer to answer. Deterministic for a given --seed. The README's
tables come from this script.

Table 1 draws a fresh baseline for every simulated check: the average over users.
Table 2 keeps one baseline per user for many days, because a frozen baseline that happened to be
lucky makes that user's false alarms more frequent than the average.
"""

import argparse
import importlib
import importlib.util
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("model_drift_plugin", ROOT / "__init__.py",
                                              submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec)
sys.modules["model_drift_plugin"] = pkg
spec.loader.exec_module(pkg)
stats = importlib.import_module("model_drift_plugin.stats")

CATS = ("reasoning", "code", "instruction", "japanese")


class Model:
    def __init__(self, rng, p_items, sigma):
        self.p = p_items
        self.len = [rng.uniform(40, 400) for _ in p_items]
        self.sigma = sigma

    def run(self, rng, wrong=frozenset()):
        out = {}
        for i, p in enumerate(self.p):
            ok = 0 if i in wrong else int(rng.random() < p)
            out[f"q{i}"] = (ok, max(1, round(self.len[i] * math.exp(rng.gauss(0, self.sigma)))))
        return out


def counts(runs):
    out = {}
    for run in runs:
        for k, (ok, tok) in run.items():
            c = out.setdefault(k, stats.ItemCounts())
            c.passes += ok
            c.n += 1
            c.tokens.append(tok)
    return out


def one_check(rng, model, base, wrong, alpha, min_drop, cats):
    """(accuracy alert with one run, (accuracy alert, length alert) after confirmation, confirmed?)."""
    r1 = model.run(rng, wrong)
    c1 = stats.compare(base, counts([r1]), cats, min_drop)
    single = bool(c1.drift_in(alpha))
    if not (c1.drift_in(stats.SCREEN_ALPHA) or c1.token_shift(stats.SCREEN_ALPHA)):
        return single, (False, False), False
    r2 = model.run(rng, wrong)
    c2 = stats.compare(base, counts([r2]), cats, min_drop)
    pooled = stats.compare(base, counts([r1, r2]), cats, min_drop)
    areas, shift = stats.confirmed(alpha, pooled, c2)
    return single, (bool(areas), shift), True


def profiles(rng, n):
    return {"strong (0.80-1.00 per question)": [rng.uniform(0.80, 1.00) for _ in range(n)],
            "mixed (0.50-1.00 per question)": [rng.uniform(0.50, 1.00) for _ in range(n)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--items", type=int, default=34, help="34 = the default set with code_execution off")
    ap.add_argument("--areas", type=int, default=3, help="3 = reasoning, instruction, japanese (code off)")
    ap.add_argument("--base-runs", type=int, default=5)
    ap.add_argument("--base-max", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-drop", type=float, default=5)
    ap.add_argument("--sigma", type=float, default=0.3, help="answer-length spread (log scale)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    areas = [c for c in CATS if c != "code"] if args.areas == 3 else list(CATS[:args.areas])
    cats = {f"q{i}": areas[i % len(areas)] for i in range(args.items)}
    print(f"{args.items} questions in {len(areas)} areas, baseline {args.base_runs} runs, alpha {args.alpha}, min drop {args.min_drop} "
          f"points, answer-length spread {args.sigma}\n")

    print(f"Table 1: average over users ({args.trials} checks for 'nothing changed', a tenth of that otherwise)\n")
    print("| model profile | questions newly wrong | accuracy alert, one run (confirm off) | days with a confirmation run "
          "| accuracy alert (default) | answer-length alert (default) |")
    print("|---|---|---|---|---|---|")
    for name, p_items in profiles(rng, args.items).items():
        for degrade in (0.0, 0.05, 0.10, 0.20):
            trials = args.trials if degrade == 0 else max(2000, args.trials // 10)
            single = conf = acc = tok = 0
            for _ in range(trials):
                model = Model(rng, p_items, args.sigma)
                base = counts([model.run(rng) for _ in range(args.base_runs)])
                wrong = frozenset(rng.sample(range(args.items), round(degrade * args.items)))
                s1, (a, t), c = one_check(rng, model, base, wrong, args.alpha, args.min_drop, cats)
                single += s1
                acc += a
                tok += t
                conf += c
            print(f"| {name} | {degrade:.0%} | {single / trials:.2%} | {conf / trials:.2%} | {acc / trials:.2%} "
                  f"| {tok / trials:.2%} |")

    print(f"\nTable 2: nothing changed, one baseline per user (grows with clean runs to {args.base_max}, then frozen), "
          f"{args.users} users x {args.days} daily checks\n")
    print("| model profile | median user | 95th percentile user | worst user |")
    print("|---|---|---|---|")
    for name, p_items in profiles(rng, args.items).items():
        rates = []
        for _ in range(args.users):
            model = Model(rng, p_items, args.sigma)
            runs = [model.run(rng) for _ in range(args.base_runs)]
            alerts = 0
            for _ in range(args.days):
                base = counts(runs)
                r1 = model.run(rng)
                c1 = stats.compare(base, counts([r1]), cats, args.min_drop)
                alert = False
                if c1.drift_in(stats.SCREEN_ALPHA) or c1.token_shift(stats.SCREEN_ALPHA):
                    r2 = model.run(rng)
                    c2 = stats.compare(base, counts([r2]), cats, args.min_drop)
                    pooled = stats.compare(base, counts([r1, r2]), cats, args.min_drop)
                    areas, shift = stats.confirmed(args.alpha, pooled, c2)
                    alert = bool(areas) or shift
                alerts += alert
                clean = not alert and not c1.drift_in(args.alpha)  # the plugin's "stable" (or "improved")
                if clean and len(runs) < args.base_max:  # the baseline keeps growing with clean runs
                    runs.append(r1)
            rates.append(alerts / args.days)
        rates.sort()
        print(f"| {name} | {rates[len(rates) // 2]:.2%} | {rates[int(0.95 * (len(rates) - 1))]:.2%} | {rates[-1]:.2%} |")


if __name__ == "__main__":
    main()
