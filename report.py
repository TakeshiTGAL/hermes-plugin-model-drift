"""Plain-text messages. They go to Telegram/Slack/etc. as-is, so: short, no markup, next step last."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SET = "hermes config set plugins.entries.model-drift-watch.settings."

HINTS = {
    "rate_limited": "Your provider is rate-limiting. Lower concurrency ({set}concurrency 2) or schedule the check at a quieter hour.",
    "network": "The provider could not be reached. Nothing to change if it was a blip; the next scheduled run tries again.",
    "timeout": "Requests timed out. Raise request_timeout ({set}request_timeout 180) if the model is slow.",
    "server_error": "The provider returned server errors (5xx). The next scheduled run tries again.",
    "auth": "The provider refused the credentials. Fix the key or login with `hermes model`, then run `hermes model-drift-watch run`.",
    "not_permitted": "Hermes did not let this plugin choose that model. Add the grant printed by `hermes model-drift-watch targets` to config.yaml.",
    "no_provider": "No model is configured. Pick one with `hermes model`, then run `hermes model-drift-watch run`.",
    "bad_request": "The provider rejected the request. If the error mentions temperature, run: {set}send_temperature false",
    "empty_response": "The model returned empty answers, usually because a reasoning model used up its answer limit while thinking. Raise the limit: {set}max_tokens_scale 4 (starts a new baseline for the questions).",
    "rerouted": "Some answers came from another model, usually Hermes falling back after a provider error. Check `hermes logs`.",
    "budget": "The check stopped at your cap. Raise it ({set}max_tokens_per_check / max_usd_per_check) or ask fewer questions (categories setting).",
    "deadline": "The check ran out of time. Raise concurrency ({set}concurrency 8) if your provider allows it; check_deadline is already capped at Hermes's tool-call limit (timeouts.tools.sequential_call) minus 30 s.",
    "checker_unavailable": "The code checker could not start. Set {set}code_execution false to skip code questions.",
    "error": "Unexpected errors; see `hermes logs` for details.",
}


def _n(x: float) -> str:
    return f"{int(round(x)):,}"


def _usd(x: Optional[float]) -> str:
    if x is None:
        return "cost unknown"
    return f"about ${x:.2f}" if x >= 0.005 else "under $0.01"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _p(p: float) -> str:
    return "p<0.0001" if p < 1e-4 else f"p={p:.4f}" if p < 0.01 else f"p={p:.2f}"


def refusal(est, s, cap_usd: Optional[float], n_items: int) -> Optional[str]:
    if n_items == 0:
        return ("[model-drift-watch] Nothing to ask: no questions are selected. "
                f"Clear the categories setting ({SET}categories '[]') or add questions to the suites folder.")
    lines = []
    if est.expected_tokens > s.max_tokens_per_check:
        lines.append(f"expected {_n(est.expected_tokens)} tokens is over your cap of {_n(s.max_tokens_per_check)}")
    if est.price and cap_usd is not None and (est.expected_usd or 0) > cap_usd:
        lines.append(f"expected {_usd(est.expected_usd)} is over your cap of ${cap_usd:.2f}")
    if not lines:
        return None
    return ("[model-drift-watch] Did not run: " + "; ".join(lines) + ". Nothing was spent.\n"
            f"Next: raise the cap ({SET}max_tokens_per_check <n> / max_usd_per_check <usd>), "
            "or ask fewer questions with the categories setting.")


def estimate_text(est, s, target_id: str, n_items: int) -> str:
    cap_usd = s.max_usd_per_check if s.max_usd_per_check > 0 else None
    price = est.price
    lines = [f"model-drift-watch estimate for {target_id}: {n_items} questions x {s.samples_per_item} sample(s).",
             f"Expected: {_n(est.expected_tokens)} tokens ({_n(est.input_tokens)} in, {_n(est.expected_output_tokens)} out, "
             f"output from {est.output_source}), {_usd(est.expected_usd)}.",
             f"Worst case, every answer at its length limit: {_n(est.worst_tokens)} tokens, {_usd(est.worst_usd)}."]
    if price:
        lines.append(f"Price: ${price.input_per_mtok:g} in / ${price.output_per_mtok:g} out per million tokens ({price.source}).")
    else:
        lines.append(f"Price unknown, so only the token cap applies. Set it with {SET}price_input_per_mtok <usd> "
                     "and price_output_per_mtok <usd>.")
    if s.confirm:
        lines.append("On a day that looks worse, a confirmation run costs the same again.")
        two_usd = 2 * (est.expected_usd or 0)
        if 2 * est.expected_tokens > s.max_tokens_per_check or (price and cap_usd is not None and two_usd > cap_usd):
            lines.append(f"Note: two runs do not fit your cap, so a confirmation would be skipped and a real drop "
                         f"reported only as 'possible'. Raise the cap to about {_n(2.2 * est.expected_tokens)} tokens"
                         + (f" / ${2.2 * (est.expected_usd or 0):.2f}" if price else "") + ".")
    over = refusal(est, s, cap_usd, n_items)
    cap = f"{_n(s.max_tokens_per_check)} tokens" + (f" / ${cap_usd:.2f}" if cap_usd is not None and price else "")
    if over:
        lines.append(f"Cap {cap}: a check would NOT start.")
    elif est.worst_tokens > s.max_tokens_per_check or (price and cap_usd is not None and (est.worst_usd or 0) > cap_usd):
        lines.append(f"Cap {cap}: fits the expected cost. If answers run long, the check stops at the cap and "
                     "reports a failed measurement, never drift.")
    else:
        lines.append(f"Cap {cap}: fits even the worst case.")
    lines.append("A scheduled check also spends one short Hermes agent turn to call the tool (see README).")
    return "\n".join(lines)


def accuracy_dict(alpha, cmp1, cmp2, pooled, rec1) -> Dict[str, Any]:
    out: Dict[str, Any] = {"answered": rec1["answered"], "correct": rec1["correct"]}
    for name, c in (("first", cmp1), ("confirmation", cmp2), ("both_runs", pooled)):
        if c is not None:
            out[name] = {"now": round(c.overall.cur_rate, 4), "baseline": round(c.overall.base_rate, 4),
                         "change_points": round(c.overall.change_points, 2), "p_lower": c.overall.p_lower,
                         "p_upper": c.overall.p_upper,
                         "ci_points": [round(x, 2) for x in c.ci_points] if c.ci_points else None,
                         "drift_in": c.drift_in(alpha), "improved_in": c.improved_in(alpha),
                         "token_ratio": c.token_ratio, "token_p": c.token_p,
                         "categories": {k: {"change_points": round(t.change_points, 2), "p_lower": t.p_lower}
                                        for k, t in c.categories.items()}}
    return out


def _spent(spent: Dict[str, Any], est, runs: int) -> str:
    where = f"in {runs} runs " if runs > 1 else ""
    return (f"Spent {_n(spent['total'])} tokens {where}(estimate {_n(est.expected_tokens)} per run), "
            f"{_usd(spent.get('usd'))}.")


def _errors(rec: Dict[str, Any]) -> str:
    errs = ", ".join(f"{k} {v}" for k, v in sorted(rec["errors"].items(), key=lambda kv: -kv[1]))
    return f"{rec['samples'] - rec['answered']} of {rec['samples']} answers could not be used ({errs})."


def _hints(rec: Dict[str, Any]) -> List[str]:
    out = []
    for code, _ in sorted(rec["errors"].items(), key=lambda kv: -kv[1])[:2]:
        hint = HINTS.get(code, HINTS["error"]).format(set=SET)
        if hint not in out:
            out.append(hint)
    return out


def _cats(c, sign: int) -> str:
    moved = sorted(((k, t.change_points) for k, t in c.categories.items() if sign * t.change_points > 0),
                   key=lambda kv: sign * -kv[1])[:3]
    return ", ".join(f"{k} {v:+.0f} points" for k, v in moved)


def _tokens(c, always: bool = False) -> str:
    if c is None or c.token_ratio is None:
        return ""
    if not always and (abs(c.token_ratio - 1) < 0.10 or (c.token_p or 1) >= 0.05):
        return ""
    change = c.token_ratio - 1
    word = "longer" if change > 0 else "shorter"
    return f"Answers are {abs(change) * 100:.0f}% {word} than in the baseline (sign test {_p(c.token_p)})."


def render(*, status: str, target, s, rec1, rec2, cmp1, cmp2, base, est, spent, price, pooled=None,
           areas=(), note: str = "", counted: bool = True, varied=None) -> str:
    t = target.id
    o = cmp1.overall if cmp1 else None
    ci = (f", 95% CI {cmp1.ci_points[0]:+.0f} to {cmp1.ci_points[1]:+.0f}" if cmp1 and cmp1.ci_points else "")
    p_shown = (o.p_upper if status == "improved" else o.p_lower) if o else 1.0
    acc_line = (f"Accuracy {_pct(o.cur_rate)} now vs {_pct(o.base_rate)} baseline "
                f"({o.change_points:+.0f} points{ci}), exact test {_p(p_shown)}") if o else ""
    lines: List[str] = []
    if status == "measurement_failed":
        lines = [f"[model-drift-watch] Could not measure {t} this time. This is not a verdict on the model.",
                 _errors(rec1), *_hints(rec1)]
    elif status in ("collecting", "baseline_ready"):
        done = min(base.runs + (1 if counted else 0), s.baseline_runs)
        right = rec1["correct"] / rec1["answered"] if rec1["answered"] else 0
        if status == "collecting":
            lines = [f"[model-drift-watch] Baseline for {t}: {done} of {s.baseline_runs} runs collected "
                     f"({_pct(right)} right this time). Testing starts after run {s.baseline_runs}."]
        else:
            lines = [f"[model-drift-watch] Baseline ready for {t}: {s.baseline_runs} runs, {_pct(right)} right in the last one. "
                     "From the next run on, every check is tested against this baseline and you hear from me "
                     "only when something changes."]
            if varied:
                n_var, n_all = varied
                lines.append(f"{n_var} of {n_all} questions were answered right only some of the time; the rest "
                             "were always right or always wrong, so they count only once they change.")
                if n_var < 5:
                    lines.append("With so few varying questions, only drops that make the model miss questions it "
                                 "used to get right can be seen. For finer detection add harder questions of your "
                                 "own (see `hermes model-drift-watch validate-suite` for the folder).")
    elif status == "model_changed":
        lines = [f"[model-drift-watch] MODEL CHANGED: answers for {t} came from \"{rec1['served_major']}\", "
                 f"but the baseline was answered by \"{base.served}\".",
                 "Usually Hermes fell back to another model after the main provider failed, or the provider "
                 "changed what runs behind the name."]
        if o:
            lines.append(f"On the same questions: {_pct(o.cur_rate)} now vs {_pct(o.base_rate)} baseline.")
        lines.append("Next: if this is the model you want, run `hermes model-drift-watch reset-baseline`; "
                     "if not, look for fallback lines in `hermes logs`.")
    elif status == "drift":
        lines = [f"[model-drift-watch] DRIFT: {t} is answering worse than its own baseline.", acc_line + "."]
        if cmp2:
            both = f"; both runs together {_p(pooled.overall.p_lower)}" if pooled else ""
            lines.append(f"A second run right after agreed: {_pct(cmp2.overall.cur_rate)} "
                         f"({cmp2.overall.change_points:+.0f} points, {_p(cmp2.overall.p_lower)}{both}).")
        named = [a for a in areas if a != "overall"]
        if named:
            lines.append("Significant in: " + ", ".join(
                f"{a} {(pooled or cmp1).categories[a].change_points:+.0f} points" for a in named) + ".")
        else:
            worst = _cats(cmp1, -1)
            if worst:
                lines.append(f"Biggest drops: {worst}.")
        if _tokens(cmp1):
            lines.append(_tokens(cmp1))
        lines.append(f"Served model: {rec1['served_major'] or 'not reported'} (baseline: {base.served or 'not reported'}).")
        lines.append("Next: if you changed this model on purpose, run `hermes model-drift-watch reset-baseline`. "
                     "Otherwise compare with another provider of the same model; `hermes model-drift-watch status` has the numbers.")
    elif status == "token_shift":
        lines = [f"[model-drift-watch] ANSWERS CHANGED: {t}. " + _tokens(pooled or cmp1, always=True) + (" A second run agreed." if cmp2 else ""),
                 acc_line + " (accuracy alone is not a significant change).",
                 "A sudden change in length often means a different reasoning effort or model behind the same name. "
                 "Next: watch the next runs; `hermes model-drift-watch status` has the numbers."]
    elif status == "suspect":
        lines = [f"[model-drift-watch] POSSIBLE DRIFT (unconfirmed): {t}. {acc_line}.",
                 _tokens(cmp1),
                 "The confirmation run did not happen or could not be measured, so this is not a verdict yet. "
                 "The next scheduled run will tell."]
        if rec2:
            lines += _hints(rec2)
    elif status == "unconfirmed":
        lines = [f"[model-drift-watch] {t}: one run looked worse ({acc_line}), but a second run did not confirm it. "
                 "Treated as noise."]
    elif status == "improved":
        lines = [f"[model-drift-watch] {t} is answering better than its baseline. {acc_line}."]
    else:
        lines = [f"[model-drift-watch] OK: {t} matches its baseline. {acc_line}."]
    if note:
        lines.append(note)
    if status == "drift" and not cmp2:
        lines.insert(2, "(Decided on one run: confirmation is turned off.)")
    if cmp1 and cmp1.pending_items and status not in ("collecting", "baseline_ready"):
        lines.append(f"{cmp1.pending_items} new question(s) are still collecting their own baseline.")
    lines.append(_spent(spent, est, 2 if rec2 else 1))
    return "\n".join(x for x in lines if x)
