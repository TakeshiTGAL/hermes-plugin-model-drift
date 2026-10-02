"""End-to-end checks against a mock model: baseline, drift, no false alarms, failures, budget."""

import pytest

from mock_model import MockModel

ALARMS = {"drift", "suspect", "token_shift", "model_changed"}


@pytest.fixture
def target(mods):
    return mods.targets.Target(provider="custom", model="mock-model", role="main", override=False)


def settings(mods, **kw):
    s = mods.runner.Settings()
    s.baseline_min_gap_hours = 0  # tests run many "days" within a second
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def check(mods, store, items, mock, s, clock, target, price=None, code=True):
    return mods.runner.run_check(mock.caller(mods.runner.Reply), target, s, items, store, price,
                                 mods.sandbox.run_python if code else None, clock=clock, sleep=clock.sleep)


def build_baseline(mods, store, items, s, clock, target, seed0=100):
    statuses = [check(mods, store, items, MockModel(items, seed=seed0 + i), s, clock, target).status
                for i in range(s.baseline_runs)]
    assert statuses == ["collecting"] * (s.baseline_runs - 1) + ["baseline_ready"]


def test_baseline_then_quiet_when_nothing_changed(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    out = check(mods, store, suite_items, MockModel(suite_items, seed=7), s, clock, target)
    assert out.status in ("stable", "unconfirmed", "improved")
    assert out.silent
    assert "[model-drift-watch]" in out.message


def test_detects_a_model_that_gets_twenty_percent_wrong(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    out = check(mods, store, suite_items, MockModel(suite_items, seed=8, degrade=0.2), s, clock, target)
    assert out.status == "drift", out.message
    assert not out.silent and "DRIFT" in out.message and "A second run right after agreed" in out.message
    assert len(out.details["runs"]) == 2  # first run + confirmation
    both = out.details["accuracy"]["both_runs"]
    assert both["change_points"] <= -10 and both["p_lower"] < 0.01


def test_detection_rate_over_ten_degraded_checks(mods, store, suite_items, clock, target):
    # a frozen 5-run baseline: otherwise degraded runs that pass unnoticed join the growing baseline
    s = settings(mods, baseline_max_runs=5)
    build_baseline(mods, store, suite_items, s, clock, target)
    statuses = [check(mods, store, suite_items, MockModel(suite_items, seed=300 + i, degrade=0.2), s, clock,
                      target).status for i in range(10)]
    assert statuses.count("drift") >= 5, statuses  # 34 questions: see README Table 1 for the rate


def test_ten_runs_of_an_unchanged_model_raise_no_alarm(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    outs = [check(mods, store, suite_items, MockModel(suite_items, seed=200 + i), s, clock, target)
            for i in range(10)]
    statuses = [o.status for o in outs]
    assert not any(st in ALARMS | {"measurement_failed"} for st in statuses), statuses
    assert all(o.silent for o in outs), statuses


def test_rate_limited_on_every_request_is_a_failed_measurement(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    out = check(mods, store, suite_items, MockModel(suite_items, seed=9, rate_limit=1.0), s, clock, target)
    assert out.status == "measurement_failed", out.message
    assert not out.silent and "DRIFT" not in out.message
    assert "Could not measure" in out.message and "not a verdict" in out.message
    asked = [i for i in suite_items if i.grader["type"] != "python"]  # code_execution is off by default
    assert len(asked) == 34 and out.details["errors"] == {"rate_limited": 34}


def test_rate_limits_are_a_failed_measurement_not_drift(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    out = check(mods, store, suite_items, MockModel(suite_items, seed=9, rate_limit=0.9), s, clock, target)
    assert out.status == "measurement_failed", out.message
    assert "not a verdict" in out.message and "rate-limiting" in out.message
    assert out.details["errors"].get("rate_limited", 0) > 0
    # the failed run must not leak into the next comparison
    nxt = check(mods, store, suite_items, MockModel(suite_items, seed=10), s, clock, target)
    assert nxt.status not in ALARMS | {"measurement_failed"}


def test_network_outage_is_a_failed_measurement(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    mock = MockModel(suite_items, seed=9, network=1.0)
    out = check(mods, store, suite_items, mock, s, clock, target)
    assert out.status == "measurement_failed"
    assert "could not be reached" in out.message
    # the breaker stops after the first 8 requests (x3 attempts) instead of retrying all 43 questions
    assert mock.calls <= (2 * s.concurrency + s.concurrency) * 3


def test_retries_recover_from_a_few_rate_limits(mods, store, suite_items, clock, target):
    s = settings(mods)
    out = check(mods, store, suite_items, MockModel(suite_items, seed=9, rate_limit=0.15), s, clock, target)
    assert out.status == "collecting"
    assert out.details["errors"] == {} or sum(out.details["errors"].values()) <= 2


def test_auth_failure_stops_after_the_first_answers(mods, store, suite_items, clock, target):
    calls = []

    class AuthenticationError(Exception):
        status_code = 401

    def caller(messages, **kw):
        calls.append(1)
        raise AuthenticationError("invalid x-api-key")

    s = settings(mods)
    out = mods.runner.run_check(caller, target, s, suite_items, store, None, None, clock=clock, sleep=clock.sleep)
    assert out.status == "measurement_failed"
    assert "refused the credentials" in out.message
    assert len(calls) <= s.concurrency


def test_fallback_answers_are_excluded(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    out = check(mods, store, suite_items, MockModel(suite_items, seed=11, reroute=0.1), s, clock, target)
    assert out.details["errors"].get("rerouted", 0) > 0
    assert out.status not in ALARMS | {"measurement_failed"}


def test_a_different_served_model_is_reported_as_such(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    out = check(mods, store, suite_items, MockModel(suite_items, seed=12, served="mock-model-2"), s, clock, target)
    assert out.status == "model_changed"
    assert '"mock-model-2"' in out.message and '"mock-model-1"' in out.message


def test_shorter_answers_are_flagged(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    out = check(mods, store, suite_items, MockModel(suite_items, seed=13, out_tokens=40), s, clock, target)
    assert out.status == "token_shift", out.message
    assert "shorter" in out.message


def test_estimate_versus_actual_tokens(mods, store, suite_items, clock, target):
    price = mods.cost.Price(5.0, 25.0, "test")
    s = settings(mods)
    first = check(mods, store, suite_items, MockModel(suite_items, seed=1), s, clock, target, price=price)
    est, spent = first.details["estimate"], first.details["spent"]
    assert est["output_source"] == "first-run guess"
    assert spent["total"] <= s.max_tokens_per_check and spent["usd"] <= s.max_usd_per_check
    second = check(mods, store, suite_items, MockModel(suite_items, seed=2), s, clock, target, price=price)
    est2, spent2 = second.details["estimate"], second.details["spent"]
    assert est2["output_source"] == "measured"
    # after one run the estimate uses the model's own measured answer lengths; the input estimate
    # stays on the safe side (the mock counts 4 characters per token, also for Japanese)
    assert est2["expected_output_tokens"] == spent2["output"]
    assert 0 <= est2["expected_tokens"] - spent2["total"] < 0.15 * spent2["total"]
    assert "Spent" in second.message and "estimate" in second.message


def test_refuses_to_start_when_the_estimate_is_over_the_cap(mods, store, suite_items, clock, target):
    mock = MockModel(suite_items, seed=1)
    out = check(mods, store, suite_items, mock, settings(mods, max_tokens_per_check=2000), clock, target)
    assert out.status == "refused" and mock.calls == 0
    assert "Nothing was spent" in out.message and "max_tokens_per_check" in out.message
    price = mods.cost.Price(5.0, 25.0, "test")
    out = check(mods, store, suite_items, mock, settings(mods, max_usd_per_check=0.01), clock, target, price=price)
    assert out.status == "refused" and mock.calls == 0


def test_never_spends_past_the_cap(mods, store, suite_items, clock, target):
    """Cap above the expected cost but below the worst case: requests stop before the cap can be passed."""
    mock = MockModel(suite_items, seed=1, out_tokens=900)  # long answers, far above the first-run guess
    s = settings(mods, max_tokens_per_check=20000)
    out = check(mods, store, suite_items, mock, s, clock, target)
    assert out.details["spent"]["total"] <= 20000
    assert out.details["errors"].get("budget", 0) > 0
    assert out.status == "measurement_failed" and "stopped at your cap" in out.message


def test_deadline_is_respected(mods, store, suite_items, clock, target):
    def slow(messages, **kw):
        clock.t += 60
        return mods.runner.Reply("ANSWER: 1", "m", 10, 10)

    s = settings(mods, check_deadline=120, concurrency=1)
    out = mods.runner.run_check(slow, target, s, suite_items, store, None, None, clock=clock, sleep=clock.sleep)
    asked = [i for i in suite_items if i.grader["type"] != "python"]  # code_execution is off by default
    assert out.details["errors"].get("deadline", 0) >= len(asked) - 5
    assert out.status == "measurement_failed"


def test_one_check_at_a_time(mods, store, suite_items, clock, target):
    s = settings(mods)
    with store.lock(target.id):
        out = check(mods, store, suite_items, MockModel(suite_items), s, clock, target)
    assert out.status == "busy" and "Nothing was spent" in out.message


def test_new_questions_wait_for_their_own_baseline(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items[:-3], s, clock, target)
    out = check(mods, store, suite_items, MockModel(suite_items, seed=20), s, clock, target)
    assert out.details["accuracy"]["first"]
    assert "3 new question(s) are still collecting" in out.message or out.silent


def test_reset_baseline_starts_over(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    store.reset_baseline(target.id, mods.runner._now(), "test")
    out = check(mods, store, suite_items, MockModel(suite_items, seed=21, degrade=0.5), s, clock, target)
    assert out.status == "collecting"  # nothing to compare with yet, so no alarm


def test_code_questions_skipped_by_default(mods, store, suite_items, clock, target):
    assert mods.runner.Settings().code_execution is False
    mock = MockModel(suite_items, seed=1)
    out = check(mods, store, suite_items, mock, settings(mods), clock, target)
    assert mock.calls == len([i for i in suite_items if i.grader["type"] != "python"]) == 34
    assert out.status == "collecting"


def test_code_questions_run_when_turned_on(mods, store, suite_items, clock, target):
    mock = MockModel(suite_items, seed=1)
    out = check(mods, store, suite_items, mock, settings(mods, code_execution=True), clock, target)
    assert mock.calls == len(suite_items) == 44
    assert out.status == "collecting" and out.details["errors"] == {}


def test_baseline_runs_must_be_spread_out(mods, store, suite_items, clock, target):
    s = settings(mods, baseline_min_gap_hours=12)
    outs = [check(mods, store, suite_items, MockModel(suite_items, seed=i), s, clock, target) for i in range(3)]
    assert [o.status for o in outs] == ["collecting"] * 3
    assert outs[0].details["baseline_runs"] == "1/5" and outs[2].details["baseline_runs"] == "1/5"
    assert "not part of the baseline" in outs[1].message


def test_confirmation_skipped_when_it_cannot_fit_the_cap(mods, store, suite_items, clock, target):
    s = settings(mods)
    build_baseline(mods, store, suite_items, s, clock, target)
    s.max_tokens_per_check = 9000  # one run (~7k) fits, two do not
    out = check(mods, store, suite_items, MockModel(suite_items, seed=8, degrade=0.3), s, clock, target)
    assert out.status == "suspect", out.message
    assert "would not fit" in out.message and len(out.details["runs"]) == 1


def test_one_deadline_for_the_whole_call(mods, store, suite_items, clock, target):
    """A shared deadline that has already passed stops the second model before it asks anything."""
    s = settings(mods)
    mock = MockModel(suite_items, seed=1)
    out = mods.runner.run_check(mock.caller(mods.runner.Reply), target, s, suite_items, store, None, None,
                                clock=clock, sleep=clock.sleep, deadline=clock() - 1)
    assert mock.calls == 0 and out.status == "measurement_failed"


def test_baseline_grows_with_clean_runs_only(mods, store, suite_items, clock, target):
    s = settings(mods, baseline_max_runs=7)
    build_baseline(mods, store, suite_items, s, clock, target)

    def base_runs():
        clean = {c["id"] for c in store.checks(target.id) if c["status"] in mods.runner.CLEAN}
        return mods.runner.baseline_from(store.runs(target.id), 0, s.baseline_runs, 0, s.baseline_max_runs, clean).runs

    assert base_runs() == 5
    drift = check(mods, store, suite_items, MockModel(suite_items, seed=50, degrade=0.4), s, clock, target)
    assert drift.status == "drift" and base_runs() == 5  # a bad run never joins the baseline
    for i in range(4):
        check(mods, store, suite_items, MockModel(suite_items, seed=60 + i), s, clock, target)
    assert base_runs() == 7  # clean runs joined, up to baseline_max_runs, then frozen


def test_changed_questions_rebuild_their_baseline_after_the_freeze(mods, store, suite_items, clock, target):
    """Regression: after the baseline froze, edited questions (e.g. a new max_tokens_scale) must collect
    their own baseline and testing must resume, instead of collecting forever in silence."""
    import dataclasses
    s = settings(mods, baseline_max_runs=6)
    build_baseline(mods, store, suite_items, s, clock, target)
    check(mods, store, suite_items, MockModel(suite_items, seed=70), s, clock, target)  # baseline now frozen at 6
    scaled = [dataclasses.replace(i, max_tokens=i.max_tokens * 4) for i in suite_items]
    outs = [check(mods, store, scaled, MockModel(scaled, seed=80 + i), s, clock, target) for i in range(s.baseline_runs)]
    assert all(o.status == "collecting" for o in outs)
    assert "collecting their own baseline: 1 of 5" in outs[0].message
    drift = check(mods, store, scaled, MockModel(scaled, seed=90, degrade=0.4), s, clock, target)
    assert drift.status == "drift", drift.message


def test_added_question_gets_tested_after_the_freeze(mods, store, suite_items, clock, target):
    s = settings(mods, baseline_max_runs=5)
    build_baseline(mods, store, suite_items[:-1], s, clock, target)
    for i in range(s.baseline_runs):
        out = check(mods, store, suite_items, MockModel(suite_items, seed=100 + i), s, clock, target)
        assert "1 new question(s)" in out.message
    out = check(mods, store, suite_items, MockModel(suite_items, seed=120), s, clock, target)
    assert "new question" not in out.message and out.details["accuracy"]["first"]
