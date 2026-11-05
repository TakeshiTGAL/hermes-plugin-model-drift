"""Every shape of the model field, and every rejection text, has one expected next step.

The two decisions used to grow by one special case at a time. This file is the table.
A row that is not listed here is not a supported input.
"""

import pytest


# cfg -> watched id, or None when Hermes has no model to watch.
MODEL_ROWS = [
    ("missing", {}, None),
    ("none", {"model": None}, None),
    ("empty string", {"model": ""}, None),
    ("whitespace", {"model": " \n\t"}, None),
    ("integer", {"model": 0}, None),
    ("bool", {"model": False}, None),
    ("list", {"model": []}, None),
    ("empty dict", {"model": {}}, None),
    ("dict blank default", {"model": {"default": "", "provider": "openai"}}, None),
    ("dict whitespace default", {"model": {"default": "  ", "provider": "openai"}}, None),
    ("string name", {"model": "gpt-4o"}, "auto:gpt-4o"),
    ("string and root provider", {"model": "gpt-4o", "provider": "openai"}, "openai:gpt-4o"),
    ("provider auto", {"model": "gpt-4o", "provider": "auto"}, "auto:gpt-4o"),
    ("blank provider", {"model": "gpt-4o", "provider": "  "}, "auto:gpt-4o"),
    ("name is the word default", {"model": "default", "provider": "ollama"}, "ollama:default"),
    ("accent", {"model": "José", "provider": "openai"}, "openai:José"),
    ("slash and question", {"model": {"provider": "custom", "default": "a/b?x"}}, "custom:a/b?x"),
    ("dict default", {"model": {"provider": "openrouter", "default": "anthropic/claude-x"}},
     "openrouter:anthropic/claude-x"),
    ("dict model key", {"model": {"provider": "openai", "model": "gpt-4o"}}, "openai:gpt-4o"),
    ("dict name key", {"model": {"provider": "custom", "name": "local-model"}}, "custom:local-model"),
    ("blank default then model key",
     {"model": {"default": "  ", "model": "gpt-4o", "provider": "openai"}}, "openai:gpt-4o"),
    ("default wins over model",
     {"model": {"default": "keep", "model": "other", "provider": "openai"}}, "openai:keep"),
    ("nested block", {"model": {"default": {"model": "gpt-4o", "provider": "openai"}}}, "openai:gpt-4o"),
    ("dict provider blank uses root",
     {"provider": "openai", "model": {"default": "gpt-4o", "provider": ""}}, "openai:gpt-4o"),
    ("padded name", {"model": "  gpt-4o  ", "provider": "openai"}, "openai:gpt-4o"),
    ("not a config", None, None),
]


# code, error_details, phrases that must appear, phrases that must not.
HINT_ROWS = [
    ("model does not exist", "bad_request",
     ["NotFoundError 404: The model `mock-model` does not exist"],
     ["does not exist", "hermes model"], ["temperature", "send_temperature"]),
    ("temperature", "bad_request",
     ["BadRequestError 400: Unsupported parameter: temperature"],
     ["temperature", "send_temperature false"], ["does not exist"]),
    ("other 400", "bad_request",
     ["BadRequestError 400: max_tokens is too large"],
     ["rejected", "hermes logs"], ["temperature", "send_temperature", "does not exist"]),
    ("no detail", "bad_request",
     [],
     ["rejected", "hermes logs"], ["temperature"]),
    ("422", "bad_request",
     ["UnprocessableEntityError 422: validation failed"],
     ["rejected"], ["temperature", "does not exist"]),
    ("model missing and temperature", "bad_request",
     ["BadRequestError 400: model does not exist and temperature was rejected"],
     ["does not exist", "send_temperature false"], []),
    ("Temperature capitalised", "bad_request",
     ["BadRequestError 400: Temperature must be 1 for this model"],
     ["send_temperature false"], ["does not exist"]),
    ("model_not_found", "bad_request",
     ["NotFoundError 404: model_not_found"],
     ["does not exist"], ["temperature"]),
    ("unknown model", "bad_request",
     ["BadRequestError 400: unknown model 'x'"],
     ["does not exist"], ["temperature"]),
    ("invalid model", "bad_request",
     ["BadRequestError 400: invalid model id"],
     ["does not exist"], ["temperature"]),
    ("404 without a model phrase", "bad_request",
     ["NotFoundError 404: "],
     ["404", "hermes model"], ["temperature", "send_temperature", "does not exist"]),
    ("unprefixed model phrase", "bad_request",
     ["model does not exist"],
     ["does not exist", "hermes model"], ["temperature"]),
    ("temperature on a different error", "bad_request",
     ["InternalServerError 500: temperature exploded", "BadRequestError 400: max_tokens is too large"],
     ["rejected", "hermes logs"], ["temperature", "does not exist"]),
    ("no such model", "bad_request",
     ["BadRequestError 400: no such model"],
     ["does not exist"], ["temperature"]),
    ("model not found", "bad_request",
     ["NotFoundError 404: model not found"],
     ["does not exist"], ["temperature"]),
    ("server error keeps its own hint", "server_error",
     ["InternalServerError 500: temperature exploded"],
     ["server errors"], ["send_temperature"]),
    ("auth keeps its own hint", "auth",
     ["AuthenticationError 401: invalid key"],
     ["refused the credentials"], ["temperature"]),
    ("no provider keeps its own hint", "no_provider",
     ["no provider configured"],
     ["No model is configured", "hermes model-drift-watch run"], ["temperature", "auto:default"]),
    ("rate limit keeps its own hint", "rate_limited",
     ["RateLimitError 429: slow down"],
     ["rate-limiting"], ["temperature"]),
    ("network keeps its own hint", "network",
     ["APIConnectionError: temperature host down"],
     ["could not be reached"], ["send_temperature"]),
]


def _hint(mods, code, details):
    rec = {"errors": {code: 1}, "error_details": details}
    lines = mods.report._hints(rec)
    assert len(lines) == 1
    return lines[0]


@pytest.mark.parametrize("kind,cfg,want", MODEL_ROWS, ids=[row[0] for row in MODEL_ROWS])
def test_model_field(mods, kind, cfg, want):
    found = mods.targets._main(cfg)
    if want is None:
        assert found is None
        assert mods.targets.discover(cfg if isinstance(cfg, dict) else {}) == []
    else:
        assert found is not None and found.id == want and found.role == "main" and found.override is False


@pytest.mark.parametrize("kind,code,details,need,ban", HINT_ROWS, ids=[row[0] for row in HINT_ROWS])
def test_rejection_hint(mods, kind, code, details, need, ban):
    text = _hint(mods, code, details)
    low = text.lower()
    for phrase in need:
        assert phrase.lower() in low, text
    for phrase in ban:
        assert phrase.lower() not in low, text


def _env(mods, tmp_path, cfg, caller=None):
    return mods.service.Env(
        get_config=lambda k, d: d, llm=None, store=mods.store.Store(tmp_path),
        hermes_config=lambda: cfg, caller=caller)


def test_status_on_the_empty_home_shape(mods, tmp_path):
    """The shape load_config_readonly returns before hermes model has been run."""
    st = mods.service.status(_env(mods, tmp_path, {"model": ""}))
    assert st["status"] == "refused"
    assert st.get("targets", []) == []
    assert "auto:default" not in st["message"]
    assert "hermes model" in st["message"]
    assert "status" in st["message"]


def test_blank_model_does_not_spend_and_hides_a_phantom_history(mods, tmp_path):
    calls = []
    env = _env(mods, tmp_path, {"model": ""}, caller=lambda *a, **k: calls.append(1))
    env.store.append_check("auto:default", {"status": "measurement_failed", "ts": "t", "message": "old"})
    assert "auto:default" not in mods.service.status(env)["message"]
    run = mods.service.run(env)
    est = mods.service.estimate(env)
    assert calls == []
    assert run["status"] == "refused" and "hermes model" in run["message"]
    assert est["status"] == "refused" and "hermes model" in est["message"]
    assert "No models found" in mods.service.targets_text(env)


def test_a_real_model_named_default_is_watched(mods, tmp_path):
    st = mods.service.status(_env(mods, tmp_path, {"model": "default", "provider": "ollama"}))
    assert st["targets"][0]["target"] == "ollama:default"
    assert "no checks yet" in st["message"]
    assert "No model is configured" not in st["message"]


def _one_target(mods):
    return mods.targets.Target(provider="custom", model="mock-model", role="main", override=False)


def test_run_check_says_the_model_is_missing_on_a_404(mods, store, suite_items, clock):
    class NotFoundError(Exception):
        status_code = 404

    def caller(messages, **kw):
        raise NotFoundError("The model `mock-model` does not exist")

    s = mods.runner.Settings()
    out = mods.runner.run_check(caller, _one_target(mods), s, suite_items[:1], store, None, None,
                                clock=clock, sleep=clock.sleep)
    assert out.status == "measurement_failed"
    assert "does not exist" in out.message
    assert "temperature" not in out.message.lower()
    assert "drift" not in out.status


def test_run_check_mentions_temperature_only_when_the_error_does(mods, store, suite_items, clock):
    class BadRequestError(Exception):
        status_code = 400

    def caller(messages, **kw):
        raise BadRequestError("Unsupported parameter: temperature")

    s = mods.runner.Settings()
    out = mods.runner.run_check(caller, _one_target(mods), s, suite_items[:1], store, None, None,
                                clock=clock, sleep=clock.sleep)
    assert out.status == "measurement_failed"
    assert "send_temperature false" in out.message
    assert "does not exist" not in out.message
