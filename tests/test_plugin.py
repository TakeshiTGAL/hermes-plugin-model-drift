"""The plugin surface: what register() declares, and that plugin.yaml agrees with it."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _yaml(path):
    try:
        from ruamel.yaml import YAML
        return YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))
    except ImportError:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))


class RecordingCtx:
    def __init__(self, config=None):
        self.tools, self.commands, self.cli, self.hooks = {}, {}, {}, []
        self.config = config or {}

    def register_tool(self, name, toolset, schema, handler, **kw):
        self.tools[name] = (toolset, schema, handler)

    def register_command(self, name, handler, **kw):
        self.commands[name] = handler

    def register_cli_command(self, name, help, setup_fn, handler_fn=None, description=""):
        self.cli[name] = (setup_fn, handler_fn)

    def register_hook(self, *a, **k):
        self.hooks.append(a)

    def get_config(self, key, default=None):
        return self.config.get(key, default)


def test_registered_capabilities_match_manifest(plugin):
    ctx = RecordingCtx()
    plugin.register(ctx)
    manifest = _yaml(ROOT / "plugin.yaml")
    assert manifest["name"] == "model-drift-watch" and manifest["manifest_version"] == 2
    assert manifest["requires_hermes"] == ">=0.21.5"
    assert sorted(ctx.tools) == sorted(manifest["provides_tools"]) == ["model_drift"]
    assert ctx.hooks == [] and "provides_hooks" not in manifest and "requires_env" not in manifest
    assert list(ctx.commands) == [manifest["name"]] and list(ctx.cli) == [manifest["name"]]
    assert not (ROOT / "catalog-entry.yaml").exists()  # the catalog entry lives in hermes-agent, pinned to a sha


def test_settings_schema_matches_code(plugin, mods):
    schema = _yaml(ROOT / "plugin.yaml")["config_schema"]
    defaults = vars(mods.runner.Settings())
    assert set(schema) == set(defaults)
    for key, spec in schema.items():
        assert spec["default"] == defaults[key], key


def test_bad_settings_fall_back_with_a_message(mods):
    s, problems = mods.runner.Settings.from_getter(
        lambda k, d: {"concurrency": 99, "alpha": 2, "notify": "sometimes", "baseline_runs": "x"}.get(k, d))
    assert s.concurrency == 16 and s.alpha == 0.01 and s.notify == "problems" and s.baseline_runs == 5
    assert len(problems) == 4


def test_tool_handler_answers_json_without_spending(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ctx = RecordingCtx()
    plugin.register(ctx)
    _, schema, handler = ctx.tools["model_drift"]
    assert schema["parameters"]["properties"]["action"]["enum"] == ["status", "estimate", "run"]
    out = json.loads(handler({"action": "nonsense"}))
    assert out["status"] == "error" and "Use status, estimate or run" in out["message"]


def test_service_end_to_end_with_mock(mods, tmp_path, suite_items):
    from mock_model import MockModel
    mock = MockModel(suite_items, seed=3)
    env = mods.service.Env(
        get_config=lambda k, d: d, llm=None, store=mods.store.Store(tmp_path),
        hermes_config=lambda: {"model": {"provider": "custom", "default": "mock-model"}},
        price_lookup=lambda p, m: (5.0, 25.0), caller=mock.caller(mods.runner.Reply))
    est = mods.service.estimate(env)
    assert "Expected:" in est["message"] and "$5 in / $25 out" in est["message"]
    first = mods.service.run(env)
    assert first["status"] == "collecting" and first["message"] == "[SILENT]"
    assert "Baseline for custom:mock-model: 1 of 5 runs collected" in first["report"]
    st = mods.service.status(env)
    assert "collecting" in st["message"] and "Not scheduled" in st["message"]


def test_targets_come_from_config(mods):
    cfg = {"model": {"provider": "openrouter", "default": "anthropic/claude-x"},
           "fallback_providers": [{"provider": "deepseek", "model": "deepseek-chat"}],
           "auxiliary": {"vision": {"provider": "auto"}, "compression": {"provider": "openrouter", "model": "g/flash"}},
           "delegation": {"provider": "nous", "model": "hermes-5"}}
    found = mods.targets.discover(cfg)
    assert [t.role for t in found] == ["main", "fallback", "auxiliary:compression", "delegation"]
    watched = mods.targets.monitored(cfg, ["deepseek:deepseek-chat", "bad"])
    assert [t.id for t in watched] == ["openrouter:anthropic/claude-x", "deepseek:deepseek-chat"]
    assert watched[0].override is False and watched[1].override is True
    snippet = mods.targets.grant_snippet(watched)
    assert "allow_model_override: true" in snippet and "- deepseek-chat" in snippet


@pytest.mark.parametrize("exc_name,status,code", [
    ("RateLimitError", 429, "rate_limited"), ("APIConnectionError", None, "network"),
    ("APITimeoutError", None, "timeout"), ("AuthenticationError", 401, "auth"),
    ("InternalServerError", 503, "server_error"), ("BadRequestError", 400, "bad_request"),
    ("PluginLlmTrustError", None, "not_permitted"),
])
def test_error_classification(mods, exc_name, status, code):
    exc = type(exc_name, (Exception,), {"status_code": status})("boom")
    assert mods.runner.classify_error(exc)[0] == code


def test_deadline_respects_hermes_tool_timeout(mods, tmp_path, suite_items):
    """A user who lowered timeouts.tools.sequential_call gets a shorter check, not a lost one."""
    seen = {}
    real = mods.runner.run_check

    def spy(*a, deadline=None, **k):
        import time
        seen["budget"] = deadline - time.monotonic()
        return real(*a, deadline=deadline, **k)

    env = mods.service.Env(
        get_config=lambda k, d: d, llm=None, store=mods.store.Store(tmp_path),
        hermes_config=lambda: {"model": {"provider": "custom", "default": "m"},
                               "timeouts": {"tools": {"sequential_call": 120}}},
        price_lookup=lambda p, m: None, caller=lambda *a, **k: (_ for _ in ()).throw(ConnectionError("x")))
    mods.service.runner.run_check = spy
    try:
        mods.service.run(env)
    finally:
        mods.service.runner.run_check = real
    assert 80 < seen["budget"] <= 90


def test_hermes_tool_limit_follows_hermes_order(mods, monkeypatch):
    lim = mods.service.hermes_tool_limit
    monkeypatch.delenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", raising=False)
    assert lim({}) == 420  # Hermes's own default when nothing is set
    assert lim({"timeouts": {"tools": {"sequential_call": 200, "concurrent_batch": 100}}}) == 200
    assert lim({"timeouts": {"tools": {"concurrent_batch": 100}}}) == 100
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "150")
    assert lim({}) == 150
    assert lim({"timeouts": {"tools": {"sequential_call": 0}}}) is None
