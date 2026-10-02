"""Load the repo root as a package (as Hermes does) and give tests a mock model and a temp store."""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PACKAGE = "model_drift_plugin"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for mock_model


def _load_plugin():
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    spec = importlib.util.spec_from_file_location(
        PACKAGE, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


PLUGIN = _load_plugin()


def sub(name):
    return importlib.import_module(f"{PACKAGE}.{name}")


@pytest.fixture
def plugin():
    return PLUGIN


@pytest.fixture
def mods():
    class M:
        runner = sub("runner")
        stats = sub("stats")
        suite = sub("suite")
        graders = sub("graders")
        sandbox = sub("sandbox")
        store = sub("store")
        cost = sub("cost")
        report = sub("report")
        service = sub("service")
        targets = sub("targets")
    return M


@pytest.fixture
def suite_items(mods):
    return mods.suite.load_suite([]).items


@pytest.fixture
def store(mods, tmp_path):
    return mods.store.Store(tmp_path / "data")


class FakeClock:
    """Monotonic clock whose sleep() advances time instantly (retries/backoff without waiting)."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def clock():
    return FakeClock()
