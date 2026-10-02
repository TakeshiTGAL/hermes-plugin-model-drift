"""A deterministic stand-in for a model, used by the tests and by the end-to-end HTTP mock.

It answers each question with the suite's own reference answer with probability ``p_right``
(per question, so some questions are "sometimes right" like on a real model), and with a wrong
answer otherwise. ``degrade`` forces a fixed share of the questions to be answered wrong, which
is what a quietly weakened model looks like to the monitor.
"""

from __future__ import annotations

import hashlib
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


def wrong_answer(item) -> str:
    if item.extract == "code":
        entry = item.grader["entry"]
        return f"```python\ndef {entry}(*args, **kwargs):\n    return None\n```"
    if item.extract == "answer_line":
        return "Let me think.\nANSWER: 999999"
    return "Sure! Here you go."


def item_p(key: str, base: float, spread: float) -> float:
    """Per-question success probability in [base - spread, base + spread], stable per key."""
    u = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return min(1.0, max(0.0, base - spread + 2 * spread * u))


class RateLimitError(Exception):
    status_code = 429


class APIConnectionError(Exception):
    pass


@dataclass
class MockModel:
    items: list
    seed: int = 0
    p_right: float = 0.9
    spread: float = 0.1
    degrade: float = 0.0  # share of questions forced wrong (picked deterministically)
    rate_limit: float = 0.0  # share of calls answered with HTTP 429
    network: float = 0.0  # share of calls failing to connect
    served: str = "mock-model-1"
    reroute: float = 0.0  # share of calls answered by another model (Hermes fallback)
    out_tokens: int = 120
    calls: int = 0
    by_prompt: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.by_prompt = {i.messages()[-1]["content"]: i for i in self.items}
        keys = sorted(i.key for i in self.items)
        rng = random.Random(f"degrade-{self.seed}")
        self.forced_wrong = set(rng.sample(keys, round(self.degrade * len(keys)))) if self.degrade else set()
        self._lock = threading.Lock()
        self._per_key: Dict[str, int] = {}

    def answer(self, content: str):
        """Return (text, served_model, input_tokens, output_tokens) or raise like a provider would."""
        item = self.by_prompt[content]
        with self._lock:  # randomness depends on (seed, question, attempt), not on thread timing
            self.calls += 1
            n = self._per_key[item.key] = self._per_key.get(item.key, 0) + 1
        rng = random.Random(f"{self.seed}|{item.key}|{n}")
        r = rng.random()
        if r < self.rate_limit:
            raise RateLimitError("429 Too Many Requests")
        if r < self.rate_limit + self.network:
            raise APIConnectionError("Connection error.")
        served = self.served
        if rng.random() < self.reroute:
            served = "fallback-model-9"
        right = item.key not in self.forced_wrong and rng.random() < item_p(item.key, self.p_right, self.spread)
        text = item.reference if right else wrong_answer(item)
        return text, served, max(1, len(content) // 4), self.out_tokens

    def caller(self, reply_cls):
        """A runner.Caller over this mock (reply_cls is the plugin's runner.Reply)."""
        def call(messages, *, max_tokens, temperature, timeout, target):
            text, served, tin, tout = self.answer(messages[-1]["content"])
            return reply_cls(text=text, model=served, input_tokens=tin, output_tokens=min(tout, max_tokens))
        return call
