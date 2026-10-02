"""Token and dollar estimates made before anything is sent."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# First-run guesses for answer length, replaced by the model's own measured lengths afterwards.
DEFAULT_OUTPUT_GUESS = {"answer_line": 220, "full": 40, "code": 220}
MESSAGE_OVERHEAD = 12  # role markers and separators per message
INPUT_SAFETY = 1.3  # reservation margin on the input estimate


def estimate_text_tokens(text: str) -> int:
    """Rough tokenizer-free count: ~4 ASCII chars per token, ~1 token per CJK character."""
    ascii_chars = cjk = other = 0
    for ch in text:
        if ord(ch) < 128:
            ascii_chars += 1
        elif unicodedata.east_asian_width(ch) in ("W", "F"):
            cjk += 1
        else:
            other += 1
    return math.ceil(ascii_chars / 4 + cjk * 1.1 + other / 2)


def estimate_input_tokens(messages: Sequence[Dict[str, str]]) -> int:
    return sum(estimate_text_tokens(m["content"]) + MESSAGE_OVERHEAD for m in messages)


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    source: str

    def usd(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


def lookup_price(provider: str, model: str, settings_in: float, settings_out: float,
                 hermes_lookup: Optional[Callable[[str, str], Optional[Tuple[float, float]]]] = None) -> Optional[Price]:
    if settings_in > 0 or settings_out > 0:
        return Price(settings_in, settings_out, "your settings")
    lookup = hermes_lookup or hermes_pricing
    try:
        found = lookup(provider, model)
    except Exception:
        found = None
    if found and (found[0] or found[1]):
        return Price(found[0], found[1], "Hermes pricing data")
    if found == (0.0, 0.0):
        return Price(0.0, 0.0, "Hermes pricing data (included in your plan)")
    return None


def hermes_pricing(provider: str, model: str) -> Optional[Tuple[float, float]]:
    """Hermes's own price table (bundled snapshot, OpenRouter or models.dev, as Hermes resolves it)."""
    from agent.usage_pricing import get_pricing_entry
    entry = get_pricing_entry(model, provider=None if provider in ("", "auto") else provider)
    if entry is None or entry.input_cost_per_million is None or entry.output_cost_per_million is None:
        return None
    return float(entry.input_cost_per_million), float(entry.output_cost_per_million)


@dataclass(frozen=True)
class Estimate:
    calls: int
    input_tokens: int
    expected_output_tokens: int
    worst_output_tokens: int
    price: Optional[Price]
    output_source: str  # "measured" | "first-run guess" | "partly measured"
    input_scale: float

    @property
    def expected_tokens(self) -> int:
        return self.input_tokens + self.expected_output_tokens

    @property
    def worst_tokens(self) -> int:
        return math.ceil(self.input_tokens * INPUT_SAFETY) + self.worst_output_tokens

    @property
    def expected_usd(self) -> Optional[float]:
        return self.price.usd(self.input_tokens, self.expected_output_tokens) if self.price else None

    @property
    def worst_usd(self) -> Optional[float]:
        return self.price.usd(self.input_tokens * INPUT_SAFETY, self.worst_output_tokens) if self.price else None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(expected_tokens=self.expected_tokens, worst_tokens=self.worst_tokens,
                 expected_usd=self.expected_usd, worst_usd=self.worst_usd)
        return d


def measured_lengths(history: List[Dict[str, Any]], last_n: int = 3) -> Tuple[Dict[str, float], float]:
    """Per-question median output tokens and the input-estimate correction from recent valid runs."""
    recent = [r for r in history if r.get("valid")][-last_n:]
    per_key: Dict[str, List[int]] = {}
    ratios: List[float] = []
    for run in recent:
        for key, rec in (run.get("items") or {}).items():
            per_key.setdefault(key, []).extend(t for t in rec.get("t", []) if t > 0)
        est, act = (run.get("usage") or {}).get("estimated_input"), (run.get("usage") or {}).get("input")
        if est and act:
            ratios.append(act / est)
    return {k: float(median(v)) for k, v in per_key.items() if v}, (float(median(ratios)) if ratios else 1.0)


def estimate(items, samples: int, price: Optional[Price], history: List[Dict[str, Any]]) -> Estimate:
    lengths, scale = measured_lengths(history)
    scale = min(max(scale, 0.5), 3.0)
    inp = exp_out = worst_out = 0
    measured = 0
    for item in items:
        inp += math.ceil(estimate_input_tokens(item.messages()) * scale) * samples
        guess = lengths.get(item.key)
        if guess is not None:
            measured += 1
        else:
            guess = DEFAULT_OUTPUT_GUESS.get(item.extract, 200)
        exp_out += math.ceil(min(guess, item.max_tokens)) * samples
        worst_out += item.max_tokens * samples
    source = "measured" if items and measured == len(items) else ("first-run guess" if not measured else "partly measured")
    return Estimate(calls=len(items) * samples, input_tokens=inp, expected_output_tokens=exp_out,
                    worst_output_tokens=worst_out, price=price, output_source=source, input_scale=scale)
