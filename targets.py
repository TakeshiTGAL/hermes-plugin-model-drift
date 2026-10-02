"""Which models to watch, read from the user's own Hermes config (never from credentials)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Target:
    provider: str
    model: str
    role: str  # "main", "fallback", "auxiliary:<task>", "delegation", "extra"
    override: bool  # True = must ask ctx.llm for this provider/model explicitly (needs the grant)

    @property
    def id(self) -> str:
        return f"{self.provider or 'auto'}:{self.model or 'default'}"


def load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly
        return load_config_readonly() or {}
    except Exception:
        return {}


def _main(cfg: Dict[str, Any]) -> Optional[Target]:
    m = cfg.get("model")
    if isinstance(m, str):
        return Target(provider=str(cfg.get("provider") or "auto"), model=m, role="main", override=False)
    if not isinstance(m, dict):
        return None
    model = str(m.get("default") or m.get("model") or "").strip()
    if not model:
        return None
    return Target(provider=str(m.get("provider") or "auto").strip(), model=model, role="main", override=False)


def parse_target(spec: Any, role: str = "extra") -> Optional[Target]:
    """'provider:model', or {'provider': ..., 'model': ...}."""
    if isinstance(spec, dict):
        provider, model = str(spec.get("provider") or "").strip(), str(spec.get("model") or "").strip()
    elif isinstance(spec, str) and ":" in spec:
        provider, model = (p.strip() for p in spec.split(":", 1))
    else:
        return None
    if not model or provider in ("", "auto", "main"):
        return None
    return Target(provider=provider, model=model, role=role, override=True)


def discover(cfg: Dict[str, Any]) -> List[Target]:
    """Every model the config names: main, fallback chain, auxiliary tasks, delegation."""
    out: List[Target] = []
    main = _main(cfg)
    if main:
        out.append(main)
    for entry in cfg.get("fallback_providers") or []:
        t = parse_target(entry, "fallback")
        if t:
            out.append(t)
    aux = cfg.get("auxiliary") or {}
    if isinstance(aux, dict):
        for task, block in sorted(aux.items()):
            if isinstance(block, dict):
                t = parse_target(block, f"auxiliary:{task}")
                if t:
                    out.append(t)
    deleg = cfg.get("delegation") or {}
    if isinstance(deleg, dict):
        t = parse_target(deleg, "delegation")
        if t:
            out.append(t)
    seen, unique = set(), []
    for t in out:
        if (t.provider, t.model) not in seen:
            seen.add((t.provider, t.model))
            unique.append(t)
    return unique


def monitored(cfg: Dict[str, Any], extra: List[Any]) -> List[Target]:
    out = [t for t in discover(cfg) if t.role == "main"]
    for spec in extra or []:
        t = parse_target(spec)
        if t and all((t.provider, t.model) != (o.provider, o.model) for o in out):
            out.append(t)
    return out


def grant_snippet(targets: List[Target]) -> str:
    providers = sorted({t.provider for t in targets if t.override})
    models = sorted({t.model for t in targets if t.override})
    if not models:
        return ""
    lines = ["plugins:", "  entries:", "    model-drift-watch:", "      llm:",
             "        allow_provider_override: true", "        allowed_providers:"]
    lines += [f"          - {p}" for p in providers]
    lines += ["        allow_model_override: true", "        allowed_models:"]
    lines += [f"          - {m}" for m in models]
    return "\n".join(lines)
