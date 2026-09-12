from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from random import Random
from typing import Any

from .strategy import StrategySpec


@dataclass(frozen=True)
class StrategyGenome:
    """Canonical, deterministic representation of a strategy for evolution."""

    strategy_id: str
    payload: tuple[tuple[str, Any], ...]

    @classmethod
    def from_strategy(cls, strategy: StrategySpec) -> StrategyGenome:
        data = strategy.model_dump(mode="json")
        canonical = _canonical(data)
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return cls(strategy_id=digest, payload=tuple(sorted(canonical.items())))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def distance(self, other: StrategyGenome) -> float:
        """Return normalized structural distance in [0, 1]."""
        left = self.as_dict()
        right = other.as_dict()
        keys = set(left) | set(right)
        if not keys:
            return 0.0
        different = sum(left.get(key) != right.get(key) for key in keys)
        return different / len(keys)


def crossover(parent_a: StrategySpec, parent_b: StrategySpec, rng: Random) -> StrategySpec:
    """Create a deterministic child by selecting compatible strategy components."""
    a = parent_a.model_dump(mode="json")
    b = parent_b.model_dump(mode="json")
    child = dict(a)

    for field in ("universe", "timeframe", "side", "position_sizing", "risk"):
        if rng.random() < 0.5:
            child[field] = b[field]

    child["indicators"] = _merge_indicators(a["indicators"], b["indicators"], rng)
    child["entry"] = _choose_signal(a["entry"], b["entry"], rng)
    child["exit"] = _choose_signal(a["exit"], b["exit"], rng)
    child["metadata"] = dict(a.get("metadata", {}))
    child["metadata"]["evolution_operator"] = "crossover"
    child["version"] = max(parent_a.version, parent_b.version) + 1
    return StrategySpec.model_validate(child)


def _merge_indicators(
    indicators_a: list[dict[str, Any]], indicators_b: list[dict[str, Any]], rng: Random
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {item["name"]: dict(item) for item in indicators_a}
    for item in indicators_b:
        name = item["name"]
        if name not in by_name or rng.random() < 0.5:
            by_name[name] = dict(item)
    return [by_name[name] for name in sorted(by_name)]


def _choose_signal(a: dict[str, Any], b: dict[str, Any], rng: Random) -> dict[str, Any]:
    left = list(a.get("all", [])) + list(a.get("any", []))
    right = list(b.get("all", [])) + list(b.get("any", []))
    if not left and not right:
        return a
    selected = left if not right or (left and rng.random() < 0.5) else right
    condition = rng.choice(selected)
    return {"all": [condition], "any": []}


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def genome_distance(left: StrategySpec, right: StrategySpec) -> float:
    return StrategyGenome.from_strategy(left).distance(StrategyGenome.from_strategy(right))
