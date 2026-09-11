from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from random import Random

from .research_queue import ResearchRequest
from .strategy import (
    Comparator,
    Condition,
    Indicator,
    PositionSizing,
    RiskLimits,
    Signal,
    StrategySpec,
)

DEFAULT_PERIODS = (5, 10, 14, 20, 50, 100, 200)


def _bump_version(strategy: StrategySpec, **updates: object) -> StrategySpec:
    return strategy.model_copy(update={"version": strategy.version + 1, **updates})


def mutate_indicator_period(strategy: StrategySpec, rng: Random | None = None) -> StrategySpec:
    """Create a small neighboring candidate by changing one indicator period."""
    rng = rng or Random()
    if not strategy.indicators:
        raise ValueError("strategy has no indicators to mutate")
    index = rng.randrange(len(strategy.indicators))
    old = strategy.indicators[index]
    choices = tuple(period for period in DEFAULT_PERIODS if period != old.period) or DEFAULT_PERIODS
    new = old.model_copy(update={"period": rng.choice(choices)})
    indicators = list(strategy.indicators)
    indicators[index] = new
    return _bump_version(strategy, indicators=indicators)


def mutate_threshold(strategy: StrategySpec, rng: Random | None = None) -> StrategySpec:
    """Perturb a numeric entry threshold by a bounded random factor."""
    rng = rng or Random()
    conditions = list(strategy.entry.all)
    if not conditions:
        raise ValueError("strategy entry has no conjunctive conditions")
    numeric = [i for i, condition in enumerate(conditions)
               if isinstance(condition.right, (int, float)) and not isinstance(condition.right, bool)]
    if not numeric:
        raise TypeError("strategy entry has no numeric threshold")
    index = rng.choice(numeric)
    condition = conditions[index]
    factor = 1.0 + rng.uniform(-0.10, 0.10)
    conditions[index] = condition.model_copy(update={"right": condition.right * factor})
    entry = strategy.entry.model_copy(update={"all": conditions})
    return _bump_version(strategy, entry=entry)


def mutate_signal_comparator(strategy: StrategySpec, rng: Random | None = None) -> StrategySpec:
    """Change one entry comparator within a semantically safe comparison family."""
    rng = rng or Random()
    conditions = list(strategy.entry.all)
    if not conditions:
        raise ValueError("strategy entry has no conjunctive conditions")
    index = rng.randrange(len(conditions))
    condition = conditions[index]
    choices = tuple(comparator for comparator in Comparator if comparator != condition.comparator)
    conditions[index] = condition.model_copy(update={"comparator": rng.choice(choices)})
    return _bump_version(strategy, entry=strategy.entry.model_copy(update={"all": conditions}))


def mutate_position_fraction(strategy: StrategySpec, rng: Random | None = None) -> StrategySpec:
    """Adjust fixed-fraction exposure while respecting the strategy risk ceiling."""
    rng = rng or Random()
    if strategy.position_sizing.method != "fixed_fraction":
        raise ValueError("strategy does not use fixed_fraction sizing")
    ceiling = min(strategy.position_sizing.max_position, strategy.risk.max_position)
    if ceiling <= 0:
        raise ValueError("strategy has no positive position ceiling")
    current = strategy.position_sizing.value
    candidates = tuple(value for value in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0) if value <= ceiling)
    if not candidates:
        raise ValueError("strategy has no valid position-size mutation")
    choices = tuple(value for value in candidates if value != current) or candidates
    sizing = strategy.position_sizing.model_copy(update={"value": rng.choice(choices)})
    return _bump_version(strategy, position_sizing=sizing)


@dataclass(frozen=True)
class StrategyCandidate:
    strategy: StrategySpec
    request_id: str
    parent_strategy_id: str | None
    mutation: str
    candidate_id: str


class StrategyGenerator:
    """Deterministic, constrained candidate generator with provenance."""

    def generate(
        self, request: ResearchRequest, symbols: list[str]
    ) -> tuple[StrategyCandidate, ...]:
        if not symbols:
            raise ValueError("symbols cannot be empty")
        constraints = set(request.constraints)
        candidates: list[StrategyCandidate] = []
        variants = (
            ("sma_fast", "sma", 10, 30),
            ("ema_fast", "ema", 10, 30),
            ("sma_slow", "sma", 20, 60),
        )
        for name, kind, fast, slow in variants:
            if "trend_only" in constraints and name == "sma_slow":
                continue
            fast_name = f"{name}_indicator"
            spec = StrategySpec(
                name=f"generated-{request.request_id}-{name}",
                universe=symbols,
                indicators=[
                    Indicator(name=fast_name, source="close", period=fast, kind=kind),
                    Indicator(name="slow_indicator", source="close", period=slow, kind="sma"),
                ],
                entry=Signal(all=[Condition(left=fast_name, comparator=Comparator.GT, right="slow_indicator")]),
                exit=Signal(all=[Condition(left=fast_name, comparator=Comparator.LT, right="slow_indicator")]),
                position_sizing=PositionSizing(method="fixed_fraction", value=0.25, max_position=0.25),
                risk=RiskLimits(max_position=0.25),
            )
            payload = {
                "request_id": request.request_id,
                "parent": request.source_strategy_id,
                "mutation": name,
                "strategy": spec.model_dump(mode="json"),
            }
            candidate_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            candidates.append(StrategyCandidate(spec, request.request_id, request.source_strategy_id, name, candidate_id))
        return tuple(candidates)
