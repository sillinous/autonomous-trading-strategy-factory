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


# Mutations stay within a direction: a "price above" rule may become ">=" or a cross,
# but never flips direction or becomes an exact float equality.
COMPARATOR_FAMILIES: tuple[frozenset[Comparator], ...] = (
    frozenset({Comparator.GT, Comparator.GTE, Comparator.CROSSES_ABOVE}),
    frozenset({Comparator.LT, Comparator.LTE, Comparator.CROSSES_BELOW}),
    frozenset({Comparator.EQ, Comparator.GTE, Comparator.LTE}),
)


def mutate_signal_comparator(strategy: StrategySpec, rng: Random | None = None) -> StrategySpec:
    """Change one entry comparator within a semantically safe comparison family."""
    rng = rng or Random()
    conditions = list(strategy.entry.all)
    if not conditions:
        raise ValueError("strategy entry has no conjunctive conditions")
    index = rng.randrange(len(conditions))
    condition = conditions[index]
    family = next(group for group in COMPARATOR_FAMILIES if condition.comparator in group)
    choices = tuple(comparator for comparator in family if comparator != condition.comparator)
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


def _cond(left: str, comparator: Comparator, right: str | float) -> Condition:
    return Condition(left=left, comparator=comparator, right=right)


def _vol_sized(stop: float | None = None) -> dict:
    return {
        "position_sizing": PositionSizing(method="volatility_target", value=0.15, max_position=1.0),
        "risk": RiskLimits(max_position=1.0, stop_loss=stop, max_drawdown=0.30),
    }


# name -> (style, indicators, entry, exit, sizing/risk)
ARCHETYPES: dict[str, tuple[str, list[Indicator], Signal, Signal, dict]] = {
    "donchian_breakout": (
        "trend",
        [Indicator(name="channel_high", kind="highest", period=55),
         Indicator(name="channel_low", kind="lowest", period=20)],
        Signal(all=[_cond("close", Comparator.GT, "channel_high")]),
        Signal(all=[_cond("close", Comparator.LT, "channel_low")]),
        _vol_sized(stop=0.10),
    ),
    "roc_momentum": (
        "trend",
        [Indicator(name="momentum", kind="roc", period=126),
         Indicator(name="trend", kind="sma", period=200)],
        Signal(all=[_cond("momentum", Comparator.GT, 0.0), _cond("close", Comparator.GT, "trend")]),
        Signal(any=[_cond("momentum", Comparator.LT, 0.0)]),
        _vol_sized(),
    ),
    "macd_cross": (
        "trend",
        # fast period defaults to 12/26 of the slow period, so period mutations stay valid
        [Indicator(name="macd_line", kind="macd", period=26),
         Indicator(name="macd_trigger", kind="macd_signal", period=26, parameters={"signal": 9})],
        Signal(all=[_cond("macd_line", Comparator.CROSSES_ABOVE, "macd_trigger")]),
        Signal(all=[_cond("macd_line", Comparator.CROSSES_BELOW, "macd_trigger")]),
        _vol_sized(stop=0.08),
    ),
    "rsi_reversion": (
        "mean_reversion",
        [Indicator(name="rsi_fast", kind="rsi", period=3),
         Indicator(name="trend", kind="sma", period=200)],
        Signal(all=[_cond("rsi_fast", Comparator.LT, 15.0), _cond("close", Comparator.GT, "trend")]),
        Signal(any=[_cond("rsi_fast", Comparator.GT, 60.0)]),
        _vol_sized(stop=0.06),
    ),
    "bollinger_reversion": (
        "mean_reversion",
        [Indicator(name="band_low", kind="bb_lower", period=20, parameters={"k": 2.0}),
         Indicator(name="band_mid", kind="sma", period=20),
         Indicator(name="trend", kind="sma", period=200)],
        Signal(all=[_cond("close", Comparator.LT, "band_low"), _cond("close", Comparator.GT, "trend")]),
        Signal(any=[_cond("close", Comparator.GT, "band_mid")]),
        _vol_sized(stop=0.06),
    ),
}


def archetype_strategy(name: str, symbols: list[str], *, label: str | None = None) -> StrategySpec:
    """Build a named, validated archetype strategy for ``symbols``."""
    if name not in ARCHETYPES:
        raise ValueError(f"unknown archetype: {name}")
    _, indicators, entry, exit_, sizing = ARCHETYPES[name]
    return StrategySpec(
        name=label or f"archetype-{name}", universe=symbols,
        indicators=[indicator.model_copy() for indicator in indicators],
        entry=entry.model_copy(deep=True), exit=exit_.model_copy(deep=True), **sizing,
    )


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
        for name, (style, *_rest) in ARCHETYPES.items():
            if "trend_only" in constraints and style != "trend":
                continue
            if "mean_reversion_only" in constraints and style != "mean_reversion":
                continue
            spec = archetype_strategy(name, symbols, label=f"generated-{request.request_id}-{name}")
            payload = {
                "request_id": request.request_id,
                "parent": request.source_strategy_id,
                "mutation": name,
                "strategy": spec.model_dump(mode="json"),
            }
            candidate_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            candidates.append(StrategyCandidate(spec, request.request_id, request.source_strategy_id, name, candidate_id))
        return tuple(candidates)
