from __future__ import annotations

from dataclasses import replace
from random import Random

from .strategy import Condition, Indicator, StrategySpec


DEFAULT_PERIODS = (5, 10, 14, 20, 50, 100, 200)


def mutate_indicator_period(strategy: StrategySpec, rng: Random | None = None) -> StrategySpec:
    """Create a small neighboring candidate without generating executable code."""
    rng = rng or Random()
    if not strategy.indicators:
        raise ValueError("strategy has no indicators to mutate")
    index = rng.randrange(len(strategy.indicators))
    old = strategy.indicators[index]
    new = old.model_copy(update={"period": rng.choice(DEFAULT_PERIODS)})
    indicators = list(strategy.indicators)
    indicators[index] = new
    return strategy.model_copy(
        update={"version": strategy.version + 1, "indicators": indicators}
    )


def mutate_threshold(strategy: StrategySpec, rng: Random | None = None) -> StrategySpec:
    """Perturb a numeric condition threshold by a bounded random factor."""
    rng = rng or Random()
    conditions = list(strategy.entry.all)
    if not conditions:
        raise ValueError("strategy entry has no conjunctive conditions")
    index = rng.randrange(len(conditions))
    condition = conditions[index]
    if not isinstance(condition.right, (int, float)) or isinstance(condition.right, bool):
        raise ValueError("selected condition has no numeric threshold")
    factor = 1.0 + rng.uniform(-0.10, 0.10)
    updated = condition.model_copy(update={"right": condition.right * factor})
    conditions[index] = updated
    entry = strategy.entry.model_copy(update={"all": conditions})
    return strategy.model_copy(update={"version": strategy.version + 1, "entry": entry})
