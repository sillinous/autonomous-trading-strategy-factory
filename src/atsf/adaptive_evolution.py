from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .research_history import ResearchHistory


@dataclass(frozen=True)
class AdaptiveEvolutionPolicy:
    """Bounded deterministic response to research stagnation."""

    stagnation_threshold: int = 3
    mutation_step: float = 0.10
    crossover_step: float = 0.05
    max_mutation_rate: float = 0.90
    min_crossover_rate: float = 0.10

    def __post_init__(self) -> None:
        if self.stagnation_threshold < 1:
            raise ValueError("stagnation_threshold must be positive")
        for name, value in (("mutation_step", self.mutation_step), ("crossover_step", self.crossover_step), ("max_mutation_rate", self.max_mutation_rate), ("min_crossover_rate", self.min_crossover_rate)):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True)
class EvolutionRates:
    """Effective variation rates for the next research generation."""

    crossover_rate: float
    mutation_rate: float
    stagnating: bool


def adapt_evolution(crossover_rate: float, mutation_rate: float, history: ResearchHistory | None, policy: AdaptiveEvolutionPolicy | None = None) -> EvolutionRates:
    """Increase exploration only after sustained stagnation, with hard bounds."""
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")
    policy = policy or AdaptiveEvolutionPolicy()
    stagnating = history is not None and history.is_stagnating(policy.stagnation_threshold)
    if not stagnating:
        return EvolutionRates(crossover_rate, mutation_rate, False)
    return EvolutionRates(
        crossover_rate=max(policy.min_crossover_rate, crossover_rate - policy.crossover_step),
        mutation_rate=min(policy.max_mutation_rate, mutation_rate + policy.mutation_step),
        stagnating=True,
    )
