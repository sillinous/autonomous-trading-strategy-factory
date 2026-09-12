from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .genome import StrategyGenome, crossover
from .generator import (
    mutate_indicator_period,
    mutate_position_fraction,
    mutate_signal_comparator,
    mutate_threshold,
)
from .lineage import LineageRecord
from .strategy import StrategySpec


@dataclass(frozen=True)
class Candidate:
    strategy: StrategySpec
    strategy_id: str
    lineage: LineageRecord

    @property
    def candidate_id(self) -> str:
        """Compatibility alias used by evaluation and robustness components."""
        return self.strategy_id


def strategy_id(strategy: StrategySpec) -> str:
    return StrategyGenome.from_strategy(strategy).strategy_id


def crossover_candidate(parent_a: Candidate, parent_b: Candidate, rng: Random) -> Candidate:
    """Create a two-parent candidate and record explicit crossover lineage."""
    if parent_a.strategy_id == parent_b.strategy_id:
        raise ValueError("crossover requires two distinct parents")
    strategy = crossover(parent_a.strategy, parent_b.strategy, rng)
    candidate_id = strategy_id(strategy)
    if candidate_id in {parent_a.strategy_id, parent_b.strategy_id}:
        raise ValueError("crossover produced a parent-identical strategy")
    lineage = LineageRecord(
        strategy_id=candidate_id,
        generation=max(parent_a.lineage.generation, parent_b.lineage.generation) + 1,
        parent_ids=(parent_a.strategy_id, parent_b.strategy_id),
        operator="crossover",
    )
    return Candidate(strategy=strategy, strategy_id=candidate_id, lineage=lineage)


def mutate_candidate(candidate: Candidate, rng: Random) -> Candidate:
    operators = (
        mutate_indicator_period,
        mutate_threshold,
        mutate_signal_comparator,
        mutate_position_fraction,
    )
    mutation = rng.choice(operators)
    try:
        strategy = mutation(candidate.strategy, rng)
        operator = mutation.__name__
    except (TypeError, ValueError):
        strategy = mutate_indicator_period(candidate.strategy, rng)
        operator = mutate_indicator_period.__name__
    candidate_id = strategy_id(strategy)
    lineage = LineageRecord(
        strategy_id=candidate_id,
        generation=candidate.lineage.generation + 1,
        parent_ids=(candidate.strategy_id,),
        operator=operator,
    )
    return Candidate(strategy=strategy, strategy_id=candidate_id, lineage=lineage)


def seed_population(strategies: list[StrategySpec]) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for strategy in strategies:
        candidate_id = strategy_id(strategy)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidates.append(Candidate(strategy, candidate_id, LineageRecord(candidate_id, 0)))
    return candidates
