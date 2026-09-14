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


def _applicable_mutations(candidate: Candidate) -> tuple:
    strategy = candidate.strategy
    operators = []
    if strategy.indicators:
        operators.append(mutate_indicator_period)
    conditions = tuple(strategy.entry.all)
    if any(
        isinstance(condition.right, (int, float)) and not isinstance(condition.right, bool)
        for condition in conditions
    ):
        operators.append(mutate_threshold)
    if conditions:
        operators.append(mutate_signal_comparator)
    if (
        strategy.position_sizing.method == "fixed_fraction"
        and min(strategy.position_sizing.max_position, strategy.risk.max_position) > 0
    ):
        operators.append(mutate_position_fraction)
    return tuple(operators)


def mutate_candidate(candidate: Candidate, rng: Random) -> Candidate:
    operators = _applicable_mutations(candidate)
    if not operators:
        raise ValueError("strategy has no applicable mutation operators")
    mutation = rng.choice(operators)
    strategy = mutation(candidate.strategy, rng)
    operator = mutation.__name__
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


def evolve_population(
    parents: list[Candidate],
    target_size: int,
    rng: Random,
    *,
    crossover_rate: float = 0.5,
    mutation_rate: float = 0.5,
    elite_count: int = 0,
) -> list[Candidate]:
    """Produce a deterministic next generation from an already-selected parent pool.

    Selection is deliberately outside this function: callers provide the parent pool,
    while this controller performs only variation, uniqueness, and lineage accounting.
    Elites are copied unchanged when requested; all other children must be novel.
    """
    if not parents:
        raise ValueError("parent population must not be empty")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if elite_count < 0 or elite_count > target_size:
        raise ValueError("elite_count must be between 0 and target_size")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")
    if crossover_rate + mutation_rate <= 0.0 and elite_count < target_size:
        raise ValueError("at least one variation rate must be positive")
    if crossover_rate > 0.0 and len(parents) < 2 and mutation_rate == 0.0:
        raise ValueError("crossover requires at least two parents")

    unique_parents: list[Candidate] = []
    seen: set[str] = set()
    for parent in parents:
        if parent.strategy_id not in seen:
            unique_parents.append(parent)
            seen.add(parent.strategy_id)
    if not unique_parents:
        raise ValueError("parent population must contain valid candidates")

    population: list[Candidate] = []
    occupied = set()
    elite_limit = min(elite_count, len(unique_parents))
    for parent in unique_parents[:elite_limit]:
        population.append(parent)
        occupied.add(parent.strategy_id)

    max_attempts = max(100, target_size * 50)
    attempts = 0
    while len(population) < target_size and attempts < max_attempts:
        attempts += 1
        roll = rng.random()
        choose_crossover = (
            crossover_rate > 0.0
            and len(unique_parents) >= 2
            and roll < crossover_rate
        )
        if choose_crossover:
            first, second = rng.sample(unique_parents, 2)
            child = crossover_candidate(first, second, rng)
        elif mutation_rate > 0.0:
            parent = rng.choice(unique_parents)
            try:
                child = mutate_candidate(parent, rng)
            except ValueError:
                if crossover_rate <= 0.0 or len(unique_parents) < 2:
                    continue
                first, second = rng.sample(unique_parents, 2)
                child = crossover_candidate(first, second, rng)
        else:
            if crossover_rate <= 0.0 or len(unique_parents) < 2:
                continue
            first, second = rng.sample(unique_parents, 2)
            child = crossover_candidate(first, second, rng)

        if child.strategy_id in occupied:
            continue
        population.append(child)
        occupied.add(child.strategy_id)

    if len(population) != target_size:
        raise ValueError("unable to produce a unique population with the configured variation operators")
    return population
