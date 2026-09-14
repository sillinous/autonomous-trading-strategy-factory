from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Callable

from .adaptive_evolution import AdaptiveEvolutionPolicy, adapt_evolution
from .orchestrator import CandidateEvaluation
from .population import Candidate, evolve_population
from .selection import SelectionPolicy, select_population

if TYPE_CHECKING:
    from .research_history import ResearchHistory


CandidateEvaluator = Callable[[Candidate], CandidateEvaluation]


@dataclass(frozen=True)
class ResearchCyclePolicy:
    """Controls one deterministic research-generation transition."""

    selection: SelectionPolicy
    crossover_rate: float = 0.5
    mutation_rate: float = 0.5
    elite_count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1")
        if self.elite_count < 0 or self.elite_count > self.selection.population_size:
            raise ValueError("elite_count must be between 0 and population_size")
        if self.crossover_rate + self.mutation_rate <= 0.0 and self.elite_count < self.selection.population_size:
            raise ValueError("at least one variation rate must be positive")


@dataclass(frozen=True)
class GenerationMetrics:
    """Auditable summary of one evaluated research generation."""

    generation: int
    candidate_count: int
    eligible_count: int
    selected_count: int
    promoted_count: int
    best_fitness: float
    mean_fitness: float
    crossover_rate: float
    mutation_rate: float
    stagnating: bool


@dataclass(frozen=True)
class ResearchCycleResult:
    """Complete output of one research-generation transition."""

    generation: int
    evaluations: tuple[CandidateEvaluation, ...]
    selected_parents: tuple[Candidate, ...]
    next_population: tuple[Candidate, ...]
    metrics: GenerationMetrics


def run_research_cycle(
    population: list[Candidate],
    evaluator: CandidateEvaluator,
    *,
    generation: int,
    seed: int,
    policy: ResearchCyclePolicy,
    history: ResearchHistory | None = None,
    adaptive_policy: AdaptiveEvolutionPolicy | None = None,
) -> ResearchCycleResult:
    """Evaluate, select, and evolve one research generation deterministically.

    The evaluator is injected so this controller stays independent of market-data
    storage, backtest infrastructure, and execution. It never submits orders or
    chooses live capital allocations.
    """
    if generation < 0:
        raise ValueError("generation must be non-negative")
    if not population:
        raise ValueError("population must not be empty")

    candidate_ids = [candidate.strategy_id for candidate in population]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("population strategy IDs must be unique")

    evaluations = tuple(evaluator(candidate) for candidate in population)
    if len(evaluations) != len(population):
        raise ValueError("evaluator must return one evaluation per candidate")

    selected = tuple(select_population(population, list(evaluations), policy.selection))
    rates = adapt_evolution(
        policy.crossover_rate,
        policy.mutation_rate,
        history,
        adaptive_policy,
    )
    rng = Random(seed)
    next_population = tuple(
        evolve_population(
            list(selected),
            policy.selection.population_size,
            rng,
            crossover_rate=rates.crossover_rate,
            mutation_rate=rates.mutation_rate,
            elite_count=policy.elite_count,
        )
    )

    eligible = [evaluation for evaluation in evaluations if evaluation.promotion.eligible]
    promoted = len(eligible)
    fitness_values = [evaluation.fitness.score for evaluation in evaluations]
    metrics = GenerationMetrics(
        generation=generation,
        candidate_count=len(population),
        eligible_count=len(eligible),
        selected_count=len(selected),
        promoted_count=promoted,
        best_fitness=max(fitness_values),
        mean_fitness=sum(fitness_values) / len(fitness_values),
        crossover_rate=rates.crossover_rate,
        mutation_rate=rates.mutation_rate,
        stagnating=rates.stagnating,
    )
    return ResearchCycleResult(
        generation=generation,
        evaluations=evaluations,
        selected_parents=selected,
        next_population=next_population,
        metrics=metrics,
    )
