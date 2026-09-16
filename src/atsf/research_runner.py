from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .adaptive_evolution import AdaptiveEvolutionPolicy
from .orchestrator import CandidateEvaluation
from .population import Candidate
from .research_checkpoint import ResearchCheckpoint
from .research_cycle import ResearchCyclePolicy, ResearchCycleResult, run_research_cycle
from .research_cycle_registry import ResearchCycleRegistry
from .research_history import ResearchHistory
from .research_provenance import GenerationProvenance, build_generation_provenance


@dataclass(frozen=True)
class ResearchRunPolicy:
    """Bounded control policy for a multi-generation research run."""

    max_generations: int = 10
    improvement_epsilon: float = 0.0
    stop_on_stagnation: int | None = None

    def __post_init__(self) -> None:
        if self.max_generations < 1:
            raise ValueError("max_generations must be positive")
        if self.improvement_epsilon < 0.0:
            raise ValueError("improvement_epsilon must be non-negative")
        if self.stop_on_stagnation is not None and self.stop_on_stagnation < 1:
            raise ValueError("stop_on_stagnation must be positive")


@dataclass(frozen=True)
class ResearchRunResult:
    """Immutable result of a bounded autonomous research run."""

    history: ResearchHistory
    generations: tuple[ResearchCycleResult, ...]
    provenance: tuple[GenerationProvenance, ...]
    checkpoints: tuple[ResearchCheckpoint, ...]
    final_population: tuple[Candidate, ...]
    stopped_on_stagnation: bool

    @property
    def final_checkpoint(self) -> ResearchCheckpoint | None:
        """Return the most recent restart point, when at least one generation ran."""
        return self.checkpoints[-1] if self.checkpoints else None


CandidateEvaluator = Callable[[Candidate], CandidateEvaluation]


def _persist_cycle(registry: ResearchCycleRegistry, result: ResearchCycleResult, seed: int) -> None:
    """Persist only deterministic research state; never execution authority."""
    registry.save_cycle_in_transaction(
        f"generation-{result.generation}",
        result.generation,
        plan={
            "seed": seed,
            "crossover_rate": result.metrics.crossover_rate,
            "mutation_rate": result.metrics.mutation_rate,
        },
        feedback={
            "candidate_count": result.metrics.candidate_count,
            "promotion_eligible_count": result.metrics.promotion_eligible_count,
            "selected_count": result.metrics.selected_count,
            "promoted_count": result.metrics.promoted_count,
            "best_fitness": result.metrics.best_fitness,
            "mean_fitness": result.metrics.mean_fitness,
            "stagnating": result.metrics.stagnating,
        },
        admissions={
            "selected_strategy_ids": tuple(candidate.strategy_id for candidate in result.selected_parents),
            "next_strategy_ids": tuple(candidate.strategy_id for candidate in result.next_population),
        },
        portfolio_feedback=None,
    )


def _run_from_state(
    population: list[Candidate],
    evaluator: CandidateEvaluator,
    *,
    generation: int,
    seed: int,
    history: ResearchHistory,
    prior_provenance: tuple[GenerationProvenance, ...],
    prior_checkpoints: tuple[ResearchCheckpoint, ...],
    cycle_policy: ResearchCyclePolicy,
    run_policy: ResearchRunPolicy,
    adaptive_policy: AdaptiveEvolutionPolicy | None,
    cycle_registry: ResearchCycleRegistry | None,
) -> ResearchRunResult:
    results: list[ResearchCycleResult] = []
    provenance = list(prior_provenance)
    checkpoints = list(prior_checkpoints)
    current = list(population)
    stopped = False

    while generation < run_policy.max_generations:
        if cycle_registry is not None:
            # A restart must prove historical integrity before it can evolve again.
            cycle_registry.verify()

        generation_seed = seed
        result = run_research_cycle(
            current,
            evaluator,
            generation=generation,
            seed=generation_seed,
            policy=cycle_policy,
            history=history,
            adaptive_policy=adaptive_policy,
        )
        history = history.record_generation(
            result,
            current,
            improvement_epsilon=run_policy.improvement_epsilon,
        )
        generation_provenance = build_generation_provenance(
            result, current, seed=generation_seed
        )
        provenance.append(generation_provenance)
        results.append(result)
        current = list(result.next_population)

        stopped = (
            run_policy.stop_on_stagnation is not None
            and history.is_stagnating(run_policy.stop_on_stagnation)
        )
        checkpoint = ResearchCheckpoint(
            next_generation=generation + 1,
            next_seed=generation_seed + 1,
            population=tuple(current),
            history=history,
            provenance=tuple(provenance),
            stopped=stopped,
        )
        if cycle_registry is not None:
            # Persist before exposing this generation as a restartable result.
            with cycle_registry.connection:
                _persist_cycle(cycle_registry, result, generation_seed)
        checkpoints.append(checkpoint)
        if stopped:
            break
        generation += 1
        seed += 1

    return ResearchRunResult(
        history=history,
        generations=tuple(results),
        provenance=tuple(provenance),
        checkpoints=tuple(checkpoints),
        final_population=tuple(current),
        stopped_on_stagnation=stopped,
    )


def run_research(
    population: list[Candidate],
    evaluator: CandidateEvaluator,
    *,
    cycle_policy: ResearchCyclePolicy,
    run_policy: ResearchRunPolicy | None = None,
    adaptive_policy: AdaptiveEvolutionPolicy | None = None,
    seed: int = 0,
    cycle_registry: ResearchCycleRegistry | None = None,
) -> ResearchRunResult:
    """Run bounded deterministic research generations without execution authority."""
    run_policy = run_policy or ResearchRunPolicy()
    return _run_from_state(
        population,
        evaluator,
        generation=0,
        seed=seed,
        history=ResearchHistory(),
        prior_provenance=(),
        prior_checkpoints=(),
        cycle_policy=cycle_policy,
        run_policy=run_policy,
        adaptive_policy=adaptive_policy,
        cycle_registry=cycle_registry,
    )


def resume_research(
    checkpoint: ResearchCheckpoint,
    evaluator: CandidateEvaluator,
    *,
    cycle_policy: ResearchCyclePolicy,
    run_policy: ResearchRunPolicy | None = None,
    adaptive_policy: AdaptiveEvolutionPolicy | None = None,
    cycle_registry: ResearchCycleRegistry | None = None,
) -> ResearchRunResult:
    """Resume a deterministic research run from a validated checkpoint.

    ``max_generations`` is the absolute generation target, so a checkpoint at
    ``next_generation=3`` with ``max_generations=5`` executes generations 3 and 4.
    """
    run_policy = run_policy or ResearchRunPolicy()
    if checkpoint.stopped:
        raise ValueError("cannot resume a stopped checkpoint")
    if checkpoint.next_generation > run_policy.max_generations:
        raise ValueError("checkpoint next_generation exceeds max_generations")
    return _run_from_state(
        list(checkpoint.population),
        evaluator,
        generation=checkpoint.next_generation,
        seed=checkpoint.next_seed,
        history=checkpoint.history,
        prior_provenance=checkpoint.provenance,
        prior_checkpoints=(checkpoint,),
        cycle_policy=cycle_policy,
        run_policy=run_policy,
        adaptive_policy=adaptive_policy,
        cycle_registry=cycle_registry,
    )
