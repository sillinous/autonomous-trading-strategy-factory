from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .adaptive_evolution import AdaptiveEvolutionPolicy
from .population import Candidate
from .research_cycle import ResearchCyclePolicy, ResearchCycleResult, run_research_cycle
from .research_history import ResearchHistory


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
    final_population: tuple[Candidate, ...]
    stopped_on_stagnation: bool


CandidateEvaluator = Callable[[Candidate], object]


def run_research(
    population: list[Candidate],
    evaluator: CandidateEvaluator,
    *,
    cycle_policy: ResearchCyclePolicy,
    run_policy: ResearchRunPolicy | None = None,
    adaptive_policy: AdaptiveEvolutionPolicy | None = None,
    seed: int = 0,
) -> ResearchRunResult:
    """Run bounded deterministic research generations without execution authority."""
    run_policy = run_policy or ResearchRunPolicy()
    history = ResearchHistory()
    results: list[ResearchCycleResult] = []
    current = list(population)
    stopped = False

    for generation in range(run_policy.max_generations):
        result = run_research_cycle(
            current,
            evaluator,
            generation=generation,
            seed=seed + generation,
            policy=cycle_policy,
            history=history,
            adaptive_policy=adaptive_policy,
        )
        history = history.record_generation(
            result,
            current,
            improvement_epsilon=run_policy.improvement_epsilon,
        )
        results.append(result)
        current = list(result.next_population)

        if (
            run_policy.stop_on_stagnation is not None
            and history.is_stagnating(run_policy.stop_on_stagnation)
        ):
            stopped = True
            break

    return ResearchRunResult(
        history=history,
        generations=tuple(results),
        final_population=tuple(current),
        stopped_on_stagnation=stopped,
    )
