from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import dataset_identity
from .experiment import ExperimentSpec
from .fitness import FitnessPolicy
from .population import Candidate, seed_population
from .registry import ExperimentRegistry
from .scheduler import GenerationResult, evolve_generation
from .strategy import StrategySpec


@dataclass(frozen=True)
class ResearchRunResult:
    generations: tuple[GenerationResult, ...]
    final_population: tuple[Candidate, ...]
    dataset_id: str
    dataset_version: str


def run_research(
    seeds: list[StrategySpec],
    data: pd.DataFrame,
    *,
    generations: int = 1,
    population_size: int = 10,
    survivor_count: int = 3,
    seed: int = 0,
    dataset_id: str = "research",
    registry: ExperimentRegistry | None = None,
    fitness_policy: FitnessPolicy | None = None,
) -> ResearchRunResult:
    """Run a deterministic, research-only evolutionary search and optionally persist it."""
    if generations <= 0:
        raise ValueError("generations must be positive")
    if population_size <= 0 or survivor_count <= 0:
        raise ValueError("population sizes must be positive")
    if survivor_count > population_size:
        raise ValueError("survivor_count cannot exceed population_size")
    identity = dataset_identity(data, dataset_id)
    population = seed_population(seeds)
    if not population:
        raise ValueError("seeds must contain at least one unique strategy")

    owned_registry = registry is None
    store = registry or ExperimentRegistry()
    results: list[GenerationResult] = []
    try:
        for candidate in population:
            store.save_strategy(candidate.strategy)
            store.save_lineage(candidate.lineage)
        for generation in range(generations):
            candidates_by_id = {candidate.strategy_id: candidate for candidate in population}
            result = evolve_generation(
                population,
                data,
                identity.dataset_id,
                identity.version,
                target_size=population_size,
                survivor_count=survivor_count,
                seed=seed + generation,
                fitness_policy=fitness_policy,
            )
            results.append(result)
            for evaluation in result.evaluations:
                candidate = candidates_by_id[evaluation.candidate_id]
                spec = ExperimentSpec(
                    candidate.strategy,
                    identity.dataset_id,
                    identity.version,
                    seed + generation,
                )
                store.save_experiment(spec, evaluation.experiment)
                store.save_evaluation_evidence(
                    evaluation.experiment.experiment_id,
                    {
                        "candidate_id": evaluation.candidate_id,
                        "walk_forward": {
                            "passed": evaluation.walk_forward.passed,
                            "oos_return": evaluation.walk_forward.oos_return,
                            "oos_sharpe": evaluation.walk_forward.oos_sharpe,
                            "oos_drawdown": evaluation.walk_forward.oos_drawdown,
                        },
                        "monte_carlo": {
                            "simulations": evaluation.monte_carlo.simulations,
                            "seed": evaluation.monte_carlo.seed,
                            "median_return": evaluation.monte_carlo.median_return,
                            "worst_return": evaluation.monte_carlo.worst_return,
                            "lower_percentile_return": evaluation.monte_carlo.lower_percentile_return,
                            "pass_rate": evaluation.monte_carlo.pass_rate,
                        },
                        "perturbation": {
                            "samples": evaluation.perturbation.samples,
                            "seed": evaluation.perturbation.seed,
                            "pass_rate": evaluation.perturbation.pass_rate,
                            "worst_score": evaluation.perturbation.worst_score,
                            "median_score": evaluation.perturbation.median_score,
                            "strategy_ids": evaluation.perturbation.strategy_ids,
                        },
                        "regime": {
                            "score": evaluation.regime.score,
                            "regime_returns": evaluation.regime.regime_returns,
                            "covered_regimes": evaluation.regime.covered_regimes,
                        },
                        "robustness": {
                            "passed": evaluation.robustness.passed,
                            "reasons": evaluation.robustness.reasons,
                            "scenarios": {
                                name: {
                                    "total_return": result.total_return,
                                    "max_drawdown": result.max_drawdown,
                                }
                                for name, result in evaluation.robustness.scenarios
                            },
                        },
                        "promotion": {
                            "stage": evaluation.promotion.stage,
                            "eligible": evaluation.promotion.eligible,
                            "reasons": evaluation.promotion.reasons,
                        },
                    },
                )
            for candidate in result.next_population:
                store.save_strategy(candidate.strategy)
                store.save_lineage(candidate.lineage)
            population = list(result.next_population)
    finally:
        if owned_registry:
            store.close()

    return ResearchRunResult(
        generations=tuple(results),
        final_population=tuple(population),
        dataset_id=identity.dataset_id,
        dataset_version=identity.version,
    )
