from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import dataset_identity
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
    identity = dataset_identity(data, "research")
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
                store.save_experiment(evaluation.experiment, evaluation.experiment)
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
