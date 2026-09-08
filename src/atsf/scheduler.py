from __future__ import annotations

from dataclasses import dataclass
from random import Random

import pandas as pd

from .fitness import FitnessPolicy
from .orchestrator import CandidateEvaluation, evaluate_candidate
from .population import Candidate, mutate_candidate


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    evaluations: tuple[CandidateEvaluation, ...]
    survivors: tuple[Candidate, ...]
    next_population: tuple[Candidate, ...]


def evolve_generation(
    population: list[Candidate],
    data: pd.DataFrame,
    dataset_id: str,
    dataset_version: str,
    target_size: int,
    survivor_count: int,
    seed: int = 0,
    fitness_policy: FitnessPolicy | None = None,
) -> GenerationResult:
    """Evaluate, rank, retain, and mutate one research-only generation."""
    if not population:
        raise ValueError("population must not be empty")
    if target_size <= 0 or survivor_count <= 0:
        raise ValueError("population sizes must be positive")
    if survivor_count > target_size:
        raise ValueError("survivor_count cannot exceed target_size")

    evaluations = tuple(
        evaluate_candidate(
            candidate,
            data,
            dataset_id,
            dataset_version,
            seed=seed,
            fitness_policy=fitness_policy,
        )
        for candidate in population
    )
    ranked = sorted(
        zip(population, evaluations),
        key=lambda pair: (
            pair[1].fitness.eligible,
            pair[1].fitness.score,
            pair[0].strategy_id,
        ),
        reverse=True,
    )
    survivors = tuple(candidate for candidate, _ in ranked[:survivor_count])
    if not survivors:
        raise RuntimeError("no survivors available for mutation")

    rng = Random(seed)
    next_population = list(survivors)
    seen = {candidate.strategy_id for candidate in next_population}
    attempts = 0
    max_attempts = max(100, target_size * 100)
    while len(next_population) < target_size and attempts < max_attempts:
        parent = rng.choice(survivors)
        child = mutate_candidate(parent, rng)
        attempts += 1
        if child.strategy_id in seen:
            continue
        seen.add(child.strategy_id)
        next_population.append(child)

    if len(next_population) < target_size:
        raise RuntimeError("unable to generate a unique next population")

    generation = max(candidate.lineage.generation for candidate in population) + 1
    return GenerationResult(
        generation=generation,
        evaluations=evaluations,
        survivors=survivors,
        next_population=tuple(next_population),
    )
