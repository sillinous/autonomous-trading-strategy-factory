from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from .population import Candidate
from .research_cycle import ResearchCycleResult


@dataclass(frozen=True)
class CandidateProvenance:
    """Stable audit record linking a strategy to its research evidence."""

    strategy_id: str
    generation: int
    parent_strategy_ids: tuple[str, ...]
    genome_digest: str
    evaluation_digest: str
    research_seed: int


@dataclass(frozen=True)
class GenerationProvenance:
    """Immutable provenance manifest for one research generation."""

    generation: int
    candidate_records: tuple[CandidateProvenance, ...]
    selected_strategy_ids: tuple[str, ...]
    next_strategy_ids: tuple[str, ...]
    metrics_digest: str


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _candidate_record(candidate: Candidate, generation: int, evaluation: Any, seed: int) -> CandidateProvenance:
    parents = tuple(getattr(candidate, "parent_strategy_ids", ()) or ())
    genome = getattr(candidate, "genome", candidate)
    evaluation_payload = {
        "candidate_id": getattr(evaluation, "candidate_id", candidate.strategy_id),
        "fitness": getattr(getattr(evaluation, "fitness", None), "score", None),
        "promotion": getattr(getattr(evaluation, "promotion", None), "eligible", None),
    }
    return CandidateProvenance(
        strategy_id=candidate.strategy_id,
        generation=generation,
        parent_strategy_ids=parents,
        genome_digest=_canonical_digest(genome),
        evaluation_digest=_canonical_digest(evaluation_payload),
        research_seed=seed,
    )


def build_generation_provenance(
    result: ResearchCycleResult,
    population: list[Candidate],
    *,
    seed: int,
) -> GenerationProvenance:
    """Build a deterministic, serialization-friendly provenance manifest."""
    by_id = {candidate.strategy_id: candidate for candidate in population}
    records = tuple(
        _candidate_record(
            by_id[evaluation.candidate_id],
            result.generation,
            evaluation,
            seed,
        )
        for evaluation in result.evaluations
    )
    metrics = {
        "generation": result.metrics.generation,
        "candidate_count": result.metrics.candidate_count,
        "eligible_count": result.metrics.eligible_count,
        "selected_count": result.metrics.selected_count,
        "promoted_count": result.metrics.promoted_count,
        "best_fitness": result.metrics.best_fitness,
        "mean_fitness": result.metrics.mean_fitness,
        "crossover_rate": result.metrics.crossover_rate,
        "mutation_rate": result.metrics.mutation_rate,
        "stagnating": result.metrics.stagnating,
    }
    return GenerationProvenance(
        generation=result.generation,
        candidate_records=records,
        selected_strategy_ids=tuple(candidate.strategy_id for candidate in result.selected_parents),
        next_strategy_ids=tuple(candidate.strategy_id for candidate in result.next_population),
        metrics_digest=_canonical_digest(metrics),
    )
