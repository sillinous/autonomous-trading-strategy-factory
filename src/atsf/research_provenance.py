from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Mapping

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


def _canonicalize(value: Any) -> Any:
    """Convert supported research objects into deterministic JSON-compatible data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("provenance values must be finite")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _canonicalize(model_dump(mode="json"))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonicalize(value.to_dict())
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return _canonicalize(value.item())
    return str(value)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _candidate_record(
    candidate: Candidate,
    generation: int,
    evaluation: Any,
    seed: int,
) -> CandidateProvenance:
    if evaluation.candidate_id != candidate.strategy_id:
        raise ValueError("evaluation candidate_id does not match candidate strategy_id")

    evidence_fields = (
        "experiment",
        "backtest",
        "validation_passed",
        "fitness",
        "walk_forward",
        "monte_carlo",
        "perturbation",
        "regime",
        "robustness",
        "promotion",
    )
    evaluation_payload = {
        field: getattr(evaluation, field, None) for field in evidence_fields
    }
    return CandidateProvenance(
        strategy_id=candidate.strategy_id,
        generation=generation,
        parent_strategy_ids=tuple(candidate.lineage.parent_ids),
        genome_digest=_canonical_digest(candidate.strategy),
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
        selected_strategy_ids=tuple(
            candidate.strategy_id for candidate in result.selected_parents
        ),
        next_strategy_ids=tuple(
            candidate.strategy_id for candidate in result.next_population
        ),
        metrics_digest=_canonical_digest(metrics),
    )
