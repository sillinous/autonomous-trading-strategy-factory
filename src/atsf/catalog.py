from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .registry import ExperimentRegistry


@dataclass(frozen=True)
class CatalogEntry:
    """A promotion-ready experiment plus its persisted robustness evidence."""

    experiment_id: str
    strategy_id: str
    dataset_id: str
    dataset_version: str
    score: float
    evidence: dict[str, Any]


def promotion_ready_experiments(
    registry: ExperimentRegistry,
    dataset_id: str | None = None,
    limit: int | None = None,
) -> tuple[CatalogEntry, ...]:
    """Return only experiments that passed the persisted research-to-paper gate.

    Missing evidence, non-paper stages, and failed promotion flags are excluded
    rather than inferred from a backtest score.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows = registry.list_experiments(dataset_id)
    entries: list[CatalogEntry] = []
    for row in rows:
        if row["status"] != "paper" or row["score"] is None:
            continue
        evidence = registry.get_evaluation_evidence(row["experiment_id"])
        if not evidence:
            continue
        promotion = evidence.get("promotion")
        if not isinstance(promotion, dict) or promotion.get("eligible") is not True:
            continue
        entries.append(
            CatalogEntry(
                experiment_id=row["experiment_id"],
                strategy_id=row["strategy_id"],
                dataset_id=row["dataset_id"],
                dataset_version=row["dataset_version"],
                score=float(row["score"]),
                evidence=evidence,
            )
        )
    entries.sort(key=lambda item: (-item.score, item.experiment_id))
    return tuple(entries[:limit] if limit is not None else entries)
