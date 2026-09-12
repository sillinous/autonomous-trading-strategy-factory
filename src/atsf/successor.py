from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .lineage import LineageRecord
from .population import Candidate
from .orchestrator import CandidateEvaluation


@dataclass(frozen=True)
class SuccessorAdmission:
    candidate_id: str
    parent_ids: tuple[str, ...]
    generation: int
    admitted: bool
    stage: str
    reasons: tuple[str, ...]


def admit_successor(
    candidate: Candidate,
    evaluation: CandidateEvaluation,
    *,
    generation: int,
    existing_ids: set[str] | frozenset[str] = frozenset(),
) -> SuccessorAdmission:
    if generation < 0:
        raise ValueError("generation must be nonnegative")
    if candidate.strategy_id != evaluation.candidate_id:
        raise ValueError("candidate and evaluation identities must match")
    if candidate.strategy_id in existing_ids:
        raise ValueError("successor already exists")

    lineage = candidate.lineage
    if lineage.strategy_id != candidate.strategy_id:
        raise ValueError("lineage strategy identity must match candidate")
    if not lineage.parent_ids:
        raise ValueError("successor requires at least one parent")
    if lineage.generation > generation:
        raise ValueError("lineage generation cannot exceed admission generation")
    if not evaluation.promotion.eligible:
        return SuccessorAdmission(
            candidate_id=candidate.strategy_id,
            parent_ids=tuple(lineage.parent_ids),
            generation=generation,
            admitted=False,
            stage=evaluation.promotion.stage,
            reasons=tuple(evaluation.promotion.reasons),
        )

    finite = (
        evaluation.fitness.score,
        evaluation.walk_forward.oos_sharpe,
        evaluation.walk_forward.oos_drawdown,
    )
    if not all(isfinite(float(value)) for value in finite):
        raise ValueError("eligible successor contains non-finite evaluation metrics")

    return SuccessorAdmission(
        candidate_id=candidate.strategy_id,
        parent_ids=tuple(lineage.parent_ids),
        generation=generation,
        admitted=True,
        stage=evaluation.promotion.stage,
        reasons=(),
    )


def successor_lineage(candidate: Candidate, admission: SuccessorAdmission) -> LineageRecord:
    if not admission.admitted:
        raise ValueError("cannot create successor lineage for rejected candidate")
    if admission.candidate_id != candidate.strategy_id:
        raise ValueError("admission identity must match candidate")
    return LineageRecord(
        strategy_id=candidate.strategy_id,
        generation=admission.generation,
        parent_ids=admission.parent_ids,
        operator=candidate.lineage.operator,
        parameters={
            **candidate.lineage.parameters,
            "admission_stage": admission.stage,
            "admitted": True,
        },
    )
