from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .fitness import FitnessPolicy
from .orchestrator import CandidateEvaluation, evaluate_candidate
from .population import Candidate
from .promotion import PromotionDecision, PromotionPolicy
from .replacement_research import ReplacementResearchResult
from .validation import ValidationPolicy


@dataclass(frozen=True)
class ReplacementEvaluationResult:
    request_id: str
    evaluations: tuple[CandidateEvaluation, ...]
    eligible_strategy_ids: tuple[str, ...]


def evaluate_replacements(
    research: ReplacementResearchResult,
    data: pd.DataFrame,
    dataset_id: str,
    dataset_version: str,
    *,
    seed: int = 0,
    fitness_policy: FitnessPolicy | None = None,
    validation_policy: ValidationPolicy | None = None,
    promotion_policy: PromotionPolicy | None = None,
) -> ReplacementEvaluationResult:
    """Send replacement hypotheses through the same deterministic gates as normal research."""
    evaluations: list[CandidateEvaluation] = []
    for offset, candidate in enumerate(research.candidates):
        if candidate.parent_strategy_id is None:
            raise ValueError(f"replacement candidate has no parent: {candidate.candidate_id}")
        from .lineage import LineageRecord

        lineage = LineageRecord(
            strategy_id=candidate.candidate_id,
            generation=1,
            parent_ids=(candidate.parent_strategy_id,),
            operator=candidate.mutation,
            parameters={"request_id": candidate.request_id, "candidate_id": candidate.candidate_id},
        )
        evaluation = evaluate_candidate(
            Candidate(strategy=candidate.strategy, strategy_id=candidate.candidate_id, lineage=lineage),
            data,
            dataset_id,
            dataset_version,
            seed=seed + offset,
            fitness_policy=fitness_policy,
            validation_policy=validation_policy,
            promotion_policy=promotion_policy,
        )
        if not evaluation.promotion.eligible:
            evaluation = replace(
                evaluation,
                promotion=PromotionDecision(stage="reject", eligible=False, reasons=evaluation.promotion.reasons),
            )
        evaluations.append(evaluation)
    eligible = tuple(evaluation.candidate_id for evaluation in evaluations if evaluation.promotion.eligible)
    return ReplacementEvaluationResult(
        request_id=research.request.request_id,
        evaluations=tuple(evaluations),
        eligible_strategy_ids=eligible,
    )
