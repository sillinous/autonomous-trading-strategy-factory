from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .fitness import FitnessPolicy
from .orchestrator import CandidateEvaluation, evaluate_candidate
from .population import Candidate
from .promotion import PromotionPolicy
from .research_executor import ResearchWorkItem, materialize_research_work, spawn_research_candidates
from .research_planner import ResearchPlan
from .research_queue import ResearchQueue
from .validation import ValidationPolicy


@dataclass(frozen=True)
class ResearchExecutionResult:
    work_items: tuple[ResearchWorkItem, ...]
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class ResearchEvaluationResult:
    execution: ResearchExecutionResult
    evaluations: tuple[CandidateEvaluation, ...]
    eligible_strategy_ids: tuple[str, ...]


def execute_research_plan(
    plan: ResearchPlan,
    population: tuple[Candidate, ...],
    *,
    seed: int = 0,
    queue: ResearchQueue | None = None,
) -> ResearchExecutionResult:
    """Materialize and execute a research plan without bypassing downstream gates.

    Candidate spawning is intentionally the terminal step of this boundary. The
    returned candidates must still pass the normal evaluation/validation,
    robustness, promotion, and paper-trading gates before becoming successors.
    """
    if not isinstance(seed, int):
        raise TypeError("research seed must be an integer")
    work_items = materialize_research_work(plan, queue=queue)
    spawned: list[Candidate] = []
    seen = {candidate.strategy_id for candidate in population}
    for offset, work_item in enumerate(work_items):
        candidates = spawn_research_candidates(
            work_item,
            population,
            seed=seed + offset,
        )
        for candidate in candidates:
            if candidate.strategy_id in seen:
                continue
            seen.add(candidate.strategy_id)
            spawned.append(candidate)
    return ResearchExecutionResult(
        work_items=work_items,
        candidates=tuple(spawned),
    )


def evaluate_research_execution(
    execution: ResearchExecutionResult,
    data: pd.DataFrame,
    dataset_id: str,
    dataset_version: str,
    *,
    seed: int = 0,
    validation_policy: ValidationPolicy | None = None,
    fitness_policy: FitnessPolicy | None = None,
    promotion_policy: PromotionPolicy | None = None,
) -> ResearchEvaluationResult:
    """Evaluate spawned research candidates through the normal deterministic gates."""
    if not isinstance(seed, int):
        raise TypeError("research seed must be an integer")
    evaluations: list[CandidateEvaluation] = []
    for offset, candidate in enumerate(execution.candidates):
        evaluation = evaluate_candidate(
            candidate,
            data,
            dataset_id,
            dataset_version,
            seed=seed + offset,
            validation_policy=validation_policy,
            fitness_policy=fitness_policy,
            promotion_policy=promotion_policy,
        )
        evaluations.append(evaluation)
    eligible = tuple(
        evaluation.candidate_id
        for evaluation in evaluations
        if evaluation.promotion.eligible
    )
    return ResearchEvaluationResult(
        execution=execution,
        evaluations=tuple(evaluations),
        eligible_strategy_ids=eligible,
    )
