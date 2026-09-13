from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import pandas as pd

from .dataset_bundle import DatasetBundleIdentity
from .experiment import ExperimentSpec
from .fitness import FitnessPolicy
from .generator import StrategyCandidate
from .lifecycle import StrategyLifecycleStage
from .lifecycle_integration import synchronize_candidate_lifecycle
from .lifecycle_store import LifecycleStore
from .population import Candidate, strategy_id
from .promotion import PromotionPolicy
from .registry import ExperimentRegistry
from .replacement_evaluation import ReplacementEvaluationResult, evaluate_replacements
from .replacement_research import ReplacementResearchResult, generate_replacements
from .research_queue import ResearchRequest
from .research_registry import ResearchRequestStore
from .successor import SuccessorAdmission, admit_successor, successor_lineage
from .validation import ValidationPolicy


@dataclass(frozen=True)
class ReplacementCycleResult:
    cycle_id: str
    request_id: str
    source_strategy_id: str
    research: ReplacementResearchResult
    evaluation: ReplacementEvaluationResult
    admissions: tuple[SuccessorAdmission, ...]

    @property
    def admitted_strategy_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.admissions if item.admitted)


class ReplacementCycleStore:
    """Immutable durable boundary for completed replacement-cycle decisions."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        self._connection = registry._connection
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS replacement_cycles (
                cycle_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                source_strategy_id TEXT NOT NULL,
                cycle_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._connection.commit()

    def save(self, result: ReplacementCycleResult) -> ReplacementCycleResult:
        payload = {
            "cycle_id": result.cycle_id,
            "request_id": result.request_id,
            "source_strategy_id": result.source_strategy_id,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "request_id": candidate.request_id,
                    "parent_strategy_id": candidate.parent_strategy_id,
                    "mutation": candidate.mutation,
                    "strategy_id": strategy_id(candidate.strategy),
                }
                for candidate in result.research.candidates
            ],
            "evaluations": [
                {
                    "candidate_id": evaluation.candidate_id,
                    "validation_passed": evaluation.validation_passed,
                    "promotion_stage": evaluation.promotion.stage,
                    "promotion_eligible": evaluation.promotion.eligible,
                    "promotion_reasons": list(evaluation.promotion.reasons),
                }
                for evaluation in result.evaluation.evaluations
            ],
            "admissions": [
                {
                    "candidate_id": admission.candidate_id,
                    "parent_ids": list(admission.parent_ids),
                    "generation": admission.generation,
                    "admitted": admission.admitted,
                    "stage": admission.stage,
                    "reasons": list(admission.reasons),
                }
                for admission in result.admissions
            ],
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        row = self._connection.execute(
            "SELECT cycle_id, source_strategy_id, cycle_json FROM replacement_cycles WHERE request_id = ?",
            (result.request_id,),
        ).fetchone()
        if row is not None:
            if tuple(row) != (result.cycle_id, result.source_strategy_id, serialized):
                raise ValueError(f"replacement cycle is immutable: {result.request_id}")
            return result
        self._connection.execute(
            "INSERT INTO replacement_cycles(cycle_id, request_id, source_strategy_id, cycle_json) VALUES (?, ?, ?, ?)",
            (result.cycle_id, result.request_id, result.source_strategy_id, serialized),
        )
        self._connection.commit()
        return result

    def get(self, request_id: str) -> dict | None:
        row = self._connection.execute(
            "SELECT * FROM replacement_cycles WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return None if row is None else dict(row)


def _cycle_id(request: ResearchRequest) -> str:
    payload = {
        "request_id": request.request_id,
        "source_strategy_id": request.source_strategy_id,
        "reason": request.reason.value,
        "priority": request.priority,
        "constraints": list(request.constraints),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]


def _candidate_for_admission(candidate: StrategyCandidate) -> Candidate:
    canonical_id = strategy_id(candidate.strategy)
    from .lineage import LineageRecord

    return Candidate(
        strategy=candidate.strategy,
        strategy_id=canonical_id,
        lineage=LineageRecord(
            strategy_id=canonical_id,
            generation=1,
            parent_ids=(candidate.parent_strategy_id,) if candidate.parent_strategy_id else (),
            operator=candidate.mutation,
            parameters={"request_id": candidate.request_id, "candidate_id": candidate.candidate_id},
        ),
    )


def _persist_evaluation_evidence(
    registry: ExperimentRegistry,
    candidate: StrategyCandidate,
    evaluation,
    dataset_id: str,
    dataset_version: str,
    seed: int,
) -> None:
    spec = ExperimentSpec(
        candidate.strategy,
        dataset_id,
        dataset_version,
        seed,
    )
    if spec.experiment_id != evaluation.experiment.experiment_id:
        raise ValueError(f"replacement experiment identity mismatch: {evaluation.candidate_id}")
    registry.save_experiment(spec, evaluation.experiment)
    registry.save_evaluation_evidence(
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
                        "total_return": scenario.total_return,
                        "max_drawdown": scenario.max_drawdown,
                    }
                    for name, scenario in evaluation.robustness.scenarios.items()
                },
            },
            "promotion": {
                "stage": evaluation.promotion.stage,
                "eligible": evaluation.promotion.eligible,
                "reasons": evaluation.promotion.reasons,
            },
            "replacement": {
                "request_id": candidate.request_id,
                "candidate_id": candidate.candidate_id,
                "parent_strategy_id": candidate.parent_strategy_id,
                "mutation": candidate.mutation,
            },
        },
    )


def run_replacement_cycle(
    registry: ExperimentRegistry,
    request: ResearchRequest,
    data: pd.DataFrame,
    symbols: list[str],
    dataset_id: str,
    dataset_version: str,
    *,
    seed: int = 0,
    fitness_policy: FitnessPolicy | None = None,
    validation_policy: ValidationPolicy | None = None,
    promotion_policy: PromotionPolicy | None = None,
) -> ReplacementCycleResult:
    """Execute one durable replacement loop without enabling live execution."""
    if request.source_strategy_id is None:
        raise ValueError("replacement request requires a source strategy")
    persisted = ResearchRequestStore(registry).get(request.request_id)
    if persisted is None or persisted != request:
        raise ValueError("replacement research request is not durably persisted")

    cycle_store = ReplacementCycleStore(registry)
    existing = cycle_store.get(request.request_id)
    if existing is not None:
        raise ValueError(f"replacement cycle already completed: {request.request_id}")

    lifecycle = LifecycleStore(registry._connection)
    source = lifecycle.get(request.source_strategy_id)
    if source is None or source.stage is not StrategyLifecycleStage.DEGRADED:
        raise ValueError("replacement cycle requires a DEGRADED source strategy")

    research = generate_replacements(
        request,
        symbols,
        request_store=ResearchRequestStore(registry),
        registry=registry,
    )
    evaluation = evaluate_replacements(
        research,
        data,
        dataset_id,
        dataset_version,
        seed=seed,
        fitness_policy=fitness_policy,
        validation_policy=validation_policy,
        promotion_policy=promotion_policy,
    )

    if registry.get_dataset(dataset_id, dataset_version) is None:
        if data.empty:
            raise ValueError("replacement research data cannot be empty")
        registry.register_dataset(
            DatasetBundleIdentity(
                dataset_id=dataset_id,
                version=dataset_version,
                symbols=tuple(symbols),
                rows=len(data),
                start=str(data.index[0]),
                end=str(data.index[-1]),
                source="replacement-cycle",
                timeframe="1d",
            ),
            source="replacement-cycle",
        )

    for offset, (candidate, candidate_evaluation) in enumerate(
        zip(research.candidates, evaluation.evaluations)
    ):
        _persist_evaluation_evidence(
            registry,
            candidate,
            candidate_evaluation,
            dataset_id,
            dataset_version,
            seed + offset,
        )

    admissions: list[SuccessorAdmission] = []
    seen_ids: set[str] = set()
    for candidate, candidate_evaluation in zip(research.candidates, evaluation.evaluations):
        admission_candidate = _candidate_for_admission(candidate)
        admission = admit_successor(
            admission_candidate,
            candidate_evaluation,
            generation=2,
            existing_ids=seen_ids,
        )
        admissions.append(admission)
        if admission.admitted:
            seen_ids.add(admission.candidate_id)
            registry.save_lineage(successor_lineage(admission_candidate, admission))
            synchronize_candidate_lifecycle(
                lifecycle,
                candidate_evaluation,
                reason_prefix=f"replacement {request.request_id}",
            )

    if not any(item.admitted for item in admissions):
        for candidate_evaluation in evaluation.evaluations:
            synchronize_candidate_lifecycle(
                lifecycle,
                candidate_evaluation,
                reason_prefix=f"replacement {request.request_id}",
            )

    lifecycle.transition(
        request.source_strategy_id,
        StrategyLifecycleStage.DEGRADED,
        StrategyLifecycleStage.RESEARCH,
        reason=f"replacement research cycle {request.request_id} completed",
    )

    result = ReplacementCycleResult(
        cycle_id=_cycle_id(request),
        request_id=request.request_id,
        source_strategy_id=request.source_strategy_id,
        research=research,
        evaluation=evaluation,
        admissions=tuple(admissions),
    )
    return cycle_store.save(result)
