from __future__ import annotations

from .lifecycle import StrategyLifecycleStage
from .lifecycle_store import LifecycleStore, PersistedLifecycle
from .orchestrator import CandidateEvaluation


def synchronize_candidate_lifecycle(
    store: LifecycleStore,
    evaluation: CandidateEvaluation,
    *,
    reason_prefix: str = "candidate evaluation",
) -> PersistedLifecycle:
    """Persist the fail-closed lifecycle implied by a completed candidate evaluation.

    Candidates begin in RESEARCH. A failed validation gate is degraded. A validated
    candidate remains VALIDATED unless every promotion gate passes, in which case it
    advances to PROMOTED. The helper is intentionally transactional so callers can
    compose it with a larger research-cycle transaction.
    """
    strategy_id = evaluation.candidate_id
    current = store.get(strategy_id)
    if current is None:
        store.save(strategy_id, StrategyLifecycleStage.RESEARCH, reason=f"{reason_prefix}: admitted")
        current = store.get(strategy_id)
        if current is None:  # pragma: no cover - defensive database failure
            raise RuntimeError("lifecycle state could not be persisted")

    if current.stage is StrategyLifecycleStage.RESEARCH:
        if not evaluation.validation_passed:
            return store.transition(
                strategy_id,
                StrategyLifecycleStage.RESEARCH,
                StrategyLifecycleStage.DEGRADED,
                reason=f"{reason_prefix}: validation failed",
            )
        current = store.transition(
            strategy_id,
            StrategyLifecycleStage.RESEARCH,
            StrategyLifecycleStage.VALIDATED,
            reason=f"{reason_prefix}: validation passed",
        )

    if current.stage is StrategyLifecycleStage.VALIDATED:
        if evaluation.promotion.eligible:
            return store.transition(
                strategy_id,
                StrategyLifecycleStage.VALIDATED,
                StrategyLifecycleStage.PROMOTED,
                reason=f"{reason_prefix}: promotion gates passed",
            )
        return current

    return current
