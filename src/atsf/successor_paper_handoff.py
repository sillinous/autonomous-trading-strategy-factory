from __future__ import annotations

from dataclasses import dataclass

from .certificate_integrity import verify_persisted_certificate
from .lifecycle import StrategyLifecycleStage
from .lifecycle_store import LifecycleStore
from .paper_admission import PaperAdmissionDecision, admit_to_paper
from .paper_admission_store import PaperAdmissionStore
from .portfolio_replay import verify_persisted_portfolio_run
from .registry import ExperimentRegistry


@dataclass(frozen=True)
class SuccessorPaperHandoffDecision:
    """Fail-closed boundary for moving one promoted successor into PAPER."""

    strategy_id: str
    run_id: str
    admitted: bool
    admission_id: str | None
    reasons: tuple[str, ...]


def handoff_successor_to_paper(
    registry: ExperimentRegistry,
    strategy_id: str,
    run_id: str,
    *,
    reason: str = "verified successor reproducibility handoff",
) -> SuccessorPaperHandoffDecision:
    """Verify successor provenance and paper-run integrity before PAPER admission."""
    reasons: list[str] = []
    if not strategy_id.strip():
        reasons.append("strategy_id is required")
    if not run_id.strip():
        reasons.append("run_id is required")
    if reasons:
        return SuccessorPaperHandoffDecision(strategy_id, run_id, False, None, tuple(reasons))

    lifecycle = LifecycleStore(registry._connection)
    admissions = PaperAdmissionStore(registry)
    try:
        state = lifecycle.get(strategy_id)
    except ValueError as exc:
        return SuccessorPaperHandoffDecision(strategy_id, run_id, False, None, (str(exc),))

    existing = admissions.get(run_id, strategy_id)
    if state is not None and state.stage is StrategyLifecycleStage.PAPER and existing is not None:
        if not admissions.verify(run_id, strategy_id):
            return SuccessorPaperHandoffDecision(
                strategy_id,
                run_id,
                False,
                existing.admission_id,
                ("existing PAPER admission failed durable verification",),
            )
        return SuccessorPaperHandoffDecision(strategy_id, run_id, True, existing.admission_id, ())

    if state is None:
        reasons.append("successor lifecycle state is missing")
    elif state.stage is not StrategyLifecycleStage.PROMOTED:
        reasons.append("successor must be in promoted lifecycle stage")

    run = registry.get_portfolio_run(run_id)
    if run is None:
        reasons.append("portfolio execution run is missing")
        return SuccessorPaperHandoffDecision(strategy_id, run_id, False, None, tuple(reasons))
    if run.get("halted"):
        reasons.append("portfolio execution run is halted")

    portfolio = registry.get_portfolio(str(run.get("portfolio_id", "")))
    if portfolio is None:
        reasons.append("portfolio provenance is missing")
        return SuccessorPaperHandoffDecision(strategy_id, run_id, False, None, tuple(reasons))
    members = portfolio.get("members")
    if not isinstance(members, dict) or strategy_id not in members:
        reasons.append("successor is not a member of the persisted portfolio")

    definition = portfolio.get("definition")
    experiment_ids = definition.get("experiment_ids") if isinstance(definition, dict) else None
    experiment_id = experiment_ids.get(strategy_id) if isinstance(experiment_ids, dict) else None
    if not experiment_id:
        reasons.append("successor experiment provenance is missing")
    else:
        experiment = registry.get_experiment(str(experiment_id))
        evidence = registry.get_evaluation_evidence(str(experiment_id))
        if experiment is None:
            reasons.append("successor experiment is missing")
        elif experiment.get("strategy_id") != strategy_id:
            reasons.append("successor experiment identity does not match strategy")
        if evidence is None:
            reasons.append("successor evaluation evidence is missing")
        else:
            promotion = evidence.get("promotion")
            if not isinstance(promotion, dict) or promotion.get("eligible") is not True:
                reasons.append("successor promotion evidence is not eligible")

    if not reasons:
        try:
            replay = verify_persisted_portfolio_run(registry, run_id)
        except (TypeError, ValueError) as exc:
            reasons.append(f"paper run verification failed: {exc}")
        else:
            if not replay.valid:
                reasons.append(f"paper run verification failed: {replay.reason}")

    certificate = registry.get_reproducibility_certificate(run_id)
    integrity = verify_persisted_certificate(registry, run_id)
    if not integrity.valid:
        reasons.append(f"reproducibility certificate is invalid: {integrity.reason}")
    elif certificate is None or certificate.get("certificate_id") != integrity.certificate_id:
        reasons.append("persisted reproducibility certificate identity is inconsistent")

    if reasons:
        return SuccessorPaperHandoffDecision(strategy_id, run_id, False, None, tuple(dict.fromkeys(reasons)))

    decision: PaperAdmissionDecision = admit_to_paper(
        strategy_id,
        source_stage=StrategyLifecycleStage.PROMOTED,
        run=run,
        certificate=certificate,
        reason=reason,
        registry=registry,
    )
    if not decision.admitted:
        return SuccessorPaperHandoffDecision(
            strategy_id, run_id, False, decision.admission_id, tuple(decision.reasons)
        )
    return SuccessorPaperHandoffDecision(
        strategy_id, run_id, True, decision.admission_id, ()
    )
