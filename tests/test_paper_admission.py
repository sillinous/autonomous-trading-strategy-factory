import pytest

from atsf.lifecycle import StrategyLifecycleStage
from atsf.paper_admission import admit_to_paper


RUN = {"run_id": "run-1", "halted": False}
CERTIFICATE = {
    "run_id": "run-1",
    "certificate_id": "cert-1",
    "verified": True,
    "dataset_version": "dataset-v1",
    "data_bundle_version": "bundle-v1",
    "execution_fingerprint": "execution-v1",
}


def test_promoted_strategy_with_verified_certificate_is_admitted():
    decision = admit_to_paper(
        "strategy-1",
        source_stage=StrategyLifecycleStage.PROMOTED,
        run=RUN,
        certificate=CERTIFICATE,
    )
    assert decision.admitted
    assert decision.reasons == ()
    assert decision.event is not None
    assert decision.event.current is StrategyLifecycleStage.PAPER


@pytest.mark.parametrize(
    ("source_stage", "run", "certificate", "expected"),
    [
        (StrategyLifecycleStage.RESEARCH, RUN, CERTIFICATE, "promoted lifecycle"),
        (StrategyLifecycleStage.VALIDATED, RUN, CERTIFICATE, "promoted lifecycle"),
        (StrategyLifecycleStage.PROMOTED, RUN, None, "certificate"),
        (StrategyLifecycleStage.PROMOTED, {"run_id": "run-1", "halted": True}, CERTIFICATE, "halted"),
        (StrategyLifecycleStage.PROMOTED, RUN, {**CERTIFICATE, "run_id": "other"}, "run_id"),
        (StrategyLifecycleStage.PROMOTED, RUN, {**CERTIFICATE, "verified": False}, "not verified"),
        (StrategyLifecycleStage.PROMOTED, RUN, {**CERTIFICATE, "execution_fingerprint": ""}, "execution_fingerprint"),
    ],
)
def test_paper_admission_fails_closed(source_stage, run, certificate, expected):
    decision = admit_to_paper(
        "strategy-1",
        source_stage=source_stage,
        run=run,
        certificate=certificate,
    )
    assert not decision.admitted
    assert any(expected in reason for reason in decision.reasons)
    assert decision.event is None
