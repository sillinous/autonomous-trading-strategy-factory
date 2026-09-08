import pytest

from atsf.paper_risk import PaperRiskController


def test_paper_risk_halts_on_drawdown():
    controller = PaperRiskController(max_drawdown=0.10)
    assert controller.check(100_000)
    assert controller.check(95_000)
    assert not controller.check(89_000)
    assert controller.state.halted
    assert controller.state.reason == "maximum drawdown breached"


def test_paper_risk_rejects_non_positive_equity():
    controller = PaperRiskController()
    assert not controller.check(0)
    assert controller.state.halted


def test_paper_risk_validates_limit():
    with pytest.raises(ValueError):
        PaperRiskController(max_drawdown=1.0)
