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


@pytest.mark.parametrize("equity", [float("nan"), float("inf"), float("-inf")])
def test_paper_risk_halts_on_non_finite_equity(equity):
    controller = PaperRiskController()
    assert not controller.check(equity)
    assert controller.state.halted
    assert controller.state.reason == "non-finite equity"


def test_paper_risk_halt_is_latched_after_equity_recovers():
    controller = PaperRiskController(max_drawdown=0.10)
    assert controller.check(100_000)
    assert not controller.check(89_000)
    assert not controller.check(100_000)
    assert controller.state.halted
    assert controller.state.reason == "maximum drawdown breached"


def test_non_finite_equity_halt_is_latched():
    controller = PaperRiskController()
    assert not controller.check(float("nan"))
    assert not controller.check(100_000)
    assert controller.state.halted
    assert controller.state.reason == "non-finite equity"


@pytest.mark.parametrize("limit", [True, "0.1", float("nan"), float("inf"), 0, 1])
def test_paper_risk_rejects_malformed_drawdown_limit(limit):
    with pytest.raises(ValueError, match="finite real number"):
        PaperRiskController(max_drawdown=limit)


@pytest.mark.parametrize("equity", [True, "100", None, complex(100, 0)])
def test_paper_risk_latches_halt_on_invalid_equity_type(equity):
    controller = PaperRiskController()
    assert not controller.check(equity)
    assert controller.state.halted
    assert controller.state.reason == "invalid equity type"
