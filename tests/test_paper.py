import pandas as pd
import pytest

from atsf.paper import PaperBroker, PaperConfig


def test_paper_broker_applies_slippage_and_fees():
    broker = PaperBroker(PaperConfig(initial_cash=1_000, commission_bps=10, slippage_bps=20))
    fill = broker.execute(pd.Timestamp("2025-01-01"), "buy", 5, 100)
    assert fill.price == pytest.approx(100.2)
    assert fill.fee == pytest.approx(0.501)
    assert broker.position == 5
    assert broker.cash == pytest.approx(498.499)


def test_paper_broker_rejects_oversized_sell():
    broker = PaperBroker()
    with pytest.raises(ValueError, match="position"):
        broker.execute(pd.Timestamp("2025-01-01"), "sell", 1, 100)


def test_paper_broker_marks_equity():
    broker = PaperBroker(PaperConfig(initial_cash=1_000))
    broker.execute(pd.Timestamp("2025-01-01"), "buy", 2, 100)
    snapshot = broker.mark(pd.Timestamp("2025-01-02"), 110)
    assert snapshot.equity > 1_000



@pytest.mark.parametrize(
    "commission_bps,slippage_bps",
    [(5_000, 0), (4_000, 1_000), (2_500, 2_500)],
)
def test_paper_broker_rejects_extreme_combined_transaction_costs(commission_bps, slippage_bps):
    with pytest.raises(ValueError, match="combined transaction costs must be below 5000 bps"):
        PaperBroker(PaperConfig(commission_bps=commission_bps, slippage_bps=slippage_bps))
