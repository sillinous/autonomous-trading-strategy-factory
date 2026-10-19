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


@pytest.mark.parametrize(
    "config",
    [
        PaperConfig(initial_cash=True),
        PaperConfig(initial_cash="1000"),
        PaperConfig(initial_cash=float("inf")),
        PaperConfig(commission_bps=True),
        PaperConfig(commission_bps="1"),
        PaperConfig(commission_bps=float("nan")),
        PaperConfig(slippage_bps=True),
        PaperConfig(slippage_bps="2"),
        PaperConfig(slippage_bps=float("inf")),
    ],
)
def test_paper_broker_rejects_malformed_configuration(config):
    with pytest.raises(ValueError):
        PaperBroker(config)


def test_paper_broker_rejects_wrong_config_type():
    with pytest.raises(ValueError, match="PaperConfig"):
        PaperBroker(object())


@pytest.mark.parametrize(
    "quantity,price",
    [
        ("1", 100),
        (True, 100),
        (float("inf"), 100),
        (1, "100"),
        (1, False),
        (1, float("nan")),
    ],
)
def test_paper_broker_rejects_malformed_order_numbers(quantity, price):
    broker = PaperBroker()
    with pytest.raises(ValueError, match="quantity and price"):
        broker.execute(pd.Timestamp("2025-01-01"), "buy", quantity, price)


@pytest.mark.parametrize("price", ["100", True, float("nan"), float("inf"), 0])
def test_paper_broker_rejects_malformed_mark_prices(price):
    broker = PaperBroker()
    with pytest.raises(ValueError, match="price"):
        broker.mark(pd.Timestamp("2025-01-01"), price)


def test_paper_broker_rejects_overflowing_execution_arithmetic():
    broker = PaperBroker(PaperConfig(initial_cash=1e308, commission_bps=0, slippage_bps=0))
    with pytest.raises(ValueError, match="finite"):
        broker.execute(pd.Timestamp("2025-01-01"), "buy", 1e308, 1e308)


def test_paper_broker_rejects_unhashable_side_without_type_error():
    broker = PaperBroker()
    with pytest.raises(ValueError, match="side"):
        broker.execute(pd.Timestamp("2025-01-01"), [], 1, 100)


def test_paper_broker_rejects_non_finite_marked_equity():
    broker = PaperBroker(PaperConfig(initial_cash=1e308, commission_bps=0, slippage_bps=0))
    broker.position = 1e308
    with pytest.raises(ValueError, match="marked equity"):
        broker.mark(pd.Timestamp("2025-01-01"), 1e308)


def test_paper_broker_rejects_overflowing_post_trade_position_atomically():
    broker = PaperBroker(PaperConfig(initial_cash=1e308, commission_bps=0, slippage_bps=0))
    with pytest.raises(ValueError, match="post-trade"):
        broker.execute(pd.Timestamp("2025-01-01"), "buy", 1e308, 0.5)
    assert broker.cash == 1e308
    assert broker.position == 0
    assert broker.fills == []


def test_paper_broker_rejects_overflowing_sell_cash_atomically():
    broker = PaperBroker(PaperConfig(initial_cash=1e308, commission_bps=0, slippage_bps=0))
    broker.position = 1e308
    with pytest.raises(ValueError, match="post-trade"):
        broker.execute(pd.Timestamp("2025-01-01"), "sell", 1e308, 1.0)
    assert broker.cash == 1e308
    assert broker.position == 1e308
    assert broker.fills == []
