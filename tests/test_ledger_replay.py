import pytest

from atsf.ledger_replay import validate_execution_ledger
from atsf.portfolio_audit import PortfolioAuditEvent


def event(sequence: int, action: str, quantity: float, price: float, fee: float) -> PortfolioAuditEvent:
    return PortfolioAuditEvent(sequence, "strategy-a", action, f"2026-01-0{sequence + 1}T00:00:00", quantity, price, fee)


def test_ledger_replay_reconciles_flat_execution() -> None:
    buy_price = 10.0
    buy_qty = 5.0
    buy_fee = buy_price * buy_qty * 1.0 / 10_000.0
    sell_price = 12.0
    sell_fee = sell_price * buy_qty * 1.0 / 10_000.0
    result = validate_execution_ledger(
        (event(0, "buy", buy_qty, buy_price, buy_fee), event(1, "sell", buy_qty, sell_price, sell_fee)),
        initial_cash_by_strategy={"strategy-a": 100.0},
        commission_bps=1.0,
        initial_cash=100.0,
        final_equity=109.989,
    )
    assert result.cash == pytest.approx(109.989)
    assert result.positions == {"strategy-a": 0.0}


def test_ledger_replay_rejects_oversell() -> None:
    with pytest.raises(ValueError, match="sell exceeding"):
        validate_execution_ledger(
            (event(0, "sell", 1.0, 10.0, 0.01),),
            initial_cash_by_strategy={"strategy-a": 100.0},
            commission_bps=1.0,
            initial_cash=100.0,
            final_equity=100.0,
        )


def test_ledger_replay_rejects_fee_mismatch() -> None:
    with pytest.raises(ValueError, match="fee does not match"):
        validate_execution_ledger(
            (event(0, "buy", 1.0, 10.0, 0.0),),
            initial_cash_by_strategy={"strategy-a": 100.0},
            commission_bps=1.0,
            initial_cash=100.0,
            final_equity=100.0,
        )


def test_ledger_replay_rejects_non_flat_end_state() -> None:
    with pytest.raises(ValueError, match="does not end flat"):
        validate_execution_ledger(
            (event(0, "buy", 1.0, 10.0, 0.01),),
            initial_cash_by_strategy={"strategy-a": 100.0},
            commission_bps=1.0,
            initial_cash=100.0,
            final_equity=100.0,
        )
