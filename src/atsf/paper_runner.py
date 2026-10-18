from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from .compiler import CompiledStrategy, compile_strategy
from .paper import PaperBroker, PaperConfig, PaperSnapshot
from .paper_risk import PaperRiskController
from .strategy import StrategySpec


@dataclass(frozen=True)
class PaperRunResult:
    snapshots: tuple[PaperSnapshot, ...]
    fills: tuple
    final_equity: float
    halted: bool = False
    halt_reason: str | None = None


def run_compiled_paper_strategy(
    data: pd.DataFrame,
    compiled: CompiledStrategy,
    *,
    config: PaperConfig | None = None,
) -> PaperRunResult:
    """Run paper-only execution with decisions filled at the next available close.

    Signals are evaluated after each bar close and queued for the following bar.
    A signal on the final bar cannot fill in-window. Any position still open at
    the window end is liquidated at the final close unless the risk controller
    has halted the run. This broker never submits live orders.
    """
    if data.empty:
        raise ValueError("data cannot be empty")
    if "close" not in data.columns:
        raise ValueError("data must contain a close column")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("paper data index must be a DatetimeIndex")
    if not data.index.is_monotonic_increasing:
        raise ValueError("paper data index must be monotonically increasing")
    if data.index.has_duplicates:
        raise ValueError("paper data index must not contain duplicates")

    normalized_data = data.copy()
    try:
        closes = pd.to_numeric(normalized_data["close"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("paper close prices must be numeric") from exc
    if closes.isna().any() or not closes.map(math.isfinite).all():
        raise ValueError("paper close prices must be finite and non-missing")
    if (closes <= 0).any():
        raise ValueError("paper close prices must be strictly positive")
    normalized_data["close"] = closes
    entry, exit_ = compiled.signals(normalized_data)
    broker = PaperBroker(config)
    risk = PaperRiskController(compiled.strategy.risk.max_drawdown)
    snapshots: list[PaperSnapshot] = []
    pending_action: str | None = None

    for timestamp, row in normalized_data.iterrows():
        price = float(row["close"])
        current = broker.mark(timestamp, price)
        if not risk.check(current.equity):
            # A runtime halt cancels pending entries and liquidates any held
            # position at the current close; preserve the halted risk state.
            if broker.position > 0:
                broker.execute(timestamp, "sell", broker.position, price)
                current = broker.mark(timestamp, price)
            snapshots.append(current)
            break

        # Execute only an order decided on a previous bar.
        if pending_action == "buy" and broker.position == 0:
            quantity = (
                broker.max_affordable_quantity(price)
                * compiled.strategy.position_sizing.max_position
            )
            if quantity > 0:
                broker.execute(timestamp, "buy", quantity, price)
        elif pending_action == "sell" and broker.position > 0:
            broker.execute(timestamp, "sell", broker.position, price)
        pending_action = None

        snapshot = broker.mark(timestamp, price)
        snapshots.append(snapshot)
        if not risk.check(snapshot.equity):
            # Preserve a triggered halt while flattening any position. Do not
            # run another risk check after liquidation, which could clear it.
            if broker.position > 0:
                broker.execute(timestamp, "sell", broker.position, price)
                snapshots[-1] = broker.mark(timestamp, price)
            break

        # Decisions made at this close are eligible to fill on the next bar.
        if bool(entry.loc[timestamp]) and broker.position == 0:
            pending_action = "buy"
        elif bool(exit_.loc[timestamp]) and broker.position > 0:
            pending_action = "sell"

    if snapshots and broker.position > 0 and not risk.state.halted:
        timestamp = normalized_data.index[-1]
        price = float(normalized_data.iloc[-1]["close"])
        broker.execute(timestamp, "sell", broker.position, price)
        snapshots[-1] = broker.mark(timestamp, price)

    state = risk.state
    return PaperRunResult(
        tuple(snapshots),
        tuple(broker.fills),
        snapshots[-1].equity,
        state.halted,
        state.reason,
    )


def run_paper_strategy(
    data: pd.DataFrame,
    strategy: StrategySpec,
    *,
    config: PaperConfig | None = None,
) -> PaperRunResult:
    """Compile a validated StrategySpec, then run only the paper representation."""
    return run_compiled_paper_strategy(data, compile_strategy(strategy), config=config)
