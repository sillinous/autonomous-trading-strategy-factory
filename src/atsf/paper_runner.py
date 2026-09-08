from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .paper import PaperBroker, PaperConfig, PaperSnapshot
from .signals import strategy_signals
from .strategy import StrategySpec


@dataclass(frozen=True)
class PaperRunResult:
    snapshots: tuple[PaperSnapshot, ...]
    fills: tuple
    final_equity: float


def run_paper_strategy(
    data: pd.DataFrame,
    strategy: StrategySpec,
    *,
    config: PaperConfig | None = None,
) -> PaperRunResult:
    """Run a StrategySpec through the deterministic paper broker only."""
    if data.empty:
        raise ValueError("data cannot be empty")
    entry, exit_ = strategy_signals(data, strategy)
    broker = PaperBroker(config)
    snapshots: list[PaperSnapshot] = []
    for timestamp, row in data.iterrows():
        price = float(row["close"])
        if bool(entry.loc[timestamp]) and broker.position == 0:
            quantity = broker.cash * strategy.position_sizing.max_position / price
            broker.execute(timestamp, "buy", quantity, price)
        elif bool(exit_.loc[timestamp]) and broker.position > 0:
            broker.execute(timestamp, "sell", broker.position, price)
        snapshots.append(broker.mark(timestamp, price))
    return PaperRunResult(tuple(snapshots), tuple(broker.fills), snapshots[-1].equity)
