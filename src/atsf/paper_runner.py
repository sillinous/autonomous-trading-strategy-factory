from __future__ import annotations

from dataclasses import dataclass

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
    """Run a compiled StrategySpec through the deterministic paper broker only."""
    if data.empty:
        raise ValueError("data cannot be empty")
    entry, exit_ = compiled.signals(data)
    broker = PaperBroker(config)
    risk = PaperRiskController(compiled.strategy.risk.max_drawdown)
    snapshots: list[PaperSnapshot] = []
    for timestamp, row in data.iterrows():
        price = float(row["close"])
        current = broker.mark(timestamp, price)
        if not risk.check(current.equity):
            snapshots.append(current)
            break
        if bool(entry.loc[timestamp]) and broker.position == 0:
            quantity = broker.cash * compiled.strategy.position_sizing.max_position / price
            broker.execute(timestamp, "buy", quantity, price)
        elif bool(exit_.loc[timestamp]) and broker.position > 0:
            broker.execute(timestamp, "sell", broker.position, price)
        snapshot = broker.mark(timestamp, price)
        snapshots.append(snapshot)
        if not risk.check(snapshot.equity):
            break

    if snapshots and broker.position > 0 and not risk.state.halted:
        timestamp = data.index[-1]
        price = float(data.iloc[-1]["close"])
        broker.execute(timestamp, "sell", broker.position, price)
        snapshots[-1] = broker.mark(timestamp, price)
        risk.check(snapshots[-1].equity)

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
    """Compile a validated strategy, then run only the compiled representation."""
    return run_compiled_paper_strategy(data, compile_strategy(strategy), config=config)
