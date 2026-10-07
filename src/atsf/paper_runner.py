from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .compiler import CompiledStrategy, compile_strategy
from .execution import ExecutionEvent, ExecutionPolicy, run_sleeve
from .paper import PaperBroker, PaperConfig, PaperFill, PaperSnapshot  # noqa: F401
from .signals import position_state
from .strategy import StrategySpec

HALT_REASON = "maximum drawdown breached"


@dataclass(frozen=True)
class PaperRunResult:
    snapshots: tuple[PaperSnapshot, ...]
    fills: tuple[PaperFill, ...]
    final_equity: float
    halted: bool = False
    halt_reason: str | None = None
    events: tuple[ExecutionEvent, ...] = ()


def run_compiled_paper_strategy(
    data: pd.DataFrame,
    compiled: CompiledStrategy,
    *,
    config: PaperConfig | None = None,
    policy: ExecutionPolicy | None = None,
    liquidate_at_end: bool = True,
) -> PaperRunResult:
    """Run a compiled StrategySpec through the shared engine and a paper broker only.

    Fills are identical to :func:`atsf.backtest.run_long_signal_backtest` on the same
    data; the only paper-specific behavior is optional end-of-sample liquidation.
    """
    if data.empty:
        raise ValueError("data cannot be empty")
    entry, exit_ = compiled.signals(data)
    broker = PaperBroker(config)
    engine = run_sleeve(compiled.strategy, data, position_state(entry, exit_), broker, policy,
                        liquidate_at_end=liquidate_at_end)
    snapshots = tuple(engine.snapshots)
    halted = engine.halted_at is not None
    return PaperRunResult(snapshots, tuple(broker.fills), snapshots[-1].equity, halted,
                          HALT_REASON if halted else None, tuple(engine.events))


def run_paper_strategy(
    data: pd.DataFrame,
    strategy: StrategySpec,
    *,
    config: PaperConfig | None = None,
    policy: ExecutionPolicy | None = None,
    liquidate_at_end: bool = True,
) -> PaperRunResult:
    """Compile a validated strategy, then run only the compiled representation."""
    return run_compiled_paper_strategy(data, compile_strategy(strategy), config=config,
                                       policy=policy, liquidate_at_end=liquidate_at_end)
