from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import pandas as pd

from .signals import strategy_signals
from .strategy import Side, StrategySpec


@dataclass(frozen=True)
class CompiledStrategy:
    """Immutable executable representation of a validated StrategySpec."""

    strategy_id: str
    compiler_version: str
    strategy: StrategySpec

    def signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Evaluate the compiled strategy without generating or executing arbitrary code."""
        if self.strategy.side is not Side.LONG:
            raise NotImplementedError("short-side compiled execution is not implemented")
        return strategy_signals(data, self.strategy)

    def manifest(self) -> dict[str, str | int]:
        """Return the reproducibility manifest for audit and paper-run records."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy.name,
            "strategy_version": self.strategy.version,
            "compiler_version": self.compiler_version,
        }


def compile_strategy(strategy: StrategySpec, *, compiler_version: str = "1") -> CompiledStrategy:
    """Compile a typed strategy definition into a deterministic execution object.

    The compiler never evaluates generated source code: executable behavior is limited
    to the trusted signal evaluator and the validated StrategySpec DSL.
    """
    if not compiler_version.strip():
        raise ValueError("compiler_version must be non-empty")
    if strategy.side is not Side.LONG:
        raise NotImplementedError("short-side compilation is not implemented")
    canonical = strategy.model_dump(mode="json")
    payload = json.dumps(
        {"compiler_version": compiler_version, "strategy": canonical},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    strategy_id = hashlib.sha256(payload).hexdigest()[:16]
    return CompiledStrategy(strategy_id, compiler_version, strategy)
