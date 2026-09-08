from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AccountingReconciliation:
    """Checks that a backtest equity curve has internally consistent returns."""

    passed: bool
    max_error: float
    reasons: tuple[str, ...]


def reconcile_equity(equity: pd.Series, returns: pd.Series, tolerance: float = 1e-10) -> AccountingReconciliation:
    """Verify that supplied simple returns reproduce the supplied equity curve."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if equity.empty or returns.empty:
        raise ValueError("equity and returns cannot be empty")
    if not equity.index.equals(returns.index):
        raise ValueError("equity and returns indexes must match")
    if (equity <= 0).any():
        raise ValueError("equity must remain positive")

    reconstructed = (1.0 + returns.astype(float)).cumprod() * float(equity.iloc[0])
    error = (reconstructed - equity.astype(float)).abs()
    max_error = float(error.max())
    reasons: list[str] = []
    if not error.index.equals(equity.index):
        reasons.append("reconstructed equity index mismatch")
    if max_error > tolerance:
        reasons.append(f"equity reconciliation error {max_error:.3e} exceeds tolerance {tolerance:.3e}")
    return AccountingReconciliation(not reasons, max_error, tuple(reasons))
