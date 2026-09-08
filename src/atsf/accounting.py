from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AccountingReconciliation:
    """Checks that a backtest equity curve has internally consistent returns."""

    passed: bool
    max_error: float
    reasons: tuple[str, ...]


def reconcile_equity(
    equity: pd.Series,
    returns: pd.Series,
    tolerance: float = 1e-10,
) -> AccountingReconciliation:
    """Verify that simple returns reproduce the supplied equity curve."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if equity.empty or returns.empty:
        raise ValueError("equity and returns cannot be empty")
    if not equity.index.equals(returns.index):
        raise ValueError("equity and returns indexes must match")

    equity_values = equity.astype(float).to_numpy()
    return_values = returns.astype(float).to_numpy()
    if not np.isfinite(equity_values).all():
        raise ValueError("equity must contain only finite values")
    if not np.isfinite(return_values).all():
        raise ValueError("returns must contain only finite values")
    if (equity_values <= 0).any():
        raise ValueError("equity must remain positive")
    if (return_values <= -1).any():
        raise ValueError("simple returns must be greater than -100%")

    reconstructed = (1.0 + returns).cumprod() * float(equity.iloc[0])
    error = (reconstructed - equity.astype(float)).abs()
    max_error = float(error.max())
    reasons: list[str] = []
    if max_error > tolerance:
        reasons.append(
            f"equity reconciliation error {max_error:.3e} exceeds tolerance {tolerance:.3e}"
        )
    return AccountingReconciliation(not reasons, max_error, tuple(reasons))
