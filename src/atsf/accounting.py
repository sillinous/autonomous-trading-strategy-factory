from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

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
    if not isinstance(tolerance, Real) or isinstance(tolerance, bool) or not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not isinstance(equity, pd.Series) or not isinstance(returns, pd.Series):
        raise ValueError("equity and returns must be pandas Series")
    if equity.empty or returns.empty:
        raise ValueError("equity and returns cannot be empty")
    if not equity.index.equals(returns.index):
        raise ValueError("equity and returns indexes must match")
    if not equity.index.is_unique:
        raise ValueError("equity and returns indexes must be unique")
    if not equity.index.is_monotonic_increasing:
        raise ValueError("equity and returns indexes must be ordered")

    for name, series in (("equity", equity), ("returns", returns)):
        if (
            not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)
        ):
            raise ValueError(f"{name} must contain real numeric values")

    equity_values = equity.to_numpy(dtype=float)
    return_values = returns.to_numpy(dtype=float)
    if not np.isfinite(equity_values).all():
        raise ValueError("equity must contain only finite values")
    if not np.isfinite(return_values).all():
        raise ValueError("returns must contain only finite values")
    if (equity_values <= 0).any():
        raise ValueError("equity must remain positive")
    if (return_values <= -1).any():
        raise ValueError("simple returns must be greater than -100%")

    reconstructed = (1.0 + returns.astype(float)).cumprod() * equity_values[0]
    error = (reconstructed - equity.astype(float)).abs()
    max_error = float(error.max())
    reasons: list[str] = []
    if max_error > tolerance:
        reasons.append(
            f"equity reconciliation error {max_error:.3e} exceeds tolerance {tolerance:.3e}"
        )
    return AccountingReconciliation(not reasons, max_error, tuple(reasons))
