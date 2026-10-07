"""Shared test fixtures."""
from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _skilled_returns() -> pd.Series:
    rng = np.random.default_rng(20260101)
    return pd.Series(rng.normal(0.0015, 0.01, 750), index=pd.bdate_range("2022-01-03", periods=750))


def skilled_walk_forward() -> SimpleNamespace:
    """OOS evidence that clears deflated-Sharpe gates for small mock populations."""
    return SimpleNamespace(oos_returns=_skilled_returns())
