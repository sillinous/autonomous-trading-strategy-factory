from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    data: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> ChronologicalSplit:
    """Split time-ordered data without shuffling or leakage."""
    if data.empty:
        raise ValueError("data must not be empty")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be monotonically increasing")

    n = len(data)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= n:
        raise ValueError("fractions produce an empty split")
    return ChronologicalSplit(
        train=data.iloc[:train_end].copy(),
        validation=data.iloc[train_end:validation_end].copy(),
        test=data.iloc[validation_end:].copy(),
    )


@dataclass(frozen=True)
class WalkForwardWindow:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def walk_forward_windows(
    data: pd.DataFrame,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[WalkForwardWindow]:
    """Create chronological rolling windows; the default advances one observation."""
    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be monotonically increasing")
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("window sizes must be positive")
    step = 1 if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    windows: list[WalkForwardWindow] = []
    start = 0
    total = train_size + validation_size + test_size
    while start + total <= len(data):
        train_end = start + train_size
        validation_end = train_end + validation_size
        test_end = validation_end + test_size
        windows.append(
            WalkForwardWindow(
                train=data.iloc[start:train_end].copy(),
                validation=data.iloc[train_end:validation_end].copy(),
                test=data.iloc[validation_end:test_end].copy(),
            )
        )
        start += step
    return windows
