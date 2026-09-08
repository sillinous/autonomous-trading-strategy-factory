from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import Comparator, Condition, Indicator, Signal, StrategySpec


SUPPORTED_INDICATORS = frozenset({"sma", "ema", "rsi"})


def _source(data: pd.DataFrame, name: str) -> pd.Series:
    if name not in data.columns:
        raise ValueError(f"missing data column: {name}")
    return pd.to_numeric(data[name], errors="raise")


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + relative_strength))
    return result.mask((average_loss == 0) & (average_gain > 0), 100.0)


def compute_indicators(data: pd.DataFrame, indicators: list[Indicator]) -> dict[str, pd.Series]:
    """Compute the small, deterministic indicator vocabulary supported by the DSL."""
    values: dict[str, pd.Series] = {}
    for indicator in indicators:
        name = indicator.name.lower()
        if name not in SUPPORTED_INDICATORS:
            raise ValueError(f"unsupported indicator: {indicator.name}")
        if indicator.period is None:
            raise ValueError(f"indicator {indicator.name} requires a period")
        source = _source(data, indicator.source)
        if name == "sma":
            result = source.rolling(indicator.period, min_periods=indicator.period).mean()
        elif name == "ema":
            result = source.ewm(span=indicator.period, adjust=False, min_periods=indicator.period).mean()
        else:
            result = _rsi(source, indicator.period)
        if indicator.name in values:
            raise ValueError(f"duplicate indicator name: {indicator.name}")
        values[indicator.name] = result
    return values


def _operand(
    data: pd.DataFrame,
    indicators: dict[str, pd.Series],
    operand: str | float | int,
) -> pd.Series | float | int:
    if isinstance(operand, str):
        if operand in indicators:
            return indicators[operand]
        return _source(data, operand)
    return operand


def _compare(
    left: pd.Series, comparator: Comparator, right: pd.Series | float | int
) -> pd.Series:
    if comparator == Comparator.GT:
        return left > right
    if comparator == Comparator.GTE:
        return left >= right
    if comparator == Comparator.LT:
        return left < right
    if comparator == Comparator.LTE:
        return left <= right
    if comparator == Comparator.EQ:
        return left == right
    right_series = right if isinstance(right, pd.Series) else pd.Series(right, index=left.index)
    previous_left = left.shift(1)
    previous_right = right_series.shift(1)
    if comparator == Comparator.CROSSES_ABOVE:
        return (left > right_series) & (previous_left <= previous_right)
    if comparator == Comparator.CROSSES_BELOW:
        return (left < right_series) & (previous_left >= previous_right)
    raise ValueError(f"unsupported comparator: {comparator.value}")


def evaluate_condition(
    data: pd.DataFrame, condition: Condition, indicators: dict[str, pd.Series]
) -> pd.Series:
    left = _operand(data, indicators, condition.left)
    right = _operand(data, indicators, condition.right)
    if not isinstance(left, pd.Series):
        raise ValueError("condition left operand must resolve to a series")
    return _compare(left, condition.comparator, right).fillna(False).astype(bool)


def evaluate_signal(
    data: pd.DataFrame, signal: Signal, indicators: dict[str, pd.Series]
) -> pd.Series:
    """Evaluate a signal as AND across ``all`` and OR across ``any`` groups."""
    all_results = [evaluate_condition(data, condition, indicators) for condition in signal.all]
    any_results = [evaluate_condition(data, condition, indicators) for condition in signal.any]
    all_result = (
        pd.concat(all_results, axis=1).all(axis=1)
        if all_results
        else pd.Series(True, index=data.index)
    )
    any_result = (
        pd.concat(any_results, axis=1).any(axis=1)
        if any_results
        else pd.Series(False, index=data.index)
    )
    return (all_result & any_result) if signal.all and signal.any else (
        all_result if signal.all else any_result
    )


def strategy_signals(data: pd.DataFrame, strategy: StrategySpec) -> tuple[pd.Series, pd.Series]:
    """Compile a StrategySpec into deterministic entry/exit boolean series."""
    indicators = compute_indicators(data, strategy.indicators)
    return (
        evaluate_signal(data, strategy.entry, indicators),
        evaluate_signal(data, strategy.exit, indicators),
    )
