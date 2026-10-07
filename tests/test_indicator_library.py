"""Every indicator is causal, warm-up aware, and numerically correct."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atsf.signals import SUPPORTED_INDICATORS, compute_indicators
from atsf.strategy import Indicator


def bars(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": 1e6}, index=pd.bdate_range("2020-01-01", periods=n))


def one(kind: str, period: int = 20, **parameters) -> Indicator:
    return Indicator(name=f"x_{kind}", kind=kind, period=period, parameters=parameters)


@pytest.mark.parametrize("kind", sorted(SUPPORTED_INDICATORS))
def test_every_indicator_is_causal(kind):
    """Values up to bar t never change when future bars are appended or altered."""
    data = bars()
    indicator = one(kind, 26) if kind.startswith("macd") else one(kind)
    full = compute_indicators(data, [indicator])[indicator.name]
    shocked = data.copy()
    shocked.iloc[200:, :4] *= 3.0
    partial = compute_indicators(shocked, [indicator])[indicator.name]
    pd.testing.assert_series_equal(full.iloc[:200], partial.iloc[:200])
    assert full.iloc[-50:].notna().all()


def test_reference_values():
    data = bars()
    close = data["close"]
    values = compute_indicators(data, [one("roc", 5), one("zscore", 10), one("bb_upper", 20, k=2),
                                       one("bb_lower", 20, k=2), one("highest", 20), one("lowest", 20)])
    assert values["x_roc"].iloc[-1] == pytest.approx(close.iloc[-1] / close.iloc[-6] - 1)
    window = close.iloc[-10:]
    assert values["x_zscore"].iloc[-1] == pytest.approx((close.iloc[-1] - window.mean()) / window.std())
    mid, std = close.iloc[-20:].mean(), close.iloc[-20:].std()
    assert values["x_bb_upper"].iloc[-1] == pytest.approx(mid + 2 * std)
    assert values["x_bb_lower"].iloc[-1] == pytest.approx(mid - 2 * std)
    assert values["x_highest"].iloc[-1] == pytest.approx(close.iloc[-21:-1].max())  # prior bars only
    assert values["x_lowest"].iloc[-1] == pytest.approx(close.iloc[-21:-1].min())


def test_breakout_is_possible_with_prior_bar_channel():
    data = bars()
    highest = compute_indicators(data, [one("highest", 20)])["x_highest"]
    assert (data["close"] > highest).any()


def test_macd_signal_is_ema_of_macd_line():
    data = bars()
    values = compute_indicators(data, [one("macd", 26, fast=12), one("macd_signal", 26, fast=12, signal=9)])
    line = values["x_macd"]
    assert values["x_macd_signal"].iloc[-1] == pytest.approx(
        line.ewm(span=9, adjust=False, min_periods=9).mean().iloc[-1])


def test_atr_requires_high_and_low():
    with pytest.raises(ValueError, match="missing data column"):
        compute_indicators(bars()[["close"]], [one("atr", 14)])


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="fast < period"):
        compute_indicators(bars(), [one("macd", 12, fast=26)])
    with pytest.raises(ValueError, match="positive number"):
        compute_indicators(bars(), [one("bb_upper", 20, k=-1)])
