"""Offline tests for the end-to-end factory and the atsf CLI."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atsf.cli import format_report, load_market_data, main
from atsf.factory import dataset_fingerprint, run_factory, summarize_backtest


@pytest.fixture(scope="module")
def market_csv(tmp_path_factory):
    rng = np.random.default_rng(42)
    n = 1800
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    frame = pd.DataFrame({
        "date": pd.bdate_range("2015-01-01", periods=n),
        "open": open_, "high": np.maximum(open_, close) * 1.004,
        "low": np.minimum(open_, close) * 0.996, "close": close, "volume": 1e6,
    })
    path = tmp_path_factory.mktemp("data") / "test.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def report(market_csv):
    data = load_market_data("TEST", csv=str(market_csv))
    return run_factory(data, "TEST", perturbation_samples=3)


def test_factory_evaluates_every_generated_strategy(report):
    assert report.n_trials == len(report.rows) == len(report.strategies) == 8
    assert {row.family for row in report.rows} >= {"donchian_breakout", "rsi_reversion",
                                                    "ma_cross_sma_fast"}
    assert all(0.0 <= row.deflated_sharpe <= 1.0 for row in report.rows)


def test_strict_pbo_blocks_promotion_from_an_overfit_population(market_csv, report):
    strict = run_factory(load_market_data("TEST", csv=str(market_csv)), "TEST",
                         perturbation_samples=3, strict_pbo=True, max_pbo=0.01)
    if strict.population_overfit:
        assert not strict.promoted
        assert all(any("PBO" in r for r in row.reasons) for row in strict.rows)


def test_rejections_always_carry_reasons(report):
    for row in report.rows:
        if not row.promoted:
            assert row.reasons


def test_factory_is_deterministic(market_csv, report):
    again = run_factory(load_market_data("TEST", csv=str(market_csv)), "TEST", perturbation_samples=3)
    assert again.to_dict() == report.to_dict()


def test_factory_requires_enough_history(market_csv):
    data = load_market_data("TEST", csv=str(market_csv)).iloc[:300]
    with pytest.raises(ValueError, match="at least 500 bars"):
        run_factory(data, "TEST")


def test_fingerprint_tracks_content(market_csv):
    data = load_market_data("TEST", csv=str(market_csv))
    changed = data.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] *= 1.0001
    assert dataset_fingerprint(data) == dataset_fingerprint(data.copy())
    assert dataset_fingerprint(data) != dataset_fingerprint(changed)


def test_report_formatting_lists_every_row(report):
    text = format_report(report)
    assert all(row.family in text for row in report.rows)
    assert "promoted to paper" in text


def test_summarize_backtest():
    equity = pd.Series([100.0, 110.0, 99.0, 121.0], index=pd.bdate_range("2020-01-01", periods=4))
    stats = summarize_backtest(equity)
    assert stats["total_return"] == pytest.approx(0.21)
    assert stats["max_drawdown"] == pytest.approx(99 / 110 - 1)


def test_cli_backtest_and_research(market_csv, tmp_path, capsys):
    assert main(["backtest", "TEST", "--csv", str(market_csv), "--archetype", "roc_momentum"]) == 0
    assert "buy&hold" in capsys.readouterr().out
    out = tmp_path / "report.json"
    assert main(["research", "TEST", "--csv", str(market_csv), "--perturbations", "2",
                 "--json", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["symbol"] == "TEST" and len(payload["rows"]) == 8


def test_cli_lists_archetypes(capsys):
    assert main(["archetypes"]) == 0
    assert "donchian_breakout" in capsys.readouterr().out


def test_cli_reports_errors_without_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01"], "open": [1], "high": [1], "low": [1], "close": [1]}).to_csv(bad, index=False)
    assert main(["research", "X", "--csv", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_stooq_bot_challenge_is_reported_clearly(monkeypatch):
    from atsf.external_data import ExternalDataGateway

    monkeypatch.setattr(ExternalDataGateway, "_get", lambda self, url, user_agent="": b"<!DOCTYPE html><html>")
    with pytest.raises(RuntimeError, match="bot challenge"):
        ExternalDataGateway().market_daily("SPY", source="stooq")


def test_yahoo_payload_is_adjusted(monkeypatch):
    from atsf.external_data import ExternalDataGateway

    payload = {"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": [1704205800, 1704292200],
        "indicators": {"quote": [{"open": [100, 102], "high": [101, 103], "low": [99, 101],
                                  "close": [100, 102], "volume": [10, 20]}],
                       "adjclose": [{"adjclose": [50, 51]}]}}]}}
    monkeypatch.setattr(ExternalDataGateway, "_get",
                        lambda self, url, user_agent="": json.dumps(payload).encode())
    envelope = ExternalDataGateway().market_daily("SPY", source="yahoo")
    records = envelope.payload["records"]
    assert [r["close"] for r in records] == [50.0, 51.0]
    assert records[0]["open"] == pytest.approx(50.0)


def test_benchmark_gate(market_csv, report):
    assert report.benchmark_sharpe is not None
    for row in report.rows:
        if row.oos_sharpe < report.benchmark_sharpe:
            assert not row.promoted
            assert any("buy-and-hold" in reason for reason in row.reasons)
    relaxed = run_factory(load_market_data("TEST", csv=str(market_csv)), "TEST",
                          perturbation_samples=3, require_benchmark=False)
    assert not any("buy-and-hold" in r for row in relaxed.rows for r in row.reasons)
