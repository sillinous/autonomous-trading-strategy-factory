from datetime import datetime

import pytest

from atsf.external_data import ExternalDataGateway, build_external_research_snapshot, ExternalDataEnvelope


def test_provider_catalog_reports_public_and_credentialed_sources(monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ATSF_SEC_USER_AGENT", raising=False)
    providers = ExternalDataGateway().providers()
    assert providers["market"]["stooq"]["configured"] is True
    assert providers["market"]["alphavantage"]["configured"] is False
    assert providers["fundamentals"]["sec"]["configured"] is False


def test_stooq_market_data_is_normalized_and_fingerprinted(monkeypatch):
    gateway = ExternalDataGateway()
    raw = b"Date,Open,High,Low,Close,Volume\n2026-01-02,10,12,9,11,100\n2026-01-03,11,13,10,12,120\n"
    monkeypatch.setattr(gateway, "_get", lambda url, **kwargs: raw)
    result = gateway.market_daily("TEST", source="stooq")
    assert result.source == "stooq"
    assert result.dataset == "daily_ohlcv"
    assert result.payload["rows"] == 2
    assert len(result.fingerprint) == 64


def test_fred_series_normalizes_observations(monkeypatch):
    gateway = ExternalDataGateway()
    raw = b"observation_date,DFF\n2026-01-01,3.5\n2026-01-02,.\n"
    monkeypatch.setattr(gateway, "_get", lambda url, **kwargs: raw)
    result = gateway.fred_series("DFF")
    assert result.source == "fred"
    assert result.payload["observations"][0]["value"] == 3.5
    assert result.payload["observations"][1]["value"] is None


def test_sec_requires_explicit_user_agent(monkeypatch):
    monkeypatch.delenv("ATSF_SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="ATSF_SEC_USER_AGENT"):
        ExternalDataGateway().sec_companyfacts("320193")


def test_news_requires_alpha_vantage_key(monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
        ExternalDataGateway().news_sentiment(tickers="AAPL")


def test_market_rejects_unknown_source():
    with pytest.raises(ValueError, match="unsupported market source"):
        ExternalDataGateway().market_daily("AAPL", source="unknown")


def test_market_rejects_empty_result(monkeypatch):
    gateway = ExternalDataGateway()
    raw = b"Date,Open,High,Low,Close,Volume\n"
    monkeypatch.setattr(gateway, "_get", lambda url, **kwargs: raw)
    with pytest.raises(ValueError):
        gateway.market_daily("TEST", source="stooq")


def test_time_bounds_are_forwarded(monkeypatch):
    gateway = ExternalDataGateway()
    seen = {}
    raw = b"Date,Open,High,Low,Close,Volume\n2026-01-02,10,12,9,11,100\n"
    def fetch(url, **kwargs):
        seen["url"] = url
        return raw
    monkeypatch.setattr(gateway, "_get", fetch)
    gateway.market_daily("TEST", source="stooq", start=datetime(2026, 1, 1), end=datetime(2026, 1, 4))
    assert "d1=20260101" in seen["url"]
    assert "d2=20260104" in seen["url"]


def test_external_research_snapshot_combines_requested_sources(monkeypatch):
    gateway = ExternalDataGateway()
    market = ExternalDataEnvelope("stooq", "daily_ohlcv", "2026-01-03T00:00:00+00:00", "a" * 64, {"symbol": "AAPL"})
    monkeypatch.setattr(gateway, "fred_series", lambda series: ExternalDataEnvelope("fred", "series", "now", series.lower(), {"series_id": series}))
    monkeypatch.setattr(gateway, "sec_companyfacts", lambda cik: ExternalDataEnvelope("sec", "companyfacts", "now", cik, {"cik": cik}))
    monkeypatch.setattr(gateway, "news_sentiment", lambda tickers=None, limit=50: ExternalDataEnvelope("alphavantage", "news_sentiment", "now", "n" * 64, {"tickers": tickers, "limit": limit}))
    snapshot = build_external_research_snapshot(
        gateway,
        market=market,
        macro_series=("DFF", "CPIAUCSL"),
        ciks=("320193",),
        news_tickers="AAPL",
        news_limit=10,
    )
    assert len(snapshot.macro) == 2
    assert len(snapshot.fundamentals) == 1
    assert snapshot.news is not None
    assert len(snapshot.fingerprint) == 64


def test_external_research_snapshot_fingerprint_changes_with_inputs():
    market = ExternalDataEnvelope("stooq", "daily_ohlcv", "now", "a" * 64, {"symbol": "AAPL"})
    first = build_external_research_snapshot(ExternalDataGateway(), market=market)
    second = build_external_research_snapshot(
        ExternalDataGateway(),
        market=market,
        macro_series=(),
    )
    assert first.fingerprint == second.fingerprint
