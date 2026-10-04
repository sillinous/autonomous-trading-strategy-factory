from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .alphavantage import AlphaVantageDailyProvider
from .data import validate_market_data


@dataclass(frozen=True)
class ExternalDataEnvelope:
    source: str
    dataset: str
    fetched_at: str
    fingerprint: str
    payload: dict


class ExternalDataGateway:
    """Single fail-closed gateway for user-facing external research data.

    Provider credentials are read only from environment variables. Returned
    payloads are normalized and fingerprinted so downstream research can bind
    itself to the exact external snapshot it consumed.
    """

    def __init__(self, *, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def _get(self, url: str, *, user_agent: str = "atsf/0.1") -> bytes:
        request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json,text/csv,*/*"})
        with urlopen(request, timeout=self.timeout) as response:
            return response.read()

    @staticmethod
    def _envelope(source: str, dataset: str, payload: dict) -> ExternalDataEnvelope:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return ExternalDataEnvelope(
            source=source,
            dataset=dataset,
            fetched_at=datetime.now(UTC).isoformat(),
            fingerprint=hashlib.sha256(canonical).hexdigest(),
            payload=payload,
        )

    def providers(self) -> dict[str, dict]:
        return {
            "market": {
                "alphavantage": {"configured": bool(os.getenv("ALPHAVANTAGE_API_KEY")), "capabilities": ["daily_ohlcv", "news_sentiment"]},
                "stooq": {"configured": True, "capabilities": ["daily_ohlcv"]},
            },
            "macro": {
                "fred": {"configured": True, "capabilities": ["series"]},
            },
            "fundamentals": {
                "sec": {"configured": bool(os.getenv("ATSF_SEC_USER_AGENT")), "capabilities": ["companyfacts"]},
            },
        }

    def market_daily(self, symbol: str, *, source: str = "stooq", start=None, end=None) -> ExternalDataEnvelope:
        symbol = symbol.strip()
        if not symbol:
            raise ValueError("symbol is required")
        source = source.strip().lower()
        if source == "alphavantage":
            key = os.getenv("ALPHAVANTAGE_API_KEY", "")
            if not key:
                raise RuntimeError("ALPHAVANTAGE_API_KEY is not configured")
            frame = AlphaVantageDailyProvider(key, timeout=self.timeout).load(symbol, start, end)
        elif source == "stooq":
            query = urlencode({"s": symbol.lower(), "d1": start.strftime("%Y%m%d") if start else None, "d2": end.strftime("%Y%m%d") if end else None, "i": "d"})
            query = query.replace("d1=None&", "").replace("d2=None&", "")
            raw = self._get(f"https://stooq.com/q/d/l/?{query}", user_agent="atsf/0.1 market-data")
            frame = pd.read_csv(io.BytesIO(raw))
            frame = frame.rename(columns={c: c.lower() for c in frame.columns})
            if "date" not in frame.columns:
                raise ValueError("Stooq response contains no date column")
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            frame = frame.set_index("date")
            if "volume" not in frame.columns:
                frame["volume"] = 0.0
            frame = validate_market_data(frame)
            if start is not None:
                frame = frame.loc[frame.index >= start]
            if end is not None:
                frame = frame.loc[frame.index < end]
            if frame.empty:
                raise ValueError("requested market-data range is empty")
        else:
            raise ValueError(f"unsupported market source: {source}")
        payload = {
            "symbol": symbol,
            "rows": len(frame),
            "start": frame.index[0].isoformat(),
            "end": frame.index[-1].isoformat(),
            "columns": list(frame.columns),
            "records": [
                {"timestamp": index.isoformat(), **{column: float(row[column]) for column in frame.columns}}
                for index, row in frame.iterrows()
            ],
        }
        return self._envelope(source, "daily_ohlcv", payload)

    def fred_series(self, series_id: str, *, start=None, end=None) -> ExternalDataEnvelope:
        series_id = series_id.strip().upper()
        if not series_id:
            raise ValueError("series_id is required")
        params = {"cosd": start.strftime("%Y-%m-%d") if start else None, "coed": end.strftime("%Y-%m-%d") if end else None}
        query = urlencode({k: v for k, v in params.items() if v})
        raw = self._get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}{('&' + query) if query else ''}", user_agent="atsf/0.1 macro-data")
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        if not rows:
            raise ValueError("FRED response contains no observations")
        observations = [{"date": row["observation_date"], "value": None if row[series_id] in {"", "."} else float(row[series_id])} for row in rows]
        return self._envelope("fred", "series", {"series_id": series_id, "observations": observations})

    def sec_companyfacts(self, cik: str) -> ExternalDataEnvelope:
        cik_digits = "".join(ch for ch in cik.strip() if ch.isdigit())
        if not cik_digits:
            raise ValueError("CIK is required")
        cik_padded = cik_digits.zfill(10)
        user_agent = os.getenv("ATSF_SEC_USER_AGENT", "")
        if not user_agent:
            raise RuntimeError("ATSF_SEC_USER_AGENT is not configured")
        raw = self._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json", user_agent=user_agent)
        payload = json.loads(raw)
        if not isinstance(payload, dict) or "facts" not in payload:
            raise ValueError("SEC response contains no company facts")
        return self._envelope("sec", "companyfacts", {"cik": cik_padded, "entityName": payload.get("entityName"), "facts": payload["facts"]})

    def news_sentiment(self, *, tickers: str | None = None, limit: int = 50) -> ExternalDataEnvelope:
        key = os.getenv("ALPHAVANTAGE_API_KEY", "")
        if not key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is not configured")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        params = {"function": "NEWS_SENTIMENT", "apikey": key, "limit": limit}
        if tickers and tickers.strip():
            params["tickers"] = tickers.strip()
        raw = self._get("https://www.alphavantage.co/query?" + urlencode(params), user_agent="atsf/0.1 news-data")
        payload = json.loads(raw)
        if "Note" in payload:
            raise RuntimeError(f"Alpha Vantage rate limit: {payload['Note']}")
        if "Information" in payload:
            raise RuntimeError(f"Alpha Vantage information: {payload['Information']}")
        feed = payload.get("feed")
        if not isinstance(feed, list):
            raise ValueError("Alpha Vantage response contains no news feed")
        return self._envelope("alphavantage", "news_sentiment", {"items": feed[:limit]})


@dataclass(frozen=True)
class ExternalResearchSnapshot:
    """Immutable collection of external inputs consumed by a research run."""

    market: ExternalDataEnvelope
    macro: tuple[ExternalDataEnvelope, ...] = ()
    fundamentals: tuple[ExternalDataEnvelope, ...] = ()
    news: ExternalDataEnvelope | None = None
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.market.fingerprint:
            raise ValueError("market snapshot fingerprint is required")
        if self.news is None and not self.macro and not self.fundamentals:
            # A market-only snapshot is valid; its fingerprint is still required.
            pass
        if not self.fingerprint:
            canonical = {
                "market": self.market.fingerprint,
                "macro": [item.fingerprint for item in self.macro],
                "fundamentals": [item.fingerprint for item in self.fundamentals],
                "news": None if self.news is None else self.news.fingerprint,
            }
            object.__setattr__(
                self,
                "fingerprint",
                hashlib.sha256(
                    json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            )


def _external_snapshot(
    gateway: ExternalDataGateway,
    *,
    market: ExternalDataEnvelope,
    macro_series: tuple[str, ...] = (),
    ciks: tuple[str, ...] = (),
    news_tickers: str | None = None,
    news_limit: int = 50,
) -> ExternalResearchSnapshot:
    """Fetch every explicitly requested external input as one immutable snapshot."""
    macro = tuple(gateway.fred_series(series) for series in macro_series)
    fundamentals = tuple(gateway.sec_companyfacts(cik) for cik in ciks)
    news = gateway.news_sentiment(tickers=news_tickers, limit=news_limit) if news_tickers else None
    return ExternalResearchSnapshot(
        market=market,
        macro=macro,
        fundamentals=fundamentals,
        news=news,
    )


@dataclass(frozen=True)
class ExternalResearchSnapshot:
    """Immutable collection of external inputs consumed by a research run."""

    market: ExternalDataEnvelope
    macro: tuple[ExternalDataEnvelope, ...] = ()
    fundamentals: tuple[ExternalDataEnvelope, ...] = ()
    news: ExternalDataEnvelope | None = None
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.market.fingerprint:
            raise ValueError("market snapshot fingerprint is required")
        if not self.fingerprint:
            canonical = {
                "market": self.market.fingerprint,
                "macro": [item.fingerprint for item in self.macro],
                "fundamentals": [item.fingerprint for item in self.fundamentals],
                "news": None if self.news is None else self.news.fingerprint,
            }
            object.__setattr__(
                self,
                "fingerprint",
                hashlib.sha256(
                    json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            )


def build_external_research_snapshot(
    gateway: ExternalDataGateway,
    *,
    market: ExternalDataEnvelope,
    macro_series: tuple[str, ...] = (),
    ciks: tuple[str, ...] = (),
    news_tickers: str | None = None,
    news_limit: int = 50,
) -> ExternalResearchSnapshot:
    """Fetch explicitly requested external inputs as one immutable snapshot."""
    macro = tuple(gateway.fred_series(series) for series in macro_series)
    fundamentals = tuple(gateway.sec_companyfacts(cik) for cik in ciks)
    news = gateway.news_sentiment(tickers=news_tickers, limit=news_limit) if news_tickers else None
    return ExternalResearchSnapshot(
        market=market,
        macro=macro,
        fundamentals=fundamentals,
        news=news,
    )
