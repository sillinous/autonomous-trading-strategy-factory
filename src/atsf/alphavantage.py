from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .data import validate_market_data


JsonTransport = Callable[[str], bytes]


class AlphaVantageDailyProvider:
    """Alpha Vantage daily OHLCV provider normalized to ATSF's canonical schema.

    The adapter is historical-data only: it uses TIME_SERIES_DAILY and never
    places orders.  Full history availability depends on the vendor plan.
    """

    source = "alphavantage"
    timeframe = "1d"
    schema_version = "ohlcv.v1"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        outputsize: str = "full",
        transport: JsonTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Alpha Vantage API key is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if outputsize not in {"compact", "full"}:
            raise ValueError("outputsize must be compact or full")
        self._api_key = api_key.strip()
        self._timeout = timeout
        self._outputsize = outputsize
        self._transport = transport or self._fetch

    def _fetch(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "atsf/0.1"})
        with urlopen(request, timeout=self._timeout) as response:
            return response.read()

    def load(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        if not symbol.strip():
            raise ValueError("symbol is required")
        params = urlencode(
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol.strip(),
                "outputsize": self._outputsize,
                "apikey": self._api_key,
            }
        )
        payload = json.loads(self._transport(f"https://www.alphavantage.co/query?{params}"))
        if "Error Message" in payload:
            raise ValueError(f"Alpha Vantage error: {payload['Error Message']}")
        if "Note" in payload:
            raise RuntimeError(f"Alpha Vantage rate limit: {payload['Note']}")
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict) or not series:
            raise ValueError("Alpha Vantage response contains no daily time series")

        rows = []
        for timestamp, values in series.items():
            rows.append(
                {
                    "timestamp": pd.Timestamp(timestamp),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "volume": float(values["5. volume"]),
                }
            )
        frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
        frame = validate_market_data(frame)
        if start is not None:
            frame = frame.loc[frame.index >= start]
        if end is not None:
            frame = frame.loc[frame.index < end]
        if frame.empty:
            raise ValueError("requested market-data range is empty")
        return frame.copy()
