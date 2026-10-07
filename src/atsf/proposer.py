"""Claude-proposed strategy hypotheses, admitted only through the typed DSL.

The model never writes code. It returns JSON that must parse as a
:class:`~atsf.strategy.StrategySpec`, compile against real data, and trade at least
once; anything else is rejected with a reason. Admitted proposals then face exactly
the same gates as generated strategies, and they count toward the trial total used
for deflated-Sharpe correction, so asking for more ideas raises the evidence bar.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.request import Request, urlopen

import pandas as pd
from pydantic import ValidationError

from .signals import SUPPORTED_INDICATORS, strategy_position
from .strategy import Side, StrategySpec

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_PROPOSALS = 12

Transport = Callable[[dict], dict]

INDICATOR_GUIDE = """\
Indicator kinds (every indicator needs a unique `name`, a `kind`, and an integer `period`):
- sma, ema: moving averages of `source` (default "close")
- rsi: Wilder RSI, 0-100
- roc: rate of change over `period` bars (0.05 = +5%)
- zscore: (source - SMA) / rolling std
- stdev: rolling standard deviation of source
- atr: average true range (uses high/low/close)
- highest, lowest: max/min of the PRIOR `period` bars (so close > highest is a breakout)
- bb_upper, bb_lower: Bollinger bands; parameters {"k": 2.0}
- macd: EMA(fast) - EMA(period); fast defaults to round(period*12/26)
- macd_signal: EMA of macd; parameters {"signal": 9}
Conditions compare `left` to `right`, where each is an indicator name, a price column
(open, high, low, close, volume), or (right only) a number. Comparators: ">", ">=", "<",
"<=", "crosses_above", "crosses_below". A Signal has `all` (AND) and/or `any` (OR) lists.
Position sizing method: "fixed_fraction" (value = fraction of equity), "volatility_target"
(value = annualized vol target, e.g. 0.15), or "equal_weight". max_position <= 1.
Risk: max_position <= 1, optional stop_loss (e.g. 0.08), optional max_drawdown (e.g. 0.3).
Execution: long-only, signals decided at the close, filled at the next bar's open."""


@dataclass(frozen=True)
class Proposal:
    strategy: StrategySpec | None
    rationale: str
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class ProposalBatch:
    model: str
    proposals: tuple[Proposal, ...]
    usage: dict = field(default_factory=dict)

    @property
    def accepted(self) -> list[StrategySpec]:
        return [p.strategy for p in self.proposals if p.accepted and p.strategy is not None]


def _http_transport(api_key: str, timeout: float = 120.0) -> Transport:
    def send(body: dict) -> dict:
        request = Request(ANTHROPIC_URL, data=json.dumps(body).encode(), method="POST", headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        })
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
            return json.load(response)
    return send


def _extract_json(text: str) -> list:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("response contains no JSON array")
    payload = json.loads(cleaned[start:end + 1])
    if not isinstance(payload, list):
        raise ValueError("response JSON is not an array")
    return payload


def build_prompt(symbol: str, data: pd.DataFrame, n: int, history: list[dict] | None) -> str:
    returns = data["close"].pct_change().dropna()
    context = {
        "symbol": symbol,
        "bars": len(data),
        "start": data.index[0].date().isoformat(),
        "end": data.index[-1].date().isoformat(),
        "annualized_return": round(float((1 + returns.mean()) ** 252 - 1), 4),
        "annualized_volatility": round(float(returns.std() * 252 ** 0.5), 4),
    }
    lines = [
        f"Propose {n} diverse, long-only daily trading strategies for {symbol}.",
        "They will be judged out-of-sample on rolling one-year windows and must beat",
        "buy-and-hold on Sharpe ratio after costs (1bp commission, 2bp slippage), with",
        "at least 20 OOS trades, robustness to parameter perturbation, and a deflated",
        "Sharpe correction for every strategy tried. Prefer simple, economically",
        "motivated rules over curve-fit ones; avoid near-duplicates of each other.",
        "",
        f"Market context: {json.dumps(context)}",
        "",
        INDICATOR_GUIDE,
    ]
    if history:
        lines += ["", "Results of strategies already tested (learn from the rejection reasons):",
                  json.dumps(history[:20], indent=1)]
    lines += [
        "",
        "Respond with ONLY a JSON array. Each element: "
        '{"rationale": "<one sentence>", "strategy": <StrategySpec JSON>}.',
        'Use "universe": ["' + symbol + '"] and "timeframe": "1d". Example strategy:',
        json.dumps({
            "name": "trend-pullback", "universe": [symbol], "timeframe": "1d",
            "indicators": [{"name": "trend", "kind": "sma", "period": 150},
                           {"name": "dip", "kind": "rsi", "period": 4}],
            "entry": {"all": [{"left": "close", "comparator": ">", "right": "trend"},
                              {"left": "dip", "comparator": "<", "right": 25}]},
            "exit": {"any": [{"left": "dip", "comparator": ">", "right": 65}]},
            "position_sizing": {"method": "volatility_target", "value": 0.15, "max_position": 1.0},
            "risk": {"max_position": 1.0, "stop_loss": 0.07, "max_drawdown": 0.3},
        }),
    ]
    return "\n".join(lines)


def admit(raw: object, symbol: str, data: pd.DataFrame) -> Proposal:
    """Validate one proposed element; never raises."""
    if not isinstance(raw, dict) or not isinstance(raw.get("strategy"), dict):
        return Proposal(None, "", False, "element is not {rationale, strategy}")
    rationale = str(raw.get("rationale", ""))[:500]
    spec_json = dict(raw["strategy"])
    spec_json["universe"] = [symbol]
    spec_json["timeframe"] = "1d"
    spec_json.setdefault("metadata", {})
    if isinstance(spec_json["metadata"], dict):
        spec_json["metadata"] = {**{str(k): str(v) for k, v in spec_json["metadata"].items()},
                                 "origin": "claude"}
    try:
        strategy = StrategySpec(**spec_json)
    except (ValidationError, TypeError, ValueError) as exc:
        return Proposal(None, rationale, False, f"invalid DSL: {str(exc).splitlines()[0][:200]}")
    if strategy.side != Side.LONG:
        return Proposal(strategy, rationale, False, "only long strategies are executable")
    unknown = {i.kind for i in strategy.indicators} - set(SUPPORTED_INDICATORS)
    if unknown:
        return Proposal(strategy, rationale, False, f"unsupported indicators: {sorted(unknown)}")
    try:
        position = strategy_position(data, strategy)
    except (ValueError, KeyError) as exc:
        return Proposal(strategy, rationale, False, f"does not compile: {exc}")
    if not bool(position.any()):
        return Proposal(strategy, rationale, False, "never enters a position on this data")
    return Proposal(strategy, rationale, True)


class ClaudeProposer:
    """Ask Claude for strategy hypotheses and admit only valid, executable ones."""

    def __init__(self, api_key: str | None = None, *, model: str | None = None,
                 transport: Transport | None = None) -> None:
        self.model = model or os.getenv("ATSF_PROPOSER_MODEL", DEFAULT_MODEL)
        if transport is None:
            key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not configured")
            transport = _http_transport(key)
        self._send = transport

    def propose(self, symbol: str, data: pd.DataFrame, n: int = 5,
                history: list[dict] | None = None) -> ProposalBatch:
        if not 1 <= n <= MAX_PROPOSALS:
            raise ValueError(f"n must be between 1 and {MAX_PROPOSALS}")
        response = self._send({
            "model": self.model,
            "max_tokens": 8000,
            "messages": [{"role": "user", "content": build_prompt(symbol, data, n, history)}],
        })
        text = "".join(block.get("text", "") for block in response.get("content", [])
                       if block.get("type") == "text")
        try:
            elements = _extract_json(text)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"proposer returned unparseable output: {exc}") from exc
        proposals: list[Proposal] = []
        seen: set[str] = set()
        for raw in elements[:n]:
            proposal = admit(raw, symbol, data)
            if proposal.accepted and proposal.strategy is not None:
                fingerprint = proposal.strategy.model_dump_json(exclude={"name", "metadata"})
                if fingerprint in seen:
                    proposal = Proposal(proposal.strategy, proposal.rationale, False, "duplicate proposal")
                seen.add(fingerprint)
            proposals.append(proposal)
        return ProposalBatch(self.model, tuple(proposals), response.get("usage", {}))
