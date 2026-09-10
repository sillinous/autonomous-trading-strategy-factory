from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from .dataset_bundle import bundle_identity
from .paper import PaperBroker, PaperConfig
from .portfolio_attribution import PortfolioAttribution
from .portfolio_audit import PortfolioAuditEvent
from .portfolio_paper import PortfolioPaperResult, run_paper_portfolio
from .portfolio_run import PortfolioRunIdentity, attribute_run, build_portfolio_run_identity
from .promotion import PromotionDecision
from .registry import ExperimentRegistry
from .signals import strategy_signals


@dataclass(frozen=True)
class PersistedPortfolioExecution:
    identity: PortfolioRunIdentity
    paper: PortfolioPaperResult
    attribution: PortfolioAttribution
    audit_events: tuple[PortfolioAuditEvent, ...]


def _sleeve_returns(frame: pd.DataFrame, strategy, *, initial_cash: float, commission_bps: float, slippage_bps: float) -> pd.Series:
    """Replay one sleeve with the same deterministic paper broker for attribution."""
    close = pd.to_numeric(frame["close"], errors="coerce")
    if frame.empty or close.isna().any() or (~close.map(isfinite)).any() or (close <= 0).any():
        raise ValueError("attribution data must contain finite positive close prices")
    broker = PaperBroker(PaperConfig(initial_cash=initial_cash, commission_bps=commission_bps, slippage_bps=slippage_bps))
    entry, exit_ = strategy_signals(frame, strategy)
    equity: list[float] = []
    for timestamp in frame.index:
        price = float(close.loc[timestamp])
        if bool(entry.loc[timestamp]) and broker.position == 0:
            quantity = broker.cash * strategy.position_sizing.max_position / price
            if quantity > 0:
                broker.execute(timestamp, "buy", quantity, price)
        elif bool(exit_.loc[timestamp]) and broker.position > 0:
            broker.execute(timestamp, "sell", broker.position, price)
        equity.append(broker.mark(timestamp, price).equity)
    if broker.position > 0:
        timestamp = frame.index[-1]
        broker.execute(timestamp, "sell", broker.position, float(close.iloc[-1]))
        equity[-1] = broker.mark(timestamp, float(close.iloc[-1])).equity
    return pd.Series(equity, index=frame.index, dtype=float).pct_change().fillna(0.0)


def execute_persisted_portfolio(
    store: ExperimentRegistry,
    portfolio_id: str,
    data: dict[str, pd.DataFrame],
    *,
    dataset_version: str,
    initial_cash: float = 100_000.0,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    max_drawdown: float | None = None,
) -> PersistedPortfolioExecution:
    """Execute exactly the persisted portfolio definition in paper mode."""
    portfolio = store.get_portfolio(portfolio_id)
    if portfolio is None:
        raise ValueError(f"unknown portfolio: {portfolio_id}")
    definition = portfolio["definition"]
    persisted_dataset_id = definition.get("dataset_id")
    if not isinstance(persisted_dataset_id, str) or not persisted_dataset_id.strip():
        raise ValueError("persisted portfolio is missing a valid dataset_id")
    if definition.get("dataset_version") != dataset_version:
        raise ValueError("dataset_version does not match the persisted portfolio")

    bundle = bundle_identity(data, persisted_dataset_id)
    if bundle.version != dataset_version:
        raise ValueError("market data content does not match the persisted dataset version")

    weights = dict(portfolio["members"])
    if not weights or any(not isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("persisted portfolio contains invalid weights")
    if sum(weights.values()) > 1.0 + 1e-12:
        raise ValueError("persisted portfolio weights exceed 100% gross exposure")
    if set(data) != set(weights):
        raise ValueError("market data must contain exactly the persisted portfolio members")
    if not isfinite(initial_cash) or initial_cash <= 0 or not isfinite(commission_bps) or commission_bps < 0 or not isfinite(slippage_bps) or slippage_bps < 0:
        raise ValueError("execution parameters must be finite and valid")
    if max_drawdown is not None and (not isfinite(max_drawdown) or not 0 < max_drawdown < 1):
        raise ValueError("max_drawdown must be between zero and one")

    experiment_ids = definition.get("experiment_ids", {})
    if set(experiment_ids) != set(weights):
        raise ValueError("persisted portfolio experiment IDs do not match its members")

    strategies = {}
    decisions = {}
    for strategy_id in weights:
        strategy = store.get_strategy(strategy_id)
        if strategy is None:
            raise ValueError(f"persisted portfolio references unknown strategy: {strategy_id}")
        evidence = store.get_evaluation_evidence(experiment_ids[strategy_id])
        if evidence is None or not isinstance(evidence.get("promotion"), dict):
            raise ValueError(f"missing promotion evidence for strategy: {strategy_id}")
        promotion = evidence["promotion"]
        decisions[strategy_id] = PromotionDecision(
            stage=str(promotion["stage"]), eligible=bool(promotion["eligible"]), reasons=tuple(promotion.get("reasons", ())),
        )
        strategies[strategy_id] = strategy

    execution_config = {
        "initial_cash": float(initial_cash), "commission_bps": float(commission_bps),
        "slippage_bps": float(slippage_bps), "max_drawdown": max_drawdown,
    }
    identity = build_portfolio_run_identity(
        portfolio_id, dataset_version, weights,
        execution_config=execution_config, data_fingerprint=bundle.version,
    )
    if store.get_portfolio_run(identity.run_id) is not None:
        raise ValueError("portfolio run already exists; execution is immutable")

    paper = run_paper_portfolio(
        data, strategies, weights, decisions=decisions,
        initial_cash=initial_cash, commission_bps=commission_bps,
        slippage_bps=slippage_bps, max_drawdown=max_drawdown,
    )

    sleeve_returns = {
        strategy_id: _sleeve_returns(
            data[strategy_id], strategies[strategy_id],
            initial_cash=initial_cash * weights[strategy_id],
            commission_bps=commission_bps, slippage_bps=slippage_bps,
        )
        for strategy_id in weights if weights[strategy_id] > 0
    }
    if not sleeve_returns:
        raise ValueError("persisted portfolio has no positive-weight strategies")
    returns = pd.concat(sleeve_returns, axis=1, join="inner").sort_index()
    returns.columns = list(sleeve_returns)
    attribution = attribute_run(returns, {key: weights[key] for key in sleeve_returns})

    audit_events = tuple(
        PortfolioAuditEvent(sequence=sequence, strategy_id=strategy_id, action=fill.side,
                            timestamp=fill.timestamp.isoformat(), quantity=float(fill.quantity),
                            price=float(fill.price), fee=float(fill.fee))
        for sequence, (strategy_id, fill) in enumerate(paper.fills)
    )
    store.save_portfolio_run(
        identity.run_id, portfolio_id, paper.final_equity, paper.halted, paper.halt_reason,
        [{"strategy_id": item.strategy_id, "return_contribution": item.return_contribution,
          "risk_contribution": item.risk_contribution} for item in attribution.contributions],
    )
    store.save_portfolio_audit_events(identity.run_id, list(audit_events))
    return PersistedPortfolioExecution(identity, paper, attribution, audit_events)
