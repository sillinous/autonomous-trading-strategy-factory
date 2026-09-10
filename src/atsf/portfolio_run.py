from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .portfolio_attribution import PortfolioAttribution, attribute_portfolio


@dataclass(frozen=True)
class PortfolioRunIdentity:
    """Deterministic identity for an immutable paper portfolio execution."""

    run_id: str
    portfolio_id: str
    dataset_version: str
    execution_fingerprint: str = ""


def portfolio_run_id(
    portfolio_id: str,
    dataset_version: str,
    strategy_ids: tuple[str, ...],
    weights: dict[str, float],
    *,
    execution_config: dict[str, Any] | None = None,
    data_fingerprint: str | None = None,
) -> str:
    if not portfolio_id or not dataset_version:
        raise ValueError("portfolio_id and dataset_version are required")
    if not strategy_ids or len(set(strategy_ids)) != len(strategy_ids):
        raise ValueError("strategy_ids must be non-empty and unique")
    if set(weights) != set(strategy_ids):
        raise ValueError("weights must match strategy_ids")
    if any(not isinstance(value, (int, float)) or not isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("weights must be finite and non-negative")
    config = execution_config or {}
    payload = {
        "portfolio_id": portfolio_id,
        "dataset_version": dataset_version,
        "strategy_ids": list(strategy_ids),
        "weights": {key: weights[key] for key in sorted(weights)},
        "execution_config": config,
        "data_fingerprint": data_fingerprint or "",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_portfolio_run_identity(
    portfolio_id: str,
    dataset_version: str,
    weights: dict[str, float],
    *,
    execution_config: dict[str, Any] | None = None,
    data_fingerprint: str | None = None,
) -> PortfolioRunIdentity:
    strategy_ids = tuple(sorted(weights))
    config = execution_config or {}
    fingerprint_payload = {"execution_config": config, "data_fingerprint": data_fingerprint or ""}
    execution_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()[:16]
    return PortfolioRunIdentity(
        run_id=portfolio_run_id(
            portfolio_id, dataset_version, strategy_ids, weights,
            execution_config=config, data_fingerprint=data_fingerprint,
        ),
        portfolio_id=portfolio_id,
        dataset_version=dataset_version,
        execution_fingerprint=execution_fingerprint,
    )


def attribute_run(returns, weights: dict[str, float]) -> PortfolioAttribution:
    """Compute deterministic strategy-level return and risk attribution."""
    return attribute_portfolio(returns, weights)
