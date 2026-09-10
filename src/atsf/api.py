from __future__ import annotations

import os
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import validate_market_data
from .portfolio_executor import execute_persisted_portfolio
from .registry import ExperimentRegistry


class MarketBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class PaperRunRequest(BaseModel):
    dataset_version: str = Field(min_length=1)
    data: dict[str, list[MarketBar]] = Field(min_length=1)
    initial_cash: float = Field(default=100_000.0, gt=0)
    commission_bps: float = Field(default=1.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)
    max_drawdown: float | None = Field(default=None, gt=0, lt=1)


class ServiceConfig(BaseModel):
    live_execution_enabled: bool = False


def _registry_from_environment() -> ExperimentRegistry:
    return ExperimentRegistry(os.getenv("ATSF_REGISTRY_PATH", "atsf.sqlite3"))


def create_app(registry: ExperimentRegistry | None = None) -> FastAPI:
    """Create the HTTP service boundary; live execution is deliberately unavailable."""
    app = FastAPI(title="Autonomous Trading Strategy Factory", version="0.1.0")
    owned_registry = registry is None
    store = registry or _registry_from_environment()

    def get_store() -> ExperimentRegistry:
        return store

    Store = Annotated[ExperimentRegistry, Depends(get_store)]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities() -> ServiceConfig:
        return ServiceConfig(live_execution_enabled=False)

    @app.get("/portfolios/{portfolio_id}")
    def portfolio(portfolio_id: str, store: Store) -> dict:
        result = store.get_portfolio(portfolio_id)
        if result is None:
            raise HTTPException(status_code=404, detail="portfolio not found")
        return result

    @app.get("/runs/{run_id}")
    def run(run_id: str, store: Store) -> dict:
        result = store.get_portfolio_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="paper run not found")
        return result

    @app.post("/portfolios/{portfolio_id}/paper-runs", status_code=201)
    def paper_run(portfolio_id: str, request: PaperRunRequest, store: Store) -> dict:
        frames: dict[str, pd.DataFrame] = {}
        try:
            for strategy_id, bars in request.data.items():
                frame = pd.DataFrame([bar.model_dump() for bar in bars]).set_index("timestamp")
                frames[strategy_id] = validate_market_data(frame)
            result = execute_persisted_portfolio(
                store,
                portfolio_id,
                frames,
                dataset_version=request.dataset_version,
                initial_cash=request.initial_cash,
                commission_bps=request.commission_bps,
                slippage_bps=request.slippage_bps,
                max_drawdown=request.max_drawdown,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "run_id": result.identity.run_id,
            "portfolio_id": result.identity.portfolio_id,
            "dataset_version": result.identity.dataset_version,
            "execution_fingerprint": result.identity.execution_fingerprint,
            "final_equity": result.paper.final_equity,
            "halted": result.paper.halted,
            "halt_reason": result.paper.halt_reason,
            "fill_count": len(result.paper.fills),
            "audit_event_count": len(result.audit_events),
        }

    @app.on_event("shutdown")
    def shutdown() -> None:
        if owned_registry:
            store.close()

    return app


app = create_app()
