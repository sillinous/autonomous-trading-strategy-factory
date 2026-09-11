from __future__ import annotations

import os
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .data import validate_market_data
from .research import run_research
from .portfolio_executor import execute_persisted_portfolio
from .portfolio_replay import verify_persisted_portfolio_run
from .reproducibility import build_reproducibility_certificate
from .registry import ExperimentRegistry
from .strategy import StrategySpec

Store = Annotated[ExperimentRegistry, Depends()]
Protected = Annotated[None, Depends(lambda x_api_key=Header(default=None): _require_api_key(x_api_key))]


def _require_api_key(value: str | None) -> None:
    expected = os.getenv("ATSF_API_KEY")
    if expected and value != expected:
        raise HTTPException(status_code=401, detail="invalid API key")


class MarketBar(BaseModel):
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class ReplayVerificationRequest(BaseModel):
    dataset_version: str
    data: dict[str, list[MarketBar]]


class ResearchRunRequest(BaseModel):
    seeds: list[StrategySpec] = Field(min_length=1)
    data: list[MarketBar] = Field(min_length=1)
    generations: int = Field(default=1, gt=0)
    population_size: int = Field(default=10, gt=0)
    survivor_count: int = Field(default=3, gt=0)
    seed: int = 0
    dataset_id: str = "research"


def create_app(store: ExperimentRegistry | None = None) -> FastAPI:
    app = FastAPI(title="Autonomous Trading Strategy Factory", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities(_auth: Protected) -> dict[str, bool]:
        return {"live_execution_enabled": False, "paper_execution_enabled": True}

    def get_store() -> ExperimentRegistry:
        if store is None:
            raise HTTPException(status_code=500, detail="registry is not configured")
        return store

    @app.get("/portfolios/{portfolio_id}")
    def portfolio(portfolio_id: str, _auth: Protected, registry: ExperimentRegistry = Depends(get_store)) -> dict:
        result = registry.get_portfolio(portfolio_id)
        if result is None:
            raise HTTPException(status_code=404, detail="portfolio not found")
        return result

    @app.get("/runs/{run_id}")
    def run(run_id: str, _auth: Protected, registry: ExperimentRegistry = Depends(get_store)) -> dict:
        result = registry.get_portfolio_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="paper run not found")
        return result

    @app.get("/runs/{run_id}/verify")
    def verify_run(run_id: str, _auth: Protected, registry: ExperimentRegistry = Depends(get_store)) -> dict:
        try:
            verification = verify_persisted_portfolio_run(registry, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "run_id": verification.run_id,
            "valid": verification.valid,
            "reason": verification.reason,
            "event_count": verification.manifest.event_count,
            "ledger_fingerprint": verification.manifest.ledger_fingerprint,
        }

    @app.post("/runs/{run_id}/verify-replay")
    def verify_replay(run_id: str, request: ReplayVerificationRequest, _auth: Protected, registry: ExperimentRegistry = Depends(get_store)) -> dict:
        try:
            frames: dict[str, pd.DataFrame] = {}
            for strategy_id, bars in request.data.items():
                frame = pd.DataFrame([bar.model_dump() for bar in bars]).set_index("timestamp")
                frames[strategy_id] = validate_market_data(frame)
            run = registry.get_portfolio_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="paper run not found")
            if run.get("provenance", {}).get("dataset_version") != request.dataset_version:
                raise HTTPException(status_code=400, detail="dataset_version does not match persisted run")
            verification = verify_persisted_portfolio_run(registry, run_id, data=frames)
        except HTTPException:
            raise
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "run_id": verification.run_id,
            "valid": verification.valid,
            "reason": verification.reason,
            "event_count": verification.manifest.event_count,
            "ledger_fingerprint": verification.manifest.ledger_fingerprint,
        }

    @app.post("/runs/{run_id}/certificate")
    def certificate(run_id: str, request: ReplayVerificationRequest, _auth: Protected, registry: ExperimentRegistry = Depends(get_store)) -> dict:
        try:
            frames: dict[str, pd.DataFrame] = {}
            for strategy_id, bars in request.data.items():
                frame = pd.DataFrame([bar.model_dump() for bar in bars]).set_index("timestamp")
                frames[strategy_id] = validate_market_data(frame)
            run = registry.get_portfolio_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="paper run not found")
            if run.get("provenance", {}).get("dataset_version") != request.dataset_version:
                raise HTTPException(status_code=400, detail="dataset_version does not match persisted run")
            verification = verify_persisted_portfolio_run(registry, run_id, data=frames)
            if not verification.valid:
                raise HTTPException(status_code=409, detail=verification.reason or "replay verification failed")
            certificate_result = build_reproducibility_certificate(registry, run, verification)
        except HTTPException:
            raise
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return certificate_result.__dict__

    @app.post("/research/runs", status_code=201)
    def research_run(request: ResearchRunRequest, _auth: Protected, registry: ExperimentRegistry = Depends(get_store)) -> dict:
        try:
            frame = validate_market_data(pd.DataFrame([bar.model_dump() for bar in request.data]).set_index("timestamp"))
            result = run_research(
                request.seeds,
                frame,
                generations=request.generations,
                population_size=request.population_size,
                survivor_count=request.survivor_count,
                seed=request.seed,
                dataset_id=request.dataset_id,
                registry=registry,
            )
            return {
                "dataset_id": result.dataset_id,
                "dataset_version": result.dataset_version,
                "portfolio_id": result.portfolio_id,
                "generations": len(result.generations),
                "final_population": [candidate.strategy_id for candidate in result.final_population],
            }
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/portfolios/{portfolio_id}/paper-runs", status_code=201)
    def paper_run(portfolio_id: str, request: ReplayVerificationRequest, _auth: Protected, registry: ExperimentRegistry = Depends(get_store)) -> dict:
        try:
            frames: dict[str, pd.DataFrame] = {}
            for strategy_id, bars in request.data.items():
                frames[strategy_id] = validate_market_data(pd.DataFrame([bar.model_dump() for bar in bars]).set_index("timestamp"))
            result = execute_persisted_portfolio(registry, portfolio_id, frames, dataset_version=request.dataset_version)
            return {"run_id": result.identity.run_id, "final_equity": result.final_equity, "halted": result.halted}
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


app = create_app()
