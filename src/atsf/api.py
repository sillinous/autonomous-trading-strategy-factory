from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .certificate_integrity import verify_persisted_certificate
from .data import dataset_identity, validate_market_data
from .feedback_registry import FeedbackEventStore
from .portfolio_executor import execute_persisted_portfolio
from .portfolio_replay import verify_persisted_portfolio_run
from .provenance_graph import build_research_provenance_graph
from .registry import ExperimentRegistry
from .research import run_research
from .reproducibility import build_reproducibility_certificate
from .strategy import StrategySpec

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
class ReplayVerificationRequest(BaseModel):
    dataset_version: str = Field(min_length=1)
    data: dict[str, list[MarketBar]] = Field(min_length=1)
class ResearchRunRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    data: list[MarketBar] = Field(min_length=1)
    seeds: list[StrategySpec] = Field(min_length=1)
    generations: int = Field(default=1, ge=1)
    population_size: int = Field(default=10, ge=1)
    survivor_count: int = Field(default=3, ge=1)
    seed: int = 0
class ServiceConfig(BaseModel):
    live_execution_enabled: bool = False

def _registry_from_environment() -> ExperimentRegistry:
    return ExperimentRegistry(os.getenv("ATSF_REGISTRY_PATH", "atsf.sqlite3"))

def create_app(registry: ExperimentRegistry | None = None) -> FastAPI:
    owned_registry = registry is None
    store = registry or _registry_from_environment()
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if owned_registry: store.close()
    app = FastAPI(title="Autonomous Trading Strategy Factory", version="0.1.0", lifespan=lifespan)
    def get_store() -> ExperimentRegistry: return store
    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        configured_key = os.getenv("ATSF_API_KEY")
        if configured_key and (x_api_key is None or not secrets.compare_digest(x_api_key, configured_key)):
            raise HTTPException(status_code=401, detail="invalid or missing API key")
    Store = Annotated[ExperimentRegistry, Depends(get_store)]
    Protected = Annotated[object, Depends(require_api_key)]
    @app.get("/health")
    def health() -> dict[str, str]: return {"status": "ok"}
    @app.get("/capabilities")
    def capabilities(_auth: Protected = None) -> ServiceConfig: return ServiceConfig(live_execution_enabled=False)
    @app.get("/portfolios/{portfolio_id}")
    def portfolio(portfolio_id: str, store: Store, _auth: Protected = None) -> dict:
        result = store.get_portfolio(portfolio_id)
        if result is None: raise HTTPException(status_code=404, detail="portfolio not found")
        return result
    @app.get("/runs/{run_id}")
    def run(run_id: str, store: Store, _auth: Protected = None) -> dict:
        result = store.get_portfolio_run(run_id)
        if result is None: raise HTTPException(status_code=404, detail="paper run not found")
        return result
    @app.get("/strategies/{strategy_id}/feedback")
    def strategy_feedback(strategy_id: str, store: Store, _auth: Protected = None) -> dict:
        if not strategy_id: raise HTTPException(status_code=400, detail="strategy_id cannot be empty")
        events = FeedbackEventStore(store).list_for_strategy(strategy_id)
        return {"strategy_id": strategy_id, "event_count": len(events), "events": events}
    @app.get("/runs/{run_id}/provenance-graph")
    def provenance_graph(run_id: str, store: Store, _auth: Protected = None) -> dict:
        run = store.get_portfolio_run(run_id)
        if run is None: raise HTTPException(status_code=404, detail="paper run not found")
        try: graph = build_research_provenance_graph(store, run)
        except (TypeError, ValueError) as exc: raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"run_id": run_id, "schema_version": graph.schema_version, "fingerprint": graph.fingerprint,
                "nodes": [{"node_id": n.node_id, "kind": n.kind, "key": n.key, "attributes": n.attributes} for n in graph.nodes],
                "edges": [{"source": e.source, "target": e.target, "relation": e.relation} for e in graph.edges]}
    @app.get("/runs/{run_id}/certificate")
    def get_certificate(run_id: str, store: Store, _auth: Protected = None) -> dict:
        certificate = store.get_reproducibility_certificate(run_id)
        if certificate is None: raise HTTPException(status_code=404, detail="reproducibility certificate not found")
        return certificate
    @app.get("/runs/{run_id}/certificate/verify")
    def verify_certificate(run_id: str, store: Store, _auth: Protected = None) -> dict:
        result = verify_persisted_certificate(store, run_id)
        if not result.valid and result.reason == "reproducibility certificate not found": raise HTTPException(status_code=404, detail=result.reason)
        return {"run_id": result.run_id, "certificate_id": result.certificate_id, "valid": result.valid, "reason": result.reason}
    @app.get("/runs/{run_id}/verify")
    def verify_run(run_id: str, store: Store, _auth: Protected = None) -> dict:
        try: verification = verify_persisted_portfolio_run(store, run_id)
        except ValueError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"run_id": verification.run_id, "valid": verification.valid, "reason": verification.reason, "event_count": verification.manifest.event_count, "ledger_fingerprint": verification.manifest.ledger_fingerprint}
    @app.post("/runs/{run_id}/verify-replay")
    def verify_replay(run_id: str, request: ReplayVerificationRequest, store: Store, _auth: Protected = None) -> dict:
        try:
            frames = {sid: validate_market_data(pd.DataFrame([bar.model_dump() for bar in bars]).set_index("timestamp")) for sid, bars in request.data.items()}
            run = store.get_portfolio_run(run_id)
            if run is None: raise HTTPException(status_code=404, detail="paper run not found")
            if run.get("provenance", {}).get("dataset_version") != request.dataset_version: raise HTTPException(status_code=400, detail="dataset_version does not match persisted run")
            verification = verify_persisted_portfolio_run(store, run_id, data=frames)
        except HTTPException: raise
        except (TypeError, ValueError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": verification.run_id, "valid": verification.valid, "reason": verification.reason, "event_count": verification.manifest.event_count, "ledger_fingerprint": verification.manifest.ledger_fingerprint}
    @app.post("/runs/{run_id}/certificate")
    def certificate(run_id: str, request: ReplayVerificationRequest, store: Store, _auth: Protected = None) -> dict:
        try:
            frames = {sid: validate_market_data(pd.DataFrame([bar.model_dump() for bar in bars]).set_index("timestamp")) for sid, bars in request.data.items()}
            run = store.get_portfolio_run(run_id)
            if run is None: raise HTTPException(status_code=404, detail="paper run not found")
            if run.get("provenance", {}).get("dataset_version") != request.dataset_version: raise HTTPException(status_code=400, detail="dataset_version does not match persisted run")
            verification = verify_persisted_portfolio_run(store, run_id, data=frames)
            if not verification.valid: raise HTTPException(status_code=409, detail=verification.reason or "replay verification failed")
            certificate_result = build_reproducibility_certificate(store, run, verification)
            store.save_reproducibility_certificate(certificate_result)
        except HTTPException: raise
        except (TypeError, ValueError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
        return certificate_result.__dict__
    @app.post("/research/runs", status_code=201)
    def research_run(request: ResearchRunRequest, store: Store, _auth: Protected = None) -> dict:
        try:
            frame = validate_market_data(pd.DataFrame([bar.model_dump() for bar in request.data]).set_index("timestamp"))
            identity = dataset_identity(frame, request.dataset_id)
            result = run_research(request.seeds, frame, generations=request.generations, population_size=request.population_size, survivor_count=request.survivor_count, seed=request.seed, dataset_id=request.dataset_id, registry=store)
        except (TypeError, ValueError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"dataset_id": identity.dataset_id, "dataset_version": identity.version, "generations": len(result.generations), "final_population_size": len(result.final_population), "portfolio_id": result.portfolio_id}
    @app.post("/portfolios/{portfolio_id}/paper-runs", status_code=201)
    def paper_run(portfolio_id: str, request: PaperRunRequest, store: Store, _auth: Protected = None) -> dict:
        try:
            frames = {sid: validate_market_data(pd.DataFrame([bar.model_dump() for bar in bars]).set_index("timestamp")) for sid, bars in request.data.items()}
            if store.get_portfolio(portfolio_id) is None: raise HTTPException(status_code=404, detail="portfolio not found")
            result = execute_persisted_portfolio(store, portfolio_id, frames, dataset_version=request.dataset_version, initial_cash=request.initial_cash, commission_bps=request.commission_bps, slippage_bps=request.slippage_bps, max_drawdown=request.max_drawdown)
        except HTTPException: raise
        except PermissionError as exc: raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": result.identity.run_id, "portfolio_id": result.identity.portfolio_id, "dataset_version": result.identity.dataset_version, "execution_fingerprint": result.identity.execution_fingerprint, "final_equity": result.paper.final_equity, "halted": result.paper.halted, "halt_reason": result.paper.halt_reason, "fill_count": len(result.paper.fills), "audit_event_count": len(result.audit_events)}
    return app

app = create_app()
