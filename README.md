# Autonomous Trading Strategy Factory

An experimental platform for autonomous quantitative strategy research, historical backtesting, robustness validation, and eventual paper/live execution.

## Quick start

```bash
pip install -e ".[dev]"
atsf archetypes                                   # list built-in strategy archetypes
atsf data SPY --start 2005-01-01 --out spy.csv    # split/dividend-adjusted daily bars (Yahoo)
atsf backtest SPY --csv spy.csv --archetype roc_momentum
atsf research SPY --csv spy.csv --json report.json
```

`atsf research` generates the candidate population (moving-average variants plus the
breakout, momentum, MACD, RSI and Bollinger archetypes), evaluates each one through
walk-forward OOS testing, Monte Carlo, parameter perturbation, regime and execution
robustness, then applies population-level overfitting controls: every promotion
decision is re-made with the deflated Sharpe ratio for the true number of trials, and
the population is rejected outright if its probability of backtest overfitting
(PBO, via CSCV) exceeds 0.5. Promotion only ever reaches the paper stage.

Execution is identical everywhere: backtests, paper runs, portfolio sleeves and
persisted-run replay all step one engine (`atsf.execution`) with next-bar fills,
adverse slippage, volatility-targeted sizing, intrabar stops and a drawdown kill switch.

## Design principles

- Strategies are represented as typed, serializable specifications rather than arbitrary generated code.
- Backtesting is deterministic and models trading frictions explicitly.
- Out-of-sample validation, walk-forward testing, perturbation, and stress testing are first-class.
- Every experiment is reproducible and retains lineage.
- AI proposes hypotheses; deterministic engines evaluate them.
- Live execution is disabled until a strategy passes explicit promotion gates.
- Persisted portfolios and paper runs are immutable and auditable.
- Market-data access is provider-neutral; vendor adapters must normalize data through the canonical schema.
- Provider-declared source, timeframe, and schema metadata are authoritative and cannot be overridden by ingestion callers.
- Strategy health feedback and replacement-research requests are persisted as immutable provenance artifacts.

## Initial scope

1. Strategy DSL and validation
2. Deterministic backtesting kernel
3. Metrics and validation framework
4. Strategy generation/evolution
5. Experiment registry
6. Deterministic strategy compiler
7. Persisted portfolio construction and attribution
8. Immutable paper-trading execution and audit ledger
9. FastAPI service boundary for research artifacts and paper execution
10. Source-neutral market-data provider contract
11. External historical-data adapter with dataset provenance
12. Deterministic provider-response cache
13. Persisted strategy-health feedback and replacement-research queue
14. End-to-end research-to-paper provenance graph and reproducibility certificates

## Service

The HTTP service is created by `atsf.api:create_app` and exposes health, capability, persisted portfolio, paper-run, run-audit, provenance-graph, certificate, and strategy-feedback endpoints. It accepts market data only for execution against an already persisted portfolio; callers cannot submit arbitrary executable strategy code.

For a local service using the default registry path:

```bash
uvicorn atsf.api:app --host 127.0.0.1 --port 8000
```

Set `ATSF_REGISTRY_PATH` to point the service at a persistent SQLite registry. The `/capabilities` endpoint explicitly reports `live_execution_enabled: false`; no live broker or live-order endpoint exists.

### API authentication

Operational endpoints support an optional shared API key through `ATSF_API_KEY`. When configured, clients must send `X-API-Key: <key>`; `/health` remains unauthenticated for container healthchecks.

For any non-local deployment, configure authentication and place the service behind a properly secured reverse proxy or private network. Do not expose the unauthenticated default directly to the public internet.

## User testing

The repository includes a controlled user-testing runbook covering Docker startup, optional API-key protection, health/readiness checks, persistence verification, paper-trading workflows, replay/provenance checks, and safety boundaries: `docs/USER_TESTING.md`.

## Docker

Build and run the paper-only service with:

```bash
docker compose up --build -d
```

The SQLite registry is stored in the named `atsf-data` volume. The container runs as a non-root user and includes an HTTP healthcheck. Docker Compose binds the API to the host loopback interface by default.

## External data gateway

User-testing builds expose the authenticated external-data surface:

- `GET /data/providers` — configured provider catalog and capabilities.
- `GET /data/market/{symbol}` — normalized real OHLCV data.
- `GET /data/macro/{series_id}` — FRED observations.
- `GET /data/fundamentals/{cik}` — SEC company facts.
- `GET /data/news` — Alpha Vantage news/sentiment when configured.
- `GET /data/snapshot` — one immutable, fingerprinted market + macro + fundamentals + news snapshot.
- `POST /research/runs/external` — runs deterministic research against externally fetched market data while retaining the external snapshot fingerprint.

The gateway never substitutes synthetic production data. Provider credentials are injected through environment variables, never stored in source control. Copy `.env.example` to your local environment and supply `ALPHAVANTAGE_API_KEY` and `ATSF_SEC_USER_AGENT` when those providers are needed. Stooq market history and FRED's public CSV endpoint require no credentials in the current gateway. Every external snapshot is fingerprinted so research can be tied to the exact input received.

## Market-data providers

`atsf.provider.MarketDataProvider` defines the vendor-neutral contract. Each provider exposes immutable `ProviderMetadata(source, timeframe, schema_version)`, which is the authoritative provenance for data it emits. `FrameMarketDataProvider` provides deterministic local/in-memory data for tests and controlled execution.

`atsf.alphavantage.AlphaVantageDailyProvider` is the first external adapter. It uses Alpha Vantage's `TIME_SERIES_DAILY` historical endpoint, normalizes the response to canonical OHLCV, validates it, supports bounded date filtering, and exposes source/timeframe/schema metadata for reproducible dataset registration. Full historical output depends on the vendor plan. See the official [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/) for current endpoint and plan details.

`atsf.cached_provider.CachedMarketDataProvider` decorates any provider with a deterministic filesystem cache and inherits the wrapped provider's metadata. Each cache entry has a manifest containing the cache-key digest and a cryptographic fingerprint of the normalized frame; missing or tampered manifests/frames are treated as cache misses. Ingestion accepts optional provenance values only as assertions and rejects mismatches, so callers cannot silently relabel a dataset.

External data must enter the system through the provider/ingestion boundary and be fingerprinted into a dataset bundle before research or paper execution. This prevents silent changes in source, timeframe, schema, or underlying bars from masquerading as the same dataset.

## Feedback and replacement research

Paper-strategy health can deterministically transition a strategy to a degraded lifecycle state and create a replacement-research request. Requests can be persisted immutably alongside feedback events, so the reason for replacement work survives process restarts. Feedback events are included in the research provenance graph and their fingerprints are verified before graph materialization.

## Status

Research-first foundation with deterministic paper execution, a real external historical-data adapter, reproducible/tamper-evident provider-response caching, persisted health feedback, replacement-research requests, and end-to-end provenance attestation. This repository is intentionally not a live-trading system and does not constitute financial advice or a guarantee of profitability.
