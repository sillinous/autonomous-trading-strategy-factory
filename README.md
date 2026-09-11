# Autonomous Trading Strategy Factory

An experimental platform for autonomous quantitative strategy research, historical backtesting, robustness validation, and eventual paper/live execution.

## Design principles

- Strategies are represented as typed, serializable specifications rather than arbitrary generated code.
- Backtesting is deterministic and models trading frictions explicitly.
- Out-of-sample validation, walk-forward testing, perturbation, and stress testing are first-class.
- Every experiment is reproducible and retains lineage.
- AI proposes hypotheses; deterministic engines evaluate them.
- Live execution is disabled until a strategy passes explicit promotion gates.
- Persisted portfolios and paper runs are immutable and auditable.
- Market-data access is provider-neutral; vendor adapters must normalize data through the canonical schema.

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

## Service

The HTTP service is created by `atsf.api:create_app` and exposes health, capability, persisted portfolio, paper-run, and run-audit endpoints. It accepts market data only for execution against an already persisted portfolio; callers cannot submit arbitrary executable strategy code.

For a local service using the default registry path:

```bash
uvicorn atsf.api:app --host 127.0.0.1 --port 8000
```

Set `ATSF_REGISTRY_PATH` to point the service at a persistent SQLite registry. The `/capabilities` endpoint explicitly reports `live_execution_enabled: false`; no live broker or live-order endpoint exists.

### API authentication

Operational endpoints support an optional shared API key through `ATSF_API_KEY`. When configured, clients must send `X-API-Key: <key>`; `/health` remains unauthenticated for container healthchecks.

For any non-local deployment, configure authentication and place the service behind a properly secured reverse proxy or private network. Do not expose the unauthenticated default directly to the public internet.

## Docker

Build and run the paper-only service with:

```bash
docker compose up --build -d
```

The SQLite registry is stored in the named `atsf-data` volume. The container runs as a non-root user and includes an HTTP healthcheck. Docker Compose binds the API to the host loopback interface by default.

## Market-data providers

`atsf.provider.MarketDataProvider` defines the vendor-neutral contract. `FrameMarketDataProvider` provides deterministic local/in-memory data for tests and controlled execution.

`atsf.alphavantage.AlphaVantageDailyProvider` is the first external adapter. It uses Alpha Vantage's `TIME_SERIES_DAILY` historical endpoint, normalizes the response to canonical OHLCV, validates it, supports bounded date filtering, and exposes source/timeframe/schema metadata for reproducible dataset registration. Full historical output depends on the vendor plan. See the official [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/) for current endpoint and plan details.

`atsf.cached_provider.CachedMarketDataProvider` decorates any provider with a deterministic filesystem cache. Cache identity includes symbol, requested range, source, timeframe, and schema version; cached frames are revalidated when read. This makes repeated research runs reuse the exact provider response without weakening dataset provenance.

External data must enter the system through the provider/ingestion boundary and be fingerprinted into a dataset bundle before research or paper execution. This prevents silent changes in source, timeframe, schema, or underlying bars from masquerading as the same dataset.

## Status

Research-first foundation with deterministic paper execution, a real external historical-data adapter, and a reproducible provider-response cache. This repository is intentionally not a live-trading system and does not constitute financial advice or a guarantee of profitability.
