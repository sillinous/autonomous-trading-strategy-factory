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

## Service

The HTTP service is created by `atsf.api:create_app` and exposes health, capability, persisted portfolio, paper-run, and run-audit endpoints. It accepts market data only for execution against an already persisted portfolio; callers cannot submit arbitrary executable strategy code.

For a local service using the default registry path:

```bash
uvicorn atsf.api:app --host 0.0.0.0 --port 8000
```

Set `ATSF_REGISTRY_PATH` to point the service at a persistent SQLite registry. The `/capabilities` endpoint explicitly reports `live_execution_enabled: false`; no live broker or live-order endpoint exists.

## Status

Research-first foundation with deterministic paper execution. This repository is intentionally not a live-trading system and does not constitute financial advice or a guarantee of profitability.
