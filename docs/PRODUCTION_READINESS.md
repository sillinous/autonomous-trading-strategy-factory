# Production Readiness Gates

## Purpose

This document defines evidence-based release gates for the Autonomous Trading Strategy Factory (ATSF). Passing software CI is necessary but does not establish trading profitability, financial suitability, or readiness for live brokerage execution.

## Verified baseline

Repository and workflow inspection on 2026-10-16 found the latest GitHub Actions test run completed successfully on `main` at commit `aeae94937e932544796e2af917f751db7d7e473e`.

The workflow passed all of its configured steps:

- Install the package and development dependencies on Python 3.11.
- Run `pytest -q --cov=atsf --cov-report=term-missing --cov-fail-under=80`.
- Run `ruff check src tests`.
- Build the production Docker image.
- Start the image and smoke-test `/health` and the operator UI root page, including the expected `ATSF Control Plane` marker.

This verifies the configured CI contract for that commit. It does not mean every production-readiness gate below has been demonstrated.

## Release gates

### Gate 1 — Build, tests, and reproducibility

- [ ] CI passes on the exact release commit.
- [ ] Dependency versions and image build inputs are controlled and reviewable.
- [ ] Research results can be reproduced from persisted strategy specification, configuration, data fingerprint, and engine version.
- [ ] A failed test or quality gate blocks release.

### Gate 2 — Market-data integrity

- [ ] Provider, timeframe, schema version, adjustment policy, retrieval time, and data fingerprint are persisted.
- [ ] Missing, duplicated, out-of-order, malformed, or stale observations are rejected or explicitly handled.
- [ ] Provider errors fail closed; no synthetic fallback is used for production research.
- [ ] Cache tampering and provenance mismatch tests pass.
- [ ] Provider licensing, rate limits, and commercial-use restrictions are documented.

### Gate 3 — Backtest integrity

- [ ] Signals use only information available at decision time.
- [ ] Order execution occurs no earlier than the configured executable timestamp.
- [ ] Fees, spread, slippage, liquidity, partial fills, and rejected orders are modeled or explicitly bounded.
- [ ] Corporate actions, splits, dividends, missing sessions, and survivorship effects are addressed for the selected universe.
- [ ] Tests cover look-ahead leakage, data leakage, and boundary conditions.

### Gate 4 — Statistical validation

- [ ] Walk-forward and untouched holdout procedures are specified before candidate selection.
- [ ] Repeated experimentation and multiple-testing/selection bias are measured and controlled.
- [ ] Results include uncertainty, sample size, turnover, drawdown, and net-of-cost performance.
- [ ] Robustness checks include parameter perturbation and distinct market regimes.
- [ ] Promotion thresholds are versioned, deterministic, and tested.
- [ ] A holdout dataset is not repeatedly reused to tune candidates.

### Gate 5 — Evolution and lineage

- [ ] Failed as well as successful candidates remain auditable.
- [ ] Parent-child lineage, mutation/crossover operator, parameters, and evaluation evidence are persisted.
- [ ] Duplicate candidates and near-duplicate strategies are detected.
- [ ] Invalid parent references and lineage cycles are rejected.
- [ ] Candidate selection preserves diversity and does not rank solely on a single in-sample metric.

### Gate 6 — Portfolio and paper execution

- [ ] Position sizing, leverage, concentration, exposure, and drawdown limits are enforced.
- [ ] Risk checks fail closed when inputs are missing or invalid.
- [ ] Paper runs have immutable audit events and deterministic replay evidence.
- [ ] Restart and recovery tests verify persisted state and avoid duplicate fills.
- [ ] Paper-versus-backtest divergence and strategy degradation are monitored.
- [ ] Kill-switch and halted-state behavior are tested.

### Gate 7 — API, security, and operations

- [ ] Operational endpoints require authentication when deployed beyond local development.
- [ ] Secrets are injected through deployment configuration and never committed or logged.
- [ ] Public liveness checks disclose no sensitive operational data.
- [ ] Deployment uses private networking or a properly secured reverse proxy.
- [ ] Persistent data has a tested backup and restore procedure.
- [ ] Logs, health/readiness checks, failure alerts, and retention policies are documented.
- [ ] API limits, request validation, and dependency vulnerabilities are reviewed.

### Gate 8 — Live execution authorization

**Live execution remains disabled by default.** Do not treat this checklist as authorization to connect a broker or trade real capital.

Before a separate live-execution project is considered, require:

- [ ] Documented paper-trading observation period and acceptance criteria.
- [ ] Independent review of strategy, portfolio, operational, and security risks.
- [ ] Explicit risk limits, exposure caps, order validation, idempotency, reconciliation, and emergency stop.
- [ ] Broker adapter tested in a sandbox and isolated from research generation.
- [ ] Human authorization outside the autonomous strategy-generation loop.
- [ ] A documented rollback and incident-response plan.

## Recommended next verification sequence

1. Capture the exact CI commit and test artifacts for each release candidate.
2. Run the documented user-testing workflow with real, fingerprinted external market data.
3. Verify research-run persistence, provenance/certificate validation, paper execution, replay, and restart recovery end to end.
4. Review test coverage around the highest-risk boundaries: data leakage, execution timing, cost modeling, promotion decisions, and duplicate execution.
5. Resolve any failed gate before advancing to the next stage.

## Status vocabulary

Use only these labels in release reports:

- **Verified:** supported by a successful automated check or reproducible evidence.
- **Implemented, unverified:** code or documentation exists, but end-to-end evidence is not yet available.
- **Blocked:** a known defect or missing prerequisite prevents progression.
- **Not started:** no implementation or evidence has been identified.
- **Not applicable:** justified and recorded for the release scope.

A successful CI run verifies software quality checks only. It does not prove a strategy will be profitable or that live trading is safe.
