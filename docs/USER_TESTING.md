# ATSF User Testing Runbook

## Testing scope

This release is for **controlled user testing of research, validation, portfolio, paper-trading, audit, replay, provenance, and operator UI workflows**.

Live brokerage execution is intentionally disabled.

## Start locally with Docker

From the repository root:

```bash
docker compose up --build -d
docker compose ps
```

The service is bound to `127.0.0.1:8000` by default and persists its SQLite registry in the `atsf-data` Docker volume.

Open:

```
http://127.0.0.1:8000/
```

## Optional API-key protection

For controlled testing, set an API key before starting Compose:

```bash
export ATSF_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose up --build -d
```

The UI accepts the key in its API-key field. The public `/health` endpoint remains available for Docker liveness; operational endpoints require the configured key.

## First checks

Verify:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

With an API key configured:

```bash
curl -H "X-API-Key: $ATSF_API_KEY" http://127.0.0.1:8000/ready
curl -H "X-API-Key: $ATSF_API_KEY" http://127.0.0.1:8000/health/details
curl -H "X-API-Key: $ATSF_API_KEY" http://127.0.0.1:8000/observability/summary
```

The service must report:

- research integrity verified
- execution authority false
- live execution disabled

## Recommended user-testing sequence

1. Open the Control Plane UI.
2. Verify service health and readiness.
3. Inspect persisted research/experiment information.
4. Exercise research-generation and strategy-lifecycle workflows using **real market data**.
5. Inspect robustness and promotion evidence.
6. Inspect portfolio lifecycle and risk evidence.
7. Run paper-trading workflows.
8. Verify audit/replay results.
9. Inspect reproducibility certificates and provenance.
10. Exercise degraded/failure paths where practical.
11. Restart the container and verify persisted state remains available.
12. Record usability defects, confusing states, missing operator information, and workflow gaps.

## Persistence test

Restart the service:

```bash
docker compose restart
```

Then reload the UI and verify persisted registry evidence remains available.

Do not delete the `atsf-data` volume during testing unless intentionally resetting the environment.

## Safety boundary

Do **not** provide broker credentials or attempt to enable live execution as part of this test phase.

The application is designed so that the operator UI cannot grant live execution authority. The live path remains behind explicit research, paper qualification, risk certification, and human authorization controls.

## Reporting defects

For each issue, capture:

- exact UI/API action
- expected result
- actual result
- timestamp
- relevant run/portfolio/strategy ID
- whether the issue survives container restart
- screenshot or response payload when useful

This gives the next development cycle enough evidence to reproduce and fix the issue.
