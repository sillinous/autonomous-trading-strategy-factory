# Autonomous Trading Strategy Factory

An experimental platform for autonomous quantitative strategy research, historical backtesting, robustness validation, and eventual paper/live execution.

## Design principles

- Strategies are represented as typed, serializable specifications rather than arbitrary generated code.
- Backtesting is deterministic and models trading frictions explicitly.
- Out-of-sample validation, walk-forward testing, perturbation, and stress testing are first-class.
- Every experiment is reproducible and retains lineage.
- AI proposes hypotheses; deterministic engines evaluate them.
- Live execution is disabled until a strategy passes explicit promotion gates.

## Initial scope

1. Strategy DSL and validation
2. Deterministic backtesting kernel
3. Metrics and validation framework
4. Strategy generation/evolution
5. Experiment registry
6. Paper-trading execution boundary

## Status

Early foundation. This repository is intentionally research-first and does not constitute financial advice or a guarantee of profitability.
