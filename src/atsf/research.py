from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .allocation import AllocationPolicy
from .data import dataset_identity
from .experiment import ExperimentSpec
from .fitness import FitnessPolicy, FitnessResult
from .population import Candidate, seed_population
from .portfolio import PortfolioPolicy
from .portfolio_builder import build_portfolio
from .ranking import rank_candidates
from .registry import ExperimentRegistry
from .scheduler import GenerationResult, evolve_generation
from .strategy import StrategySpec


@dataclass(frozen=True)
class ResearchRunResult:
    generations: tuple[GenerationResult, ...]
    final_population: tuple[Candidate, ...]
    dataset_id: str
    dataset_version: str
    portfolio_id: str | None = None


def _build_research_portfolio(
    result: GenerationResult,
    dataset_id: str,
    dataset_version: str,
    store: ExperimentRegistry,
) -> str | None:
    eligible = [evaluation for evaluation in result.evaluations if evaluation.promotion.eligible]
    if not eligible:
        return None

    return_series = {
        evaluation.candidate_id: evaluation.walk_forward.oos_returns
        for evaluation in eligible
        if evaluation.walk_forward.oos_returns is not None
    }
    if not return_series:
        return None
    returns = pd.concat(return_series, axis=1, join="inner").sort_index()
    returns.columns = list(return_series)
    selection_policy = PortfolioPolicy()
    if len(returns) < selection_policy.min_history:
        return None

    ranked_inputs = [
        (
            evaluation.candidate_id,
            FitnessResult(
                score=evaluation.fitness.score,
                eligible=True,
                reasons=evaluation.promotion.reasons,
            ),
            1.0 if evaluation.robustness.passed else 0.0,
            0.0,
        )
        for evaluation in eligible
    ]
    ranked = rank_candidates(ranked_inputs)
    allocation_policy = AllocationPolicy()
    portfolio = build_portfolio(
        ranked,
        returns,
        portfolio_policy=selection_policy,
        allocation_policy=allocation_policy,
    )

    definition = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "generation": result.generation,
        "selection_policy": {
            "max_strategies": selection_policy.max_strategies,
            "max_average_correlation": selection_policy.max_average_correlation,
            "min_history": selection_policy.min_history,
        },
        "allocation_policy": {
            "max_total_weight": allocation_policy.max_total_weight,
            "min_weight": allocation_policy.min_weight,
            "max_weight": allocation_policy.max_weight,
            "volatility_floor": allocation_policy.volatility_floor,
        },
        "ranked": [
            {
                "strategy_id": item.candidate_id,
                "fitness_score": item.fitness_score,
                "robustness_score": item.robustness_score,
                "diversity_score": item.diversity_score,
                "final_score": item.final_score,
            }
            for item in portfolio.ranked
        ],
        "selected": list(portfolio.selection.selected),
        "rejected": list(portfolio.selection.rejected),
        "average_correlation": portfolio.selection.average_correlation,
        "weights": portfolio.allocation.weights,
        "estimated_volatility": portfolio.allocation.estimated_volatility,
        "total_weight": portfolio.allocation.total_weight,
    }
    canonical = json.dumps(definition, sort_keys=True, allow_nan=False, separators=(",", ":"))
    portfolio_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    store.save_portfolio(portfolio_id, definition, portfolio.allocation.weights)
    return portfolio_id


def run_research(
    seeds: list[StrategySpec],
    data: pd.DataFrame,
    *,
    generations: int = 1,
    population_size: int = 10,
    survivor_count: int = 3,
    seed: int = 0,
    dataset_id: str = "research",
    registry: ExperimentRegistry | None = None,
    fitness_policy: FitnessPolicy | None = None,
) -> ResearchRunResult:
    """Run deterministic evolutionary research and persist a reproducible portfolio when possible."""
    if generations <= 0:
        raise ValueError("generations must be positive")
    if population_size <= 0 or survivor_count <= 0:
        raise ValueError("population sizes must be positive")
    if survivor_count > population_size:
        raise ValueError("survivor_count cannot exceed population_size")
    identity = dataset_identity(data, dataset_id)
    population = seed_population(seeds)
    if not population:
        raise ValueError("seeds must contain at least one unique strategy")

    owned_registry = registry is None
    store = registry or ExperimentRegistry()
    results: list[GenerationResult] = []
    portfolio_id: str | None = None
    try:
        for candidate in population:
            store.save_strategy(candidate.strategy)
            store.save_lineage(candidate.lineage)
        for generation in range(generations):
            candidates_by_id = {candidate.strategy_id: candidate for candidate in population}
            result = evolve_generation(
                population,
                data,
                identity.dataset_id,
                identity.version,
                target_size=population_size,
                survivor_count=survivor_count,
                seed=seed + generation,
                fitness_policy=fitness_policy,
            )
            results.append(result)
            for evaluation in result.evaluations:
                candidate = candidates_by_id[evaluation.candidate_id]
                spec = ExperimentSpec(
                    candidate.strategy,
                    identity.dataset_id,
                    identity.version,
                    seed + generation,
                )
                store.save_experiment(spec, evaluation.experiment)
                store.save_evaluation_evidence(
                    evaluation.experiment.experiment_id,
                    {
                        "candidate_id": evaluation.candidate_id,
                        "walk_forward": {
                            "passed": evaluation.walk_forward.passed,
                            "oos_return": evaluation.walk_forward.oos_return,
                            "oos_sharpe": evaluation.walk_forward.oos_sharpe,
                            "oos_drawdown": evaluation.walk_forward.oos_drawdown,
                        },
                        "monte_carlo": {
                            "simulations": evaluation.monte_carlo.simulations,
                            "seed": evaluation.monte_carlo.seed,
                            "median_return": evaluation.monte_carlo.median_return,
                            "worst_return": evaluation.monte_carlo.worst_return,
                            "lower_percentile_return": evaluation.monte_carlo.lower_percentile_return,
                            "pass_rate": evaluation.monte_carlo.pass_rate,
                        },
                        "perturbation": {
                            "samples": evaluation.perturbation.samples,
                            "seed": evaluation.perturbation.seed,
                            "pass_rate": evaluation.perturbation.pass_rate,
                            "worst_score": evaluation.perturbation.worst_score,
                            "median_score": evaluation.perturbation.median_score,
                            "strategy_ids": evaluation.perturbation.strategy_ids,
                        },
                        "regime": {
                            "score": evaluation.regime.score,
                            "regime_returns": evaluation.regime.regime_returns,
                            "covered_regimes": evaluation.regime.covered_regimes,
                        },
                        "robustness": {
                            "passed": evaluation.robustness.passed,
                            "reasons": evaluation.robustness.reasons,
                            "scenarios": {
                                name: {
                                    "total_return": scenario.total_return,
                                    "max_drawdown": scenario.max_drawdown,
                                }
                                for name, scenario in evaluation.robustness.scenarios
                            },
                        },
                        "promotion": {
                            "stage": evaluation.promotion.stage,
                            "eligible": evaluation.promotion.eligible,
                            "reasons": evaluation.promotion.reasons,
                        },
                    },
                )
            portfolio_id = _build_research_portfolio(
                result,
                identity.dataset_id,
                identity.version,
                store,
            ) or portfolio_id
            for candidate in result.next_population:
                store.save_strategy(candidate.strategy)
                store.save_lineage(candidate.lineage)
            population = list(result.next_population)
    finally:
        if owned_registry:
            store.close()

    return ResearchRunResult(
        generations=tuple(results),
        final_population=tuple(population),
        dataset_id=identity.dataset_id,
        dataset_version=identity.version,
        portfolio_id=portfolio_id,
    )
