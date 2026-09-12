from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .allocation import AllocationPolicy
from .data import dataset_identity
from .dataset_bundle import DATA_SCHEMA_VERSION, bundle_identity
from .experiment import ExperimentSpec
from .fitness import FitnessPolicy, FitnessResult
from .population import Candidate, seed_population
from .portfolio import PortfolioPolicy
from .portfolio_builder import build_portfolio
from .ranking import rank_candidates
from .registry import ExperimentRegistry
from .research_budget import ResearchBudgetPolicy
from .research_cycle_registry import ResearchCycleRegistry
from .research_director import ResearchDirectorPolicy
from .research_feedback import ResearchFeedbackResult, build_research_feedback
from .research_planner import ResearchPlan, build_research_plan
from .scheduler import GenerationResult, evolve_generation
from .strategy import StrategySpec
from .successor import SuccessorAdmission, admit_successor


@dataclass(frozen=True)
class ResearchRunResult:
    generations: tuple[GenerationResult, ...]
    final_population: tuple[Candidate, ...]
    dataset_id: str
    dataset_version: str
    portfolio_id: str | None = None
    research_plans: tuple[ResearchPlan, ...] = ()
    research_feedback: tuple[ResearchFeedbackResult, ...] = ()


def _build_research_portfolio(
    result: GenerationResult,
    dataset_id: str,
    dataset_version: str,
    source_data: pd.DataFrame,
    store: ExperimentRegistry,
    *,
    source: str,
    timeframe: str,
    schema_version: str,
) -> str | None:
    eligible = [
        evaluation
        for evaluation in result.evaluations
        if evaluation.promotion.eligible
    ]
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
    data_bundle = {strategy_id: source_data for strategy_id in portfolio.allocation.weights}
    bundle = bundle_identity(
        data_bundle,
        dataset_id,
        source=source,
        timeframe=timeframe,
        schema_version=schema_version,
    )

    definition = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "data_bundle_version": bundle.version,
        "data_source": source,
        "data_timeframe": timeframe,
        "data_schema_version": schema_version,
        "generation": result.generation,
        "experiment_ids": {
            evaluation.candidate_id: evaluation.experiment.experiment_id
            for evaluation in eligible
        },
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
    canonical = json.dumps(
        definition,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    portfolio_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    store.save_portfolio(portfolio_id, definition, portfolio.allocation.weights)
    return portfolio_id


def _cycle_payload(plan: ResearchPlan, feedback: ResearchFeedbackResult, admissions: tuple[SuccessorAdmission, ...]) -> tuple[str, dict, dict, list[dict]]:
    plan_payload = {
        "generation": plan.generation,
        "requests": [
            {
                "request_id": request.request_id,
                "reason": request.reason,
                "priority": request.priority,
                "source_strategy_id": request.source_strategy_id,
            }
            for request in plan.requests
        ],
        "allocations": [
            {
                "request_id": allocation.request_id,
                "units": allocation.units,
                "score": allocation.score,
            }
            for allocation in plan.allocations
        ],
    }
    feedback_payload = {
        "generation": feedback.generation,
        "signals": [
            {
                "strategy_id": signal.strategy_id,
                "reason": signal.reason,
                "fitness": signal.fitness,
                "robustness": signal.robustness,
                "novelty": signal.novelty,
                "uncertainty": signal.uncertainty,
                "capacity_gap": signal.capacity_gap,
            }
            for signal in feedback.signals
        ],
    }
    admissions_payload = [
        {
            "candidate_id": admission.candidate_id,
            "parent_ids": list(admission.parent_ids),
            "generation": admission.generation,
            "admitted": admission.admitted,
            "stage": admission.stage,
            "reasons": list(admission.reasons),
        }
        for admission in admissions
    ]
    canonical = json.dumps(
        {"plan": plan_payload, "feedback": feedback_payload, "admissions": admissions_payload},
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    cycle_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return cycle_id, plan_payload, feedback_payload, admissions_payload


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
    director_policy: ResearchDirectorPolicy | None = None,
    budget_policy: ResearchBudgetPolicy | None = None,
) -> ResearchRunResult:
    """Run deterministic evolutionary research and persist portfolio/research cycles."""
    if generations <= 0:
        raise ValueError("generations must be positive")
    if population_size <= 0 or survivor_count <= 0:
        raise ValueError("population sizes must be positive")
    if survivor_count > population_size:
        raise ValueError("survivor_count cannot exceed population_size")
    frame_identity = dataset_identity(data, dataset_id)
    population = seed_population(seeds)
    if not population:
        raise ValueError("seeds must contain at least one unique strategy")
    timeframes = {candidate.strategy.timeframe for candidate in population}
    if len(timeframes) != 1:
        raise ValueError("all research strategies must use the same timeframe")
    timeframe = next(iter(timeframes))
    source = "research_input"
    schema_version = DATA_SCHEMA_VERSION
    registered_identity = bundle_identity(
        {"research": data},
        dataset_id,
        source=source,
        timeframe=timeframe,
        schema_version=schema_version,
    )

    owned_registry = registry is None
    store = registry or ExperimentRegistry()
    cycle_store = ResearchCycleRegistry(store._connection)
    results: list[GenerationResult] = []
    research_plans: list[ResearchPlan] = []
    research_feedback: list[ResearchFeedbackResult] = []
    portfolio_id: str | None = None
    try:
        store.register_dataset(registered_identity, source=source)
        store.require_dataset(registered_identity.dataset_id, registered_identity.version)
        for candidate in population:
            store.save_strategy(candidate.strategy)
            store.save_lineage(candidate.lineage)
        for generation in range(generations):
            candidates_by_id = {candidate.strategy_id: candidate for candidate in population}
            result = evolve_generation(
                population,
                data,
                registered_identity.dataset_id,
                registered_identity.version,
                target_size=population_size,
                survivor_count=survivor_count,
                seed=seed + generation,
                fitness_policy=fitness_policy,
            )
            results.append(result)
            feedback = build_research_feedback(
                result.evaluations,
                generation=result.generation,
                population=tuple(population),
            )
            research_feedback.append(feedback)
            plan = build_research_plan(
                result,
                director_policy=director_policy,
                budget_policy=budget_policy,
                population=population,
            )
            research_plans.append(plan)
            admissions: list[SuccessorAdmission] = []
            for evaluation in result.evaluations:
                candidate = candidates_by_id[evaluation.candidate_id]
                admissions.append(
                    admit_successor(
                        candidate,
                        evaluation,
                        generation=result.generation,
                        existing_ids=frozenset(),
                    )
                )
            cycle_id, plan_payload, feedback_payload, admissions_payload = _cycle_payload(
                plan,
                feedback,
                tuple(admissions),
            )
            cycle_store.save_cycle(
                cycle_id,
                result.generation,
                plan=plan_payload,
                feedback=feedback_payload,
                admissions=admissions_payload,
            )
            for evaluation in result.evaluations:
                candidate = candidates_by_id[evaluation.candidate_id]
                spec = ExperimentSpec(
                    candidate.strategy,
                    registered_identity.dataset_id,
                    registered_identity.version,
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
                        "research_plan": {
                            "generation": plan.generation,
                            "requests": [request.request_id for request in plan.requests],
                            "allocations": {
                                allocation.request_id: allocation.units
                                for allocation in plan.allocations
                            },
                        },
                        "research_feedback": {
                            "generation": feedback.generation,
                            "signals": [
                                {
                                    "strategy_id": signal.strategy_id,
                                    "reason": signal.reason,
                                    "fitness": signal.fitness,
                                    "robustness": signal.robustness,
                                    "novelty": signal.novelty,
                                    "uncertainty": signal.uncertainty,
                                    "capacity_gap": signal.capacity_gap,
                                }
                                for signal in feedback.signals
                                if signal.strategy_id == evaluation.candidate_id
                            ],
                        },
                        "successor_admission": {
                            "cycle_id": cycle_id,
                            "admitted": next(
                                admission.admitted
                                for admission in admissions
                                if admission.candidate_id == evaluation.candidate_id
                            ),
                        },
                    },
                )
            portfolio_id = _build_research_portfolio(
                result,
                registered_identity.dataset_id,
                registered_identity.version,
                data,
                store,
                source=source,
                timeframe=timeframe,
                schema_version=schema_version,
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
        dataset_id=frame_identity.dataset_id,
        dataset_version=registered_identity.version,
        portfolio_id=portfolio_id,
        research_plans=tuple(research_plans),
        research_feedback=tuple(research_feedback),
    )