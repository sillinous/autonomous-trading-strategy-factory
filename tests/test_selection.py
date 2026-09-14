from types import SimpleNamespace

import pandas as pd
import pytest

from atsf.selection import SelectionPolicy, pareto_front, select_population


def evaluation(
    candidate_id: str,
    score: float,
    drawdown: float = 0.10,
    pass_rate: float = 0.90,
    robustness: float = 0.95,
    regime: float = -0.01,
    eligible: bool = True,
):
    baseline = pd.Series([1.0, 100.0])
    stressed = pd.Series([1.0, 100.0 * robustness])
    robustness_result = SimpleNamespace(
        baseline=SimpleNamespace(equity=baseline),
        scenarios=(("baseline", SimpleNamespace(equity=baseline)), ("stress", SimpleNamespace(equity=stressed))),
    )
    return SimpleNamespace(
        candidate_id=candidate_id,
        fitness=SimpleNamespace(score=score),
        backtest=SimpleNamespace(max_drawdown=drawdown),
        monte_carlo=SimpleNamespace(pass_rate=pass_rate),
        regime=SimpleNamespace(score=regime),
        robustness=robustness_result,
        promotion=SimpleNamespace(eligible=eligible),
        validation_passed=True,
    )


def candidate(candidate_id: str):
    return SimpleNamespace(strategy_id=candidate_id)


def test_pareto_front_removes_dominated_candidate():
    strong = evaluation("strong", 2.0, drawdown=0.10, pass_rate=0.95, robustness=0.98)
    dominated = evaluation("dominated", 1.0, drawdown=0.20, pass_rate=0.80, robustness=0.90)
    assert [item.candidate_id for item in pareto_front([strong, dominated])] == ["strong"]


def test_selection_is_deterministic_and_prefers_pareto_candidates():
    candidates = [candidate("a"), candidate("b"), candidate("c")]
    evaluations = [
        evaluation("a", 1.0, drawdown=0.20, pass_rate=0.80),
        evaluation("b", 1.5, drawdown=0.10, pass_rate=0.90),
        evaluation("c", 1.2, drawdown=0.15, pass_rate=0.85),
    ]
    policy = SelectionPolicy(population_size=2)
    selected = select_population(candidates, evaluations, policy)
    assert [item.strategy_id for item in selected] == ["b", "c"]


def test_selection_applies_hard_gates():
    candidates = [candidate("bad"), candidate("good")]
    evaluations = [
        evaluation("bad", 10.0, eligible=False),
        evaluation("good", 1.0),
    ]
    selected = select_population(candidates, evaluations, SelectionPolicy(population_size=1))
    assert [item.strategy_id for item in selected] == ["good"]


def test_selection_fails_closed_when_not_enough_eligible_candidates():
    candidates = [candidate("a"), candidate("b")]
    evaluations = [evaluation("a", 1.0)]
    with pytest.raises(ValueError, match="equal length"):
        select_population(candidates, evaluations, SelectionPolicy(population_size=1))


def test_selection_validates_population_policy():
    with pytest.raises(ValueError, match="population_size"):
        SelectionPolicy(population_size=0)
    with pytest.raises(ValueError, match="pass_rate"):
        SelectionPolicy(population_size=1, min_monte_carlo_pass_rate=1.1)
