import pandas as pd

from atsf.feedback_loop import process_strategy_health
from atsf.feedback_provenance import build_feedback_provenance
from atsf.feedback_registry import FeedbackEventStore
from atsf.lifecycle import StrategyLifecycle
from atsf.monitoring import DegradationReport
from atsf.portfolio_executor import execute_persisted_portfolio
from atsf.provenance_graph import build_research_provenance_graph
from atsf.replacement_research import generate_replacements
from atsf.research_queue import ResearchQueue
from atsf.research_registry import ResearchRequestStore
from atsf.registry import ExperimentRegistry
from atsf.successor_registry import SuccessorCandidateStore

from tests.test_portfolio_executor import make_data, seed_persisted_portfolio


def test_successor_candidates_are_attached_to_feedback_provenance_graph() -> None:
    registry = ExperimentRegistry()
    first, second, dataset_version, _bundle = seed_persisted_portfolio(registry)
    result = execute_persisted_portfolio(
        registry,
        "portfolio-1",
        {first: make_data(), second: make_data()},
        dataset_version=dataset_version,
    )
    run = registry.get_portfolio_run(result.identity.run_id)
    assert run is not None

    report = DegradationReport(
        degraded=True,
        observations=20,
        total_return=-0.12,
        max_drawdown=-0.22,
        volatility=0.08,
        reasons=("minimum return breached",),
    )
    queue = ResearchQueue()
    action = process_strategy_health(first, StrategyLifecycle(), report, queue)
    assert action.research_request is not None

    request_store = ResearchRequestStore(registry)
    request_store.save(action.research_request)
    feedback = build_feedback_provenance(first, action, report, previous_state="active")
    FeedbackEventStore(registry).save(feedback)

    generated = generate_replacements(
        action.research_request,
        ["TEST"],
        request_store=request_store,
        registry=registry,
    )
    assert generated.strategy_ids
    candidate_store = SuccessorCandidateStore(registry)
    persisted_candidates = candidate_store.list_for_request(action.research_request.request_id)
    assert len(persisted_candidates) == len(generated.candidates)

    graph = build_research_provenance_graph(registry, run)
    successor_nodes = [node for node in graph.nodes if node.kind == "successor_candidate"]
    request_nodes = [node for node in graph.nodes if node.kind == "research_request"]
    assert len(successor_nodes) == len(generated.candidates)
    assert any(node.key == action.research_request.request_id for node in request_nodes)

    successor_ids = {node.attributes["candidate_id"] for node in successor_nodes}
    assert successor_ids == {candidate["candidate_id"] for candidate in persisted_candidates}
    assert all(
        any(edge.target == node.node_id and edge.relation == "generated_successor" for edge in graph.edges)
        for node in successor_nodes
    )
    assert all(
        any(edge.target == node.node_id and edge.relation == "defines_strategy" for edge in graph.edges)
        for node in successor_nodes
    )
    registry.close()


def test_successor_persistence_tampering_fails_closed_in_graph() -> None:
    registry = ExperimentRegistry()
    first, second, dataset_version, _bundle = seed_persisted_portfolio(registry)
    result = execute_persisted_portfolio(
        registry,
        "portfolio-1",
        {first: make_data(), second: make_data()},
        dataset_version=dataset_version,
    )
    run = registry.get_portfolio_run(result.identity.run_id)
    assert run is not None
    report = DegradationReport(
        degraded=True,
        observations=20,
        total_return=-0.12,
        max_drawdown=-0.22,
        volatility=0.08,
        reasons=("minimum return breached",),
    )
    queue = ResearchQueue()
    action = process_strategy_health(first, StrategyLifecycle(), report, queue)
    request_store = ResearchRequestStore(registry)
    request_store.save(action.research_request)
    FeedbackEventStore(registry).save(build_feedback_provenance(first, action, report, previous_state="active"))
    generate_replacements(action.research_request, ["TEST"], request_store=request_store, registry=registry)

    registry._connection.execute(
        "UPDATE successor_candidates SET strategy_id = ? WHERE request_id = ?",
        ("missing-successor", action.research_request.request_id),
    )
    registry._connection.commit()

    try:
        build_research_provenance_graph(registry, run)
    except ValueError as exc:
        assert "successor strategy is absent" in str(exc)
    else:
        raise AssertionError("tampered successor provenance was accepted")
    registry.close()
