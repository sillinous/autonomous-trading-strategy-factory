import json

import pytest

from atsf.feedback_loop import process_strategy_health
from atsf.feedback_provenance import build_feedback_provenance
from atsf.feedback_registry import FeedbackEventStore
from atsf.lifecycle import StrategyLifecycle
from atsf.monitoring import DegradationReport
from atsf.portfolio_executor import execute_persisted_portfolio
from atsf.portfolio_replay import verify_persisted_portfolio_run
from atsf.provenance_graph import build_research_provenance_graph
from atsf.reproducibility import build_reproducibility_certificate
from atsf.registry import ExperimentRegistry
from atsf.research_queue import ResearchQueue
from tests.test_portfolio_executor import make_data, seed_persisted_portfolio


def execute(registry: ExperimentRegistry):
    first, second, dataset_version, _bundle_version = seed_persisted_portfolio(registry)
    result = execute_persisted_portfolio(
        registry,
        "portfolio-1",
        {first: make_data(), second: make_data()},
        dataset_version=dataset_version,
    )
    return result


def certificate_for(registry: ExperimentRegistry, run_id: str):
    verification = verify_persisted_portfolio_run(registry, run_id)
    assert verification.valid is True
    run = registry.get_portfolio_run(run_id)
    return build_reproducibility_certificate(registry, run, verification)


def make_feedback_event(strategy_id: str):
    report = DegradationReport(
        degraded=True,
        observations=30,
        total_return=-0.12,
        max_drawdown=-0.22,
        volatility=0.08,
        reasons=("minimum return breached",),
    )
    lifecycle = StrategyLifecycle()
    queue = ResearchQueue()
    action = process_strategy_health(strategy_id, lifecycle, report, queue)
    return build_feedback_provenance(
        strategy_id,
        action,
        report,
        previous_state="active",
    )


def test_reproducibility_certificate_is_deterministic() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    first = certificate_for(registry, result.identity.run_id)
    second = certificate_for(registry, result.identity.run_id)
    assert first == second
    assert len(first.certificate_id) == 24
    assert first.verified is True
    assert len(first.research_fingerprint) == 24
    assert len(first.provenance_graph_fingerprint) == 24
    assert len(first.lineage_fingerprint) == 16
    assert len(first.attribution_fingerprint) == 16
    assert first.event_count == len(result.audit_events)
    registry.close()


def test_provenance_graph_covers_research_to_paper_chain() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    run = registry.get_portfolio_run(result.identity.run_id)
    graph = build_research_provenance_graph(registry, run)

    kinds = {node.kind for node in graph.nodes}
    assert {
        "dataset",
        "data_bundle",
        "strategy",
        "lineage",
        "experiment",
        "evaluation_evidence",
        "promotion",
        "portfolio",
        "paper_run",
        "attribution",
        "audit_event",
        "fill_lineage",
        "execution_manifest",
        "replay_verification",
    } <= kinds
    relations = {edge.relation for edge in graph.edges}
    assert {
        "tested",
        "produced_evidence",
        "evaluated_for",
        "promoted_to_candidate",
        "executed_as",
        "records",
        "explained_by",
        "verifies",
    } <= relations
    assert len(graph.fingerprint) == 24
    assert graph == build_research_provenance_graph(registry, run)
    registry.close()


def test_provenance_graph_binds_persisted_feedback_and_research_request() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    run = registry.get_portfolio_run(result.identity.run_id)
    baseline = build_research_provenance_graph(registry, run)
    strategy_id = run["attribution"][0]["strategy_id"]

    event = make_feedback_event(strategy_id)
    FeedbackEventStore(registry).save(event)
    changed = build_research_provenance_graph(registry, run)

    kinds = {node.kind for node in changed.nodes}
    assert {"feedback_event", "lifecycle_state", "research_request"} <= kinds
    relations = {edge.relation for edge in changed.edges}
    assert {
        "experienced_feedback",
        "transitioned_to",
        "generated_research_request",
    } <= relations
    assert changed.fingerprint != baseline.fingerprint
    assert any(
        node.kind == "feedback_event" and node.key == event.event_id
        for node in changed.nodes
    )
    assert any(
        node.kind == "research_request" and node.key == event.research_request_id
        for node in changed.nodes
    )
    registry.close()


def test_provenance_graph_rejects_tampered_feedback_event() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    run = registry.get_portfolio_run(result.identity.run_id)
    strategy_id = run["attribution"][0]["strategy_id"]
    event = make_feedback_event(strategy_id)
    store = FeedbackEventStore(registry)
    store.save(event)
    registry._connection.execute(
        "UPDATE strategy_feedback_events SET event_json = ? WHERE event_id = ?",
        ('{"fingerprint":"tampered"}', event.event_id),
    )
    registry._connection.commit()
    with pytest.raises(ValueError, match="feedback provenance is invalid"):
        build_research_provenance_graph(registry, run)
    registry.close()


def test_provenance_graph_rejects_lineage_cycles() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    run = registry.get_portfolio_run(result.identity.run_id)
    strategy_id = run["attribution"][0]["strategy_id"]
    registry._connection.execute(
        "UPDATE lineage SET parent_ids_json = ? WHERE strategy_id = ?",
        (json.dumps([strategy_id]), strategy_id),
    )
    registry._connection.commit()
    with pytest.raises(ValueError, match="lineage cycle"):
        build_research_provenance_graph(registry, run)
    registry.close()


def test_certificate_binds_research_provenance() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    baseline = certificate_for(registry, result.identity.run_id)

    run = registry.get_portfolio_run(result.identity.run_id)
    portfolio = registry.get_portfolio(run["portfolio_id"])
    experiment_id = next(iter(portfolio["definition"]["experiment_ids"].values()))

    evidence = registry.get_evaluation_evidence(experiment_id)
    evidence["certificate_test_marker"] = "changed"
    registry.save_evaluation_evidence(experiment_id, evidence)
    evidence_changed = certificate_for(registry, result.identity.run_id)
    assert evidence_changed.research_fingerprint != baseline.research_fingerprint
    assert evidence_changed.provenance_graph_fingerprint != baseline.provenance_graph_fingerprint
    assert evidence_changed.certificate_id != baseline.certificate_id

    registry._connection.execute(
        "UPDATE lineage SET parameters_json = ? WHERE strategy_id = (SELECT strategy_id FROM experiments WHERE experiment_id = ?)",
        (json.dumps({"certificate_test_marker": "changed"}), experiment_id),
    )
    registry._connection.commit()
    lineage_changed = certificate_for(registry, result.identity.run_id)
    assert lineage_changed.lineage_fingerprint != evidence_changed.lineage_fingerprint
    assert lineage_changed.provenance_graph_fingerprint != evidence_changed.provenance_graph_fingerprint
    assert lineage_changed.research_fingerprint != evidence_changed.research_fingerprint
    registry.close()


def test_certificate_binds_attribution_and_portfolio_definition() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    baseline = certificate_for(registry, result.identity.run_id)
    run = registry.get_portfolio_run(result.identity.run_id)

    registry._connection.execute(
        "UPDATE portfolio_attribution SET return_contribution = return_contribution + 0.001 WHERE run_id = ?",
        (run["run_id"],),
    )
    registry._connection.commit()
    attribution_changed = certificate_for(registry, result.identity.run_id)
    assert attribution_changed.attribution_fingerprint != baseline.attribution_fingerprint
    assert attribution_changed.certificate_id != baseline.certificate_id

    portfolio = registry.get_portfolio(run["portfolio_id"])
    definition = dict(portfolio["definition"])
    definition["certificate_test_marker"] = "changed"
    registry._connection.execute(
        "UPDATE portfolios SET definition_json = ? WHERE portfolio_id = ?",
        (json.dumps(definition, sort_keys=True), run["portfolio_id"]),
    )
    registry._connection.commit()
    portfolio_changed = certificate_for(registry, result.identity.run_id)
    assert portfolio_changed.provenance_graph_fingerprint != attribution_changed.provenance_graph_fingerprint
    assert portfolio_changed.research_fingerprint != attribution_changed.research_fingerprint
    assert portfolio_changed.certificate_id != attribution_changed.certificate_id
    registry.close()


def test_certificate_requires_complete_provenance() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    run = registry.get_portfolio_run(result.identity.run_id)
    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)
    registry._connection.execute("DELETE FROM lineage")
    registry._connection.commit()
    with pytest.raises(ValueError, match="strategy provenance is incomplete"):
        build_reproducibility_certificate(registry, run, verification)
    registry.close()


def test_unverified_run_cannot_be_certified() -> None:
    registry = ExperimentRegistry()
    result = execute(registry)
    verification = verify_persisted_portfolio_run(registry, result.identity.run_id)
    bad = verification.__class__(verification.run_id, False, verification.manifest, "tampered")
    run = registry.get_portfolio_run(result.identity.run_id)
    with pytest.raises(ValueError, match="unverified"):
        build_reproducibility_certificate(registry, run, bad)
    registry.close()
