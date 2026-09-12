from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .feedback_provenance import FeedbackProvenance, verify_feedback_provenance
from .feedback_registry import FeedbackEventStore
from .registry import ExperimentRegistry
from .research_registry import ResearchRequestStore
from .successor_registry import SuccessorCandidateStore

PROVENANCE_GRAPH_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ProvenanceNode:
    node_id: str
    kind: str
    key: str
    attributes: dict[str, Any]


@dataclass(frozen=True)
class ProvenanceEdge:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class ResearchProvenanceGraph:
    schema_version: str
    nodes: tuple[ProvenanceNode, ...]
    edges: tuple[ProvenanceEdge, ...]
    fingerprint: str


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _node_id(kind: str, key: str) -> str:
    return hashlib.sha256(_canonical({"kind": kind, "key": key}).encode()).hexdigest()[:16]


def _node(kind: str, key: str, attributes: dict[str, Any]) -> ProvenanceNode:
    return ProvenanceNode(_node_id(kind, key), kind, key, attributes)


def _edge(source: ProvenanceNode, target: ProvenanceNode, relation: str) -> ProvenanceEdge:
    return ProvenanceEdge(source.node_id, target.node_id, relation)


def _lineage_nodes(store: ExperimentRegistry, strategy_ids: set[str]):
    nodes: dict[str, ProvenanceNode] = {}
    edges: list[ProvenanceEdge] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise ValueError(f"strategy lineage cycle detected: {identifier}")
        if identifier in visited:
            return
        strategy = store.get_strategy(identifier)
        lineage = store.get_lineage(identifier)
        if strategy is None or lineage is None:
            raise ValueError(f"strategy provenance is incomplete: {identifier}")
        visiting.add(identifier)
        strategy_node = _node("strategy", identifier, {"strategy_id": identifier, "definition": strategy.model_dump(mode="json")})
        lineage_node = _node("lineage", identifier, {"strategy_id": identifier, "generation": lineage.generation, "parent_ids": list(lineage.parent_ids), "operator": lineage.operator, "parameters": lineage.parameters})
        nodes.setdefault(identifier, strategy_node)
        nodes[f"lineage:{identifier}"] = lineage_node
        edges.append(_edge(lineage_node, strategy_node, "defines"))
        for parent_id in lineage.parent_ids:
            visit(parent_id)
            edges.append(_edge(nodes[parent_id], strategy_node, "parent_of"))
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in sorted(strategy_ids):
        visit(identifier)
    return nodes, edges


def build_research_provenance_graph(store: ExperimentRegistry, run: dict[str, Any]) -> ResearchProvenanceGraph:
    """Materialize and validate the persisted research-to-paper lineage graph."""
    run_id = str(run.get("run_id") or "")
    portfolio_id = str(run.get("portfolio_id") or "")
    if not run_id or not portfolio_id:
        raise ValueError("run and portfolio identifiers are required")
    portfolio = store.get_portfolio(portfolio_id)
    if portfolio is None:
        raise ValueError("portfolio provenance is missing")
    definition = portfolio.get("definition")
    members = portfolio.get("members")
    if not isinstance(definition, dict) or not isinstance(members, dict) or not members:
        raise ValueError("portfolio provenance is incomplete")
    provenance = run.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("execution provenance is invalid")
    dataset_id, dataset_version, bundle_version, execution_fingerprint = (provenance.get(k) for k in ("dataset_id", "dataset_version", "data_bundle_version", "execution_fingerprint"))
    if not all(isinstance(value, str) and value for value in (dataset_id, dataset_version, bundle_version, execution_fingerprint)):
        raise ValueError("execution provenance is incomplete")
    config = provenance.get("execution_config")
    if not isinstance(config, dict):
        raise ValueError("execution configuration is invalid")
    experiment_map = definition.get("experiment_ids")
    if not isinstance(experiment_map, dict) or not experiment_map:
        raise ValueError("portfolio experiment provenance is missing")

    root = _node("paper_run", run_id, {"run_id": run_id, "portfolio_id": portfolio_id, "final_equity": run.get("final_equity"), "halted": bool(run.get("halted")), "halt_reason": run.get("halt_reason"), "execution_fingerprint": execution_fingerprint})
    portfolio_node = _node("portfolio", portfolio_id, {"portfolio_id": portfolio_id, "definition": definition, "members": members})
    dataset_node = _node("dataset", f"{dataset_id}:{dataset_version}", {"dataset_id": dataset_id, "dataset_version": dataset_version, "data_bundle_version": bundle_version, "source": definition.get("data_source"), "timeframe": definition.get("data_timeframe"), "schema_version": definition.get("data_schema_version")})
    bundle_node = _node("data_bundle", bundle_version, {"data_bundle_version": bundle_version, "dataset_id": dataset_id, "dataset_version": dataset_version})
    nodes = {root.key: root, f"portfolio:{portfolio_id}": portfolio_node, f"dataset:{dataset_id}:{dataset_version}": dataset_node, f"bundle:{bundle_version}": bundle_node}
    edges = [_edge(dataset_node, bundle_node, "bundles"), _edge(bundle_node, portfolio_node, "input_to"), _edge(portfolio_node, root, "executed_as")]

    strategy_ids: set[str] = set(members)
    for candidate_id, experiment_id_value in sorted(experiment_map.items()):
        experiment_id = str(experiment_id_value)
        experiment = store.get_experiment(experiment_id)
        evidence = store.get_evaluation_evidence(experiment_id)
        if experiment is None or evidence is None:
            raise ValueError(f"experiment provenance is incomplete: {experiment_id}")
        strategy_id = str(experiment["strategy_id"])
        if str(candidate_id) != strategy_id:
            raise ValueError(f"experiment strategy mismatch: {experiment_id}")
        if str(experiment["dataset_id"]) != dataset_id or str(experiment["dataset_version"]) != dataset_version:
            raise ValueError(f"experiment dataset mismatch: {experiment_id}")
        strategy_ids.add(strategy_id)
        experiment_node = _node("experiment", experiment_id, {"experiment_id": experiment_id, "strategy_id": strategy_id, "dataset_id": experiment["dataset_id"], "dataset_version": experiment["dataset_version"], "seed": experiment["seed"], "status": experiment["status"], "score": experiment["score"], "reason": experiment["reason"]})
        evidence_node = _node("evaluation_evidence", experiment_id, {"experiment_id": experiment_id, "evidence": evidence})
        promotion = evidence.get("promotion") if isinstance(evidence, dict) else None
        if not isinstance(promotion, dict) or not isinstance(promotion.get("stage"), str) or not isinstance(promotion.get("eligible"), bool):
            raise ValueError(f"promotion provenance is incomplete: {experiment_id}")
        promotion_node = _node("promotion", experiment_id, {"experiment_id": experiment_id, **promotion})
        nodes[f"experiment:{experiment_id}"] = experiment_node
        nodes[f"evidence:{experiment_id}"] = evidence_node
        nodes[f"promotion:{experiment_id}"] = promotion_node
        edges.extend([_edge(experiment_node, evidence_node, "produced_evidence"), _edge(evidence_node, promotion_node, "evaluated_for"), _edge(experiment_node, portfolio_node, "candidate_for"), _edge(experiment_node, dataset_node, "uses_dataset")])
        if promotion["eligible"]:
            if strategy_id not in members:
                raise ValueError(f"eligible experiment not represented by portfolio member: {experiment_id}")
            edges.append(_edge(promotion_node, portfolio_node, "promoted_to_candidate"))
        elif strategy_id in members:
            raise ValueError(f"ineligible experiment selected into portfolio: {experiment_id}")
        edges.append(ProvenanceEdge(f"__strategy__strategy:{strategy_id}", experiment_node.node_id, "tested"))

    lineage_nodes, lineage_edges = _lineage_nodes(store, strategy_ids)
    nodes.update({f"strategy:{key}": value for key, value in lineage_nodes.items() if not key.startswith("lineage:")})
    nodes.update({key: value for key, value in lineage_nodes.items() if key.startswith("lineage:")})
    edges.extend(lineage_edges)

    normalized_edges: list[ProvenanceEdge] = []
    for edge in edges:
        source = edge.source
        if source.startswith("__strategy__"):
            strategy_key = source.removeprefix("__strategy__")
            strategy_node = nodes.get(strategy_key)
            if strategy_node is None:
                raise ValueError(f"missing strategy graph node: {strategy_key}")
            source = strategy_node.node_id
        normalized_edges.append(ProvenanceEdge(source, edge.target, edge.relation))

    feedback_store = FeedbackEventStore(store)
    request_store = ResearchRequestStore(store)
    successor_store = SuccessorCandidateStore(store)
    for strategy_id in sorted(strategy_ids):
        for event_payload in feedback_store.list_for_strategy(strategy_id):
            try:
                event = FeedbackProvenance(event_id=str(event_payload["event_id"]), strategy_id=str(event_payload["strategy_id"]), previous_state=str(event_payload["previous_state"]), resulting_state=str(event_payload["resulting_state"]), report_fingerprint=str(event_payload["report_fingerprint"]), research_request_id=None if event_payload.get("research_request_id") is None else str(event_payload["research_request_id"]), research_fingerprint=None if event_payload.get("research_fingerprint") is None else str(event_payload["research_fingerprint"]), fingerprint=str(event_payload["fingerprint"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"feedback provenance is invalid: {strategy_id}") from exc
            if event.strategy_id != strategy_id or not verify_feedback_provenance(event):
                raise ValueError(f"feedback provenance is invalid: {event.event_id}")
            feedback_node = _node("feedback_event", event.event_id, {"event_id": event.event_id, "strategy_id": event.strategy_id, "previous_state": event.previous_state, "resulting_state": event.resulting_state, "report_fingerprint": event.report_fingerprint, "research_request_id": event.research_request_id, "research_fingerprint": event.research_fingerprint, "fingerprint": event.fingerprint})
            nodes[f"feedback:{event.event_id}"] = feedback_node
            strategy_node = nodes.get(f"strategy:{strategy_id}")
            if strategy_node is None:
                raise ValueError(f"missing strategy graph node: {strategy_id}")
            normalized_edges.append(_edge(strategy_node, feedback_node, "experienced_feedback"))
            state_node = _node("lifecycle_state", f"{event.event_id}:{event.resulting_state}", {"state": event.resulting_state, "event_id": event.event_id})
            nodes[f"state:{event.event_id}"] = state_node
            normalized_edges.append(_edge(feedback_node, state_node, "transitioned_to"))
            if event.research_request_id is not None:
                request = request_store.get(event.research_request_id)
                if request is None or request.source_strategy_id != strategy_id:
                    request_node = _node("research_request", event.research_request_id, {"request_id": event.research_request_id, "source_strategy_id": strategy_id, "feedback_event_id": event.event_id, "research_fingerprint": event.research_fingerprint, "persisted": False})
                    nodes[f"research_request:{event.research_request_id}"] = request_node
                    normalized_edges.append(_edge(feedback_node, request_node, "generated_research_request"))
                    continue
                request_node = _node("research_request", request.request_id, {"request_id": request.request_id, "source_strategy_id": request.source_strategy_id, "reason": request.reason.value, "priority": request.priority, "constraints": list(request.constraints), "feedback_event_id": event.event_id, "research_fingerprint": event.research_fingerprint})
                nodes[f"research_request:{request.request_id}"] = request_node
                normalized_edges.append(_edge(feedback_node, request_node, "generated_research_request"))
                for row in successor_store.list_for_request(request.request_id):
                    candidate_id = str(row["candidate_id"])
                    candidate_strategy_id = str(row["strategy_id"])
                    candidate_request_id = str(row["request_id"])
                    if candidate_request_id != request.request_id:
                        raise ValueError(f"successor request mismatch: {candidate_id}")
                    if candidate_strategy_id not in strategy_ids or store.get_strategy(candidate_strategy_id) is None or store.get_lineage(candidate_strategy_id) is None:
                        raise ValueError(f"successor strategy is absent: {candidate_id}")
                    candidate_node = _node("successor_candidate", candidate_id, {"candidate_id": candidate_id, "request_id": request.request_id, "parent_strategy_id": row["parent_strategy_id"], "mutation": row["mutation"], "strategy_id": candidate_strategy_id})
                    nodes[f"successor:{candidate_id}"] = candidate_node
                    normalized_edges.append(_edge(request_node, candidate_node, "generated_successor"))
                    strategy_node = nodes.get(f"strategy:{candidate_strategy_id}")
                    if strategy_node is None:
                        raise ValueError(f"successor strategy is absent: {candidate_id}")
                    normalized_edges.append(_edge(candidate_node, strategy_node, "defines_strategy"))

    attribution = run.get("attribution")
    if not isinstance(attribution, list) or not attribution:
        raise ValueError("portfolio attribution is missing")
    for item in attribution:
        strategy_id = str(item.get("strategy_id"))
        if strategy_id not in members:
            raise ValueError(f"attribution references non-member strategy: {strategy_id}")
        attribution_node = _node("attribution", f"{run_id}:{strategy_id}", {"run_id": run_id, **item})
        nodes[f"attribution:{strategy_id}"] = attribution_node
        normalized_edges.append(_edge(attribution_node, root, "contributes_to"))

    audit_events = run.get("audit_events")
    if not isinstance(audit_events, list):
        raise ValueError("audit ledger is missing")
    for item in audit_events:
        event_id = str(item.get("event_id") or "")
        strategy_id = str(item.get("strategy_id") or "")
        if not event_id or strategy_id not in members:
            raise ValueError("audit provenance is incomplete")
        audit_node = _node("audit_event", f"{run_id}:{event_id}", {"run_id": run_id, **item})
        nodes[f"audit:{event_id}"] = audit_node
        normalized_edges.append(_edge(root, audit_node, "records"))

    fill_lineage = config.get("fill_lineage")
    if not isinstance(fill_lineage, list):
        raise ValueError("fill lineage is missing")
    for item in fill_lineage:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key] for key in ("event_id", "decision_id", "reason")):
            raise ValueError("fill lineage is incomplete")
        lineage_node = _node("fill_lineage", f"{run_id}:{item['event_id']}", {"run_id": run_id, **item})
        nodes[f"fill:{item['event_id']}"] = lineage_node
        audit_node = nodes.get(f"audit:{item['event_id']}")
        if audit_node is None:
            raise ValueError(f"fill lineage references unknown audit event: {item['event_id']}")
        normalized_edges.append(_edge(audit_node, lineage_node, "explained_by"))

    manifest = _node("execution_manifest", run_id, {"event_count": config.get("ledger_event_count"), "ledger_fingerprint": config.get("ledger_fingerprint")})
    nodes[f"manifest:{run_id}"] = manifest
    normalized_edges.append(_edge(root, manifest, "attested_by"))
    verification_node = _node("replay_verification", run_id, {"run_id": run_id, "verified": True, "ledger_fingerprint": config.get("ledger_fingerprint")})
    nodes[f"verification:{run_id}"] = verification_node
    normalized_edges.append(_edge(verification_node, root, "verifies"))

    ordered_nodes = tuple(sorted(nodes.values(), key=lambda node: (node.kind, node.key, node.node_id)))
    ordered_edges = tuple(sorted(set(normalized_edges), key=lambda edge: (edge.source, edge.target, edge.relation)))
    payload = {"schema_version": PROVENANCE_GRAPH_SCHEMA_VERSION, "nodes": [{"node_id": n.node_id, "kind": n.kind, "key": n.key, "attributes": n.attributes} for n in ordered_nodes], "edges": [e.__dict__ for e in ordered_edges]}
    fingerprint = hashlib.sha256(_canonical(payload).encode()).hexdigest()[:24]
    return ResearchProvenanceGraph(PROVENANCE_GRAPH_SCHEMA_VERSION, ordered_nodes, ordered_edges, fingerprint)
