from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

from .lineage import LineageRecord
from .population import Candidate, strategy_id
from .research_history import GenerationRecord, ResearchHistory
from .research_provenance import CandidateProvenance, GenerationProvenance
from .strategy import StrategySpec

_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchCheckpoint:
    """Portable, integrity-checked state from which research can be resumed."""

    next_generation: int
    next_seed: int
    population: tuple[Candidate, ...]
    history: ResearchHistory
    provenance: tuple[GenerationProvenance, ...]
    stopped: bool = False
    schema_version: int = _SCHEMA_VERSION
    state_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema version")
        if self.next_generation < 0:
            raise ValueError("next_generation must be non-negative")
        if not self.population:
            raise ValueError("checkpoint population must not be empty")
        ids = [candidate.strategy_id for candidate in self.population]
        if len(ids) != len(set(ids)):
            raise ValueError("checkpoint population strategy IDs must be unique")
        if len(self.provenance) != len(self.history.records):
            raise ValueError("checkpoint provenance and history lengths must match")
        if self.history.records and self.next_generation != self.history.latest.generation + 1:
            raise ValueError("next_generation must follow the latest history generation")

        expected = self.compute_state_digest()
        if self.state_digest and self.state_digest != expected:
            raise ValueError("checkpoint state digest mismatch")
        if not self.state_digest:
            object.__setattr__(self, "state_digest", expected)

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "next_generation": self.next_generation,
            "next_seed": self.next_seed,
            "population": [_candidate_to_dict(candidate) for candidate in self.population],
            "history": [asdict(record) for record in self.history.records],
            "provenance": [_provenance_to_dict(item) for item in self.provenance],
            "stopped": self.stopped,
        }

    def compute_state_digest(self) -> str:
        return _digest(self._state_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._state_payload()
        payload["state_digest"] = self.compute_state_digest()
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResearchCheckpoint:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be an object")
        supplied_digest = payload.get("state_digest", "")
        state = dict(payload)
        state.pop("state_digest", None)
        if supplied_digest and supplied_digest != _digest(state):
            raise ValueError("checkpoint state digest mismatch")
        try:
            population = tuple(_candidate_from_dict(item) for item in state["population"])
            history = ResearchHistory(
                records=tuple(GenerationRecord(**item) for item in state["history"])
            )
            provenance = tuple(_provenance_from_dict(item) for item in state["provenance"])
            return cls(
                next_generation=int(state["next_generation"]),
                next_seed=int(state["next_seed"]),
                population=population,
                history=history,
                provenance=provenance,
                stopped=bool(state.get("stopped", False)),
                schema_version=int(state["schema_version"]),
                state_digest=supplied_digest,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid research checkpoint payload") from exc

    @classmethod
    def from_json(cls, payload: str) -> ResearchCheckpoint:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid checkpoint JSON") from exc
        return cls.from_dict(value)


def _candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "strategy": candidate.strategy.model_dump(mode="json"),
        "strategy_id": candidate.strategy_id,
        "lineage": {
            "strategy_id": candidate.lineage.strategy_id,
            "generation": candidate.lineage.generation,
            "parent_ids": list(candidate.lineage.parent_ids),
            "operator": candidate.lineage.operator,
            "parameters": candidate.lineage.parameters,
        },
    }


def _candidate_from_dict(payload: dict[str, Any]) -> Candidate:
    strategy = StrategySpec.model_validate(payload["strategy"])
    candidate_id = str(payload["strategy_id"])
    if strategy_id(strategy) != candidate_id:
        raise ValueError("candidate strategy ID does not match strategy genome")
    lineage_data = payload["lineage"]
    lineage = LineageRecord(
        strategy_id=str(lineage_data["strategy_id"]),
        generation=int(lineage_data["generation"]),
        parent_ids=tuple(lineage_data.get("parent_ids", ())),
        operator=str(lineage_data.get("operator", "seed")),
        parameters=dict(lineage_data.get("parameters", {})),
    )
    if lineage.strategy_id != candidate_id:
        raise ValueError("candidate lineage ID does not match candidate ID")
    return Candidate(strategy=strategy, strategy_id=candidate_id, lineage=lineage)


def _provenance_to_dict(item: GenerationProvenance) -> dict[str, Any]:
    return {
        "generation": item.generation,
        "candidate_records": [asdict(record) for record in item.candidate_records],
        "selected_strategy_ids": list(item.selected_strategy_ids),
        "next_strategy_ids": list(item.next_strategy_ids),
        "metrics_digest": item.metrics_digest,
    }


def _provenance_from_dict(payload: dict[str, Any]) -> GenerationProvenance:
    return GenerationProvenance(
        generation=int(payload["generation"]),
        candidate_records=tuple(
            CandidateProvenance(
                strategy_id=str(record["strategy_id"]),
                generation=int(record["generation"]),
                parent_strategy_ids=tuple(record.get("parent_strategy_ids", ())),
                genome_digest=str(record["genome_digest"]),
                evaluation_digest=str(record["evaluation_digest"]),
                research_seed=int(record["research_seed"]),
            )
            for record in payload["candidate_records"]
        ),
        selected_strategy_ids=tuple(payload["selected_strategy_ids"]),
        next_strategy_ids=tuple(payload["next_strategy_ids"]),
        metrics_digest=str(payload["metrics_digest"]),
    )
