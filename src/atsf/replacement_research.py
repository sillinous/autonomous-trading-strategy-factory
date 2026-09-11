from __future__ import annotations

from dataclasses import dataclass

from .generator import StrategyCandidate, StrategyGenerator
from .lineage import LineageRecord
from .research_queue import ResearchRequest
from .research_registry import ResearchRequestStore
from .registry import ExperimentRegistry
from .successor_registry import SuccessorCandidateStore


@dataclass(frozen=True)
class ReplacementResearchResult:
    request: ResearchRequest
    candidates: tuple[StrategyCandidate, ...]
    strategy_ids: tuple[str, ...] = ()


def generate_replacements(
    request: ResearchRequest,
    symbols: list[str],
    *,
    generator: StrategyGenerator | None = None,
    request_store: ResearchRequestStore | None = None,
    registry: ExperimentRegistry | None = None,
) -> ReplacementResearchResult:
    """Generate typed successor hypotheses and optionally persist their lineage."""
    if request_store is not None:
        persisted = request_store.get(request.request_id)
        if persisted is None:
            raise ValueError(f"research request is not persisted: {request.request_id}")
        if persisted != request:
            raise ValueError(f"research request mismatch: {request.request_id}")
    candidates = (generator or StrategyGenerator()).generate(request, symbols)
    if not candidates:
        raise ValueError(f"no replacement candidates generated: {request.request_id}")
    if registry is None:
        return ReplacementResearchResult(request=request, candidates=candidates)

    candidate_store = SuccessorCandidateStore(registry)
    strategy_ids: list[str] = []
    for candidate in candidates:
        strategy_identifier = registry.save_strategy(candidate.strategy)
        registry.save_lineage(
            LineageRecord(
                strategy_id=strategy_identifier,
                generation=1,
                parent_ids=(() if candidate.parent_strategy_id is None else (candidate.parent_strategy_id,)),
                operator=candidate.mutation,
                parameters={"request_id": candidate.request_id, "candidate_id": candidate.candidate_id},
            )
        )
        candidate_store.save(candidate, strategy_identifier)
        strategy_ids.append(strategy_identifier)
    return ReplacementResearchResult(request=request, candidates=candidates, strategy_ids=tuple(strategy_ids))
