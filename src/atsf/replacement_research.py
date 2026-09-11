from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .generator import StrategyCandidate, StrategyGenerator
from .research_queue import ResearchRequest
from .research_registry import ResearchRequestStore


@dataclass(frozen=True)
class ReplacementResearchResult:
    request: ResearchRequest
    candidates: tuple[StrategyCandidate, ...]


def generate_replacements(
    request: ResearchRequest,
    symbols: list[str],
    *,
    generator: StrategyGenerator | None = None,
    request_store: ResearchRequestStore | None = None,
) -> ReplacementResearchResult:
    """Generate typed successor hypotheses from a persisted replacement request."""
    if request_store is not None:
        persisted = request_store.get(request.request_id)
        if persisted is None:
            raise ValueError(f"research request is not persisted: {request.request_id}")
        if persisted != request:
            raise ValueError(f"research request mismatch: {request.request_id}")
    candidates = (generator or StrategyGenerator()).generate(request, symbols)
    if not candidates:
        raise ValueError(f"no replacement candidates generated: {request.request_id}")
    return ReplacementResearchResult(request=request, candidates=candidates)
