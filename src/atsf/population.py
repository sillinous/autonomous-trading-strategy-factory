from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from random import Random

from .generator import mutate_indicator_period, mutate_threshold
from .lineage import LineageRecord
from .strategy import StrategySpec


@dataclass(frozen=True)
class Candidate:
    strategy: StrategySpec
    strategy_id: str
    lineage: LineageRecord


def strategy_id(strategy: StrategySpec) -> str:
    """Return a compact stable identity for the canonical strategy definition."""
    payload = json.dumps(
        strategy.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def mutate_candidate(candidate: Candidate, rng: Random) -> Candidate:
    """Apply one constrained mutation and preserve genealogy."""
    mutation = rng.choice((mutate_indicator_period, mutate_threshold))
    strategy = mutation(candidate.strategy, rng)
    candidate_id = strategy_id(strategy)
    lineage = LineageRecord(
        strategy_id=candidate_id,
        generation=candidate.lineage.generation + 1,
        parent_ids=(candidate.strategy_id,),
        operator=mutation.__name__,
    )
    return Candidate(strategy=strategy, strategy_id=candidate_id, lineage=lineage)


def seed_population(strategies: list[StrategySpec]) -> list[Candidate]:
    """Create generation-zero candidates from trusted seed specifications."""
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for strategy in strategies:
        candidate_id = strategy_id(strategy)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidates.append(
            Candidate(
                strategy=strategy,
                strategy_id=candidate_id,
                lineage=LineageRecord(strategy_id=candidate_id, generation=0),
            )
        )
    return candidates
