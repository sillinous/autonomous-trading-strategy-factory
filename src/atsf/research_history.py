from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import TYPE_CHECKING

from .genome import genome_distance
from .population import Candidate

if TYPE_CHECKING:
    from .research_cycle import ResearchCycleResult


@dataclass(frozen=True)
class GenerationRecord:
    """Immutable control-plane record for one completed research generation."""

    generation: int
    candidate_count: int
    selected_count: int
    promoted_count: int
    best_fitness: float
    mean_fitness: float
    mean_genome_distance: float
    min_genome_distance: float
    max_genome_distance: float
    unique_strategy_count: int
    new_strategy_count: int
    best_strategy_id: str
    generations_without_improvement: int


@dataclass(frozen=True)
class ResearchHistory:
    """Deterministic, persistence-agnostic history of research generations."""

    records: tuple[GenerationRecord, ...] = ()

    @property
    def latest(self) -> GenerationRecord | None:
        return self.records[-1] if self.records else None

    @property
    def best_fitness(self) -> float | None:
        if not self.records:
            return None
        return max(record.best_fitness for record in self.records)

    @property
    def generations_without_improvement(self) -> int:
        return self.latest.generations_without_improvement if self.latest else 0

    def record_generation(
        self,
        result: ResearchCycleResult,
        population: list[Candidate],
        *,
        improvement_epsilon: float = 0.0,
    ) -> ResearchHistory:
        """Append an auditable generation record without mutating prior history."""
        if improvement_epsilon < 0 or not isfinite(improvement_epsilon):
            raise ValueError("improvement_epsilon must be finite and non-negative")
        if not population:
            raise ValueError("population must not be empty")
        if result.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.records and result.generation <= self.latest.generation:
            raise ValueError("generation must increase monotonically")

        ids = [candidate.strategy_id for candidate in population]
        if len(set(ids)) != len(ids):
            raise ValueError("population strategy IDs must be unique")

        distances = [
            genome_distance(left.strategy, right.strategy)
            for left, right in combinations(population, 2)
        ]
        if distances:
            mean_distance = sum(distances) / len(distances)
            minimum_distance = min(distances)
            maximum_distance = max(distances)
        else:
            mean_distance = minimum_distance = maximum_distance = 0.0

        best = max(
            result.evaluations,
            key=lambda evaluation: (evaluation.fitness.score, evaluation.candidate_id),
        )
        parent_ids = {candidate.strategy_id for candidate in result.selected_parents}
        new_strategy_count = sum(1 for candidate_id in ids if candidate_id not in parent_ids)

        previous_best = self.best_fitness
        if previous_best is None or result.metrics.best_fitness > previous_best + improvement_epsilon:
            stagnation = 0
        else:
            stagnation = self.generations_without_improvement + 1

        record = GenerationRecord(
            generation=result.generation,
            candidate_count=len(population),
            selected_count=result.metrics.selected_count,
            promoted_count=result.metrics.promoted_count,
            best_fitness=float(result.metrics.best_fitness),
            mean_fitness=float(result.metrics.mean_fitness),
            mean_genome_distance=mean_distance,
            min_genome_distance=minimum_distance,
            max_genome_distance=maximum_distance,
            unique_strategy_count=len(set(ids)),
            new_strategy_count=new_strategy_count,
            best_strategy_id=best.candidate_id,
            generations_without_improvement=stagnation,
        )
        return ResearchHistory(records=self.records + (record,))

    def is_stagnating(self, threshold: int) -> bool:
        if threshold < 1:
            raise ValueError("threshold must be positive")
        return self.generations_without_improvement >= threshold
