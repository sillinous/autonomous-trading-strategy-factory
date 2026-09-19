from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from .research_checkpoint import ResearchCheckpoint
from .research_checkpoint_store import ResearchCheckpointStore
from .research_cycle import ResearchCycleResult
from .research_cycle_registry import ResearchCycleRecord, ResearchCycleRegistry


@dataclass(frozen=True)
class ResearchControlPlaneRecord:
    """Verified durable pair of one research cycle and its restart checkpoint."""

    cycle: ResearchCycleRecord
    checkpoint: ResearchCheckpoint


class ResearchControlPlane:
    """Transactional coordinator for research-cycle and checkpoint persistence."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        cycle_registry: ResearchCycleRegistry | None = None,
        checkpoint_store: ResearchCheckpointStore | None = None,
    ) -> None:
        self._connection = connection
        self.cycle_registry = cycle_registry or ResearchCycleRegistry(connection)
        self.checkpoint_store = checkpoint_store or ResearchCheckpointStore(connection)
        if self.cycle_registry.connection is not connection or self.checkpoint_store.connection is not connection:
            raise ValueError("research control-plane stores must share a connection")

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def verify(self) -> None:
        """Verify both ledgers and their cross-store checkpoint bindings."""
        self.cycle_registry.verify()
        self.checkpoint_store.verify()
        checkpoints = self._connection.execute(
            "SELECT next_generation FROM research_checkpoints ORDER BY next_generation"
        ).fetchall()
        for (next_generation,) in checkpoints:
            checkpoint = self.checkpoint_store.load(int(next_generation))
            cycle_id = f"generation-{checkpoint.next_generation - 1}"
            cycle = self.cycle_registry.get_cycle(cycle_id)
            if cycle is None:
                raise ValueError("durable checkpoint references a missing research cycle")
            try:
                plan = json.loads(cycle.plan_json)
            except json.JSONDecodeError as exc:
                raise ValueError("durable research-cycle plan is invalid") from exc
            if plan.get("checkpoint_digest") != checkpoint.state_digest:
                raise ValueError("durable checkpoint does not match research-cycle evidence")

    def persist_generation(
        self,
        result: ResearchCycleResult,
        *,
        seed: int,
        checkpoint: ResearchCheckpoint,
    ) -> ResearchControlPlaneRecord:
        """Atomically persist a completed cycle and its exact restart checkpoint."""
        expected_next_generation = result.generation + 1
        if checkpoint.next_generation != expected_next_generation:
            raise ValueError("checkpoint generation does not follow research cycle")
        if checkpoint.state_digest == "":
            raise ValueError("checkpoint state digest is required")
        cycle_id = f"generation-{result.generation}"
        with self._connection:
            self.cycle_registry.verify()
            self.checkpoint_store.verify()
            self.cycle_registry.save_cycle_in_transaction(
                cycle_id,
                result.generation,
                plan={
                    "seed": seed,
                    "crossover_rate": result.metrics.crossover_rate,
                    "mutation_rate": result.metrics.mutation_rate,
                    "checkpoint_digest": checkpoint.state_digest,
                },
                feedback={
                    "candidate_count": result.metrics.candidate_count,
                    "promotion_eligible_count": result.metrics.promotion_eligible_count,
                    "selected_count": result.metrics.selected_count,
                    "promoted_count": result.metrics.promoted_count,
                    "best_fitness": result.metrics.best_fitness,
                    "mean_fitness": result.metrics.mean_fitness,
                    "stagnating": result.metrics.stagnating,
                },
                admissions={
                    "selected_strategy_ids": tuple(candidate.strategy_id for candidate in result.selected_parents),
                    "next_strategy_ids": tuple(candidate.strategy_id for candidate in result.next_population),
                },
                portfolio_feedback=None,
            )
            self.checkpoint_store.save_in_transaction(
                checkpoint, cycle_id=cycle_id
            )
        cycle = self.cycle_registry.get_cycle(cycle_id)
        if cycle is None:
            raise ValueError("persisted research cycle could not be reloaded")
        self.verify()
        return ResearchControlPlaneRecord(cycle=cycle, checkpoint=checkpoint)

    def latest_checkpoint(self) -> ResearchCheckpoint | None:
        self.verify()
        return self.checkpoint_store.latest()


    def persist_and_verify(
        self,
        result: ResearchCycleResult,
        *,
        seed: int,
        checkpoint: ResearchCheckpoint,
    ) -> ResearchControlPlaneRecord:
        """Persist one generation and immediately verify the complete durable state."""
        return self.persist_generation(
            result,
            seed=seed,
            checkpoint=checkpoint,
        )
