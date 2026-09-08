from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .lifecycle import StrategyLifecycle, StrategyState
from .registry import ExperimentRegistry


class ControlAction(StrEnum):
    ACTIVATE = "activate"
    DEGRADE = "degrade"
    HALT = "halt"


@dataclass(frozen=True)
class StrategyControlRecord:
    strategy_id: str
    state: StrategyState
    reason: str
    sequence: int


class StrategyControlPlane:
    """Persistent control-plane facade over strategy lifecycle state."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        self.registry = registry
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.registry._connection.execute(
            """CREATE TABLE IF NOT EXISTS strategy_control (
                strategy_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                reason TEXT NOT NULL,
                sequence INTEGER NOT NULL
            )"""
        )
        self.registry._connection.execute(
            """CREATE TABLE IF NOT EXISTS strategy_control_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                previous_state TEXT,
                current_state TEXT NOT NULL,
                reason TEXT NOT NULL
            )"""
        )
        self.registry._connection.commit()

    def record(self, strategy_id: str, state: StrategyState, reason: str) -> StrategyControlRecord:
        row = self.registry._connection.execute(
            "SELECT state, sequence FROM strategy_control WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        previous = row["state"] if row else None
        sequence = int(row["sequence"]) + 1 if row else 1
        if previous == StrategyState.HALTED.value and state != StrategyState.HALTED:
            raise PermissionError("halted strategies cannot be reactivated automatically")
        self.registry._connection.execute(
            """INSERT OR REPLACE INTO strategy_control(strategy_id, state, reason, sequence)
            VALUES (?, ?, ?, ?)""",
            (strategy_id, state.value, reason, sequence),
        )
        self.registry._connection.execute(
            """INSERT INTO strategy_control_events
            (strategy_id, previous_state, current_state, reason)
            VALUES (?, ?, ?, ?)""",
            (strategy_id, previous, state.value, reason),
        )
        self.registry._connection.commit()
        return StrategyControlRecord(strategy_id, state, reason, sequence)

    def get(self, strategy_id: str) -> StrategyControlRecord | None:
        row = self.registry._connection.execute(
            "SELECT * FROM strategy_control WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        if row is None:
            return None
        return StrategyControlRecord(
            strategy_id, StrategyState(row["state"]), row["reason"], row["sequence"]
        )

    def events(self, strategy_id: str) -> list[dict[str, object]]:
        rows = self.registry._connection.execute(
            """SELECT previous_state, current_state, reason
            FROM strategy_control_events WHERE strategy_id = ? ORDER BY event_id""",
            (strategy_id,),
        ).fetchall()
        return [dict(row) for row in rows]
