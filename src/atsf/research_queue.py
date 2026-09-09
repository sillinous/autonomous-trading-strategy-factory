from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ResearchReason(StrEnum):
    DEGRADED = "degraded"
    HALTED = "halted"
    DIVERSIFICATION = "diversification"
    CAPACITY = "capacity"


@dataclass(frozen=True)
class ResearchRequest:
    request_id: str
    source_strategy_id: str | None
    reason: ResearchReason
    priority: int
    constraints: tuple[str, ...] = ()


class ResearchQueue:
    """Deterministic queue for autonomous strategy replacement research."""

    def __init__(self) -> None:
        self._items: dict[str, ResearchRequest] = {}

    def enqueue(self, request: ResearchRequest) -> ResearchRequest:
        if not request.request_id:
            raise ValueError("request_id cannot be empty")
        if request.priority < 0:
            raise ValueError("priority cannot be negative")
        existing = self._items.get(request.request_id)
        if existing is None or request.priority < existing.priority:
            self._items[request.request_id] = request
        return self._items[request.request_id]

    def pending(self) -> tuple[ResearchRequest, ...]:
        return tuple(
            sorted(
                self._items.values(),
                key=lambda item: (item.priority, item.request_id),
            )
        )

    def complete(self, request_id: str) -> ResearchRequest:
        try:
            return self._items.pop(request_id)
        except KeyError as exc:
            raise KeyError(f"unknown research request: {request_id}") from exc

    def __len__(self) -> int:
        return len(self._items)
