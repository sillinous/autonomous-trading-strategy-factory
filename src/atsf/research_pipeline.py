from __future__ import annotations

from dataclasses import dataclass

from .population import Candidate
from .research_executor import ResearchWorkItem, materialize_research_work, spawn_research_candidates
from .research_planner import ResearchPlan
from .research_queue import ResearchQueue


@dataclass(frozen=True)
class ResearchExecutionResult:
    work_items: tuple[ResearchWorkItem, ...]
    candidates: tuple[Candidate, ...]


def execute_research_plan(
    plan: ResearchPlan,
    population: tuple[Candidate, ...],
    *,
    seed: int = 0,
    queue: ResearchQueue | None = None,
) -> ResearchExecutionResult:
    """Materialize and execute a research plan without bypassing downstream gates.

    Candidate spawning is intentionally the terminal step of this boundary. The
    returned candidates must still pass the normal evaluation/validation,
    robustness, promotion, and paper-trading gates before becoming successors.
    """
    if not isinstance(seed, int):
        raise TypeError("research seed must be an integer")
    work_items = materialize_research_work(plan, queue=queue)
    spawned: list[Candidate] = []
    seen = {candidate.strategy_id for candidate in population}
    for offset, work_item in enumerate(work_items):
        candidates = spawn_research_candidates(
            work_item,
            population,
            seed=seed + offset,
        )
        for candidate in candidates:
            if candidate.strategy_id in seen:
                continue
            seen.add(candidate.strategy_id)
            spawned.append(candidate)
    return ResearchExecutionResult(
        work_items=work_items,
        candidates=tuple(spawned),
    )
