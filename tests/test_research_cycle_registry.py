from __future__ import annotations

import json
import sqlite3

import pytest

from atsf.research_cycle_registry import ResearchCycleRegistry


def test_cycle_registry_is_durable_and_ordered() -> None:
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)

    first = registry.save_cycle(
        "cycle:b",
        2,
        plan={"requests": ["r2"]},
        feedback={"signals": ["s2"]},
        admissions={"strategy_ids": ["c2"]},
    )
    registry.save_cycle(
        "cycle:a",
        1,
        plan={"requests": ["r1"]},
        feedback={"signals": ["s1"]},
        admissions={"strategy_ids": ["c1"]},
    )

    assert registry.get_cycle("cycle:b") == first
    assert [item.cycle_id for item in registry.list_cycles()] == ["cycle:a", "cycle:b"]
    assert json.loads(first.plan_json) == {"requests": ["r2"]}


def test_identical_replay_is_idempotent_but_changed_replay_fails_closed() -> None:
    connection = sqlite3.connect(":memory:")
    registry = ResearchCycleRegistry(connection)
    kwargs = {
        "plan": {"requests": ["r"]},
        "feedback": {"signals": ["s"]},
        "admissions": {"strategy_ids": ["c"]},
    }

    first = registry.save_cycle("cycle:1", 1, **kwargs)
    assert registry.save_cycle("cycle:1", 1, **kwargs) == first

    with pytest.raises(ValueError, match="immutable"):
        registry.save_cycle("cycle:1", 1, plan={"requests": ["changed"]}, **{k: v for k, v in kwargs.items() if k != "plan"})


def test_cycle_validation_fails_closed() -> None:
    registry = ResearchCycleRegistry(sqlite3.connect(":memory:"))

    with pytest.raises(ValueError, match="cycle_id"):
        registry.save_cycle("", 0, plan={}, feedback={}, admissions={})
    with pytest.raises(ValueError, match="nonnegative"):
        registry.save_cycle("cycle", -1, plan={}, feedback={}, admissions={})
