import pytest

from atsf.control_plane import StrategyControlPlane
from atsf.lifecycle import StrategyState
from atsf.registry import ExperimentRegistry


def test_control_plane_persists_strategy_state_and_events():
    registry = ExperimentRegistry()
    plane = StrategyControlPlane(registry)
    first = plane.record("strategy-1", StrategyState.ACTIVE, "paper approved")
    second = plane.record("strategy-1", StrategyState.DEGRADED, "drawdown threshold")
    assert first.sequence == 1
    assert second.sequence == 2
    assert plane.get("strategy-1").state == StrategyState.DEGRADED
    assert len(plane.events("strategy-1")) == 2
    registry.close()


def test_halted_strategy_cannot_be_reactivated_automatically():
    registry = ExperimentRegistry()
    plane = StrategyControlPlane(registry)
    plane.record("strategy-1", StrategyState.HALTED, "risk breach")
    with pytest.raises(PermissionError, match="halted"):
        plane.record("strategy-1", StrategyState.ACTIVE, "new signal")
    registry.close()
