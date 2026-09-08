from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .strategy import StrategySpec


@dataclass(frozen=True)
class ExperimentSpec:
    strategy: StrategySpec
    dataset_id: str
    dataset_version: str
    seed: int = 0

    def canonical_payload(self) -> dict:
        return {
            "strategy": self.strategy.model_dump(mode="json"),
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "seed": self.seed,
        }

    @property
    def experiment_id(self) -> str:
        payload = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    status: str
    score: float | None = None
    reason: str | None = None
