from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LineageRecord:
    strategy_id: str
    generation: int
    parent_ids: tuple[str, ...] = ()
    operator: str = "seed"
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.strategy_id in self.parent_ids:
            raise ValueError(f"strategy lineage cycle detected: {self.strategy_id}")
        if self.operator == "seed" and self.parent_ids:
            raise ValueError("seed strategies cannot have parents")
        if self.operator != "seed" and not self.parent_ids:
            raise ValueError("derived strategies must have at least one parent")
