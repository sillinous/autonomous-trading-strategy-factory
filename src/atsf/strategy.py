from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IndicatorKind = Literal["sma", "ema", "rsi"]


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class Comparator(str, Enum):
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


class Indicator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: IndicatorKind | None = None
    source: str = "close"
    period: int | None = Field(default=None, gt=0)
    parameters: dict[str, float | int | str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_kind(cls, values: object) -> object:
        if isinstance(values, dict) and values.get("kind") is None:
            name = values.get("name")
            if name in {"sma", "ema", "rsi"}:
                values = dict(values)
                values["kind"] = name
        return values

    @model_validator(mode="after")
    def validate_kind(self) -> Indicator:
        if self.kind is None:
            raise ValueError("indicator kind is required when name is not a built-in indicator name")
        return self


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: str = Field(min_length=1)
    comparator: Comparator
    right: float | int | str


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    all: list[Condition] = Field(default_factory=list)
    any: list[Condition] = Field(default_factory=list)

    @model_validator(mode="after")
    def has_conditions(self) -> Signal:
        if not self.all and not self.any:
            raise ValueError("signal must contain at least one condition")
        return self


class PositionSizing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["fixed_fraction", "equal_weight", "volatility_target"]
    value: float = Field(gt=0)
    max_position: float = Field(default=1.0, gt=0, le=1)


class RiskLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_leverage: float = Field(default=1.0, gt=0)
    max_position: float = Field(default=1.0, gt=0, le=1)
    stop_loss: float | None = Field(default=None, gt=0, lt=1)
    max_drawdown: float | None = Field(default=None, gt=0, lt=1)


class StrategySpec(BaseModel):
    """Serializable, execution-neutral strategy definition."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    universe: list[str] = Field(min_length=1)
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d"] = "1d"
    indicators: list[Indicator] = Field(default_factory=list)
    entry: Signal
    exit: Signal
    side: Side = Side.LONG
    position_sizing: PositionSizing
    risk: RiskLimits = Field(default_factory=RiskLimits)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_risk_consistency(self) -> StrategySpec:
        if self.risk.max_position > self.position_sizing.max_position:
            raise ValueError("risk.max_position cannot exceed position_sizing.max_position")
        names = [indicator.name for indicator in self.indicators]
        if len(names) != len(set(names)):
            raise ValueError("indicator names must be unique")
        return self
