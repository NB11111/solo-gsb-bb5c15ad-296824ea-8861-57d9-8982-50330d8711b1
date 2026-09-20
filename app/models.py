"""Pydantic request/response schemas."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class IndicatorRange(BaseModel):
    """Allowed concentration interval for one indicator (mg/L)."""

    min: float = Field(..., ge=0, description="lower bound, mg/L")
    max: float = Field(..., ge=0, description="upper bound, mg/L")

    @model_validator(mode="after")
    def _ordered(self) -> "IndicatorRange":
        if self.min > self.max:
            raise ValueError("min must be <= max")
        return self


class Source(BaseModel):
    """One candidate water source."""

    id: str = Field(..., min_length=1, max_length=64, description="unique source id")
    max_volume: float = Field(..., ge=0, description="available volume, m3")
    cost: float = Field(..., ge=0, description="cost per m3")
    calcium: float = Field(..., ge=0, description="calcium concentration, mg/L")
    magnesium: float = Field(..., ge=0, description="magnesium concentration, mg/L")
    alkalinity: float = Field(..., ge=0, description="alkalinity, mg/L as CaCO3")
    batch_size: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "optional delivery batch size in m3; when set, the volume drawn "
            "from this source must be a non-negative integer multiple of it "
            "and not exceed max_volume; when omitted the volume is continuous"
        ),
    )
    minimum_run_volume: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "optional activation threshold in m3; when set, the volume drawn "
            "from this source is either 0 or at least this value; modeled "
            "with a binary enable variable inside the same MIP (never "
            "enforced after solving)"
        ),
    )

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v: Any) -> str:
        return str(v).strip()

    @classmethod
    def _finite_positive(cls, v: Any, field: str) -> Any:
        """Reject NaN/+inf/-inf explicitly and non-finite numeric strings.

        Pydantic's ``gt=0`` already excludes negatives, zero and null/non
        numeric input; this adds the finite-number requirement.
        """
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError(f"{field} must be a finite positive number")
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v  # let Pydantic produce its standard type error
        if not math.isfinite(f):
            raise ValueError(f"{field} must be a finite positive number")
        return f

    @field_validator("batch_size", mode="before")
    @classmethod
    def _coerce_batch_size(cls, v: Any) -> Any:
        return cls._finite_positive(v, "batch_size")

    @field_validator("minimum_run_volume", mode="before")
    @classmethod
    def _coerce_minimum_run_volume(cls, v: Any) -> Any:
        return cls._finite_positive(v, "minimum_run_volume")


class BlendRequest(BaseModel):
    """Blending problem definition."""

    target_volume: float = Field(..., gt=0, description="required blend volume, m3")
    calcium: Optional[IndicatorRange] = None
    magnesium: Optional[IndicatorRange] = None
    alkalinity: Optional[IndicatorRange] = None
    sources: List[Source] = Field(..., min_length=2, max_length=6)
    minimum_active_sources: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "require at least this many sources that declare a "
            "minimum_run_volume to be used at or above their threshold; "
            "must not exceed the number of such sources"
        ),
    )

    @field_validator("minimum_active_sources", mode="before")
    @classmethod
    def _coerce_minimum_active_sources(cls, v: Any) -> Any:
        # bool is an int subclass; reject it explicitly rather than silently
        # coercing true/false to 1/0.
        if isinstance(v, bool):
            raise ValueError("minimum_active_sources must be a non-negative integer")
        return v

    @model_validator(mode="after")
    def _unique_ids_and_active_count(self) -> "BlendRequest":
        ids = [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("source ids must be unique")
        gated = sum(1 for s in self.sources if s.minimum_run_volume is not None)
        if (
            self.minimum_active_sources is not None
            and self.minimum_active_sources > gated
        ):
            raise ValueError(
                f"minimum_active_sources ({self.minimum_active_sources}) must not "
                f"exceed the number of sources with a minimum_run_volume ({gated})"
            )
        return self


class PlanCreateRequest(BlendRequest):
    """Blend request plus the idempotency key used to persist the plan."""

    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=255)


class Allocation(BaseModel):
    source_id: str
    volume: float
    cost: float


class OptimalPlan(BaseModel):
    status: Literal["optimal"] = "optimal"
    plan_id: Optional[int] = None
    created_at: Optional[str] = None
    total_volume: float
    total_cost: float
    blend: List[Allocation]
    indicators: Dict[str, float]


class InfeasiblePlan(BaseModel):
    status: Literal["infeasible"] = "infeasible"
    plan_id: Optional[int] = None
    created_at: Optional[str] = None
    message: str
    achievable_ranges: Dict[str, Any]


class ContingencyAllocation(BaseModel):
    source_id: str
    original_volume: float
    new_volume: float
    volume_delta: float
    new_cap: float
    cost: float


class ContingencyPlan(BaseModel):
    """Optimal supply-reduction contingency re-allocation (not persisted)."""

    status: Literal["optimal"] = "optimal"
    plan_id: int
    total_volume: float
    total_cost: float
    cost_delta: float
    total_adjustment: float
    blend: List[ContingencyAllocation]
    indicators: Dict[str, float]


class ContingencyInfeasible(BaseModel):
    """No feasible allocation exists under the reduced supply caps."""

    status: Literal["infeasible"] = "infeasible"
    plan_id: int
    message: str
    achievable_ranges: Dict[str, Any]
