"""
_shapes.py — response models shared across routers.

Serialisation lives here, in the API, not in the domain library. That separation
is what keeps elai-confidence-core embeddable in ELAI's existing stack.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from elai_confidence_core import RenderedMetric, ScoredObservation


class MetricOut(BaseModel):
    metric: str
    state: str
    display: str = Field(..., description="Render this verbatim. Do not reformat.")
    value: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    unit: Optional[str] = None
    note: Optional[str] = None
    last_verified_date: Optional[str] = None
    last_verified_das: Optional[int] = None
    rule_version: str

    @classmethod
    def of(cls, m: RenderedMetric) -> "MetricOut":
        return cls(
            metric=m.metric, state=m.state, display=m.display, value=m.value,
            low=m.low, high=m.high, unit=m.unit, note=m.note,
            last_verified_date=m.last_verified_date.isoformat() if m.last_verified_date else None,
            last_verified_das=m.last_verified_das, rule_version=m.rule_version,
        )


class ObservationOut(BaseModel):
    obs_date: str
    das: int
    stage: str
    state: str
    rung_used: str
    rung_label: str
    source: str
    ndvi: Optional[float] = None
    ndvi_uncertainty: Optional[float] = None
    valid_pixel_fraction: float
    cloudy: bool
    days_since_last_clear_optical: int
    suppressed_indices: List[str] = []
    reason: str
    marginal_cost_inr: float
    rule_version: str

    @classmethod
    def of(cls, s: ScoredObservation) -> "ObservationOut":
        return cls(
            obs_date=s.obs_date.isoformat(), das=s.das, stage=s.stage, state=s.state,
            rung_used=s.rung_used, rung_label=s.rung_label, source=s.source,
            ndvi=s.ndvi, ndvi_uncertainty=s.ndvi_uncertainty,
            valid_pixel_fraction=s.valid_pixel_fraction, cloudy=s.cloudy,
            days_since_last_clear_optical=s.days_since_last_clear_optical,
            suppressed_indices=s.suppressed_indices, reason=s.reason,
            marginal_cost_inr=s.marginal_cost_inr, rule_version=s.rule_version,
        )


class FarmOut(BaseModel):
    farm_id: str
    farm_name: str
    farmer_name: str
    mobile_masked: str = Field(..., description="DPDP: full numbers are never returned.")
    country: str
    state: str
    district: str
    mandal: str
    area_ha: float
    lat: float
    lon: float
    polygon: List[List[float]]
    crop: str
    variety: str
    season: str
    irrigation_type: str
    sowing_date: str
    sowing_date_source: str
    boundary_source: str
    predicted_harvest_date: str
    client_id: str
    client_tier: str
    sla_tier: str
