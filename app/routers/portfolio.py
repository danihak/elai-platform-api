"""portfolio.py — screen S1. Book-level confidence mix and the farm table."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter

from elai_confidence_core import BLIND, DEGRADED, VERIFIED, derive

from ..seed import yield_estimate
from ..store.repository import repo
from ._shapes import MetricOut

router = APIRouter(prefix="/v1/portfolio", tags=["portfolio"])


@router.get("", summary="Portfolio view with the confidence mix")
def portfolio(crop: Optional[str] = None, district: Optional[str] = None,
              client_id: Optional[str] = None, state: Optional[str] = None) -> dict:
    """The confidence mix is weighted by AREA, not by farm count.

    A hundred small Blind plots and one large Blind plot are different problems,
    and a farm-count bar hides that. LenderBook does the same roll-up weighted by
    rupees.
    """
    farms = repo.farms(crop=crop, district=district, client_id=client_id)

    rows: List[dict] = []
    mix_area = {VERIFIED: 0.0, DEGRADED: 0.0, BLIND: 0.0}
    mix_count = {VERIFIED: 0, DEGRADED: 0, BLIND: 0}
    total_area = 0.0

    for f in farms:
        latest = repo.latest(f.farm_id)
        if latest is None:
            continue
        if state and latest.state.lower() != state.lower():
            continue

        point = yield_estimate(f, repo.raw_observations(f.farm_id)[-1])
        y = derive("predicted_productivity", point, [latest], unit="MT/ha")

        mix_area[latest.state] += f.area_ha
        mix_count[latest.state] += 1
        total_area += f.area_ha

        rows.append({
            "farm_id": f.farm_id,
            "farm_name": f.farm_name,
            "farmer_name": f.farmer_name,
            "district": f.district,
            "crop": f.crop,
            "area_ha": f.area_ha,
            "das": latest.das,
            "stage": latest.stage,
            "state": latest.state,
            "cloudy": latest.cloudy,
            "days_since_last_clear_optical": latest.days_since_last_clear_optical,
            "last_verified_date": latest.last_verified_date.isoformat() if latest.last_verified_date else None,
            "rung_label": latest.rung_label,
            # The field supervisor's question is spatial — "where do I go
            # today" — and it cannot be answered without boundaries. A cluster
            # of blind farms in two mandals is one trip, not twelve, and that
            # is only visible on a map.
            "polygon": [[round(x, 6), round(y, 6)] for x, y in f.polygon],
            "centroid": [
                round(sum(x for x, _ in f.polygon) / len(f.polygon), 6),
                round(sum(y for _, y in f.polygon) / len(f.polygon), 6),
            ],
            "yield": MetricOut.of(y),
        })

    return {
        "data_source": getattr(repo, "data_source", "unknown"),
        "farm_count": len(rows),
        "total_area_ha": round(total_area, 2),
        "confidence_mix_by_area": {
            k: round(v / total_area, 4) if total_area else 0.0 for k, v in mix_area.items()
        },
        "confidence_mix_by_count": mix_count,
        "rows": rows,
    }
