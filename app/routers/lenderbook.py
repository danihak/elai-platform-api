"""
lenderbook.py — book-level roll-up and the harvest-timed call list.

The rule this module enforces: Blind does not mean risky. It means unknown.
Blind exposure is reported separately from risk and never feeds a risk band.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

from fastapi import APIRouter

from elai_confidence_core import BLIND, DEGRADED, VERIFIED

from ..store.repository import AS_OF, repo

router = APIRouter(prefix="/v1/lenderbook", tags=["lenderbook"])

#: Stand-in loan book. In production this is the lender's own upload, matched to
#: plots with a stored match tier.
LOANS = {
    "TS-MZ-0044": {"outstanding_inr": 145_000, "due_date": date(2026, 10, 20), "match_tier": "strong"},
    "TS-MZ-0051": {"outstanding_inr": 232_000, "due_date": date(2026, 10, 15), "match_tier": "exact"},
    "TS-MZ-0063": {"outstanding_inr": 98_000, "due_date": date(2026, 11, 5), "match_tier": "medium"},
    "TS-CT-0107": {"outstanding_inr": 310_000, "due_date": date(2026, 12, 18), "match_tier": "exact"},
    "TS-CT-0119": {"outstanding_inr": 187_000, "due_date": date(2026, 12, 10), "match_tier": "strong"},
    "TS-CT-0126": {"outstanding_inr": 265_000, "due_date": date(2026, 12, 22), "match_tier": "weak"},
}


@router.get("/book", summary="Book view with exposure by confidence state")
def book(client_id: Optional[str] = None) -> dict:
    """Exposure mix is weighted by RUPEES, not farm count.

    Unmatched plots are excluded from every figure and reported separately rather
    than quietly dropped, which is what most vendors do.
    """
    exposure = {VERIFIED: 0.0, DEGRADED: 0.0, BLIND: 0.0}
    rows: List[dict] = []
    excluded = []

    for f in repo.farms(client_id=client_id):
        loan = LOANS.get(f.farm_id)
        if not loan:
            excluded.append({"farm_id": f.farm_id, "reason": "unmatched"})
            continue
        latest = repo.latest(f.farm_id)
        if latest is None:
            excluded.append({"farm_id": f.farm_id, "reason": "no observations"})
            continue

        exposure[latest.state] += loan["outstanding_inr"]
        rows.append({
            "farm_id": f.farm_id,
            "borrower": f.farmer_name,
            "district": f.district,
            "crop": f.crop,
            "outstanding_inr": loan["outstanding_inr"],
            "due_date": loan["due_date"].isoformat(),
            "predicted_harvest_date": f.predicted_harvest_date.isoformat(),
            "days_due_before_harvest": (loan["due_date"] - f.predicted_harvest_date).days,
            "confidence_state": latest.state,
            "match_tier": loan["match_tier"],
            "last_verified_date": latest.last_verified_date.isoformat() if latest.last_verified_date else None,
        })

    total = sum(exposure.values())
    return {
        "as_of": AS_OF.isoformat(),
        "borrowers": len(rows),
        "total_exposure_inr": round(total, 2),
        "exposure_by_state_inr": {k: round(v, 2) for k, v in exposure.items()},
        "exposure_by_state_share": {
            k: round(v / total, 4) if total else 0.0 for k, v in exposure.items()
        },
        "excluded_from_figures": excluded,
        "rule": (
            "Blind means unknown, not risky. It never reduces a score, never holds "
            "a disbursement on its own, and never appears in a risk band. It "
            "triggers verification."
        ),
        "rows": rows,
    }


@router.get("/call-list", summary="Borrowers to call, ranked by harvest window")
def call_list(client_id: Optional[str] = None) -> dict:
    """Ranked by when the crop is ready, not by an abstract risk score.

    The officer's question is "who do I call this week", not "who is riskiest".
    """
    rows = []
    for f in repo.farms(client_id=client_id):
        loan = LOANS.get(f.farm_id)
        latest = repo.latest(f.farm_id)
        if not loan or latest is None:
            continue
        gap = (loan["due_date"] - f.predicted_harvest_date).days
        rows.append({
            "farm_id": f.farm_id,
            "borrower": f.farmer_name,
            "mobile_masked": f.mobile_masked,
            "outstanding_inr": loan["outstanding_inr"],
            "predicted_harvest_date": f.predicted_harvest_date.isoformat(),
            "due_date": loan["due_date"].isoformat(),
            "due_before_harvest": gap < 0,
            "harvest_confidence": latest.state,
            "action": (
                "Call now: loan matures before the predicted harvest window"
                if gap < 0 else "Call in the harvest week"
            ),
            "note": (
                "Harvest date is Blind this week; treat the window as indicative"
                if latest.state == BLIND else None
            ),
        })
    rows.sort(key=lambda r: r["predicted_harvest_date"])
    return {"as_of": AS_OF.isoformat(), "count": len(rows), "rows": rows}
