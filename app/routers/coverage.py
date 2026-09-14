"""coverage.py — blind-days accounting and the economic escalation rule.

This is what turns an SLA from a sentence into a contract. The promise is not
accuracy; it is a valid observation every N days, or we tell you.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from elai_confidence_core import BLIND, DEGRADED, VERIFIED
from elai_confidence_core.estimators import (
    NISAR_L, NISAR_S, SENTINEL1, effective_radar_revisit,
)

from ..seed import REVISIT_DAYS
from ..store.repository import repo

router = APIRouter(prefix="/v1/coverage", tags=["coverage"])

SLA_TIERS = {
    "standard": {"max_days_between_valid": 10, "blind_days_budget": 30, "escalation": False},
    "assured": {"max_days_between_valid": 7, "blind_days_budget": 18, "escalation": False},
    "premium": {"max_days_between_valid": 5, "blind_days_budget": 10, "escalation": True},
}

TASKING_COST_INR_PER_FARM = 45.0  # TBD — indicative, not a quote


def _blind_days(series) -> int:
    """Days in the season with no decision-grade observation."""
    if not series:
        return 0
    days = 0
    for a, b in zip(series, series[1:]):
        if a.state == BLIND:
            days += (b.obs_date - a.obs_date).days
    if series[-1].state == BLIND:
        # The season has not ended, so the trailing blind stretch runs at least
        # until the next scheduled pass.
        days += REVISIT_DAYS
    return days


def _farm_coverage(farm_id: str) -> Optional[dict]:
    f = repo.farm(farm_id)
    if not f:
        return None
    series = repo.scored(farm_id)
    if not series:
        return None

    tier = SLA_TIERS.get(f.sla_tier, SLA_TIERS["standard"])
    blind = _blind_days(series)
    latest = series[-1]

    in_breach = (
        blind > tier["blind_days_budget"]
        or latest.days_since_last_clear_optical > tier["max_days_between_valid"]
    )
    escalate = bool(tier["escalation"] and in_breach and latest.state != VERIFIED)

    counts = {VERIFIED: 0, DEGRADED: 0, BLIND: 0}
    for s in series:
        counts[s.state] += 1

    return {
        "farm_id": farm_id,
        "sla_tier": f.sla_tier,
        "promise": f"A valid observation at least every {tier['max_days_between_valid']} days, or we tell you the field is Blind and for how long.",
        "observations": len(series),
        "state_counts": counts,
        "blind_days_this_season": blind,
        "blind_days_budget": tier["blind_days_budget"],
        "days_since_last_valid": latest.days_since_last_clear_optical,
        "in_breach": in_breach,
        "escalation_available": tier["escalation"],
        "escalation_recommended": escalate,
        "escalation_cost_inr": TASKING_COST_INR_PER_FARM if escalate else None,
        "escalation_rationale": (
            f"Blind {blind} days against a {tier['blind_days_budget']} day budget on a "
            f"{f.sla_tier} tier. Tasking is an economic trigger, not a technical default."
        ) if escalate else None,
    }


@router.get("", summary="Coverage and SLA status across the set")
def coverage_all() -> dict:
    farms = repo.farms()
    rows = [c for c in (_farm_coverage(f.farm_id) for f in farms) if c]
    breaching = [r for r in rows if r["in_breach"]]
    escalations = [r for r in rows if r["escalation_recommended"]]

    return {
        "farms": len(rows),
        "in_breach": len(breaching),
        "escalations_recommended": len(escalations),
        "escalation_cost_if_all_approved_inr": round(
            sum(r["escalation_cost_inr"] or 0 for r in escalations), 2),
        "radar_sources_available": [
            {
                "key": s.key, "label": s.label, "band": s.band,
                "repeat_days": s.repeat_days,
                "publication_latency_hours": s.publication_latency_hours,
                "marginal_cost_inr": s.marginal_cost_inr,
                "access": s.access, "notes": s.notes,
            }
            for s in (SENTINEL1, NISAR_S, NISAR_L)
        ],
        "effective_radar_revisit_days": round(
            effective_radar_revisit([SENTINEL1, NISAR_S, NISAR_L]), 1),
        "revisit_caveat": (
            "Approximate. NISAR publishes daily PROCESSED products; the revisit is "
            "about 12 days. Daily coverage of every farm would be an over-claim."
        ),
        "rows": rows,
    }


@router.get("/{farm_id}", summary="Coverage and SLA status for one farm")
def coverage_one(farm_id: str) -> dict:
    c = _farm_coverage(farm_id)
    if not c:
        raise HTTPException(404, "farm not found")
    return c
