"""
trust.py — TrustLayer. The confidence state as a bare contract.

Sold to institutions that already buy a competitor's feed. Deliberately small:
six endpoints, everything else a query parameter.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from elai_confidence_core import CURRENT

from ..store.repository import AS_OF, repo

router = APIRouter(prefix="/v1/trust", tags=["trustlayer"])


class BatchRequest(BaseModel):
    farm_ids: List[str]


def _state_row(farm_id: str) -> dict:
    latest = repo.latest(farm_id)
    if latest is None:
        return {"farm_id": farm_id, "state": None, "error": "no observations"}
    return {
        "farm_id": farm_id,
        "state": latest.state,
        "as_of": latest.obs_date.isoformat(),
        "das": latest.das,
        "stage": latest.stage,
        "source": latest.source,
        "rung": latest.rung_used,
        "uncertainty_ndvi": latest.ndvi_uncertainty,
        "days_since_last_clear_optical": latest.days_since_last_clear_optical,
        "last_verified_date": latest.last_verified_date.isoformat() if latest.last_verified_date else None,
        "reason": latest.reason,
        "rule_version": latest.rule_version,
    }


@router.get("/state/{farm_id}", summary="Current state for one farm")
def state(farm_id: str) -> dict:
    if not repo.farm(farm_id):
        raise HTTPException(404, "farm not found")
    return _state_row(farm_id)


@router.post("/states:batch", summary="State for up to 500 farms")
def states_batch(req: BatchRequest) -> dict:
    if len(req.farm_ids) > 500:
        raise HTTPException(413, "batch limit is 500 farms")
    return {
        "as_of": AS_OF.isoformat(),
        "rule_version": CURRENT.version,
        "count": len(req.farm_ids),
        "rows": [_state_row(fid) for fid in req.farm_ids],
    }


@router.get("/events/sample", summary="Webhook payload shapes")
def event_samples() -> dict:
    """Sandbox fixtures for the six webhook events.

    Deterministic on purpose. An integrator cannot write a test against a live
    satellite feed, so without fixed fixtures nobody ever exercises the Blind
    branch until a monsoon hits with real money attached.
    """
    return {
        "delivery": "at least once, ordered per farm, HMAC signed",
        "retries": ["1s", "10s", "1m", "10m", "1h", "dead letter + alert"],
        "consumer_requirement": "idempotent on event_id",
        "events": [
            {"event": "state.changed", "farm_id": "TS-MZ-0044", "from": "Verified",
             "to": "Degraded", "reason": "optical unusable, estimated from radar"},
            {"event": "blind.entered", "farm_id": "TS-CT-0107", "days_expected": 9},
            {"event": "blind.exited", "farm_id": "TS-CT-0107", "blind_days": 11},
            {"event": "sla.breached", "farm_id": "TS-CT-0119", "tier": "premium",
             "blind_days": 12, "escalation_available": True},
            {"event": "groundtruth.received", "farm_id": "TS-MZ-0051",
             "responder": "farmer", "label_type": "false_positive"},
            {"event": "coverage.changed", "farm_id": "TS-MZ-0063",
             "observable": False, "reason": "plot unmatched"},
        ],
    }
