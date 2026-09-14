"""feedback.py — the ground-truth ingress. Not an endpoint; a writer."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..store.repository import repo

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


class FeedbackIn(BaseModel):
    farm_id: str
    responder: str = Field("farmer", description="farmer | field_officer")
    confirms_flag: Optional[bool] = Field(
        None, description="A 'no' is worth more than a 'yes'. It is a labelled false positive."
    )
    note: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    photo_ref: Optional[str] = None
    actual_productivity_mt_ha: Optional[float] = None
    consent_research_use: bool = Field(
        True, description="If false, the reply still moves this farm's badge but is excluded from training."
    )


@router.post("", summary="Submit ground truth from a farmer or field officer")
def submit(fb: FeedbackIn) -> dict:
    f = repo.farm(fb.farm_id)
    if not f:
        raise HTTPException(404, "farm not found")

    before = repo.latest(fb.farm_id)
    record = fb.model_dump()
    record["obs_date"] = before.obs_date.isoformat() if before else ""
    record["das"] = before.das if before else 0
    repo.add_feedback(record)

    return {
        "accepted": True,
        "farm_id": fb.farm_id,
        "state_at_time_of_reply": before.state if before else None,
        "label_type": (
            "false_positive" if fb.confirms_flag is False
            else "confirmation" if fb.confirms_flag is True else "observation"
        ),
        "usable_for_training": fb.consent_research_use,
        "message": (
            "Recorded. A disagreement is more useful than a confirmation — it is a "
            "labelled false positive at a known crop and stage."
            if fb.confirms_flag is False else "Recorded."
        ),
        "ground_truth_records": len(repo.feedback(fb.farm_id)),
    }
