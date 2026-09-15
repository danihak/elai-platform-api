"""KisanVoice endpoints — the farmer surface and the ground-truth ingress."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.kisanvoice import (Channel, FarmerReply, Language,
                                   TWO_TAP_PROMPTS, compose, ingest_reply,
                                   should_message)
from ..store.repository import repo

router = APIRouter(prefix="/v1/kisanvoice", tags=["kisanvoice"])


@router.get("/queue", summary="Who gets a message this week, and why")
def queue(language: str = "te") -> dict:
    """The send list, with a reason on every line and every exclusion.

    Messaging every farmer every week trains them to ignore it, so the rule is
    that a message goes out when there is something worth saying or something
    worth asking — and being unable to see a field is worth saying.
    """
    out: List[dict] = []
    for farm in repo.farms():
        latest = repo.latest(farm.farm_id)
        if latest is None:
            continue
        days = latest.days_since_last_clear_optical
        send, why = should_message(latest.state, days, last_messaged_days=30)

        prompt = ("canopy_check" if latest.state in ("Blind", "Degraded")
                  else "sowing_confirm")
        entry = {
            "farm_id": farm.farm_id,
            "farmer_name": farm.farmer_name,
            "state": latest.state,
            "days_unseen": days,
            "send": send,
            "reason": why,
        }
        if send:
            msg = compose(
                farm_id=farm.farm_id, farmer_name=farm.farmer_name,
                state=latest.state, language=Language(language),
                days_unseen=days,
                last_seen=(latest.last_verified_date.strftime("%d %b")
                           if latest.last_verified_date else None),
                prompt_id=prompt, channel=Channel.WHATSAPP_VOICE,
                rule_version=latest.rule_version,
            )
            entry["message"] = msg.body
            entry["options"] = msg.prompt.options if msg.prompt else []
            entry["prompt_id"] = prompt
            # Asserted on every send rather than trusted. A jargon leak is the
            # kind of regression that survives review and reaches a farmer.
            entry["jargon_leaks"] = msg.leaks_jargon()
        out.append(entry)

    sending = [e for e in out if e["send"]]
    return {
        "language": language,
        "considered": len(out),
        "sending": len(sending),
        "rule": ("a message goes out when there is something worth saying or "
                 "something worth asking; a clear reading is neither"),
        "queue": out,
    }


@router.get("/prompts", summary="The two-tap questions, and why each exists")
def prompts() -> dict:
    """Every question, its options in three languages, and the label it writes.

    A question that produces no usable label does not earn a farmer's time, so
    each one states what it writes back and why that is worth asking.
    """
    return {
        "prompts": [
            {
                "prompt_id": p.prompt_id,
                "question": p.question,
                "options": p.options,
                "writes_label": p.writes,
                "why": p.why,
            }
            for p in TWO_TAP_PROMPTS.values()
        ]
    }


class ReplyIn(BaseModel):
    farm_id: str
    prompt_id: str
    option_id: str
    received: str
    language: str = "te"
    free_text: str = ""
    photo_url: Optional[str] = None
    gps: Optional[Dict[str, float]] = None


@router.post("/reply", summary="A farmer's tap becomes a training label")
def reply(body: ReplyIn) -> dict:
    """Ground truth in, and an explicit statement of what it does NOT change.

    A farmer reply is ground truth about the field. It is not a licence to
    upgrade a satellite observation: if the satellite could not see the field,
    it still could not see the field. What changes is that an independent label
    now exists for that date — which is worth more than the observation would
    have been.
    """
    latest = repo.latest(body.farm_id)
    if latest is None:
        raise HTTPException(404, "farm not found or has no observations")

    out = ingest_reply(
        FarmerReply(
            farm_id=body.farm_id, prompt_id=body.prompt_id,
            option_id=body.option_id, received=body.received,
            language=Language(body.language), free_text=body.free_text,
            photo_url=body.photo_url, gps=body.gps,
        ),
        current_state=latest.state,
    )
    if not out["accepted"]:
        raise HTTPException(422, out["reason"])
    return out
