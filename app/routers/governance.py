"""Sign-off endpoints. Krupa's "who signs off", as a working surface."""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.signoff import (REQUIRED, BlastRadius, Evidence, Proposal, Role,
                                Stage, new_proposal)

router = APIRouter(prefix="/v1/governance", tags=["governance"])

STORE = pathlib.Path(__file__).parent.parent.parent / "data" / "proposals.json"


def _load() -> Dict[str, dict]:
    if not STORE.exists():
        return {}
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save(blob: Dict[str, dict]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(blob, indent=1, default=str), encoding="utf-8")


def _rehydrate(d: dict) -> Proposal:
    from dataclasses import fields as dc_fields
    from ..services.signoff import Approval

    ev = Evidence(**d["evidence"])
    bl = BlastRadius(**d["blast"])
    p = Proposal(
        proposal_id=d["proposal_id"], kind=d["kind"], title=d["title"],
        proposed_by=d["proposed_by"], proposed_at=d["proposed_at"],
        rule_version_from=d["rule_version_from"],
        rule_version_to=d["rule_version_to"],
        diff=d["diff"], evidence=ev, blast=bl, stage=Stage(d["stage"]),
        history=d.get("history", []),
    )
    p.approvals = [Approval(role=Role(a["role"]), approver=a["approver"],
                            at=a["at"], comment=a.get("comment", ""))
                   for a in d.get("approvals", [])]
    return p


class EvidenceIn(BaseModel):
    validation_run_id: str
    measured_at: str
    holdout_condition: str
    n_splits: int
    rmse: Optional[float] = None
    rmse_sd: Optional[float] = None
    rmse_upper: Optional[float] = None
    ceiling: Optional[float] = None
    notes: str = ""


class BlastIn(BaseModel):
    farms_affected: int = 0
    farms_changing_state: int = 0
    farms_losing_a_number: int = 0
    exposure_re_rated_inr: float = 0.0
    clients_affected: List[str] = Field(default_factory=list)


class ProposalIn(BaseModel):
    kind: str
    title: str
    proposed_by: str
    rule_version_from: str
    rule_version_to: str
    diff: Dict[str, object]
    evidence: EvidenceIn
    blast: BlastIn


class ApprovalIn(BaseModel):
    role: str
    approver: str
    comment: str = ""


@router.get("/roles", summary="Who must approve what")
def roles() -> dict:
    """The matrix, exposed so nobody has to read the source to know.

    No single person can carry a change end to end. A code review asks whether
    a diff is correct; this asks whether the world changes.
    """
    return {
        "matrix": {k: [r.value for r in v] for k, v in REQUIRED.items()},
        "rules": [
            "the proposer cannot approve their own change in any role",
            "one person cannot hold two roles on one proposal",
            "a proposal whose validation is missing, stale, or measured on a "
            "mixed hold-out cannot be approved by anyone",
            "a change that takes a number away from a client also needs the "
            "agronomist, whatever its category",
            "the release path is approved -> shadow -> limited -> released, "
            "and no stage may be skipped",
        ],
    }


@router.get("", summary="All proposals")
def list_proposals(stage: Optional[str] = None) -> dict:
    blob = _load()
    items = [v for v in blob.values() if stage is None or v["stage"] == stage]
    return {"count": len(items), "proposals": items}


@router.post("", summary="Raise a change for review", status_code=201)
def create(body: ProposalIn) -> dict:
    p = new_proposal(
        kind=body.kind, title=body.title, proposed_by=body.proposed_by,
        rule_version_from=body.rule_version_from,
        rule_version_to=body.rule_version_to, diff=body.diff,
        evidence=Evidence(**body.evidence.model_dump()),
        blast=BlastRadius(**body.blast.model_dump()),
    )
    p.submit()
    blob = _load()
    blob[p.proposal_id] = p.audit_record()["proposal"]
    _save(blob)
    return {
        "proposal_id": p.proposal_id,
        "stage": p.stage.value,
        "required": [r.value for r in p.required_roles()],
        "outstanding": [r.value for r in p.outstanding()],
        # Surfaced before anyone is asked to sign. An approver who cannot see
        # why a proposal is unapprovable is being asked to rubber-stamp.
        "blockers": p.blockers(),
    }


@router.get("/{proposal_id}", summary="One proposal, with its audit record")
def get_one(proposal_id: str) -> dict:
    blob = _load()
    if proposal_id not in blob:
        raise HTTPException(404, "proposal not found")
    p = _rehydrate(blob[proposal_id])
    return p.audit_record()


@router.post("/{proposal_id}/approve", summary="Sign as one role")
def approve(proposal_id: str, body: ApprovalIn) -> dict:
    blob = _load()
    if proposal_id not in blob:
        raise HTTPException(404, "proposal not found")
    p = _rehydrate(blob[proposal_id])
    try:
        p.approve(Role(body.role), body.approver, body.comment)
    except ValueError as exc:
        # 409, not 400: the request is well formed, the STATE refuses it.
        raise HTTPException(409, str(exc))
    blob[proposal_id] = p.audit_record()["proposal"]
    _save(blob)
    return {"proposal_id": p.proposal_id, "stage": p.stage.value,
            "outstanding": [r.value for r in p.outstanding()]}


@router.post("/{proposal_id}/promote", summary="Move along the release path")
def promote(proposal_id: str, to: str, by: str) -> dict:
    blob = _load()
    if proposal_id not in blob:
        raise HTTPException(404, "proposal not found")
    p = _rehydrate(blob[proposal_id])
    try:
        p.promote(Stage(to), by)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    blob[proposal_id] = p.audit_record()["proposal"]
    _save(blob)
    return {"proposal_id": p.proposal_id, "stage": p.stage.value}
