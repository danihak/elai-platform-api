"""
signoff.py — who signs off, and what they are signing.

KRUPA'S QUESTION, LITERALLY. "What counts as validated before an index or model
is trusted enough to show a client, who signs off?" The first half was answered
by the validation gate. This is the second half, and until now the answer was
nobody: thresholds changed by editing rules.py and pushing.

WHAT A CHANGE HAS TO CLEAR. Four roles, and no single person can carry a change
end to end:

  DS lead        owns statistical validity. Did it clear the ceiling on a
                 hold-out stratified by the condition it is deployed in.
  Agronomist     owns agronomic plausibility. Is a PSRI threshold of 0.22 at
                 grain fill a real senescence signal or a fitted artefact.
  Product head   owns what the client sees. A change can be statistically sound
                 and still make a number mean something different on a screen.
  Client         accepts before institutional rollout, where the contract says so.

WHY IT IS NOT A PULL REQUEST APPROVAL. A code review asks whether the diff is
correct. This asks whether the WORLD changes: how many farms move state, how much
exposure re-rates, whether any farm goes from a number to no number. Those are
computed and attached to the proposal before anyone is asked to approve, because
an approver who cannot see the blast radius is rubber-stamping.

THE RULE THAT MATTERS MOST. A proposal carries its own evidence. If the
validation run behind it is missing, stale, or was measured on the wrong
condition, the proposal cannot be approved at all — not by anyone, not with a
comment. That is the check that would have caught the 15% narrowing I invented,
which took a rejected cotton model under the gate with nothing measured behind it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional


class Role(str, Enum):
    DS_LEAD = "ds_lead"
    AGRONOMIST = "agronomist"
    PRODUCT_HEAD = "product_head"
    CLIENT = "client"


class Stage(str, Enum):
    DRAFT = "draft"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    SHADOW = "shadow"
    LIMITED = "limited"
    RELEASED = "released"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


#: Who must approve what. A threshold change needs the agronomist; a model
#: release needs the DS lead. Both need the product head, because both change
#: what a client reads off a screen.
REQUIRED: Dict[str, List[Role]] = {
    "threshold": [Role.DS_LEAD, Role.AGRONOMIST, Role.PRODUCT_HEAD],
    "model": [Role.DS_LEAD, Role.PRODUCT_HEAD],
    "stage_weight": [Role.AGRONOMIST, Role.PRODUCT_HEAD],
    "ladder": [Role.DS_LEAD, Role.PRODUCT_HEAD],
    "sla": [Role.PRODUCT_HEAD, Role.CLIENT],
    "precision_policy": [Role.PRODUCT_HEAD],
}

#: Changes that additionally require a named client to accept before they reach
#: that client's tenant. Anything altering what a number means commercially.
CLIENT_ACCEPTANCE = {"sla", "precision_policy"}


@dataclass
class Evidence:
    """What was measured, on what, and when.

    `holdout_condition` is here because it is the thing that was wrong before
    anyone noticed: a model measured on a mixed hold-out scored 0.090 and the
    same model on a Kharif-only hold-out scored 0.156. A proposal that does not
    say which condition it was judged on has not been judged.
    """

    validation_run_id: str
    measured_at: str
    holdout_condition: str
    n_splits: int
    rmse: Optional[float] = None
    rmse_sd: Optional[float] = None
    rmse_upper: Optional[float] = None
    ceiling: Optional[float] = None
    conformal_coverage: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def problems(self, max_age_days: int = 30) -> List[str]:
        out: List[str] = []
        if not self.validation_run_id:
            out.append("no validation run attached")
        if self.holdout_condition in ("", "mixed"):
            out.append(
                "hold-out was not stratified by deployment condition; a mixed "
                "hold-out fills with easy clear-sky days and flatters the model")
        if self.n_splits < 5:
            out.append(
                f"only {self.n_splits} split(s); a single split on a small "
                f"hold-out is luck, not a measurement")
        if self.rmse_upper is None:
            out.append("no upper bound; the mean alone hides the spread")
        elif self.ceiling is not None and self.rmse_upper > self.ceiling:
            out.append(
                f"upper bound {self.rmse_upper} exceeds the ceiling {self.ceiling}")
        try:
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(self.measured_at)).days
            if age > max_age_days:
                out.append(f"validation is {age} days old; limit is {max_age_days}")
        except (TypeError, ValueError):
            out.append("validation timestamp is unreadable")
        return out


@dataclass
class BlastRadius:
    """What changes in the world if this ships.

    Computed before approval is requested. An approver looking only at a diff
    cannot tell whether a threshold move re-states twelve farms or twelve
    thousand.
    """

    farms_affected: int = 0
    farms_changing_state: int = 0
    farms_losing_a_number: int = 0
    exposure_re_rated_inr: float = 0.0
    clients_affected: List[str] = field(default_factory=list)

    @property
    def severe(self) -> bool:
        """Changes that take a number away from a client need more care.

        Going from a printed value to no value is the most disruptive thing this
        system does, and it is also the thing it is for.
        """
        return self.farms_losing_a_number > 0 or self.exposure_re_rated_inr > 1e7


@dataclass
class Approval:
    role: Role
    approver: str
    at: str
    comment: str = ""


@dataclass
class Proposal:
    proposal_id: str
    kind: str
    title: str
    proposed_by: str
    proposed_at: str
    rule_version_from: str
    rule_version_to: str
    diff: Dict[str, object]
    evidence: Evidence
    blast: BlastRadius
    stage: Stage = Stage.DRAFT
    approvals: List[Approval] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)

    # ---------------------------------------------------------------- checks

    def required_roles(self) -> List[Role]:
        roles = list(REQUIRED.get(self.kind, [Role.PRODUCT_HEAD]))
        if self.kind in CLIENT_ACCEPTANCE and Role.CLIENT not in roles:
            roles.append(Role.CLIENT)
        # A change that takes numbers away from clients gets agronomic review
        # even when its category would not normally need it.
        if self.blast.severe and Role.AGRONOMIST not in roles:
            roles.append(Role.AGRONOMIST)
        return roles

    def outstanding(self) -> List[Role]:
        have = {a.role for a in self.approvals}
        return [r for r in self.required_roles() if r not in have]

    def blockers(self) -> List[str]:
        """Reasons this cannot be approved by anyone, in any order."""
        return self.evidence.problems()

    # ---------------------------------------------------------------- actions

    def submit(self, at: Optional[str] = None) -> "Proposal":
        self.stage = Stage.AWAITING_REVIEW
        self._log("submitted", at)
        return self

    def approve(self, role: Role, approver: str, comment: str = "",
                at: Optional[str] = None) -> "Proposal":
        """Record one approval.

        Three refusals, and each exists because of something that actually
        happened or nearly did:

        - Evidence problems block everyone. No approver may wave through a
          proposal whose measurement is missing or was taken on the wrong
          condition.
        - The proposer cannot approve their own change in any role. The person
          who invented a 15% adjustment is exactly the person least able to see
          that it was invented.
        - One person cannot hold two roles on the same proposal. Four
          signatures from two people is two signatures.
        """
        problems = self.blockers()
        if problems:
            raise ValueError(
                "cannot approve, the evidence does not support it: "
                + "; ".join(problems))

        if approver == self.proposed_by:
            raise ValueError(
                f"{approver} proposed this change and cannot also approve it")

        if any(a.approver == approver for a in self.approvals):
            raise ValueError(
                f"{approver} has already signed as "
                f"{next(a.role.value for a in self.approvals if a.approver == approver)}; "
                f"one person cannot hold two roles on one proposal")

        if role not in self.required_roles():
            raise ValueError(
                f"{role.value} is not required for a {self.kind} change; "
                f"required: {', '.join(r.value for r in self.required_roles())}")

        if any(a.role == role for a in self.approvals):
            raise ValueError(f"{role.value} has already approved")

        self.approvals.append(Approval(
            role=role, approver=approver, comment=comment, at=at or _now()))
        self._log(f"approved by {approver} as {role.value}", at)

        if not self.outstanding():
            self.stage = Stage.APPROVED
            self._log("fully approved; ready for shadow", at)
        return self

    def reject(self, approver: str, reason: str, at: Optional[str] = None) -> "Proposal":
        self.stage = Stage.REJECTED
        self._log(f"rejected by {approver}: {reason}", at)
        return self

    def promote(self, to: Stage, by: str, at: Optional[str] = None) -> "Proposal":
        """Move along the release path. Order is enforced, never skipped.

        approved -> shadow -> limited -> released

        Shadow means computed and logged but not shown. Limited means one client
        sees it. Nothing reaches everyone without passing through both, however
        confident anyone is.
        """
        order = [Stage.APPROVED, Stage.SHADOW, Stage.LIMITED, Stage.RELEASED]
        if self.stage not in order:
            raise ValueError(f"cannot promote from {self.stage.value}")
        if to not in order:
            raise ValueError(f"{to.value} is not a release stage")
        if order.index(to) != order.index(self.stage) + 1:
            raise ValueError(
                f"cannot go from {self.stage.value} straight to {to.value}; "
                f"the path is {' -> '.join(s.value for s in order)}")
        self.stage = to
        self._log(f"promoted to {to.value} by {by}", at)
        return self

    def withdraw(self, by: str, reason: str, at: Optional[str] = None) -> "Proposal":
        """Pull a released change back. Always available, at any stage."""
        self.stage = Stage.WITHDRAWN
        self._log(f"withdrawn by {by}: {reason}", at)
        return self

    # ---------------------------------------------------------------- record

    def _log(self, what: str, at: Optional[str] = None) -> None:
        self.history.append({"at": at or _now(), "what": what,
                             "stage": self.stage.value})

    def audit_record(self) -> Dict[str, object]:
        """The whole thing, hashed, for an institutional audit trail.

        A lender asked six months later why a number changed gets the proposal,
        its evidence, every signature, and a digest proving nothing was edited
        afterwards.
        """
        body = {
            "proposal": asdict(self),
            "required_roles": [r.value for r in self.required_roles()],
        }
        payload = json.dumps(body, sort_keys=True, default=str).encode()
        body["digest"] = hashlib.sha256(payload).hexdigest()[:16]
        return body


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_proposal(
    *,
    kind: str,
    title: str,
    proposed_by: str,
    rule_version_from: str,
    rule_version_to: str,
    diff: Dict[str, object],
    evidence: Evidence,
    blast: BlastRadius,
) -> Proposal:
    seed = f"{kind}:{title}:{proposed_by}:{_now()}"
    return Proposal(
        proposal_id="P-" + hashlib.sha256(seed.encode()).hexdigest()[:8],
        kind=kind, title=title, proposed_by=proposed_by, proposed_at=_now(),
        rule_version_from=rule_version_from, rule_version_to=rule_version_to,
        diff=diff, evidence=evidence, blast=blast,
    )
