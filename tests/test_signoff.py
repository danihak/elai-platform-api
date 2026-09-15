"""Sign-off. The answer to "who signs off" from Krupa's Part 2a."""

import os, sys
import pytest
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.signoff import (BlastRadius, Evidence, Role, Stage,
                                  new_proposal)


def good_evidence(**kw):
    base = dict(
        validation_run_id="fit-2026-09-15-a",
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        holdout_condition="kharif", n_splits=30,
        rmse=0.104, rmse_sd=0.011, rmse_upper=0.115, ceiling=0.12,
    )
    base.update(kw)
    return Evidence(**base)


def proposal(kind="threshold", evidence=None, blast=None, by="ds@elai"):
    return new_proposal(
        kind=kind, title="PSRI stressed threshold at grain fill 0.22 -> 0.20",
        proposed_by=by, rule_version_from="2026.09.14-a",
        rule_version_to="2026.09.16-a",
        diff={"maize.grain_fill.PSRI.stressed": [0.22, 0.20]},
        evidence=evidence or good_evidence(),
        blast=blast or BlastRadius(farms_affected=6, farms_changing_state=2,
                                   exposure_re_rated_inr=265000),
    )


# ------------------------------------------------------- the evidence gate

def test_a_proposal_without_validation_cannot_be_approved_by_anyone():
    p = proposal(evidence=good_evidence(validation_run_id=""))
    with pytest.raises(ValueError, match="no validation run"):
        p.approve(Role.DS_LEAD, "asha@elai")


def test_a_mixed_holdout_blocks_approval():
    """The failure that started all of this.

    A model measured on a mixed hold-out scored 0.090; the same model on a
    Kharif-only hold-out scored 0.156. A proposal that does not say which
    condition it was judged on has not been judged.
    """
    p = proposal(evidence=good_evidence(holdout_condition="mixed"))
    with pytest.raises(ValueError, match="not stratified"):
        p.approve(Role.DS_LEAD, "asha@elai")


def test_a_single_split_blocks_approval():
    p = proposal(evidence=good_evidence(n_splits=1))
    with pytest.raises(ValueError, match="luck, not a measurement"):
        p.approve(Role.DS_LEAD, "asha@elai")


def test_an_upper_bound_over_the_ceiling_blocks_approval():
    p = proposal(evidence=good_evidence(rmse=0.118, rmse_upper=0.131))
    with pytest.raises(ValueError, match="exceeds the ceiling"):
        p.approve(Role.DS_LEAD, "asha@elai")


def test_stale_validation_blocks_approval():
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(timespec="seconds")
    p = proposal(evidence=good_evidence(measured_at=old))
    with pytest.raises(ValueError, match="days old"):
        p.approve(Role.DS_LEAD, "asha@elai")


# ------------------------------------------------------- separation of duty

def test_the_proposer_cannot_approve_their_own_change():
    p = proposal(by="asha@elai")
    with pytest.raises(ValueError, match="cannot also approve"):
        p.approve(Role.DS_LEAD, "asha@elai")


def test_one_person_cannot_hold_two_roles():
    """Four signatures from two people is two signatures."""
    p = proposal().submit()
    p.approve(Role.DS_LEAD, "asha@elai")
    with pytest.raises(ValueError, match="two roles"):
        p.approve(Role.AGRONOMIST, "asha@elai")


def test_a_role_that_is_not_required_cannot_sign():
    p = proposal(kind="model").submit()
    with pytest.raises(ValueError, match="not required"):
        p.approve(Role.CLIENT, "bank@client")


# ------------------------------------------------------- the happy path

def test_a_threshold_change_needs_three_roles_and_then_is_approved():
    p = proposal().submit()
    assert p.stage == Stage.AWAITING_REVIEW
    assert len(p.outstanding()) == 3
    p.approve(Role.DS_LEAD, "asha@elai")
    p.approve(Role.AGRONOMIST, "vijay@elai")
    assert p.stage == Stage.AWAITING_REVIEW
    p.approve(Role.PRODUCT_HEAD, "danish@elai")
    assert p.stage == Stage.APPROVED
    assert p.outstanding() == []


def test_release_path_cannot_be_skipped():
    """Shadow then limited then released. However confident anyone is."""
    p = proposal().submit()
    for role, who in ((Role.DS_LEAD, "asha@elai"), (Role.AGRONOMIST, "vijay@elai"),
                      (Role.PRODUCT_HEAD, "danish@elai")):
        p.approve(role, who)
    with pytest.raises(ValueError, match="straight to"):
        p.promote(Stage.RELEASED, "danish@elai")
    p.promote(Stage.SHADOW, "danish@elai")
    p.promote(Stage.LIMITED, "danish@elai")
    p.promote(Stage.RELEASED, "danish@elai")
    assert p.stage == Stage.RELEASED


# ------------------------------------------------------- blast radius

def test_taking_a_number_away_pulls_in_the_agronomist():
    """Going from a printed value to no value is the most disruptive thing
    this system does — and also the thing it is for."""
    p = proposal(kind="model",
                 blast=BlastRadius(farms_affected=6, farms_losing_a_number=3))
    assert Role.AGRONOMIST in p.required_roles()


def test_an_ordinary_model_change_does_not_need_the_agronomist():
    p = proposal(kind="model", blast=BlastRadius(farms_affected=6))
    assert Role.AGRONOMIST not in p.required_roles()


def test_sla_changes_need_the_client():
    p = proposal(kind="sla")
    assert Role.CLIENT in p.required_roles()


# ------------------------------------------------------- audit

def test_the_audit_record_is_hashed():
    p = proposal().submit()
    rec = p.audit_record()
    assert len(rec["digest"]) == 16
    assert rec["proposal"]["rule_version_from"] == "2026.09.14-a"


def test_a_released_change_can_always_be_withdrawn():
    p = proposal().submit()
    for role, who in ((Role.DS_LEAD, "asha@elai"), (Role.AGRONOMIST, "vijay@elai"),
                      (Role.PRODUCT_HEAD, "danish@elai")):
        p.approve(role, who)
    p.promote(Stage.SHADOW, "danish@elai")
    p.withdraw("danish@elai", "field team disputes the senescence signal")
    assert p.stage == Stage.WITHDRAWN
    assert "disputes" in p.history[-1]["what"]
