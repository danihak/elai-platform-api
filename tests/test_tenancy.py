"""Multi-tenancy — Krupa's "we serve multiple clients"."""

import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.tenancy import (NON_NEGOTIABLE, OverrideRefused, Tenant,
                                  Vertical, render_for, what_a_client_can_change)


def lender():
    return Tenant("t-coop-01", "Karimnagar District Co-op", Vertical.LENDER)


def insurer():
    return Tenant("t-ins-01", "Agri Insurance Co", Vertical.INSURER)


# ------------------------------------------- what clients may change

def test_a_client_can_set_its_own_sla_tier():
    t = lender().set("sla_tier", "premium").set("blind_days_budget", 10)
    assert t.resolved()["sla_tier"] == "premium"
    assert t.resolved()["blind_days_budget"] == 10


def test_a_client_can_choose_languages_and_channels():
    t = lender().set("languages", ["te", "hi"]).set("channels", ["whatsapp_voice"])
    assert t.resolved()["languages"] == ["te", "hi"]


# ------------------------------------------- what no client may change

def test_no_client_can_raise_the_decision_grade_ceiling():
    """The single most important refusal in the system."""
    with pytest.raises(OverrideRefused, match="right to see numbers"):
        lender().set("decision_grade_ceiling", 0.25)


def test_no_client_can_redefine_the_confidence_states():
    with pytest.raises(OverrideRefused, match="mean nothing in any of them"):
        lender().set("confidence_states", ["good", "bad"])


def test_no_client_can_loosen_the_precision_policy():
    """This is the mechanism that stops a blind field printing 7.2."""
    with pytest.raises(OverrideRefused, match="it is the product"):
        insurer().set("precision_policy", {"blind": "point_value"})


def test_no_client_can_set_its_own_stress_thresholds():
    with pytest.raises(OverrideRefused, match="negotiated"):
        lender().set("stress_thresholds", {"maize": {"NDVI": 0.2}})


def test_no_client_can_pick_an_uncertified_conformal_level():
    with pytest.raises(OverrideRefused, match="was not certified"):
        insurer().set("conformal_levels", [0.99])


def test_every_refusal_carries_a_reason():
    """A refusal without one reads as obstruction rather than a position."""
    for key, reason in NON_NEGOTIABLE.items():
        assert len(reason) > 40, key
        with pytest.raises(OverrideRefused) as exc:
            lender().set(key, "anything")
        assert reason.split(";")[0][:30] in str(exc.value)


def test_an_unknown_setting_is_refused_not_silently_stored():
    with pytest.raises(OverrideRefused, match="not a recognised setting"):
        lender().set("make_numbers_look_better", True)


# ------------------------------------------- different screens, same facts

def test_a_lender_and_an_insurer_see_different_metrics():
    assert lender().may_see("exposure_at_risk")
    assert not insurer().may_see("exposure_at_risk")
    assert insurer().may_see("observation_history")
    assert not lender().may_see("observation_history")


def test_both_see_the_confidence_state():
    """Different screens. Never different facts."""
    for t in (lender(), insurer(), Tenant("t", "x", Vertical.EXPORTER)):
        assert t.may_see("confidence_state")


def test_the_display_string_is_passed_through_verbatim():
    """A tenant decides WHETHER a metric appears. Never HOW a number is written."""
    out = render_for(lender(), "yield_band", "6.3 to 8.1 MT/ha", "Degraded")
    assert out["display"] == "6.3 to 8.1 MT/ha"


def test_a_hidden_metric_says_why():
    out = render_for(lender(), "observation_history", "…", "Degraded")
    assert out["visible"] is False
    assert "visible set" in out["reason"]


# ------------------------------------------- the split, stated

def test_the_split_is_published():
    """Sales needs this as much as engineering does."""
    out = what_a_client_can_change()
    assert "right to be lied to" in out["principle"]
    assert "the confidence engine and its rule version" in out["what_is_shared"]
    assert set(out["non_negotiable"]) == set(NON_NEGOTIABLE)
