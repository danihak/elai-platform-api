"""
test_agent.py — the properties that make the chatbot safe to put in front of a
farmer. These are the tests to point at when asked how explanations are evaluated.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.main import app
from app.services.llm import Circuit, check_grounded
from app.services.tools import run_tool, tool_definitions

c = TestClient(app)


# ------------------------------------------------------------------ tools

def test_tool_surface_is_small_and_fixed():
    names = {d["name"] for d in tool_definitions()}
    assert names == {
        "get_farm_context", "get_latest_observation", "get_observation_history",
        "get_cloud_state", "get_pop_reference",
    }


def test_pop_reference_contains_no_chemicals_or_rates():
    """The only agronomic content the model can cite must be safe by construction."""
    banned = ("urea", "dap", "npk", "ml", "gram", "kg/", "litre", "dose", "spray")
    for crop in ("maize", "cotton"):
        for stage in ("vegetative", "flowering", "grain_fill"):
            out = run_tool("get_pop_reference", {"crop": crop, "stage": stage})
            text = out["guidance"].lower()
            for word in banned:
                assert word not in text, f"{crop}/{stage} guidance mentions {word}"


def test_tools_return_the_state_not_a_bare_number():
    out = run_tool("get_latest_observation", {"farm_id": "TS-MZ-0044"})
    assert out["confidence_state"] in ("Verified", "Degraded", "Blind")
    assert "state_meaning" in out
    assert "rule_version" in out


# ------------------------------------------------------------ groundedness

def test_groundedness_accepts_figures_the_tools_returned():
    evidence = [{"tool": "get_latest_observation",
                 "result": {"ndvi": 0.62, "days_since_last_clear_optical": 13}}]
    ok, bad = check_grounded("NDVI was 0.62, and the field has not been seen for 13 days.", evidence)
    assert ok and not bad


def test_groundedness_rejects_an_invented_figure():
    """The failure that actually hurts someone: a plausible number nobody measured."""
    evidence = [{"tool": "get_latest_observation", "result": {"ndvi": 0.62}}]
    ok, bad = check_grounded("Yield is tracking at 8.7 MT/ha.", evidence)
    assert not ok
    assert "8.7" in bad


def test_groundedness_allows_a_fraction_spoken_as_a_percentage():
    evidence = [{"tool": "get_latest_observation", "result": {"valid_fraction": 0.19}}]
    ok, _ = check_grounded("Only 19% of the field was visible.", evidence)
    assert ok


# --------------------------------------------------------------- endpoint

def test_explain_always_returns_an_answer_without_an_api_key():
    """No key, provider down, circuit open, groundedness failed — all produce an answer."""
    r = c.post("/v1/agent/explain", json={"farm_id": "TS-MZ-0044", "audience": "farmer"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["generator"] == "deterministic"
    assert body["refused"] is False


def test_dosing_question_is_refused_before_any_model_call():
    r = c.post("/v1/agent/explain", json={
        "farm_id": "TS-MZ-0044", "question": "how much urea should I spray this week"})
    body = r.json()
    assert body["refused"] is True
    assert body["escalated"] is True
    assert "no_dosing" in body["guardrails_applied"]


def test_every_answer_carries_a_rule_version():
    body = c.post("/v1/agent/explain", json={"farm_id": "TS-MZ-0044"}).json()
    assert body["rule_version"]
    assert body["confidence_state"] in ("Verified", "Degraded", "Blind")


def test_blind_farmer_answer_gives_no_value():
    """A Blind field must not produce a number in prose either."""
    port = c.get("/v1/portfolio").json()
    blind = [r for r in port["rows"] if r["state"] == "Blind"]
    if not blind:
        return
    body = c.post("/v1/agent/explain", json={
        "farm_id": blind[0]["farm_id"], "audience": "farmer"}).json()
    assert "not been able to see" in body["answer"] or "not going to guess" in body["answer"]


# ----------------------------------------------------------- circuit breaker

def test_circuit_opens_after_three_failures_and_cools_off():
    cb = Circuit()
    assert not cb.is_open
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open
    cb.cool_off_s = 0.0
    assert not cb.is_open


def test_a_number_next_to_a_unit_must_always_be_grounded():
    """Regression: small integers used to slip through.

    'Yield is about 7 MT/ha' passed the old check purely because 7 is a small
    integer — the exact class of invented figure a lender would act on.
    """
    ev = [{"tool": "get_latest_observation", "result": {"ndvi": 0.62}}]
    for claim in ("Yield is about 7 MT/ha.", "Stress covers 4 hectares.", "It has been 9 days."):
        ok, bad = check_grounded(claim, ev)
        assert not ok, f"ungrounded claim accepted: {claim}"


def test_groundedness_is_deliberately_strict_about_rounding():
    """'About two weeks' from 13 days is a rounding, and it is rejected.

    This costs some fallbacks on farmer-facing phrasing, and that is the intended
    trade. The system prompt tells the model to use exact figures; a product whose
    entire thesis is not over-claiming should not quietly round in prose either.
    Saying '13 days' is also more verifiable for the farmer.
    """
    ev = [{"tool": "get_cloud_state", "result": {"days_since_last_clear_optical": 13}}]
    ok, _ = check_grounded("It has been about 2 weeks.", ev)
    assert not ok
    ok, _ = check_grounded("It has been 13 days.", ev)
    assert ok


def test_groundedness_accepts_sensible_rounding_of_float_artifacts():
    """Regression from the first live Claude run.

    Tools return 0.059500000000000004 because of float arithmetic. The model
    writes "0.0595", which is the honest way to say it — and the checker rejected
    it for being better formatted than the JSON. Every real answer was being
    thrown away.
    """
    ev = [{"tool": "get_latest_observation",
           "result": {"ndvi_error_band": 0.07 * 0.85, "ndvi": 0.716}}]
    ok, bad = check_grounded("NDVI 0.716, band 0.0595.", ev)
    assert ok, f"rejected a correctly rounded figure: {bad}"


def test_groundedness_ignores_digits_inside_identifiers():
    """TS-MZ-0044 is a farm id, not a claim about data."""
    ev = [{"tool": "get_farm_context", "result": {"farm_id": "TS-MZ-0044"}}]
    ok, bad = check_grounded("Farm TS-MZ-0044 was reviewed.", ev)
    assert ok, f"flagged identifier digits: {bad}"
