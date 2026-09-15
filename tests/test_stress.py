"""Stage-weighted stress. The answers to Krupa's 2a questions, as tests."""

import os, sys
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elai_confidence_core import BLIND, DEGRADED, VERIFIED, CURRENT, Observation, score
from app.services.stress import MIN_WEIGHT_COVERAGE, assess, explain_weights


def obs(**kw):
    base = dict(farm_id="X", obs_date=date(2026, 9, 6), das=86, crop="maize",
                stage="grain_fill", valid_pixel_fraction=0.95,
                mean_cloud_probability=0.03, cloudy=False, ndvi=0.62, ndwi=0.34,
                psri=0.10, gci=1.35, red_edge=0.40, days_since_last_clear_optical=2,
                last_clear_optical_date=date(2026, 9, 4), last_clear_optical_das=84,
                last_clear_optical_ndvi=0.63)
    base.update(kw)
    return score(Observation(**base))


def test_no_hardcoded_area_fraction_remains():
    """Regression: area under stress was area_ha * 0.18 for every farm, always."""
    import pathlib, re
    src = (pathlib.Path(__file__).parent.parent / "app" / "routers" / "farms.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert not re.search(r"area_ha\s*\*\s*0\.18", code)
    assert "area_under_stress" not in code


def test_stage_weights_actually_drive_the_flag():
    """NDWI outweighs NDVI at grain fill, so water stress should dominate."""
    a = assess(obs(ndwi=0.16, ndvi=0.62), "maize")
    assert a.driving_index == "NDWI"
    b = assess(obs(ndvi=0.30, ndwi=0.34), "maize")
    assert b.driving_index == "NDVI"


def test_the_same_reading_scores_differently_by_stage():
    """The direct answer to 'weighted by growth stage'.

    PSRI at 0.30 is senescence. At grain fill that is stress; at maturation it is
    what the crop is supposed to be doing.
    """
    gf = assess(obs(stage="grain_fill", psri=0.30), "maize")
    mat = assess(obs(stage="maturation", das=110, psri=0.30), "maize")
    gf_psri = next(c for c in gf.contributions if c.index == "PSRI")
    mat_psri = next(c for c in mat.contributions if c.index == "PSRI")
    assert gf_psri.severity > mat_psri.severity


def test_direction_is_respected():
    """PSRI rising is stress; NDVI rising is not."""
    high_psri = assess(obs(psri=0.30), "maize")
    low_psri = assess(obs(psri=0.05), "maize")
    assert high_psri.score > low_psri.score


def test_blind_observation_yields_no_stress_verdict():
    a = assess(obs(valid_pixel_fraction=0.04, cloudy=True, ndvi=None, ndwi=None,
                   psri=None, gci=None, red_edge=None,
                   days_since_last_clear_optical=40), "maize")
    assert a.state == BLIND
    assert a.score is None
    assert a.band == "Not observed"


def test_thin_evidence_is_not_reported_as_healthy():
    """The distinction that matters: 'we cannot tell' is not 'the crop is fine'.

    A radar-estimated observation carries NDVI alone — about a fifth of the
    evidence grain fill calls for. Declaring no stress on that is the same
    over-claim as printing a yield on an unseen field.
    """
    a = assess(obs(ndwi=None, psri=None, gci=None, red_edge=None), "maize")
    assert a.weight_covered < MIN_WEIGHT_COVERAGE
    assert a.band == "Insufficient evidence"
    assert a.score is None
    assert "not the same as no stress" in a.reason


def test_weights_renormalise_over_available_indices():
    """A partial observation is scored on its own terms, not diluted to zero."""
    full = assess(obs(ndvi=0.30, ndwi=0.16, psri=0.30, gci=0.70, red_edge=0.22), "maize")
    partial = assess(obs(ndvi=0.30, ndwi=0.16, psri=0.30, gci=None, red_edge=None), "maize")
    assert full.score > 0.5 and partial.score > 0.5


def test_stage_weighting_is_queryable():
    """Krupa asked how weighting works; it is data, not a slide."""
    w = explain_weights("maize", "grain_fill")
    assert w["indices"][0]["index"] == "NDWI"
    assert w["indices"][0]["weight"] == 0.35
    assert all("threshold_status" in i for i in w["indices"])


def test_every_assessment_carries_a_rule_version():
    assert assess(obs(), "maize").rule_version == CURRENT.version
