"""
stress.py — crop stress from stage-weighted indices.

REPLACES A FABRICATION. Area under stress was `area_ha * 0.18` — a constant
dressed up as a measurement, and the last invented number in the system.

MAKES THE STAGE MATRIX LIVE. `STAGE_INDEX_WEIGHTS` sat in rules.py read by
nothing. It now decides which index drives the headline flag, which is the
direct answer to "how does the relative importance of different stress indices
get weighted by growth stage".

THE MODEL. Each index has a direction and a stage-specific threshold. NDVI and
GCI falling is stress; PSRI rising is senescence, which is stress early and
normal at maturity; NDWI falling is water stress and matters most at flowering.
Each index contributes a 0-1 severity, weighted by its relevance at this stage,
and the weighted sum is the stress score.

Indices the stage suppresses contribute nothing and their weight is redistributed
rather than counted as zero stress — a suppressed index is unknown, not healthy.

WHAT IS STILL TBD. The thresholds are agronomic judgements, not fitted values.
They are marked, they are in one place, and they are exactly the kind of number
that should move through the sign-off workflow rather than a pull request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from elai_confidence_core import BLIND, DEGRADED, VERIFIED, CURRENT, ScoredObservation

#: Direction of stress for each index. "down" means falling values indicate
#: stress; "up" means rising values do.
DIRECTION: Dict[str, str] = {
    "NDVI": "down",
    "GCI": "down",
    "NDWI": "down",
    "PSRI": "up",
    "RED_EDGE": "down",
}

#: Per crop, per stage, per index: (healthy_value, stressed_value).
#: A reading at or beyond `stressed` scores 1.0; at or beyond `healthy` scores
#: 0.0; linear in between.
#:
#: TBD — agronomic thresholds, to be set with the agronomist and moved through
#: the sign-off workflow. They are deliberately in one table so a change is
#: visible and reviewable rather than scattered through the code.
THRESHOLDS: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
    "maize": {
        "germination": {"NDVI": (0.30, 0.15), "NDWI": (0.25, 0.10)},
        "vegetative": {"NDVI": (0.55, 0.32), "GCI": (1.30, 0.70), "NDWI": (0.35, 0.18),
                       "RED_EDGE": (0.38, 0.22)},
        "flowering": {"NDVI": (0.70, 0.45), "NDWI": (0.40, 0.20), "GCI": (1.60, 0.90),
                      "RED_EDGE": (0.45, 0.26)},
        "grain_fill": {"NDVI": (0.62, 0.38), "NDWI": (0.36, 0.17), "PSRI": (0.08, 0.22),
                       "GCI": (1.40, 0.75)},
        "maturation": {"PSRI": (0.20, 0.38), "NDVI": (0.40, 0.22), "NDWI": (0.28, 0.12)},
        "harvest": {"PSRI": (0.30, 0.50), "NDVI": (0.30, 0.15)},
    },
    "cotton": {
        "germination": {"NDVI": (0.28, 0.14), "NDWI": (0.24, 0.10)},
        "vegetative": {"NDVI": (0.50, 0.30), "GCI": (1.20, 0.65), "NDWI": (0.33, 0.16),
                       "RED_EDGE": (0.35, 0.20)},
        "flowering": {"NDVI": (0.65, 0.42), "NDWI": (0.38, 0.19), "GCI": (1.50, 0.85),
                      "RED_EDGE": (0.42, 0.24)},
        "grain_fill": {"NDVI": (0.58, 0.35), "NDWI": (0.34, 0.16), "PSRI": (0.09, 0.24),
                       "GCI": (1.30, 0.70)},
        "maturation": {"PSRI": (0.22, 0.40), "NDVI": (0.38, 0.20), "NDWI": (0.26, 0.11)},
    },
}

#: Below this share of the stage's index weight, no stress verdict is issued.
#:
#: A Degraded observation carries an NDVI estimate from radar and nothing else,
#: so at flowering — where NDWI matters more than NDVI — a single index is about
#: a fifth of the evidence the stage calls for. Declaring "no stress detected"
#: on that is the same class of over-claim as printing a yield on a field nobody
#: has seen. The honest output is that the question cannot be answered, which is
#: a different statement from "the crop is fine".
#:
#: TBD — the level is a product decision about how much evidence is enough, and
#: belongs in the sign-off workflow.
MIN_WEIGHT_COVERAGE = 0.35

#: Stress score bands. Also TBD, also reviewable in one place.
BANDS: List[Tuple[float, str]] = [
    (0.60, "Severe"),
    (0.35, "Moderate"),
    (0.15, "Mild"),
    (0.00, "No stress detected"),
]


@dataclass
class IndexContribution:
    index: str
    value: float
    weight: float
    severity: float
    threshold: Tuple[float, float]
    direction: str


@dataclass
class StressAssessment:
    """The headline flag, and every input that produced it."""

    score: Optional[float]
    band: str
    driving_index: Optional[str]
    contributions: List[IndexContribution]
    stage: str
    suppressed: List[str]
    unavailable: List[str]
    weight_covered: float
    state: str
    rule_version: str
    reason: str

    @property
    def area_under_stress_fraction(self) -> Optional[float]:
        """Share of the field treated as stressed.

        A whole-field mean index cannot locate stress spatially, so this is the
        score itself rather than a per-pixel count — and it is labelled as a
        field-level severity, not a mapped area. Reporting it as an area would
        imply a spatial measurement we have not made. Per-pixel statistics from
        the Sentinel Hub raster would give a real area; that is a later change.
        """
        return self.score


def _severity(value: float, healthy: float, stressed: float, direction: str) -> float:
    if direction == "up":
        if value <= healthy:
            return 0.0
        if value >= stressed:
            return 1.0
        return (value - healthy) / (stressed - healthy)
    if value >= healthy:
        return 0.0
    if value <= stressed:
        return 1.0
    return (healthy - value) / (healthy - stressed)


def assess(obs: ScoredObservation, crop: str, rules=CURRENT) -> StressAssessment:
    """Stage-weighted stress for one scored observation.

    The confidence state is inherited, not recomputed: a stress figure derived
    from a Blind observation is itself Blind, and no score is returned at all.
    """
    stage = obs.stage
    weights = rules.stage_index_weights.get(stage, {})
    suppressed = list(obs.suppressed_indices)
    thresholds = THRESHOLDS.get(crop, {}).get(stage, {})

    if obs.state == BLIND:
        return StressAssessment(
            score=None, band="Not observed", driving_index=None, contributions=[],
            stage=stage, suppressed=suppressed, unavailable=[], weight_covered=0.0,
            state=BLIND, rule_version=rules.version,
            reason="field not observed; no stress assessment is possible",
        )

    contributions: List[IndexContribution] = []
    unavailable: List[str] = []
    total_weight = 0.0

    for index, weight in weights.items():
        if weight <= 0:
            continue
        if index in suppressed:
            # Not meaningful at this stage. Its weight is redistributed below
            # rather than scored zero — suppressed is unknown, not healthy.
            continue
        value = obs.indices.get(index)
        band = thresholds.get(index)
        if value is None or band is None:
            unavailable.append(index)
            continue
        sev = _severity(value, band[0], band[1], DIRECTION.get(index, "down"))
        contributions.append(IndexContribution(
            index=index, value=value, weight=weight, severity=sev,
            threshold=band, direction=DIRECTION.get(index, "down"),
        ))
        total_weight += weight

    if not contributions or total_weight <= 0:
        return StressAssessment(
            score=None, band="Not assessable", driving_index=None, contributions=[],
            stage=stage, suppressed=suppressed, unavailable=unavailable,
            weight_covered=0.0, state=BLIND, rule_version=rules.version,
            reason=(f"no index with a stage weight was available at {stage}; "
                    f"missing {', '.join(unavailable) or 'all'}"),
        )

    # Weights are renormalised over what was actually available, so a partial
    # observation is scored on its own terms rather than diluted toward zero.
    score = sum(c.weight * c.severity for c in contributions) / total_weight
    driver = max(contributions, key=lambda c: c.weight * c.severity)
    band = next(name for cut, name in BANDS if score >= cut)

    stage_total = sum(w for w in weights.values() if w > 0)
    covered = total_weight / stage_total if stage_total else 0.0

    if covered < MIN_WEIGHT_COVERAGE:
        return StressAssessment(
            score=None, band="Insufficient evidence", driving_index=driver.index,
            contributions=contributions, stage=stage, suppressed=suppressed,
            unavailable=unavailable, weight_covered=round(covered, 3),
            state=DEGRADED if obs.state == VERIFIED else obs.state,
            rule_version=rules.version,
            reason=(f"only {covered:.0%} of the stage's index weight was available "
                    f"at {stage} (have {', '.join(c.index for c in contributions)}; "
                    f"missing {', '.join(unavailable)}). Not enough to issue a stress "
                    f"verdict — this is not the same as no stress."),
        )

    # A partial observation cannot be Verified even if the pass was clear: the
    # stress figure rests on a fraction of the evidence the stage calls for.
    state = obs.state
    if covered < 0.60 and state == VERIFIED:
        state = DEGRADED

    reason = (
        f"{band.lower()} at {stage}; driven by {driver.index} "
        f"({driver.value:.3f} against {driver.threshold[0]:.2f} healthy / "
        f"{driver.threshold[1]:.2f} stressed, stage weight {driver.weight:.2f}); "
        f"{covered:.0%} of the stage's index weight available"
    )
    if unavailable:
        reason += f"; missing {', '.join(unavailable)}"

    return StressAssessment(
        score=round(score, 3), band=band, driving_index=driver.index,
        contributions=contributions, stage=stage, suppressed=suppressed,
        unavailable=unavailable, weight_covered=round(covered, 3),
        state=state, rule_version=rules.version, reason=reason,
    )


def explain_weights(crop: str, stage: str, rules=CURRENT) -> Dict:
    """The stage-weighting answer, as data.

    This is what Krupa asked for directly: how the relative importance of
    different stress indices gets weighted by growth stage. It is queryable
    rather than described in a slide.
    """
    weights = rules.stage_index_weights.get(stage, {})
    thresholds = THRESHOLDS.get(crop, {}).get(stage, {})
    return {
        "crop": crop,
        "stage": stage,
        "rule_version": rules.version,
        "indices": [
            {
                "index": index,
                "weight": weight,
                "direction": DIRECTION.get(index, "down"),
                "healthy": thresholds.get(index, (None, None))[0],
                "stressed": thresholds.get(index, (None, None))[1],
                "suppressed_at_stage": index in rules.stage_suppressed_indices.get(stage, []),
                "threshold_status": "TBD — pending agronomist sign-off"
                                    if index in thresholds else "not set for this stage",
            }
            for index, weight in sorted(weights.items(), key=lambda kv: -kv[1])
            if weight > 0
        ],
        "bands": [{"at_or_above": cut, "label": name} for cut, name in BANDS],
    }
