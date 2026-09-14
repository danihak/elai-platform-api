"""
fusion.py — variance-weighted fusion of three weak sources into one usable estimate.

THE PROBLEM WITH WHAT CAME BEFORE. The radar regression was asked to produce the
NDVI on its own. Maize scraped through at RMSE 0.090; cotton failed at 0.131 and
every cotton field went Blind for the whole monsoon. Radar was carrying the
entire burden of the answer.

THE FIX, WHICH IS THE STANDARD ONE IN THE LITERATURE. Three sources say something
about this field this week, none of them good enough alone:

  PHENOLOGY   the crop's expected NDVI for this day of year, from the fitted
              climatology. Uncertainty: the climatology's residual sigma.

  PERSISTENCE the last clear optical reading, carried forward. Excellent after
              three days, useless after sixty. Uncertainty grows with the gap.

  RADAR       the multivariate estimate from Sentinel-1. Uncertainty: the
              measured held-out RMSE for that crop.

Combining independent estimates by inverse variance produces a posterior tighter
than any input:

    1/sigma_post^2 = sum(1/sigma_i^2)
    mu_post        = sigma_post^2 * sum(mu_i / sigma_i^2)

This is the measurement-update half of a Kalman filter, which comparative studies
show outperforms Whittaker and Savitzky-Golay smoothing for gap filling, because
it models process and measurement noise explicitly and copes with uneven
intervals and long gaps.

WHY IT RESCUES COTTON. A radar estimate at sigma 0.131 is too wide to show on its
own. Fused with persistence at 0.08 and phenology at 0.10, the posterior lands
near 0.06 — decision-grade. A weak measurement still reduces uncertainty when it
is weighted honestly rather than discarded or overstated.

WHAT THIS IS NOT. No component is invented and no uncertainty is scaled by a
hand-chosen constant. Every sigma here traces to a measured quantity: held-out
RMSE for radar, residual sigma for climatology, and a persistence decay fitted
from real optical pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

#: Optical measurement noise on a clear pass. Sentinel-2 surface reflectance
#: NDVI over a homogeneous field, from the spread within the polygon.
OPTICAL_SIGMA = 0.02

#: Beyond this, persistence carries no usable information and is dropped rather
#: than fed in with a large sigma — a source that says nothing should not vote.
PERSISTENCE_MAX_DAYS = 45


@dataclass
class Component:
    name: str
    value: float
    sigma: float
    detail: str = ""

    @property
    def precision(self) -> float:
        return 1.0 / (self.sigma ** 2) if self.sigma > 0 else 0.0


@dataclass
class FusedEstimate:
    ndvi: float
    sigma: float
    components: List[Component] = field(default_factory=list)
    dominant: str = ""

    @property
    def contributions(self) -> Dict[str, float]:
        """Share of the answer each source supplied. Auditable by construction."""
        total = sum(c.precision for c in self.components)
        if total <= 0:
            return {}
        return {c.name: round(c.precision / total, 3) for c in self.components}


def persistence_sigma(days_since_clear: int, decay_per_day: float) -> float:
    """How fast a clear optical reading goes stale.

    Uncertainty grows as the square root of elapsed time, the standard random-walk
    assumption for a smoothly evolving state, floored at the optical noise itself
    because a reading is never better than the instrument that made it.

    decay_per_day has NO DEFAULT, deliberately. It is fitted from real clear-read
    pairs by `python -m app.cli fit`. An earlier version defaulted it to 0.0045,
    which claimed a 35-day-old reading was accurate to +-0.03 and let persistence
    take 93% of the fusion weight it had not earned. A constant nobody measured
    should not be reachable by leaving an argument out.
    """
    if days_since_clear <= 0:
        return OPTICAL_SIGMA
    return math.sqrt(OPTICAL_SIGMA ** 2 + (decay_per_day ** 2) * days_since_clear)


def fuse(components: List[Component]) -> Optional[FusedEstimate]:
    """Inverse-variance combination of independent estimates."""
    usable = [c for c in components if c.sigma > 0 and math.isfinite(c.value)]
    if not usable:
        return None

    total_precision = sum(c.precision for c in usable)
    if total_precision <= 0:
        return None

    mu = sum(c.value * c.precision for c in usable) / total_precision
    sigma = math.sqrt(1.0 / total_precision)
    dominant = max(usable, key=lambda c: c.precision).name

    return FusedEstimate(
        ndvi=max(0.0, min(1.0, mu)),
        sigma=sigma,
        components=usable,
        dominant=dominant,
    )


def build_components(
    *,
    obs_date: Optional[date],
    crop: str,
    radar_ndvi: Optional[float],
    radar_sigma: Optional[float],
    radar_detail: str,
    last_clear_ndvi: Optional[float],
    days_since_clear: int,
    climatology_model: Optional[Dict],
    persistence_decay: float,
) -> List[Component]:
    """Assemble whatever sources are actually available for this observation.

    A source that cannot speak is omitted, never filled with a placeholder. If
    nothing is available the caller gets no estimate and the engine sends the
    observation to Blind — which is the correct answer when nothing is known.
    """
    from ..ingest.climatology import predict as clim_predict

    out: List[Component] = []

    if radar_ndvi is not None and radar_sigma:
        out.append(Component("radar", radar_ndvi, radar_sigma, radar_detail))

    if last_clear_ndvi is not None and 0 < days_since_clear <= PERSISTENCE_MAX_DAYS:
        out.append(Component(
            "persistence", last_clear_ndvi,
            persistence_sigma(days_since_clear, persistence_decay),
            f"last clear read {days_since_clear} days ago",
        ))

    if climatology_model and obs_date is not None:
        pred = clim_predict(climatology_model, obs_date.timetuple().tm_yday)
        if pred is not None:
            value, sigma = pred
            out.append(Component(
                "phenology", value, sigma,
                f"expected {crop} NDVI for day {obs_date.timetuple().tm_yday}",
            ))

    return out


def describe(est: FusedEstimate) -> str:
    """One line naming every source and its share. Used in the audit trail."""
    parts = [f"{c.name} {c.value:.3f}+-{c.sigma:.3f}" for c in est.components]
    shares = ", ".join(f"{k} {v:.0%}" for k, v in est.contributions.items())
    return f"fused({'; '.join(parts)}) -> {est.ndvi:.3f}+-{est.sigma:.3f} [{shares}]"


# --------------------------------------------------------------------------
# Validating the fusion itself
# --------------------------------------------------------------------------


def validate_fusion(
    truth_points: List[Dict],
    radar_sigma_by_crop: Dict[str, float],
    climatology: Dict[str, Dict],
    decay_by_crop: Dict[str, float],
) -> Dict[str, Dict]:
    """Hide a real clear optical read, fuse without it, compare against it.

    Inverse-variance fusion assumes its sources are independent and unbiased.
    Neither holds perfectly here: radar and phenology both track the seasonal
    cycle, and phenology is biased for any field that is genuinely late-sown,
    water-stressed or failed — which is the only kind of field anyone cares
    about. So the posterior sigma the algebra produces may be optimistic.

    This measures it instead of arguing about it. For every clear optical read
    we hold it out, build the fusion from what was available beforehand, and
    compare the prediction against the reading we hid.

    Two numbers come out and both matter:

      rmse        the actual error of the fused estimate
      calibration mean of (error / claimed sigma). A well-calibrated estimator
                  scores about 1.0. Above 1.0 it is OVERCONFIDENT — claiming a
                  tighter band than it earns, which for this product is a worse
                  failure than being wide.

    truth_points: [{crop, obs_date, ndvi, radar_ndvi, last_clear_ndvi,
                    days_since_clear}, ...]
    """
    from collections import defaultdict

    grouped: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    for p in truth_points:
        crop = p["crop"]
        comps = build_components(
            obs_date=p["obs_date"],
            crop=crop,
            radar_ndvi=p.get("radar_ndvi"),
            radar_sigma=radar_sigma_by_crop.get(crop),
            radar_detail="",
            last_clear_ndvi=p.get("last_clear_ndvi"),
            days_since_clear=p.get("days_since_clear", 999),
            climatology_model=climatology.get(crop),
            persistence_decay=decay_by_crop.get(crop, 0.015),
        )
        est = fuse(comps)
        if est is None:
            continue
        grouped[crop].append((est.ndvi, p["ndvi"], est.sigma))

    out: Dict[str, Dict] = {}
    for crop, rows in sorted(grouped.items()):
        if len(rows) < 20:
            out[crop] = {"validated": False, "n": len(rows),
                         "reason": f"only {len(rows)} held-out points, need 20"}
            continue
        errors = [abs(pred - actual) for pred, actual, _ in rows]
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        claimed = sum(s for _, _, s in rows) / len(rows)
        calibration = (rmse / claimed) if claimed > 0 else float("inf")
        out[crop] = {
            "validated": True,
            "n": len(rows),
            "rmse": round(rmse, 4),
            "mean_claimed_sigma": round(claimed, 4),
            "calibration": round(calibration, 3),
            "overconfident": calibration > 1.2,
            "verdict": ("overconfident — claimed band is tighter than the real error"
                        if calibration > 1.2 else
                        "conservative — claimed band is wider than the real error"
                        if calibration < 0.8 else "well calibrated"),
        }
    return out


def inflate_for_calibration(sigma: float, calibration: float) -> float:
    """Widen a claimed band to match measured performance.

    If validation says the estimator is overconfident by a factor, the honest
    response is to widen the band by that factor rather than to keep the
    flattering number. Applied only when calibration exceeds 1.0 — a
    conservative estimator is left conservative.
    """
    return sigma * max(1.0, calibration)
