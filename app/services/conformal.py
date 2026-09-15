"""
conformal.py — distribution-free coverage guarantees for the fused estimate.

WHAT THIS REPLACES. The fusion produces a sigma from inverse-variance algebra,
and `inflate_for_calibration` widened it by the mean ratio of real error to
claimed sigma. That is a reasonable correction but it guarantees nothing: a mean
ratio of 1.0 is consistent with a band that contains the truth 60% of the time
and one that contains it 99% of the time.

WHAT CONFORMAL PREDICTION GIVES INSTEAD. A finite-sample, distribution-free
guarantee: if the calibration data and the new observation are exchangeable, the
interval contains the true value at least (1 - alpha) of the time. No assumption
that errors are Gaussian, no assumption the model is correct, no asymptotics.
For a product whose entire claim is "trust this number", that is the difference
between a measurement and a proof.

THE VARIANT USED, AND WHY. Normalised (locally adaptive) split conformal. The
nonconformity score is

    s_i = |y_i - yhat_i| / sigma_i

dividing the error by the sigma the fusion already claims for that observation.
A plain absolute-residual score would produce ONE fixed interval width for every
farm — far too wide for a field seen last week, far too narrow for one unseen
for forty days. Normalising keeps the shape of the uncertainty the fusion
already models and calibrates only its scale.

The multiplier is then the ceil((n+1)(1-alpha))/n quantile of the calibration
scores. That finite-sample correction is what makes the guarantee exact rather
than asymptotic, and it is why small calibration sets produce honestly wider
intervals instead of quietly failing.

WHERE THE GUARANTEE STOPS, STATED PLAINLY. Exchangeability is a real assumption
and it is the one most likely to break here:

  - Coverage is MARGINAL, not conditional. 90% across all farms does not mean
    90% for every farm. A systematically unusual field can be under-covered
    while the average holds. Per-crop calibration mitigates this; it does not
    remove it.
  - A new season, a new district or a new crop is not exchangeable with the
    calibration set. Recalibration is required, and the code refuses to
    extrapolate to a crop it has no scores for.
  - Time-series data is not independent. Consecutive observations of one field
    are correlated, so the effective sample size is smaller than the row count.
    Calibration is therefore blocked by field rather than pooled at random.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Levels calibrated by default. 0.80 for internal triage, 0.90 for client
#: display, 0.95 for anything that may end up in a claims file.
DEFAULT_LEVELS: Tuple[float, ...] = (0.80, 0.90, 0.95)

#: Stability floor, ABOVE the mathematical requirement.
#:
#: The mathematically required sample size is whatever makes the quantile index
#: exist: ceil((n+1)*level) <= n, which is n >= 19 at 95% and n >= 9 at 90%. A
#: margin above that is sensible because a quantile estimated from very few
#: points is itself noisy — the guarantee still holds in expectation, but the
#: multiplier swings between recalibrations.
#:
#: An earlier version used a flat 50, which was a round number rather than a
#: derived one, and it refused to certify a crop with 49 points while certifying
#: one with 50. The floor is now per-level and derived, with this as the
#: stability margin.
MIN_STABLE = 30


def required_n(level: float) -> int:
    """Smallest n for which the (1-alpha) conformal quantile exists."""
    n = 1
    while math.ceil((n + 1) * level) > n:
        n += 1
    return n


@dataclass
class ConformalModel:
    """Calibrated multipliers for one crop."""

    crop: str
    n: int
    multipliers: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    usable: bool = True
    reason: str = ""

    def multiplier(self, level: float) -> Optional[float]:
        return self.multipliers.get(f"{level:.2f}")


def _quantile_index(n: int, level: float) -> Optional[int]:
    """ceil((n+1)(1-alpha)) as a 1-based rank, or None if it exceeds n.

    When it exceeds n the calibration set is too small to certify that level.
    Returning None rather than clamping to the maximum score is deliberate: the
    clamped version looks like a guarantee and is not one.
    """
    k = math.ceil((n + 1) * level)
    return k if k <= n else None


def calibrate(
    residuals: List[Dict],
    levels: Sequence[float] = DEFAULT_LEVELS,
    min_stable: int = MIN_STABLE,
) -> Dict[str, ConformalModel]:
    """residuals: [{crop, actual, predicted, sigma}, ...] from HELD-OUT data.

    The calibration set must not have been used to fit the estimator. Reusing
    training points inflates coverage and quietly voids the guarantee.
    """
    by_crop: Dict[str, List[float]] = {}
    for r in residuals:
        sigma = r.get("sigma")
        if not sigma or sigma <= 0:
            continue
        score = abs(r["actual"] - r["predicted"]) / sigma
        by_crop.setdefault(r["crop"], []).append(score)

    out: Dict[str, ConformalModel] = {}
    for crop, scores in sorted(by_crop.items()):
        n = len(scores)
        needed = max(min_stable, required_n(max(levels)))
        if n < needed:
            out[crop] = ConformalModel(
                crop=crop, n=n, usable=False,
                reason=(f"only {n} calibration points; need {needed} "
                        f"(quantile exists at {required_n(max(levels))}, "
                        f"stability floor {min_stable})"),
            )
            continue

        scores.sort()
        model = ConformalModel(crop=crop, n=n)
        for level in levels:
            k = _quantile_index(n, level)
            if k is None:
                model.diagnostics[f"{level:.2f}"] = {
                    "certifiable": 0.0,
                    "note": float("nan"),
                }
                continue
            model.multipliers[f"{level:.2f}"] = round(scores[k - 1], 4)
        out[crop] = model
    return out


def evaluate(
    residuals: List[Dict],
    models: Dict[str, ConformalModel],
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Measure achieved coverage on a set the calibration never saw.

    Two standard diagnostics:

      PICP  prediction interval coverage probability — the share of true values
            that actually fell inside. Should sit at or just above the nominal
            level. Materially below means the guarantee is not holding, which in
            practice means exchangeability is broken.

      MPIW  mean prediction interval width. Coverage is trivial to achieve with
            an infinitely wide band, so width is what makes the interval useful.
            The two must always be reported together.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for crop, model in sorted(models.items()):
        rows = [r for r in residuals if r["crop"] == crop and r.get("sigma")]
        if not rows or not model.usable:
            continue
        per_level: Dict[str, Dict[str, float]] = {}
        for level in levels:
            m = model.multiplier(level)
            if m is None:
                continue
            covered = sum(
                1 for r in rows
                if abs(r["actual"] - r["predicted"]) <= m * r["sigma"]
            )
            widths = [2 * m * r["sigma"] for r in rows]
            picp = covered / len(rows)
            per_level[f"{level:.2f}"] = {
                "nominal": level,
                "picp": round(picp, 4),
                "mpiw": round(sum(widths) / len(widths), 4),
                "n": len(rows),
                "holds": bool(picp >= level - 0.03),
            }
        out[crop] = per_level
    return out


def interval(
    predicted: float,
    sigma: float,
    model: Optional[ConformalModel],
    level: float = 0.90,
) -> Optional[Tuple[float, float, float]]:
    """(low, high, multiplier), or None when this level cannot be certified.

    Returning None is the correct behaviour for an uncalibrated crop. Falling
    back to the raw fusion sigma here would present an uncertified band in the
    same shape as a certified one, and nothing downstream could tell them apart.
    """
    if model is None or not model.usable:
        return None
    m = model.multiplier(level)
    if m is None:
        return None
    half = m * sigma
    return max(0.0, predicted - half), min(1.0, predicted + half), m


def blocked_split(
    points: List[Dict],
    calibration_fraction: float = 0.4,
    seed: int = 4471,
) -> Tuple[List[Dict], List[Dict]]:
    """Split by FIELD, not by row.

    Observations of one field across a season are strongly correlated. A random
    row split puts neighbouring dates from the same field on both sides, which
    makes calibration look better than it is and shrinks the effective sample
    far below the row count. Splitting whole fields keeps the two sets closer to
    exchangeable.
    """
    import random

    rng = random.Random(seed)
    fields = sorted({p.get("field_id", "unknown") for p in points})
    rng.shuffle(fields)
    cut = max(1, int(len(fields) * calibration_fraction))
    cal_fields = set(fields[:cut])
    cal = [p for p in points if p.get("field_id", "unknown") in cal_fields]
    test = [p for p in points if p.get("field_id", "unknown") not in cal_fields]
    return cal, test


def to_json(models: Dict[str, ConformalModel]) -> Dict:
    return {
        crop: {
            "n": m.n,
            "usable": m.usable,
            "reason": m.reason,
            "multipliers": m.multipliers,
        }
        for crop, m in models.items()
    }


def from_json(blob: Dict) -> Dict[str, ConformalModel]:
    out = {}
    for crop, d in blob.items():
        model = ConformalModel(crop=crop, n=d.get("n", 0),
                               usable=d.get("usable", False),
                               reason=d.get("reason", ""))
        model.multipliers = {k: float(v) for k, v in d.get("multipliers", {}).items()}
        out[crop] = model
    return out
