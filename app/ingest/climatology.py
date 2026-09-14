"""
climatology.py — the expected NDVI curve for a crop, fitted from history.

WHY THIS EXISTS. Regressing NDVI on radar alone asks radar to explain the entire
signal, including the part that is completely predictable: crops are green in
September and bare in May. Most of the variance in an NDVI series is seasonal,
and radar does not need to explain any of it.

So the estimator is restructured. Climatology carries the seasonal shape, and
radar only has to estimate the DEPARTURE from it — which is the part anyone
actually cares about, because a departure from the expected curve is what stress
looks like.

METHOD. Harmonic regression on day of year:

    ndvi(doy) = a0
              + a1*cos(2pi*doy/365) + b1*sin(2pi*doy/365)
              + a2*cos(4pi*doy/365) + b2*sin(4pi*doy/365)

Two harmonics: the first captures one growing season a year, the second captures
the double-cropped Kharif/Rabi pattern that is normal across Telangana. Fitted
per crop from every clear optical observation in the two-year training pull.

Pure Python least squares, so nothing new is added to the dependency list.

HONEST LIMIT. A climatology is an average. A field that is genuinely late-sown or
failed will depart from it, and that departure is the signal, not an error —
which is exactly why the fusion step keeps the prior weak relative to a real
measurement.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

TERMS = 5  # intercept + two harmonics


def _basis(doy: int) -> List[float]:
    w = 2.0 * math.pi * doy / 365.25
    return [1.0, math.cos(w), math.sin(w), math.cos(2 * w), math.sin(2 * w)]


def _solve(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / pv
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * out[c] for c in range(r + 1, n))
        out[r] = s / m[r][r]
    return out


def fit_climatology(observations: List[Dict], min_samples: int = 40) -> Dict[str, Dict]:
    """observations: [{crop, date, ndvi}, ...] from clear optical reads only.

    Returns one entry per crop with the harmonic coefficients, the residual
    standard deviation, and the sample count. The residual sigma matters as much
    as the curve: it is the prior's uncertainty in the fusion step, and it is
    what stops a climatology from overruling a real measurement.
    """
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for o in observations:
        if o.get("ndvi") is None:
            continue
        d = date.fromisoformat(o["date"]) if isinstance(o["date"], str) else o["date"]
        grouped[o["crop"]].append((d.timetuple().tm_yday, o["ndvi"]))

    out: Dict[str, Dict] = {}
    for crop, rows in sorted(grouped.items()):
        if len(rows) < min_samples:
            out[crop] = {"fitted": False, "n": len(rows),
                         "reason": f"only {len(rows)} clear reads, need {min_samples}"}
            continue

        ata = [[0.0] * TERMS for _ in range(TERMS)]
        atb = [0.0] * TERMS
        for doy, ndvi in rows:
            x = _basis(doy)
            for i in range(TERMS):
                atb[i] += x[i] * ndvi
                for j in range(TERMS):
                    ata[i][j] += x[i] * x[j]
        for i in range(1, TERMS):
            ata[i][i] += 1e-6

        coef = _solve(ata, atb)
        if coef is None:
            out[crop] = {"fitted": False, "n": len(rows), "reason": "singular"}
            continue

        resid = [ndvi - sum(c * b for c, b in zip(coef, _basis(doy)))
                 for doy, ndvi in rows]
        sigma = math.sqrt(sum(r * r for r in resid) / len(resid))
        mean = sum(n for _, n in rows) / len(rows)
        var = sum((n - mean) ** 2 for _, n in rows) / len(rows)

        out[crop] = {
            "fitted": True,
            "n": len(rows),
            "coefficients": [round(c, 6) for c in coef],
            "residual_sigma": round(sigma, 4),
            "variance_explained": round(1 - (sigma ** 2) / var, 4) if var > 0 else 0.0,
        }
    return out


def predict(model: Dict, day_of_year: int) -> Optional[Tuple[float, float]]:
    """Expected NDVI and its uncertainty for this crop on this day of year."""
    if not model.get("fitted"):
        return None
    value = sum(c * b for c, b in zip(model["coefficients"], _basis(day_of_year)))
    return max(0.0, min(1.0, value)), model["residual_sigma"]


# --------------------------------------------------------------------------
# How fast a reading goes stale — measured, not assumed
# --------------------------------------------------------------------------


def fit_persistence_decay(
    series_by_field: Dict[Tuple[str, str], List[Tuple[date, float]]],
    max_gap_days: int = 60,
) -> Dict[str, Dict]:
    """Per-day standard deviation of NDVI change, from real clear-read pairs.

    The persistence component of the fusion needs to know how quickly a reading
    stops being true. Under a random walk the variance of the change grows
    linearly with elapsed time:

        var(ndvi(t+k) - ndvi(t)) = (decay_per_day^2) * k

    So every pair of clear optical reads on the same field gives one estimate of
    that slope. Fitting a line through the origin over all pairs gives
    decay_per_day.

    This replaces a constant that was previously invented. With that constant a
    35-day-old reading claimed an uncertainty of +-0.03, which is absurd for a
    crop whose NDVI moves by 0.3 over the same period — and it made persistence
    dominate the fusion by an order of magnitude it had not earned.
    """
    # Pairs are formed WITHIN a field and then pooled per crop. Pooling the
    # points first and pairing afterwards compares one field against another and
    # measures spatial difference as if it were temporal change — which produced
    # a decay of 0.031 NDVI per day, implying a crop changes by 0.3 in ten days.
    accum: Dict[str, List[float]] = {}
    for (crop, _field_id), series in sorted(series_by_field.items()):
        pts = sorted(series, key=lambda x: x[0])
        acc = accum.setdefault(crop, [0.0, 0.0, 0.0])
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                gap = (pts[j][0] - pts[i][0]).days
                if gap <= 0 or gap > max_gap_days:
                    continue
                delta = pts[j][1] - pts[i][1]
                acc[0] += gap * (delta ** 2)
                acc[1] += gap * gap
                acc[2] += 1

    out: Dict[str, Dict] = {}
    for crop, (num, den, n) in sorted(accum.items()):
        if n < 20 or den <= 0:
            out[crop] = {"fitted": False, "n": n,
                         "reason": f"only {n} usable pairs, need 20"}
            continue
        var_per_day = num / den
        out[crop] = {
            "fitted": True,
            "n": int(n),
            "decay_per_day": round(math.sqrt(max(var_per_day, 0.0)), 5),
            "max_gap_days": max_gap_days,
        }
    return out
