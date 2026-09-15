"""
fit.py — fits the radar-to-NDVI estimator on real clear-day pairs.

Rewritten after the first fit on real data refuted the original design.

WHAT THE DATA SAID. Regressing NDVI on VH backscatter alone gave a maize slope
of -0.0004 — no relationship at all — and held-out RMSE of 0.13 to 0.14 against
a 0.12 decision-grade ceiling. Cotton carried some signal, maize essentially
none.

WHY. C-band backscatter responds to soil moisture as strongly as to vegetation.
After monsoon rain a wet bare field scatters much like a canopy, so VH alone
cannot separate them. The standard remedy is the cross-polarisation ratio and
the Radar Vegetation Index, which normalise out much of the moisture and
geometry effect.

WHAT CHANGED. Multivariate least squares over physically motivated features:

  vh_db     cross-polarised backscatter, sensitive to volume scattering
  vv_db     co-polarised, more sensitive to surface and moisture
  ratio_db  vh_db - vv_db, the cross-pol ratio; in dB a ratio is a difference
  rvi       4*VH / (VV + VH) in LINEAR power, the Radar Vegetation Index —
            bounded roughly 0 to 1 and the closest radar analogue to NDVI

Solved with normal equations and a small ridge term, in pure Python, so nothing
here adds a dependency.

Still no ELAI ground truth required. On a clear day the optical NDVI is the
label, so optical teaches radar. Ground truth is for the yield model, which is a
different problem.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

FEATURES = ("vh_db", "vv_db", "ratio_db", "rvi")

#: Months that count as Kharif in Telangana. Sowing June, harvest through
#: October for cotton.
KHARIF_MONTHS = {"06", "07", "08", "09", "10"}


def is_kharif(iso_date: str) -> bool:
    return iso_date[5:7] in KHARIF_MONTHS


def _to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def build_features(sar_vh: Optional[float], sar_vv: Optional[float]) -> Optional[Dict[str, float]]:
    """Physical radar features from VV and VH in decibels.

    Returns None when either polarisation is missing. The row is dropped rather
    than back-filled — a fabricated feature is exactly the quiet guess this
    product exists to prevent.
    """
    if sar_vh is None or sar_vv is None:
        return None
    vh_lin, vv_lin = _to_linear(sar_vh), _to_linear(sar_vv)
    denom = vv_lin + vh_lin
    if denom <= 0:
        return None
    return {
        "vh_db": sar_vh,
        "vv_db": sar_vv,
        "ratio_db": sar_vh - sar_vv,
        "rvi": 4.0 * vh_lin / denom,
    }


def _solve(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Gaussian elimination with partial pivoting."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / pv
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * out[c] for c in range(r + 1, n))
        out[r] = s / m[r][r]
    return out


def _fit_ols(rows: List[Dict], features: Sequence[str], ridge: float = 1e-6) -> Optional[List[float]]:
    """Returns [intercept, *coefficients], or None if the system is singular."""
    k = len(features) + 1
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for r in rows:
        x = [1.0] + [r["features"][f] for f in features]
        y = r["ndvi"]
        for i in range(k):
            atb[i] += x[i] * y
            for j in range(k):
                ata[i][j] += x[i] * x[j]
    for i in range(1, k):
        ata[i][i] += ridge  # stabilises correlated features; intercept untouched
    return _solve(ata, atb)


def _predict(coef: Sequence[float], feats: Dict[str, float], features: Sequence[str]) -> float:
    return coef[0] + sum(c * feats[f] for c, f in zip(coef[1:], features))


def _rmse(pairs: List[Tuple[float, float]]) -> float:
    if not pairs:
        return float("inf")
    return math.sqrt(sum((p - a) ** 2 for p, a in pairs) / len(pairs))


def _r2(pairs: List[Tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    actual = [a for _, a in pairs]
    mean = sum(actual) / len(actual)
    ss_tot = sum((a - mean) ** 2 for a in actual)
    ss_res = sum((p - a) ** 2 for p, a in pairs)
    return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot


def _prepare(pairs: List[Dict]) -> List[Dict]:
    out = []
    for p in pairs:
        feats = build_features(p.get("sar_vh"), p.get("sar_vv"))
        if feats is None or p.get("ndvi") is None:
            continue
        date = p.get("date") or p.get("obs_date") or ""
        out.append({
            "crop": p["crop"], "stage": p.get("stage", "unknown"),
            "ndvi": p["ndvi"], "features": feats,
            # The condition the observation was taken in, so the hold-out can be
            # stratified by the condition the model is deployed in.
            "condition": p.get("condition") or
                         ("kharif" if date and is_kharif(date) else "other"),
        })
    return out


#: How many stratified splits to average a score over.
#:
#: A Kharif-only hold-out on these farms is 7 to 36 rows. One split on 23 rows
#: produced an RMSE of 0.098 for cotton where the mean over 30 splits was 0.132 —
#: the gate admitted a model on noise. A single split is not a measurement.
N_SPLITS = 30


def _fit_group(rows, features, min_samples, holdout, rng, deploy_only=None) -> Dict:
    """deploy_only: if given, the HOLD-OUT is drawn only from rows matching it.

    A random hold-out across the whole year flatters the model. Measured on
    these six farms: a radar model evaluated on a mixed hold-out scored 0.090
    for maize; the same model on a Kharif-only hold-out scored 0.156. The gap is
    not noise, it is the test set quietly filling with easy clear-sky Rabi days
    that the model will never encounter in the season it is deployed in.

    So the hold-out is stratified by DEPLOYMENT CONDITION. The model may train on
    anything; it is judged on the condition it will actually face.
    """
    if len(rows) < min_samples:
        return {"fitted": False, "n": len(rows),
                "reason": f"only {len(rows)} clear-day pairs, need {min_samples}"}
    import statistics

    eligible_all = ([r for r in rows if r.get("condition") == deploy_only]
                    if deploy_only else rows)
    other_all = ([r for r in rows if r.get("condition") != deploy_only]
                 if deploy_only else [])

    if deploy_only and len(eligible_all) < 20:
        return {"fitted": False, "n": len(rows),
                "reason": (f"only {len(eligible_all)} rows in the deployment "
                           f"condition '{deploy_only}'; need 20")}

    scores, r2s, biases = [], [], []
    last_coef, n_train, n_test = None, 0, 0

    # Averaged over repeated stratified splits, because the hold-out is small
    # enough that a single draw is luck rather than evidence.
    for i in range(N_SPLITS):
        srng = random.Random(4471 + i)
        pool = eligible_all[:]
        srng.shuffle(pool)
        cut = int(len(pool) * (1 - holdout))
        train, test = pool[:cut] + other_all, pool[cut:]
        if len(train) <= len(features) + 1 or len(test) < 5:
            continue
        coef = _fit_ols(train, features)
        if coef is None:
            continue
        preds = [(_predict(coef, r["features"], features), r["ndvi"]) for r in test]
        scores.append(_rmse(preds))
        r2s.append(_r2(preds))
        biases.append(sum(p - a for p, a in preds) / len(preds))
        last_coef, n_train, n_test = coef, len(train), len(test)

    if not scores or last_coef is None:
        return {"fitted": False, "n": len(rows), "reason": "no usable split"}

    mean_rmse = statistics.mean(scores)
    sd_rmse = statistics.pstdev(scores) if len(scores) > 1 else 0.0

    # The model is finally fitted on everything, but judged on the repeated
    # hold-out above.
    coef = _fit_ols(eligible_all + other_all, features) or last_coef

    return {
        "fitted": True, "n": len(rows), "n_train": n_train, "n_holdout": n_test,
        "n_splits": len(scores),
        "holdout_condition": deploy_only or "mixed",
        "features": list(features),
        "intercept": round(coef[0], 6),
        "coefficients": {f: round(c, 6) for f, c in zip(features, coef[1:])},
        "rmse": round(mean_rmse, 4),
        "rmse_sd": round(sd_rmse, 4),
        # The figure the gate judges: one standard deviation above the mean.
        # With hold-outs this small, passing on the mean means passing half the
        # time by luck.
        "rmse_upper": round(mean_rmse + sd_rmse, 4),
        "r2": round(statistics.mean(r2s), 4),
        "bias": round(statistics.mean(biases), 4),
    }


def fit_per_crop_stage(pairs, min_samples: int = 30, holdout: float = 0.3,
                       seed: int = 4471, features: Sequence[str] = FEATURES,
                       deploy_only: Optional[str] = "kharif") -> Dict[str, Dict]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in _prepare(pairs):
        grouped[f"{r['crop']}:{r['stage']}"].append(r)
    return {k: _fit_group(v, features, min_samples, holdout, rng, deploy_only)
            for k, v in sorted(grouped.items())}


def fit_pooled_by_crop(pairs, min_samples: int = 30, holdout: float = 0.3,
                       seed: int = 4471, features: Sequence[str] = FEATURES,
                       deploy_only: Optional[str] = "kharif") -> Dict[str, Dict]:
    """One relation per crop, pooled across stages.

    Labelled `crop:*` wherever it surfaces, because backscatter responds
    differently to a seedling and a closed canopy. Used where stage-level
    samples are thin, and for training blocks where no sowing date exists so
    stage cannot be derived at all.
    """
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in _prepare(pairs):
        grouped[r["crop"]].append(r)
    return {f"{k}:*": _fit_group(v, features, min_samples, holdout, rng, deploy_only)
            for k, v in sorted(grouped.items())}


def compare_feature_sets(pairs, min_samples: int = 30) -> Dict[str, Dict]:
    """Fit each crop under several feature sets so the choice is evidenced.

    If VH alone were adequate, this table would show it.
    """
    sets = {
        "vh_only": ("vh_db",),
        "vv_only": ("vv_db",),
        "ratio_only": ("ratio_db",),
        "rvi_only": ("rvi",),
        "vh_vv": ("vh_db", "vv_db"),
        "full": FEATURES,
    }
    out: Dict[str, Dict] = {}
    for name, feats in sets.items():
        res = fit_pooled_by_crop(pairs, min_samples=min_samples, features=feats)
        out[name] = {k: {"rmse": v.get("rmse"), "r2": v.get("r2"), "n": v.get("n")}
                     for k, v in res.items() if v.get("fitted")}
    return out


def to_estimator_table(results: Dict[str, Dict], max_rmse: float = 0.12) -> Dict:
    """Only combinations whose held-out RMSE clears the ceiling are validated.

    Fitted but wide stays out, and the engine routes those observations to Blind
    rather than showing a number nobody should act on.
    """
    models, validated, rejected = {}, [], {}
    for combo, r in results.items():
        if not r.get("fitted"):
            rejected[combo] = r.get("reason", "not fitted")
            continue
        models[combo] = {
            "features": r["features"], "intercept": r["intercept"],
            "coefficients": r["coefficients"], "rmse": r["rmse"],
            "rmse_sd": r.get("rmse_sd"), "rmse_upper": r.get("rmse_upper"),
            "r2": r.get("r2"),
        }
        # Judged on mean + 1 sd across repeated splits, not on one draw.
        judged = r.get("rmse_upper", r["rmse"])
        if judged <= max_rmse:
            validated.append(combo)
        else:
            rejected[combo] = (
                f"RMSE {r['rmse']}±{r.get('rmse_sd', 0)} over "
                f"{r.get('n_splits', 1)} splits; upper bound {judged} exceeds {max_rmse}")
    return {"models": models, "validated": sorted(validated),
            "rejected": rejected, "max_rmse": max_rmse}
