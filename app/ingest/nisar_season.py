"""
nisar_season.py — a full season of L-band, and what can honestly be done with it.

THREE THINGS, IN ORDER OF HOW MUCH THEY CAN BE CLAIMED.

1. THE SERIES. Every NISAR acquisition over each farm, reduced to HH, HV and an
   L-band Radar Vegetation Index. This is a measurement and needs no calibration
   against anything.

2. THE COINCIDENCES. Days where NISAR and a clear optical pass saw the same farm.
   These are the only rows that can teach L-band what NDVI looks like. There will
   not be many: NISAR L-band opened on 17 June 2026, Kharif is 24% clear, and
   roughly a third of granules are off-swath for a given farm. The count is
   reported before anything is fitted, because a fit on five pairs is not a fit.

3. THE FIT, IF THERE IS ENOUGH. Same Kharif-stratified, repeated-split gate as
   every other model. If the pairs are too few the correct output is the number
   of pairs and a statement that no model can be built yet — not a model built
   anyway.

ON CROP HEALTH. RVI is a vegetation index in its own right. It rises as a canopy
develops volume structure and falls at senescence, and unlike NDVI it is
unaffected by cloud. What it cannot do without calibration is state a yield or a
stress percentage. So health is reported as DIRECTION AND DEVIATION — is this
field tracking its own history and its neighbours, or departing from them — which
is answerable from the series alone and is what a field officer actually acts on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple


def to_linear(db: Optional[float]) -> Optional[float]:
    return None if db is None else 10.0 ** (db / 10.0)


def rvi_dual_pol(hh_db: Optional[float], hv_db: Optional[float]) -> Optional[float]:
    """Radar Vegetation Index for a dual-pol acquisition.

        RVI = 4 * HV / (HH + HV)   in LINEAR power, not decibels

    Bounded roughly 0 to 1. Near zero for a smooth bare surface, rising as a
    canopy develops the volume structure that converts co-polarised energy into
    cross-polarised return.

    It is the closest radar analogue to NDVI and — the point here — it is
    computed from a measurement taken through cloud.
    """
    hh, hv = to_linear(hh_db), to_linear(hv_db)
    if hh is None or hv is None:
        return None
    denom = hh + hv
    if denom <= 0:
        return None
    return max(0.0, min(1.0, 4.0 * hv / denom))


def geometry_of(granule: str) -> str:
    """The acquisition geometry, from the granule name.

    NISAR_L2_PR_GCOV_030_084_A_011_4005_DHDH_A_2026...
                         ^track ^direction

    WHY THIS EXISTS. Pooling geometries produced a series that alternated
    between 0.13 and 0.85 every six days on one farm — a six-fold swing that no
    crop performs. Radar backscatter depends on the angle the sensor looks from,
    so an ascending pass and a descending pass over the same field on the same
    day are two different measurements, not two samples of one.

    A series that mixes them is not a time series. It is two time series
    interleaved, and any trend read off it is an artefact of which direction the
    satellite happened to fly last.
    """
    parts = granule.split("_")
    if len(parts) < 7:
        return "unknown"
    return f"{parts[5]}{parts[6]}"      # e.g. "084A", "077D"


@dataclass
class LBandObservation:
    farm_id: str
    date: str
    hh_db: float
    hv_db: float
    pixels: int
    granule: str

    @property
    def geometry(self) -> str:
        return geometry_of(self.granule)

    @property
    def ratio_db(self) -> float:
        return self.hv_db - self.hh_db

    @property
    def rvi(self) -> Optional[float]:
        return rvi_dual_pol(self.hh_db, self.hv_db)


# --------------------------------------------------------------------------
# Coincidences with optical
# --------------------------------------------------------------------------


def find_coincidences(
    lband: List[LBandObservation],
    optical_rows: List[Dict],
    tolerance_days: int = 2,
    min_valid_fraction: float = 0.80,
) -> List[Dict]:
    """Days where L-band and a trustworthy optical read saw the same farm.

    Two days of tolerance: long enough to catch a pairing given different orbits,
    short enough that a crop has not meaningfully changed. The gap is recorded on
    every row so it can be audited or down-weighted later.
    """
    clear = [
        (date.fromisoformat(r["date"]), r)
        for r in optical_rows
        if r.get("ndvi") is not None
        and r.get("valid_pixel_fraction", 0) >= min_valid_fraction
    ]
    if not clear:
        return []

    out: List[Dict] = []
    for ob in lband:
        if not ob.date:
            continue
        d = date.fromisoformat(ob.date)
        best, gap = None, None
        for od, row in clear:
            g = abs((od - d).days)
            if gap is None or g < gap:
                best, gap = row, g
        if best is None or gap is None or gap > tolerance_days:
            continue
        out.append({
            "farm_id": ob.farm_id,
            "date": ob.date,
            "gap_days": gap,
            "hh_db": ob.hh_db,
            "hv_db": ob.hv_db,
            "ratio_db": ob.ratio_db,
            "rvi": ob.rvi,
            "ndvi": best["ndvi"],
            "pixels": ob.pixels,
        })
    return out


# --------------------------------------------------------------------------
# Health from the series alone
# --------------------------------------------------------------------------


@dataclass
class HealthReading:
    farm_id: str
    date: str
    rvi: float
    trend_per_day: Optional[float]
    peer_median: Optional[float]
    deviation: Optional[float]
    verdict: str
    basis: str


def split_by_geometry(series: List[LBandObservation]) -> Dict[str, List[LBandObservation]]:
    """One series per track and look direction.

    Everything downstream — trends, peer comparison, any future model — operates
    within a geometry, never across one.
    """
    out: Dict[str, List[LBandObservation]] = {}
    for o in series:
        out.setdefault(o.geometry, []).append(o)
    return {k: sorted(v, key=lambda x: x.date) for k, v in sorted(out.items())}


def dominant_geometry(series: List[LBandObservation]) -> Optional[str]:
    """The geometry with the most observations — the one worth reading a trend from."""
    groups = split_by_geometry(series)
    if not groups:
        return None
    return max(groups, key=lambda k: len(groups[k]))


def assess_health(
    farm_id: str,
    series: List[LBandObservation],
    peers: Dict[str, List[LBandObservation]],
    window: int = 3,
    require_agreement: bool = True,
) -> Optional[HealthReading]:
    """Direction and deviation, not a yield or a stress percentage.

    Two questions that are answerable from an uncalibrated index, and that are
    the ones a field officer acts on:

      Is this field's canopy still developing, or has it turned?
      Is it doing what its neighbours are doing this week?

    A yield figure or a stress percentage would need the index calibrated against
    ground truth, which does not exist for L-band on these farms. Reporting
    direction and peer deviation is what the data supports, and saying so is the
    difference between a measurement and a guess.
    """
    # Within one geometry only. A trend computed across look directions
    # measures the satellite's flight path, not the crop.
    geom = dominant_geometry(series)
    if geom is None:
        return None
    own = split_by_geometry(series).get(geom, [])
    pts = [(o.date, o.rvi) for o in own if o.rvi is not None]
    if not pts:
        return None

    latest_date, latest_rvi = pts[-1]

    trend = None
    if len(pts) >= 2:
        recent = pts[-window:] if len(pts) >= window else pts
        d0 = date.fromisoformat(recent[0][0])
        d1 = date.fromisoformat(recent[-1][0])
        days = (d1 - d0).days
        if days > 0:
            trend = (recent[-1][1] - recent[0][1]) / days

    # Peers are compared within the SAME geometry too. A farm looked at from
    # track 084 ascending cannot be compared against one looked at from 077
    # descending — that difference alone was six times larger than any crop
    # signal in this data.
    peer_vals: List[float] = []
    for pid, plist in peers.items():
        if pid == farm_id:
            continue
        same_geom = [o for o in plist if o.geometry == geom]
        near = [o.rvi for o in same_geom
                if o.rvi is not None and o.date
                and abs((date.fromisoformat(o.date) - date.fromisoformat(latest_date)).days) <= 8]
        peer_vals.extend(v for v in near if v is not None)

    peer_median = None
    deviation = None
    if len(peer_vals) >= 2:
        peer_vals.sort()
        peer_median = peer_vals[len(peer_vals) // 2]
        deviation = latest_rvi - peer_median

    # CROSS-GEOMETRY AGREEMENT.
    #
    # Separating the series stopped a farm looking like it collapsed every six
    # days, but it did not stop a second error: one farm reads 0.10 on the
    # descending track and 0.79 on the ascending one, both perfectly stable. It
    # is not stressed — it scatters strongly in one look direction, most likely
    # because its rows align with that geometry. Judged on the dominant track
    # alone it scored -0.348 against its peers and was called "below
    # neighbours", which would have sent a field officer to a healthy field.
    #
    # So a comparative verdict now requires the SAME finding on every geometry
    # that has enough data. A claim that holds from one angle and not the other
    # is a claim about the angle.
    agreement = _agreement(farm_id, series, peers, window)

    if deviation is not None and deviation < -0.06:
        verdict = ("below neighbours" if agreement.get("below")
                   else "inconclusive — geometries disagree")
    elif deviation is not None and deviation > 0.06:
        verdict = ("above neighbours" if agreement.get("above")
                   else "inconclusive — geometries disagree")
    elif trend is not None and trend < -0.004:
        verdict = ("canopy declining" if agreement.get("declining", True)
                   else "inconclusive — geometries disagree")
    elif trend is not None and trend > 0.004:
        verdict = ("canopy developing" if agreement.get("developing", True)
                   else "inconclusive — geometries disagree")
    else:
        verdict = "steady"

    basis_bits = [f"{len(pts)} L-band reads on geometry {geom}"]

    # Only report the agreement check when a comparative verdict was actually
    # attempted. Saying "not confirmed across geometries" beneath a verdict of
    # "steady" implies a claim was tested and failed, when none was made.
    claimed = verdict not in ("steady",)
    if agreement.get("checked") and claimed:
        basis_bits.append(
            "confirmed on " + " and ".join(agreement["geometries"])
            if "inconclusive" not in verdict
            else "the finding holds on "
                 + agreement["geometries"][0] + " but not on "
                 + agreement["geometries"][-1])
    elif agreement.get("checked"):
        basis_bits.append(
            "checked on " + " and ".join(agreement["geometries"]))
    if trend is not None:
        basis_bits.append(f"trend {trend:+.4f} RVI/day")
    if peer_median is not None:
        basis_bits.append(f"{len(peer_vals)} peer reads, median {peer_median:.3f}")

    return HealthReading(
        farm_id=farm_id,
        date=latest_date,
        rvi=latest_rvi,
        trend_per_day=trend,
        peer_median=peer_median,
        deviation=deviation,
        verdict=verdict,
        basis="; ".join(basis_bits),
    )


# --------------------------------------------------------------------------
# Fitting, only if the evidence supports it
# --------------------------------------------------------------------------


def _agreement(
    farm_id: str,
    series: List[LBandObservation],
    peers: Dict[str, List[LBandObservation]],
    window: int,
) -> Dict[str, object]:
    """Does the same finding hold on every geometry with enough data.

    Returns flags for each verdict type, plus what was checked. A farm with only
    one usable geometry returns `checked: False` and the caller falls back to
    the single-geometry reading — stated, not silently.
    """
    groups = split_by_geometry(series)
    usable = {g: o for g, o in groups.items()
              if len([x for x in o if x.rvi is not None]) >= 3}
    if len(usable) < 2:
        return {"checked": False, "below": True, "above": True,
                "declining": True, "developing": True,
                "why": "only one geometry has enough reads"}

    below, above, declining, developing = [], [], [], []

    for geom, obs in usable.items():
        pts = [(o.date, o.rvi) for o in sorted(obs, key=lambda x: x.date)
               if o.rvi is not None]
        latest_date, latest = pts[-1]

        peer_vals: List[float] = []
        for pid, plist in peers.items():
            if pid == farm_id:
                continue
            for o in plist:
                if o.geometry != geom or o.rvi is None or not o.date:
                    continue
                gap = abs((date.fromisoformat(o.date)
                           - date.fromisoformat(latest_date)).days)
                if gap <= 8:
                    peer_vals.append(o.rvi)

        if len(peer_vals) >= 2:
            peer_vals.sort()
            med = peer_vals[len(peer_vals) // 2]
            below.append(latest - med < -0.06)
            above.append(latest - med > 0.06)

        recent = pts[-window:] if len(pts) >= window else pts
        days = (date.fromisoformat(recent[-1][0])
                - date.fromisoformat(recent[0][0])).days
        if days > 0:
            slope = (recent[-1][1] - recent[0][1]) / days
            declining.append(slope < -0.004)
            developing.append(slope > 0.004)

    def all_true(xs):
        return bool(xs) and all(xs)

    return {
        "checked": True,
        "geometries": sorted(usable),
        "below": all_true(below),
        "above": all_true(above),
        "declining": all_true(declining),
        "developing": all_true(developing),
        "confirmed": any([all_true(below), all_true(above),
                          all_true(declining), all_true(developing)]),
        "why": "the finding holds on one geometry but not the other",
    }


def fit_lband_to_ndvi(
    coincidences: List[Dict],
    min_pairs: int = 30,
    splits: int = 30,
    holdout: float = 0.3,
) -> Dict:
    """L-band features to NDVI, judged the same way as everything else.

    Refuses below `min_pairs` rather than producing a number. The C-band model
    was believed for a whole evening on the strength of a single split over 23
    rows; that mistake is not worth repeating on a new sensor.
    """
    import random
    import statistics

    rows = [c for c in coincidences
            if c.get("ndvi") is not None and c.get("rvi") is not None]
    if len(rows) < min_pairs:
        return {
            "fitted": False,
            "n": len(rows),
            "reason": (f"only {len(rows)} L-band/optical coincidences; need "
                       f"{min_pairs}. NISAR L-band opened on 17 June 2026 and "
                       f"Kharif is 24% clear, so coincidences are structurally "
                       f"scarce this season."),
        }

    feats = ("hh_db", "hv_db", "ratio_db", "rvi")

    def design(r):
        return [1.0] + [r[f] for f in feats]

    def solve(train):
        k = len(feats) + 1
        ata = [[0.0] * k for _ in range(k)]
        atb = [0.0] * k
        for r in train:
            x, y = design(r), r["ndvi"]
            for i in range(k):
                atb[i] += x[i] * y
                for j in range(k):
                    ata[i][j] += x[i] * x[j]
        for i in range(1, k):
            ata[i][i] += 1e-6
        m = [row[:] + [atb[i]] for i, row in enumerate(ata)]
        for col in range(k):
            piv = max(range(col, k), key=lambda rr: abs(m[rr][col]))
            if abs(m[piv][col]) < 1e-12:
                return None
            m[col], m[piv] = m[piv], m[col]
            for rr in range(col + 1, k):
                f = m[rr][col] / m[col][col]
                for cc in range(col, k + 1):
                    m[rr][cc] -= f * m[col][cc]
        out = [0.0] * k
        for rr in range(k - 1, -1, -1):
            s = m[rr][k] - sum(m[rr][cc] * out[cc] for cc in range(rr + 1, k))
            out[rr] = s / m[rr][rr]
        return out

    rmses, r2s = [], []
    coef = None
    for i in range(splits):
        rng = random.Random(4471 + i)
        pool = rows[:]
        rng.shuffle(pool)
        cut = int(len(pool) * (1 - holdout))
        train, test = pool[:cut], pool[cut:]
        if len(train) <= len(feats) + 1 or len(test) < 5:
            continue
        c = solve(train)
        if c is None:
            continue
        preds = [(sum(a * b for a, b in zip(c, design(r))), r["ndvi"]) for r in test]
        err = [p - a for p, a in preds]
        rmses.append(math.sqrt(sum(e * e for e in err) / len(err)))
        actual = [a for _, a in preds]
        mean = sum(actual) / len(actual)
        ss_tot = sum((a - mean) ** 2 for a in actual)
        r2s.append(0.0 if ss_tot == 0 else 1 - sum(e * e for e in err) / ss_tot)
        coef = c

    if not rmses or coef is None:
        return {"fitted": False, "n": len(rows), "reason": "no usable split"}

    mean_rmse = statistics.mean(rmses)
    sd = statistics.pstdev(rmses) if len(rmses) > 1 else 0.0
    return {
        "fitted": True,
        "n": len(rows),
        "n_splits": len(rmses),
        "features": list(feats),
        "intercept": round(coef[0], 6),
        "coefficients": {f: round(c, 6) for f, c in zip(feats, coef[1:])},
        "rmse": round(mean_rmse, 4),
        "rmse_sd": round(sd, 4),
        "rmse_upper": round(mean_rmse + sd, 4),
        "r2": round(statistics.mean(r2s), 4),
    }
