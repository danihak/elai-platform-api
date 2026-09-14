"""
seed.py — six Telangana farms with a full Kharif 2026 observation series.

Shaped deliberately like the DRISHTI walkthrough: Telangana, maize and cotton,
Kharif 2026, DAS-stamped observations, cloud flags, per-farm polygons named after
farmers. Nothing here is real ELAI data.

The cloud pattern is the important part. It is not random: it reproduces the
monsoon signature visible in the walkthrough, where every observation from late
August into September is flagged Cloudy. That window is what the whole product
argument is about, so the fixtures have to contain it.

Every value generated here is deterministic, seeded per farm, so a demo shows the
same numbers every time it is run.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from elai_confidence_core import Observation

REVISIT_DAYS = 5  # Sentinel-2 effective revisit with both satellites


@dataclass
class Farm:
    farm_id: str
    farm_name: str
    farmer_name: str
    mobile_masked: str
    district: str
    mandal: str
    area_ha: float
    lat: float
    lon: float
    crop: str
    variety: str
    irrigation_type: str
    sowing_date: date
    predicted_harvest_date: date
    client_id: str
    client_tier: str
    sla_tier: str
    boundary_source: str
    sowing_date_source: str
    baseline_yield_mt_ha: float
    polygon: List[Tuple[float, float]] = field(default_factory=list)
    state: str = "Telangana"
    country: str = "India"
    season: str = "Kharif 2026"


def _square(lat: float, lon: float, ha: float) -> List[Tuple[float, float]]:
    """A rough square polygon of the right area, for map rendering."""
    side_m = math.sqrt(ha * 10_000)
    dlat = (side_m / 2) / 111_320
    dlon = (side_m / 2) / (111_320 * math.cos(math.radians(lat)))
    return [
        (lon - dlon, lat - dlat),
        (lon + dlon, lat - dlat),
        (lon + dlon, lat + dlat),
        (lon - dlon, lat + dlat),
        (lon - dlon, lat - dlat),
    ]


FARMS: List[Farm] = [
    Farm("TS-MZ-0044", "Ramulu North Block", "K. Ramulu", "+91 ●●●●● ●2318",
         "Karimnagar", "Choppadandi", 0.81, 18.5127, 79.0912, "maize", "DHM-117",
         "Rainfed", date(2026, 6, 12), date(2026, 10, 4), "CLI-RRB-01", "lender",
         "assured", "hand_drawn", "farmer_reported", 7.2),
    Farm("TS-MZ-0051", "Sarojini Field", "M. Sarojini", "+91 ●●●●● ●7741",
         "Karimnagar", "Choppadandi", 1.24, 18.5203, 79.0988, "maize", "DHM-117",
         "Borewell", date(2026, 6, 9), date(2026, 10, 1), "CLI-RRB-01", "lender",
         "assured", "hand_drawn", "farmer_reported", 8.1),
    Farm("TS-MZ-0063", "Yadagiri Plot 2", "B. Yadagiri", "+91 ●●●●● ●1102",
         "Jagtial", "Korutla", 0.62, 18.8214, 78.7126, "maize", "Bio-9637",
         "Rainfed", date(2026, 6, 18), date(2026, 10, 10), "CLI-RRB-01", "lender",
         "standard", "gps_walked", "field_officer", 6.4),
    Farm("TS-CT-0107", "Lakshmi Cotton", "P. Lakshmi", "+91 ●●●●● ●5590",
         "Warangal", "Parkal", 1.05, 18.2041, 79.6883, "cotton", "NCS-855",
         "Rainfed", date(2026, 6, 21), date(2026, 12, 2), "CLI-NBFC-02", "lender",
         "premium", "hand_drawn", "farmer_reported", 2.3),
    Farm("TS-CT-0119", "Venkatesh South", "G. Venkatesh", "+91 ●●●●● ●8834",
         "Warangal", "Parkal", 0.74, 18.1987, 79.6951, "cotton", "NCS-855",
         "Borewell", date(2026, 6, 15), date(2026, 11, 26), "CLI-NBFC-02", "lender",
         "premium", "hand_drawn", "farmer_reported", 2.6),
    Farm("TS-CT-0126", "Anjamma Block", "T. Anjamma", "+91 ●●●●● ●2207",
         "Warangal", "Parkal", 1.42, 18.2112, 79.6802, "cotton", "RCH-659",
         "Rainfed", date(2026, 6, 24), date(2026, 12, 6), "CLI-NBFC-02", "lender",
         "standard", "gps_walked", "field_officer", 2.1),
]

for _f in FARMS:
    _f.polygon = _square(_f.lat, _f.lon, _f.area_ha)

FARMS_BY_ID: Dict[str, Farm] = {f.farm_id: f for f in FARMS}


# --------------------------------------------------------------------------
# Growth stages
# --------------------------------------------------------------------------

MAIZE_STAGES = [(0, "germination"), (20, "vegetative"), (52, "flowering"),
                (72, "grain_fill"), (105, "maturation"), (118, "harvest")]
COTTON_STAGES = [(0, "germination"), (25, "vegetative"), (60, "flowering"),
                 (100, "grain_fill"), (145, "maturation"), (160, "harvest")]


def stage_for(crop: str, das: int) -> str:
    table = MAIZE_STAGES if crop == "maize" else COTTON_STAGES
    current = table[0][1]
    for threshold, name in table:
        if das >= threshold:
            current = name
    return current


# --------------------------------------------------------------------------
# Cloud pattern
# --------------------------------------------------------------------------

def _cloud_probability(d: date, rng: random.Random) -> float:
    """Monsoon signature for a Telangana Kharif.

    Heavy from late June, peaking through August, clearing from late September.
    The late-August to mid-September plateau is what the walkthrough showed, and
    it is the window the product argument lives in.
    """
    doy = d.timetuple().tm_yday
    peak = 230  # mid August
    spread = 42.0
    seasonal = math.exp(-((doy - peak) ** 2) / (2 * spread ** 2))
    base = 0.12 + 0.80 * seasonal
    return max(0.0, min(0.99, base + rng.uniform(-0.18, 0.18)))


def _ndvi_curve(crop: str, das: int) -> float:
    """A plausible double-logistic growth curve, per crop."""
    if crop == "maize":
        green_up, senescence, peak = 28.0, 100.0, 0.82
    else:
        green_up, senescence, peak = 38.0, 140.0, 0.76
    rise = 1 / (1 + math.exp(-(das - green_up) / 9.0))
    fall = 1 / (1 + math.exp((das - senescence) / 11.0))
    return round(0.12 + (peak - 0.12) * rise * fall, 3)


# --------------------------------------------------------------------------
# Observation series
# --------------------------------------------------------------------------

def observations_for(farm: Farm, as_of: date, radar_enabled: bool = True) -> List[Observation]:
    """Generate the season-to-date series for one farm."""
    rng = random.Random(farm.farm_id)
    out: List[Observation] = []

    last_clear_date: Optional[date] = None
    last_clear_das: Optional[int] = None
    last_clear_ndvi: Optional[float] = None

    d = farm.sowing_date
    while d <= as_of:
        das = (d - farm.sowing_date).days
        stage = stage_for(farm.crop, das)
        cloud_p = _cloud_probability(d, rng)
        valid_fraction = max(0.0, min(1.0, 1.0 - cloud_p + rng.uniform(-0.08, 0.08)))
        cloudy = cloud_p > 0.35

        usable = valid_fraction >= 0.50
        true_ndvi = _ndvi_curve(farm.crop, das)
        ndvi = round(true_ndvi + rng.uniform(-0.02, 0.02), 3) if usable else None

        days_since = 999 if last_clear_date is None else (d - last_clear_date).days

        # Radar passes are independent of cloud. Sentinel-1 plus NISAR S and L
        # give roughly one usable radar read every few days.
        sources: Tuple[str, ...] = ()
        sar_vh = sar_vv = None
        if radar_enabled and rng.random() < 0.72:
            pool = ["sentinel1", "nisar_s", "nisar_l"]
            sources = tuple(s for s in pool if rng.random() < 0.6) or ("sentinel1",)
            sar_vh = round(-22.0 + 14.0 * true_ndvi + rng.uniform(-0.9, 0.9), 2)
            sar_vv = round(sar_vh + rng.uniform(4.0, 7.0), 2)

        out.append(Observation(
            farm_id=farm.farm_id,
            obs_date=d,
            das=das,
            crop=farm.crop,
            stage=stage,
            valid_pixel_fraction=round(valid_fraction, 3),
            mean_cloud_probability=round(cloud_p, 3),
            cloudy=cloudy,
            ndvi=ndvi,
            ndwi=round(0.30 + 0.25 * true_ndvi, 3) if usable else None,
            psri=round(max(0.0, 0.35 - 0.30 * true_ndvi), 3) if usable else None,
            gci=round(2.4 * true_ndvi, 3) if usable else None,
            red_edge=round(0.55 * true_ndvi + 0.08, 3) if usable else None,
            sar_available=bool(sources),
            sar_vv=sar_vv,
            sar_vh=sar_vh,
            radar_sources=sources,
            days_since_last_clear_optical=days_since,
            last_clear_optical_date=last_clear_date,
            last_clear_optical_das=last_clear_das,
            last_clear_optical_ndvi=last_clear_ndvi,
        ))

        if usable and ndvi is not None:
            last_clear_date, last_clear_das, last_clear_ndvi = d, das, ndvi

        d += timedelta(days=REVISIT_DAYS)

    return out


def yield_estimate(farm: Farm, obs: Observation) -> float:
    """A crude yield point estimate, scaled by how the canopy is tracking.

    Stands in for the trained yield model. Clearly not the real thing, and
    labelled as such wherever it surfaces.
    """
    expected = _ndvi_curve(farm.crop, obs.das)
    actual = obs.ndvi if obs.ndvi is not None else expected
    ratio = 1.0 if expected <= 0 else max(0.6, min(1.25, actual / expected))
    return round(farm.baseline_yield_mt_ha * ratio, 2)
