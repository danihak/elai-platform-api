"""
sentinelhub.py — pulls real observations from the Sentinel Hub Statistical API.

Run once, locally, with credentials in a .env. The result is cached to
data/observations.json and committed, so the deployed service serves real
satellite data without holding any credentials at runtime.

Two requests per farm:

  Sentinel-2 L2A  NDVI and the clear-pixel fraction. The evalscript masks cloud,
                  shadow and snow via SCL before the statistics are computed, so
                  the returned mean is a CLEAR-PIXEL mean rather than an average
                  contaminated by cloud tops. sampleCount against noDataCount
                  then gives the valid fraction directly — which is exactly the
                  input the confidence engine scores on.

  Sentinel-1 GRD  VV and VH in decibels. Independent of cloud, which is the whole
                  point of rung 2.

NOT YET RUN AGAINST THE LIVE API. Written from the documented request shapes; the
build environment has no route to services.sentinel-hub.com. Expect the first run
to surface something — run `python -m app.cli verify` first, which does a single
cheap call and prints the raw response on failure.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

BASE = os.getenv("SH_BASE_URL", "https://services.sentinel-hub.com")
TOKEN_URL = f"{BASE}/auth/realms/main/protocol/openid-connect/token"
STATS_URL = f"{BASE}/api/v1/statistics"

# Copernicus Data Space is a drop-in alternative:
#   SH_BASE_URL=https://sh.dataspace.copernicus.eu
# with its own OAuth client. Same request shapes.

CRS = "http://www.opengis.net/def/crs/EPSG/0/4326"


# --------------------------------------------------------------------------
# Evalscripts
# --------------------------------------------------------------------------

# SCL classes dropped: 3 cloud shadow, 8 cloud medium, 9 cloud high,
# 10 thin cirrus, 11 snow. Class 0 is no-data, 1 is saturated.
S2_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B04", "B08", "B11", "SCL", "dataMask"] }],
    output: [
      { id: "ndvi",  bands: 1, sampleType: "FLOAT32" },
      { id: "ndwi",  bands: 1, sampleType: "FLOAT32" },
      { id: "gci",   bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(s) {
  var bad = (s.SCL == 0 || s.SCL == 1 || s.SCL == 3 ||
             s.SCL == 8 || s.SCL == 9 || s.SCL == 10 || s.SCL == 11);
  var clear = (bad || s.dataMask == 0) ? 0 : 1;

  var ndvi = (s.B08 + s.B04) == 0 ? 0 : (s.B08 - s.B04) / (s.B08 + s.B04);
  var ndwi = (s.B08 + s.B11) == 0 ? 0 : (s.B08 - s.B11) / (s.B08 + s.B11);
  var gci  = s.B03 == 0 ? 0 : (s.B08 / s.B03) - 1;

  return {
    ndvi: [ndvi],
    ndwi: [ndwi],
    gci: [gci],
    dataMask: [clear]
  };
}
"""

# Landsat 8/9 OLI. Different band numbers to Sentinel-2: red is B04, NIR is B05,
# SWIR1 is B06, green is B03. The QA_PIXEL bitmask carries cloud, shadow, cirrus
# and snow flags.
#
# WHY THIS EXISTS. Sentinel-2 alone revisits every 5 days. Landsat 8 and 9 pass
# on entirely different days, so combining them roughly halves the expected gap
# between looks. That does not beat cloud — it beats SAMPLING, which is the
# binding constraint here: 119 clear Kharif pairs for cotton and 64 for maize is
# too few to validate anything, and no amount of patience fixes it.
#
# The cost is resolution. Landsat is 30 m against Sentinel-2's 10 m, so a 0.8 ha
# smallholder plot is a handful of pixels and is reported with that caveat. The
# 1 km training blocks are unaffected, and they are where the sample shortage
# actually bites.
LANDSAT_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B04", "B05", "B06", "BQA", "dataMask"] }],
    output: [
      { id: "ndvi",  bands: 1, sampleType: "FLOAT32" },
      { id: "ndwi",  bands: 1, sampleType: "FLOAT32" },
      { id: "gci",   bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

// QA_PIXEL bits: 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow.
function masked(qa) {
  var bits = [1, 2, 3, 4, 5];
  for (var i = 0; i < bits.length; i++) {
    if ((qa & (1 << bits[i])) !== 0) return true;
  }
  return false;
}

function evaluatePixel(s) {
  var clear = (masked(s.BQA) || s.dataMask == 0) ? 0 : 1;
  var ndvi = (s.B05 + s.B04) == 0 ? 0 : (s.B05 - s.B04) / (s.B05 + s.B04);
  var ndwi = (s.B05 + s.B06) == 0 ? 0 : (s.B05 - s.B06) / (s.B05 + s.B06);
  var gci  = s.B03 == 0 ? 0 : (s.B05 / s.B03) - 1;
  return { ndvi: [ndvi], ndwi: [ndwi], gci: [gci], dataMask: [clear] };
}
"""

S1_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH", "dataMask"] }],
    output: [
      { id: "vv", bands: 1, sampleType: "FLOAT32" },
      { id: "vh", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function toDb(linear) {
  return linear <= 0 ? -40 : 10 * Math.log(linear) / Math.LN10;
}

function evaluatePixel(s) {
  return {
    vv: [toDb(s.VV)],
    vh: [toDb(s.VH)],
    dataMask: [s.dataMask]
  };
}
"""


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


class SentinelHubError(RuntimeError):
    pass


@dataclass
class Client:
    client_id: str
    client_secret: str
    _token: Optional[str] = None
    _expires_at: float = 0.0

    @classmethod
    def from_env(cls) -> "Client":
        cid = os.getenv("SH_CLIENT_ID")
        secret = os.getenv("SH_CLIENT_SECRET")
        if not cid or not secret:
            raise SentinelHubError(
                "SH_CLIENT_ID and SH_CLIENT_SECRET are not set.\n"
                "Create an OAuth client at https://apps.sentinel-hub.com/dashboard "
                "under User Settings, then put both in a local .env file."
            )
        return cls(client_id=cid, client_secret=secret)

    def token(self) -> str:
        """Tokens last about an hour; reused until 60s before expiry."""
        if self._token and time.time() < self._expires_at - 60:
            return self._token

        resp = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            headers={"content-type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        if resp.status_code != 200:
            raise SentinelHubError(
                f"Token request failed ({resp.status_code}). Response:\n{resp.text[:600]}"
            )
        data = resp.json()
        self._token = data["access_token"]
        self._expires_at = time.time() + float(data.get("expires_in", 3600))
        return self._token

    # ---------------------------------------------------------------- stats

    def statistics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = httpx.post(
            STATS_URL,
            headers={
                "Authorization": f"Bearer {self.token()}",
                "content-type": "application/json",
                "accept": "application/json",
            },
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            raise SentinelHubError(
                f"Statistical API returned {resp.status_code}.\n"
                f"Request keys: {sorted(payload.keys())}\n"
                f"Response:\n{resp.text[:1200]}"
            )
        return resp.json()


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------


def _bounds(polygon: List[List[float]]) -> Dict[str, Any]:
    return {
        "geometry": {"type": "Polygon", "coordinates": [polygon]},
        "properties": {"crs": CRS},
    }


def _stats_payload(
    polygon: List[List[float]],
    collection: str,
    evalscript: str,
    start: date,
    end: date,
    interval_days: int,
    data_filter: Optional[Dict[str, Any]] = None,
    resolution_m: int = 10,
) -> Dict[str, Any]:
    # resx/resy are in the units of the BOUNDS CRS. We pass geometry in
    # EPSG:4326, so they must be degrees, not metres. Passing 10 here means ten
    # DEGREES — one pixel for the whole farm, no cloud fraction, no spatial
    # averaging, and a statistic that means nothing.
    res_deg = resolution_m / 111_320.0
    data_entry: Dict[str, Any] = {"type": collection}
    if data_filter:
        data_entry["dataFilter"] = data_filter

    return {
        "input": {"bounds": _bounds(polygon), "data": [data_entry]},
        "aggregation": {
            "timeRange": {
                "from": f"{start.isoformat()}T00:00:00Z",
                "to": f"{end.isoformat()}T23:59:59Z",
            },
            "aggregationInterval": {"of": f"P{interval_days}D"},
            "evalscript": evalscript,
            "resx": res_deg,
            "resy": res_deg,
        },
        "calculations": {"default": {}},
    }


def _band_stats(interval: Dict[str, Any], output_id: str) -> Optional[Dict[str, Any]]:
    outputs = interval.get("outputs", {})
    if output_id not in outputs:
        return None
    bands = outputs[output_id].get("bands", {})
    if not bands:
        return None
    return next(iter(bands.values())).get("stats")


def _valid_fraction(stats: Optional[Dict[str, Any]]) -> float:
    """Share of pixels that survived the cloud mask.

    sampleCount is every pixel in the polygon; noDataCount is those the
    evalscript masked out. Their ratio is the clear fraction — the exact input
    the confidence engine scores on.
    """
    if not stats:
        return 0.0
    total = stats.get("sampleCount", 0)
    nodata = stats.get("noDataCount", 0)
    if not total:
        return 0.0
    return max(0.0, min(1.0, (total - nodata) / total))


# --------------------------------------------------------------------------
# Per-farm pulls
# --------------------------------------------------------------------------


def fetch_optical(
    client: Client,
    polygon: List[List[float]],
    start: date,
    end: date,
    interval_days: int = 5,
) -> List[Dict[str, Any]]:
    payload = _stats_payload(
        polygon,
        "sentinel-2-l2a",
        S2_EVALSCRIPT,
        start,
        end,
        interval_days,
        data_filter={"mosaickingOrder": "leastCC"},
    )
    out: List[Dict[str, Any]] = []
    for interval in client.statistics(payload).get("data", []):
        if interval.get("outputs") is None:
            continue  # wholly cloudy interval — Sentinel Hub returns no outputs
        ndvi_stats = _band_stats(interval, "ndvi")
        valid = _valid_fraction(ndvi_stats)
        out.append({
            "date": interval["interval"]["from"][:10],
            "valid_pixel_fraction": round(valid, 4),
            "mean_cloud_probability": round(1.0 - valid, 4),
            "ndvi": round(ndvi_stats["mean"], 4) if ndvi_stats and valid > 0 else None,
            "ndvi_stddev": round(ndvi_stats.get("stDev", 0.0), 4) if ndvi_stats and valid > 0 else None,
            "ndwi": _mean(interval, "ndwi") if valid > 0 else None,
            "gci": _mean(interval, "gci") if valid > 0 else None,
        })
    return out


def _mean(interval: Dict[str, Any], output_id: str) -> Optional[float]:
    stats = _band_stats(interval, output_id)
    return round(stats["mean"], 4) if stats and "mean" in stats else None


def fetch_landsat(
    client: Client,
    polygon: List[List[float]],
    start: date,
    end: date,
    interval_days: int = 1,
) -> List[Dict[str, Any]]:
    """Landsat 8 and 9 surface reflectance, same outputs as the Sentinel-2 pull.

    Resolution is 30 m rather than 10 m, so rows are tagged `sensor: landsat`
    and `resolution_m: 30`. A consumer that cares — anything at smallholder
    plot scale — can filter them out; the training blocks can use them.
    """
    payload = _stats_payload(
        polygon, "landsat-ot-l2", LANDSAT_EVALSCRIPT, start, end, interval_days,
        data_filter={"mosaickingOrder": "leastCC"},
        resolution_m=30,
    )
    out: List[Dict[str, Any]] = []
    for interval in client.statistics(payload).get("data", []):
        if interval.get("outputs") is None:
            continue
        ndvi_stats = _band_stats(interval, "ndvi")
        valid = _valid_fraction(ndvi_stats)
        out.append({
            "date": interval["interval"]["from"][:10],
            "sensor": "landsat",
            "resolution_m": 30,
            "valid_pixel_fraction": round(valid, 4),
            "mean_cloud_probability": round(1.0 - valid, 4),
            "ndvi": round(ndvi_stats["mean"], 4) if ndvi_stats and valid > 0 else None,
            "ndwi": _mean(interval, "ndwi") if valid > 0 else None,
            "gci": _mean(interval, "gci") if valid > 0 else None,
        })
    return out


def merge_optical(
    sentinel: List[Dict[str, Any]],
    landsat: List[Dict[str, Any]],
    min_gap_days: int = 1,
) -> List[Dict[str, Any]]:
    """Combine both optical sources into one series, Sentinel-2 preferred.

    Where both sensors saw the field on the same day, Sentinel-2 wins: 10 m beats
    30 m and the cloud masks are not identical. Landsat contributes only on days
    Sentinel-2 did not cover, which is the entire point — the gain is extra
    LOOKS, not a better look.
    """
    from datetime import date as _date

    merged = {r["date"]: dict(r, sensor=r.get("sensor", "sentinel2"),
                              resolution_m=r.get("resolution_m", 10))
              for r in sentinel}

    s2_dates = sorted(_date.fromisoformat(d) for d in merged)
    added = 0
    for r in landsat:
        d = _date.fromisoformat(r["date"])
        if any(abs((d - s).days) < min_gap_days for s in s2_dates):
            continue
        merged[r["date"]] = r
        added += 1

    out = sorted(merged.values(), key=lambda x: x["date"])
    for r in out:
        r.setdefault("sensor", "sentinel2")
        r.setdefault("resolution_m", 10)
    return out


def fetch_radar(
    client: Client,
    polygon: List[List[float]],
    start: date,
    end: date,
    interval_days: int = 5,
) -> List[Dict[str, Any]]:
    payload = _stats_payload(
        polygon,
        "sentinel-1-grd",
        S1_EVALSCRIPT,
        start,
        end,
        interval_days,
        data_filter={
            "acquisitionMode": "IW",
            "polarization": "DV",
            "resolution": "HIGH",
        },
        resolution_m=20,
    )
    out: List[Dict[str, Any]] = []
    for interval in client.statistics(payload).get("data", []):
        if interval.get("outputs") is None:
            continue
        vv, vh = _mean(interval, "vv"), _mean(interval, "vh")
        if vv is None and vh is None:
            continue
        out.append({
            "date": interval["interval"]["from"][:10],
            "sar_vv": vv,
            "sar_vh": vh,
            "radar_sources": ["sentinel1"],
        })
    return out


def verify(client: Client, polygon: List[List[float]]) -> Dict[str, Any]:
    """One cheap call, to prove credentials and request shape before a full pull."""
    end = date.today()
    start = end - timedelta(days=30)
    payload = _stats_payload(
        polygon, "sentinel-2-l2a", S2_EVALSCRIPT, start, end, 10,
        data_filter={"mosaickingOrder": "leastCC"},
    )
    raw = client.statistics(payload)
    intervals = raw.get("data", [])
    return {
        "intervals_returned": len(intervals),
        "with_data": sum(1 for i in intervals if i.get("outputs")),
        "first": intervals[0] if intervals else None,
    }


# --------------------------------------------------------------------------
# Pairing optical and radar
# --------------------------------------------------------------------------


def match_radar_to_optical(
    optical: List[Dict[str, Any]],
    radar: List[Dict[str, Any]],
    tolerance_days: int = 3,
) -> List[Dict[str, Any]]:
    """Join each optical read to its NEAREST radar read within a tolerance.

    The first version keyed a dictionary on the exact interval start date. Since
    Sentinel-1 and Sentinel-2 almost never start an interval on the same day,
    nearly every radar reading failed to join and the training set collapsed to
    a handful of pairs.

    Three days is a deliberate compromise: long enough to catch the pairing,
    short enough that the crop has not meaningfully changed in between. The gap
    is recorded on every row so it can be audited, and a wider gap could later
    be down-weighted in the fit.
    """
    if not radar:
        return [dict(o, sar_vv=None, sar_vh=None, radar_sources=[],
                     radar_gap_days=None) for o in optical]

    radar_dated = sorted(
        ((date.fromisoformat(r["date"]), r) for r in radar), key=lambda x: x[0]
    )
    out: List[Dict[str, Any]] = []
    for o in optical:
        od = date.fromisoformat(o["date"])
        best, best_gap = None, None
        for rd, r in radar_dated:
            gap = abs((rd - od).days)
            if best_gap is None or gap < best_gap:
                best, best_gap = r, gap
        row = dict(o)
        if best is not None and best_gap is not None and best_gap <= tolerance_days:
            row.update({
                "sar_vv": best.get("sar_vv"),
                "sar_vh": best.get("sar_vh"),
                "radar_sources": best.get("radar_sources", ["sentinel1"]),
                "radar_gap_days": best_gap,
            })
        else:
            row.update({"sar_vv": None, "sar_vh": None, "radar_sources": [],
                        "radar_gap_days": best_gap})
        out.append(row)
    return out
