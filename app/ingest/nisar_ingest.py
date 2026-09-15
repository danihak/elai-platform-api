"""
nisar_ingest.py — L-band backscatter over our farms, without downloading 324 GB.

WHY THIS BECAME CHEAP. The first assessment said a NISAR pipeline was days of
work: gigabyte HDF5 granules, no statistics API, write your own chunk reader.
That was wrong because the tooling already exists.

ASF's own guidance is that GDAL's /vsicurl driver subsets NISAR HDF5 in place,
and openSEPPO wraps exactly that:

    seppo_nisar_search           find granule URLs via NASA Earthdata CMR
    seppo_nisar_gcov_convert     subset a GCOV granule straight from S3,
                                 -projwin for a bounding box, -vars to pick
                                 polarisations, -dB for decibels
    seppo_earthaccess_credentials  manage the Earthdata token

So a 0.8 hectare farm costs a few megabytes out of an 8 GB granule, and the
whole thing is `pip install openseppo`.

WHY L-BAND AT ALL. Measured on our own data: every cloudy run in Kharif lasts
longer than three days, the median is 10 days and the longest is 70. No optical
sensor at any revisit rate gets through that — we checked Landsat and the gaps
are structurally too long. C-band radar then failed a Kharif hold-out for both
crops. L-band is the last remaining lever, and it is free.

WHAT IS STILL HONEST TO SAY. NISAR L2 products are PROVISIONAL with a validated
reprocessing campaign expected, and no peer-reviewed agricultural result on real
NISAR data exists yet. Anything built here is research until it clears the same
Kharif-stratified gate as everything else.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Polarisations to extract. HV is the cross-polarised channel and carries the
#: volume-scattering signal that tracks canopy; HH is co-polarised and more
#: sensitive to the surface. Their ratio is the feature that rescued C-band, and
#: it should matter more at L-band, not less.
DEFAULT_VARS = ["HHHH", "HVHV"]

#: Extra margin around a farm polygon so the subset window always contains it
#: after reprojection. About 100 m.
BBOX_PAD_DEG = 0.001


def _run(cmd: List[str], timeout: int = 900) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]} not found. Run: pip install openseppo rasterio xarray h5py earthaccess"
    except subprocess.TimeoutExpired:
        return 124, f"{cmd[0]} timed out after {timeout}s"


def check_environment() -> Dict[str, object]:
    """What is installed, and is there an Earthdata login to use.

    Reported rather than assumed, because every previous NISAR step failed on
    something environmental and silently looked like an absence of data.
    """
    out: Dict[str, object] = {}
    for tool in ("seppo_nisar_search", "seppo_nisar_gcov_convert",
                 "seppo_earthaccess_credentials"):
        out[tool] = shutil.which(tool) is not None
    for mod in ("rasterio", "xarray", "h5py", "earthaccess"):
        try:
            __import__(mod)
            out[mod] = True
        except ImportError:
            out[mod] = False

    netrc = Path.home() / ("_netrc" if os.name == "nt" else ".netrc")
    out["netrc_present"] = netrc.exists()
    out["netrc_has_earthdata"] = (
        netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text(errors="ignore")
    )
    out["ready"] = all(v for k, v in out.items() if k != "netrc_present")
    return out


def bbox_for(polygon: List[Tuple[float, float]], pad: float = BBOX_PAD_DEG) -> Tuple[float, float, float, float]:
    """(ul_lon, ul_lat, lr_lon, lr_lat) in the order -projwin expects."""
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    return (min(lons) - pad, max(lats) + pad, max(lons) + pad, min(lats) - pad)


def search(
    bbox: Tuple[float, float, float, float],
    start: date,
    end: date,
    product: str = "GCOV",
    limit: int = 200,
) -> Tuple[List[str], str]:
    """Granule URLs over a bounding box. Returns (urls, log)."""
    ul_lon, ul_lat, lr_lon, lr_lat = bbox
    code, log = _run([
        "seppo_nisar_search",
        "--product", product,
        "--ullr", str(ul_lon), str(ul_lat), str(lr_lon), str(lr_lat),
        "--start_time_after", start.isoformat(),
        "--start_time_before", end.isoformat(),
        "--limit", str(limit),
        "--format", "url",
    ])
    if code != 0:
        return [], log
    urls = [l.strip() for l in log.splitlines()
            if l.strip().startswith(("s3://", "https://")) and ".h5" in l]
    return urls, log


def subset(
    url: str,
    bbox: Tuple[float, float, float, float],
    outdir: Path,
    variables: Optional[List[str]] = None,
    downscale: int = 1,
) -> Tuple[bool, str]:
    """Pull just our window out of one granule, in decibels.

    `-projwin` is the whole trick: GDAL reads only the byte ranges covering the
    window, so an 8 GB granule costs megabytes. `-dB` matches the units the
    estimator's features already use, so nothing downstream converts anything.
    """
    ul_lon, ul_lat, lr_lon, lr_lat = bbox
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "seppo_nisar_gcov_convert",
        "-i", url,
        "-o", str(outdir) + os.sep,
        "-vars", *(variables or DEFAULT_VARS),
        "-dB",
        "-of", "COG",
        "-projwin", str(ul_lon), str(ul_lat), str(lr_lon), str(lr_lat),
        "-projwin_srs", "EPSG:4326",
        "--no_vrt", "--no_time_series",
    ]
    if downscale > 1:
        cmd += ["-d", str(downscale)]
    code, log = _run(cmd)
    return code == 0, log


def polygon_mean(tif: Path, polygon: List[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    """Mean backscatter inside the farm boundary.

    Masked to the polygon rather than the bounding box, because a 0.8 ha plot
    inside a square window is mostly other people's fields. NISAR posts GCOV at
    20 m, so a small plot is a handful of pixels and the pixel count is returned
    alongside the mean — a mean over three pixels is not the same measurement as
    a mean over three hundred, and the caller should be able to tell.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.mask import mask
    except ImportError:
        return None

    geom = {"type": "Polygon", "coordinates": [[list(p) for p in polygon]]}
    try:
        with rasterio.open(tif) as src:
            data, _ = mask(src, [geom], crop=True, filled=True, nodata=np.nan)
    except Exception:  # noqa: BLE001 — a window that misses the polygon is normal
        return None

    band = data[0].astype("float64")
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return None
    return {
        "mean_db": float(np.mean(valid)),
        "std_db": float(np.std(valid)),
        "pixels": int(valid.size),
    }


@dataclass
class NisarObservation:
    farm_id: str
    date: str
    hh_db: Optional[float]
    hv_db: Optional[float]
    pixels: int
    granule: str

    @property
    def ratio_db(self) -> Optional[float]:
        if self.hh_db is None or self.hv_db is None:
            return None
        return self.hv_db - self.hh_db

    def as_pair_row(self) -> Dict[str, object]:
        """Shaped like the Sentinel-1 rows the fitter already consumes.

        L-band is mapped onto the same vv/vh slots so it flows through the
        existing feature builder unchanged — HH stands in for VV as the
        co-polarised channel, HV for VH as the cross-polarised one. The `sensor`
        field keeps them separable, because a model fitted on C-band must never
        be silently applied to L-band.
        """
        return {
            "date": self.date,
            "sensor": "nisar_l",
            "sar_vv": self.hh_db,
            "sar_vh": self.hv_db,
            "radar_sources": ["nisar_l"],
            "pixels": self.pixels,
            "granule": self.granule,
        }


def ingest_farm(
    farm_id: str,
    polygon: List[Tuple[float, float]],
    start: date,
    end: date,
    workdir: Optional[Path] = None,
    max_granules: int = 20,
) -> Tuple[List[NisarObservation], List[str]]:
    """Search, subset and reduce to per-farm backscatter. Returns (rows, notes)."""
    notes: List[str] = []
    bbox = bbox_for(polygon)

    urls, log = search(bbox, start, end)
    if not urls:
        notes.append(f"no granules returned; {log.strip()[:300]}")
        return [], notes
    notes.append(f"{len(urls)} granules found")
    urls = urls[:max_granules]

    tmp = workdir or Path(tempfile.mkdtemp(prefix=f"nisar_{farm_id}_"))
    rows: List[NisarObservation] = []

    for url in urls:
        name = url.rsplit("/", 1)[-1].replace(".h5", "")
        parts = name.split("_")
        acquired = parts[11][:8] if len(parts) > 11 else ""
        iso = f"{acquired[:4]}-{acquired[4:6]}-{acquired[6:8]}" if len(acquired) == 8 else ""

        ok, sublog = subset(url, bbox, tmp / name)
        if not ok:
            notes.append(f"{name[:40]}: subset failed — {sublog.strip()[-200:]}")
            continue

        stats: Dict[str, Optional[Dict[str, float]]] = {}
        for pol in DEFAULT_VARS:
            hits = list((tmp / name).rglob(f"*{pol}*.tif"))
            stats[pol] = polygon_mean(hits[0], polygon) if hits else None

        hh, hv = stats.get("HHHH"), stats.get("HVHV")
        if hh is None and hv is None:
            notes.append(f"{name[:40]}: window did not intersect the polygon")
            continue

        rows.append(NisarObservation(
            farm_id=farm_id,
            date=iso,
            hh_db=hh["mean_db"] if hh else None,
            hv_db=hv["mean_db"] if hv else None,
            pixels=max(hh["pixels"] if hh else 0, hv["pixels"] if hv else 0),
            granule=name,
        ))

    return rows, notes
