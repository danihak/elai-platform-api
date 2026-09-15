"""
nisar_h5.py — read NISAR GCOV over a farm with nothing but h5py and numpy.

WHY THIS EXISTS. The openSEPPO route needs rasterio, rasterio needs GDAL, and
GDAL's DLLs are blocked by endpoint security on a managed Windows machine. That
is a policy decision on the device, not something to be worked around with
another pip install.

So this takes the dependency away instead. GCOV is a geocoded HDF5 product: it
carries its own coordinate arrays and projection, which is everything needed to
locate a farm inside it. h5py reads HDF5 over HTTPS with byte-range requests, so
opening an 8 GB granule and slicing a 40x40 pixel window transfers kilobytes.

Dependencies: h5py, numpy, fsspec. All present, none blocked.

WHAT IT DOES NOT DO. No reprojection library. NISAR grids are UTM, so the
WGS84-to-UTM conversion is implemented directly below from the standard series
expansion — about forty lines, accurate to well under a metre, and with no
PROJ binary to be blocked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# WGS84
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_K0 = 0.9996


def utm_zone(lon: float) -> int:
    return int((lon + 180) / 6) + 1


def epsg_for(lat: float, lon: float) -> int:
    zone = utm_zone(lon)
    return (32600 if lat >= 0 else 32700) + zone


def wgs84_to_utm(lat: float, lon: float, zone: Optional[int] = None) -> Tuple[float, float, int]:
    """Forward UTM projection, standard series expansion.

    Implemented rather than imported because PROJ ships as a binary and binaries
    are what the device policy blocks. The series is accurate to centimetres
    well inside a zone, which is far beyond what a 20 m radar posting needs.
    """
    zone = zone or utm_zone(lon)
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    n = _A / math.sqrt(1 - _E2 * math.sin(lat_r) ** 2)
    t = math.tan(lat_r) ** 2
    c = _E2 / (1 - _E2) * math.cos(lat_r) ** 2
    a = math.cos(lat_r) * (lon_r - lon0)

    m = _A * (
        (1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256) * lat_r
        - (3 * _E2 / 8 + 3 * _E2**2 / 32 + 45 * _E2**3 / 1024) * math.sin(2 * lat_r)
        + (15 * _E2**2 / 256 + 45 * _E2**3 / 1024) * math.sin(4 * lat_r)
        - (35 * _E2**3 / 3072) * math.sin(6 * lat_r)
    )

    easting = _K0 * n * (
        a + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * _E2 / (1 - _E2)) * a**5 / 120
    ) + 500000.0

    northing = _K0 * (
        m + n * math.tan(lat_r) * (
            a**2 / 2 + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * _E2 / (1 - _E2)) * a**6 / 720
        )
    )
    if lat < 0:
        northing += 10000000.0
    return easting, northing, zone


# --------------------------------------------------------------------------
# Opening a granule
# --------------------------------------------------------------------------

GRID_ROOT = "/science/LSAR/GCOV/grids"


def open_granule(url: str, token: Optional[str] = None):
    """Open a NISAR granule over HTTPS without downloading it.

    fsspec issues HTTP range requests, so h5py reads only the chunks it is asked
    for. Opening the file costs a few hundred kilobytes of metadata regardless of
    whether the granule is 6 GB or 8 GB.
    """
    import fsspec
    import h5py

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    fs = fsspec.filesystem("https", client_kwargs={"headers": headers})
    # A large block size matters: HDF5 metadata is scattered, and small reads
    # turn one open into hundreds of round trips.
    handle = fs.open(url, "rb", block_size=8 * 1024 * 1024, cache_type="blockcache")
    return h5py.File(handle, "r"), handle


def list_grids(h5) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if GRID_ROOT not in h5:
        return out
    for freq in h5[GRID_ROOT]:
        node = h5[f"{GRID_ROOT}/{freq}"]
        out[freq] = [k for k in node if getattr(node[k], "ndim", 0) == 2]
    return out


@dataclass
class Window:
    x0: int
    x1: int
    y0: int
    y1: int
    pixels: int
    posting_m: float


def locate(h5, freq: str, polygon: Sequence[Tuple[float, float]],
           pad_px: int = 2) -> Optional[Window]:
    """Turn a lat/lon polygon into array indices inside the granule's grid.

    The coordinate arrays are one-dimensional and small — a few tens of
    kilobytes — so reading them in full costs nothing and removes any need to
    parse the geotransform.
    """
    import numpy as np

    g = h5[f"{GRID_ROOT}/{freq}"]
    if "xCoordinates" not in g or "yCoordinates" not in g:
        return None

    xs = np.asarray(g["xCoordinates"][:], dtype="float64")
    ys = np.asarray(g["yCoordinates"][:], dtype="float64")

    lat0 = sum(p[1] for p in polygon) / len(polygon)
    lon0 = sum(p[0] for p in polygon) / len(polygon)
    zone = utm_zone(lon0)

    es, ns = [], []
    for lon, lat in polygon:
        e, n, _ = wgs84_to_utm(lat, lon, zone)
        es.append(e)
        ns.append(n)

    x0 = int(np.searchsorted(xs, min(es))) - pad_px
    x1 = int(np.searchsorted(xs, max(es))) + pad_px
    # y is usually descending in a north-up grid.
    if ys[0] > ys[-1]:
        y0 = int(np.searchsorted(-ys, -max(ns))) - pad_px
        y1 = int(np.searchsorted(-ys, -min(ns))) + pad_px
    else:
        y0 = int(np.searchsorted(ys, min(ns))) - pad_px
        y1 = int(np.searchsorted(ys, max(ns))) + pad_px

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(len(xs), x1), min(len(ys), y1)
    if x1 <= x0 or y1 <= y0:
        return None

    posting = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 20.0
    return Window(x0, x1, y0, y1, (x1 - x0) * (y1 - y0), posting)


def read_window(h5, freq: str, variable: str, w: Window) -> Optional[Dict[str, float]]:
    """Slice one polarisation over the window and reduce it to a mean in dB.

    GCOV stores gamma-nought as linear power, so the conversion to decibels
    happens here — the estimator's features are all in dB and nothing downstream
    should have to know which units a source arrived in.
    """
    import numpy as np

    path = f"{GRID_ROOT}/{freq}/{variable}"
    if path not in h5:
        return None

    block = np.asarray(h5[path][w.y0:w.y1, w.x0:w.x1], dtype="float64")
    valid = block[np.isfinite(block) & (block > 0)]
    if valid.size == 0:
        return None

    db = 10.0 * np.log10(valid)
    return {
        "mean_db": float(np.mean(db)),
        "std_db": float(np.std(db)),
        "pixels": int(valid.size),
        "posting_m": w.posting_m,
    }


def sample_farm(
    url: str,
    polygon: Sequence[Tuple[float, float]],
    variables: Sequence[str] = ("HHHH", "HVHV"),
    token: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Dict[str, float]]], str]:
    """One granule, one farm. Returns (per-polarisation stats, note)."""
    try:
        h5, handle = open_granule(url, token)
    except Exception as exc:  # noqa: BLE001
        return None, f"could not open: {type(exc).__name__}: {exc}"

    try:
        grids = list_grids(h5)
        if not grids:
            return None, "no GCOV grids found in this granule"
        freq = "frequencyA" if "frequencyA" in grids else sorted(grids)[0]

        w = locate(h5, freq, polygon)
        if w is None:
            return None, "farm polygon falls outside this granule's grid"

        out: Dict[str, Dict[str, float]] = {}
        for v in variables:
            stats = read_window(h5, freq, v, w)
            if stats:
                out[v] = stats

        if not out:
            return None, (f"window found ({w.pixels} px) but no valid backscatter "
                          f"— likely a masked or off-swath area")
        return out, (f"{freq}, window {w.x1-w.x0}x{w.y1-w.y0} px at "
                     f"{w.posting_m:.0f} m posting")
    finally:
        try:
            h5.close()
            handle.close()
        except Exception:  # noqa: BLE001
            pass
