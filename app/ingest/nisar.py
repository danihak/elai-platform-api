"""
nisar.py — is there NISAR data over our farms, and is it worth ingesting?

WHY THIS IS A SEPARATE, CHEAP STEP. NISAR has no Sentinel-Hub-style statistics
API. You search a catalogue, download whole granules (HDF5, gigabyte scale), and
extract your own statistics. That is days of work, so the first question is not
"how do we ingest it" but "is there anything to ingest over Karimnagar, Jagtial
and Warangal, in our season, at all".

This module answers that with metadata queries only. Nothing is downloaded.

WHY IT MATTERS FOR THIS PRODUCT. Cotton failed radar validation at RMSE 0.131
against a 0.12 ceiling, and the physics says why: Sentinel-1 is C-band, about a
5.6 cm wavelength, which scatters strongly off the wet soil visible between
cotton's widely spaced rows. NISAR L-band is about 24 cm and penetrates further
into the canopy, so it is the physically motivated fix for the specific failure
our own data produced — not a general wish for more data.

WHAT IS HONESTLY UNKNOWN. As of September 2026 the L-band products are marked
PROVISIONAL and a validated reprocessing campaign is expected. No peer-reviewed
agricultural NDVI or biomass study using real NISAR data exists yet. Anything
built on this is research, not a production claim.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

#: Bounding box covering all three districts our farms and training blocks sit in.
#: Karimnagar and Jagtial in the north, Warangal in the south.
TELANGANA_BBOX = {
    "min_lon": 78.55,
    "min_lat": 17.90,
    "max_lon": 79.85,
    "max_lat": 18.90,
}

#: Products worth checking, in order of usefulness to us.
#:
#:   GCOV  geocoded polarimetric covariance. Already on a map grid and carries
#:         the terms needed for an L-band radar vegetation index. This is the
#:         one we want.
#:   GSLC  geocoded single-look complex. Phase preserved, heavier to handle.
#:   RSLC  range-doppler single-look complex. Requires geocoding ourselves.
CANDIDATE_PRODUCTS = ["GCOV", "GSLC", "RSLC"]


def product_size(value) -> Dict[str, float]:
    """Split a NISAR `bytes` entry into the science product and everything else.

    ASF returns a dict keyed by filename. The .h5 is the science product and is
    6 to 8 GB; the browse PNGs, QA stats and KMLs together are a few megabytes.
    Summing them all into one number hides the only figure that matters for
    feasibility, which is the size of the thing you would actually have to move.
    """
    out = {"h5_gb": 0.0, "qa_stats_mb": 0.0, "browse_mb": 0.0, "other_mb": 0.0}
    if not isinstance(value, dict):
        return out
    for name, meta in value.items():
        size = meta.get("bytes", 0) if isinstance(meta, dict) else 0
        try:
            size = float(size)
        except (TypeError, ValueError):
            continue
        lower = name.lower()
        if lower.endswith("_qa_stats.h5"):
            out["qa_stats_mb"] += size / 1e6
        elif lower.endswith(".h5"):
            out["h5_gb"] += size / 1e9
        elif lower.endswith(".png"):
            out["browse_mb"] += size / 1e6
        else:
            out["other_mb"] += size / 1e6
    return out


def _size_bytes(value) -> float:
    """asf_search changed `bytes` from a number to a dict between versions.

    Parsing it with a bare float() threw a TypeError, which the caller then
    reported as "nothing to ingest" — turning a SUCCESSFUL query into a false
    negative. Size is cosmetic; it must never be able to discard results.
    """
    if value is None:
        return 0.0
    if isinstance(value, dict):
        for key in ("bytes", "size", "value", "length"):
            if key in value:
                try:
                    return float(value[key])
                except (TypeError, ValueError):
                    continue
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _wkt(bbox: Dict[str, float]) -> str:
    return (
        f"POLYGON(({bbox['min_lon']} {bbox['min_lat']},"
        f"{bbox['max_lon']} {bbox['min_lat']},"
        f"{bbox['max_lon']} {bbox['max_lat']},"
        f"{bbox['min_lon']} {bbox['max_lat']},"
        f"{bbox['min_lon']} {bbox['min_lat']}))"
    )


@dataclass
class Availability:
    product: str
    granules: int
    first_date: Optional[str]
    last_date: Optional[str]
    total_gb: float
    sample: List[Dict]
    error: Optional[str] = None


def check(
    start: date = date(2026, 6, 1),
    end: date = date(2026, 9, 30),
    bbox: Dict[str, float] = TELANGANA_BBOX,
    products: List[str] = None,
) -> List[Availability]:
    """Metadata-only search. Requires `pip install asf_search`. No credentials.

    Searching ASF is open; only downloading needs an Earthdata login. So this
    tells us whether the whole idea is viable before anyone registers for
    anything.
    """
    try:
        import asf_search as asf
    except ImportError:
        print("asf_search is not installed. Run:  pip install asf_search",
              file=sys.stderr)
        return []

    wkt = _wkt(bbox)
    out: List[Availability] = []

    for product in (products or CANDIDATE_PRODUCTS):
        try:
            # The asf_search NISAR constants have moved between releases, so try
            # the documented forms in order rather than assuming one.
            results = None
            errors: List[str] = []
            for attempt in (
                lambda: asf.search(dataset="NISAR", processingLevel=product,
                                   intersectsWith=wkt,
                                   start=start.isoformat(), end=end.isoformat(),
                                   maxResults=200),
                lambda: asf.search(platform="NISAR", processingLevel=product,
                                   intersectsWith=wkt,
                                   start=start.isoformat(), end=end.isoformat(),
                                   maxResults=200),
                lambda: asf.search(processingLevel=f"NISAR_L2_{product}",
                                   intersectsWith=wkt,
                                   start=start.isoformat(), end=end.isoformat(),
                                   maxResults=200),
            ):
                try:
                    results = attempt()
                    if results is not None:
                        break
                except Exception as exc:  # noqa: BLE001 — try the next form
                    errors.append(f"{type(exc).__name__}: {exc}")
                    continue

            if results is None:
                # Distinguish "could not ask" from "asked, nothing there". The
                # first version collapsed both into "no query form accepted",
                # which reported an empty result when the real problem was no
                # route to the server. A search that never ran must never look
                # like a search that came back empty.
                blob = " | ".join(errors).lower()
                # A JSON decode failure from an HTTP client is almost always a
                # proxy or error page arriving where JSON was expected — a
                # network symptom, not the API rejecting a valid query.
                if any(k in blob for k in ("resolve", "connection", "timeout",
                                           "ssl", "network", "unreachable",
                                           "getaddrinfo", "proxy", "403", "denied",
                                           "jsondecode", "expecting value",
                                           "max retries", "name or service")):
                    reason = "COULD NOT REACH ASF — network, not an empty result"
                else:
                    reason = "query rejected by ASF"
                out.append(Availability(product, 0, None, None, 0.0, [],
                                        error=f"{reason}: {errors[-1] if errors else 'unknown'}"))
                continue

            props = [r.properties for r in results]
            dates = sorted(p.get("startTime", "")[:10] for p in props if p.get("startTime"))
            total_bytes = sum(product_size(p.get("bytes"))["h5_gb"] * 1e9 for p in props)

            out.append(Availability(
                product=product,
                granules=len(props),
                first_date=dates[0] if dates else None,
                last_date=dates[-1] if dates else None,
                total_gb=round(total_bytes / 1e9, 2),
                sample=[{
                    "name": p.get("fileName") or p.get("sceneName"),
                    "date": (p.get("startTime") or "")[:10],
                    "gb": round(product_size(p.get("bytes"))["h5_gb"], 2),
                    "polarisation": "+".join(p.get("mainBandPolarization") or []) or None,
                    "orbit": p.get("orbit"),
                    "track": p.get("pathNumber"),
                    "direction": p.get("flightDirection"),
                    "url": p.get("url"),
                } for p in props[:3]],
            ))
        except Exception as exc:  # noqa: BLE001
            # The query may well have succeeded and the failure be ours. Say so,
            # rather than letting a parsing bug read as an absence of data.
            out.append(Availability(
                product, 0, None, None, 0.0, [],
                error=(f"PARSE FAILED AFTER QUERY — our bug, not an empty "
                       f"result: {type(exc).__name__}: {exc}")))
    return out


def report(results: List[Availability]) -> None:
    if not results:
        print("No query ran.")
        return

    print(f"NISAR L-band over Telangana "
          f"({TELANGANA_BBOX['min_lat']}-{TELANGANA_BBOX['max_lat']}N, "
          f"{TELANGANA_BBOX['min_lon']}-{TELANGANA_BBOX['max_lon']}E)\n")

    any_data = False
    unreachable = False
    for r in results:
        if r.error:
            if "COULD NOT REACH" in r.error or "PARSE FAILED" in r.error:
                unreachable = True
            print(f"  {r.product:6} {r.error[:150]}")
            continue
        if r.granules == 0:
            print(f"  {r.product:6} no granules in the window")
            continue
        any_data = True
        print(f"  {r.product:6} {r.granules:4} granules  "
              f"{r.first_date} to {r.last_date}  ~{r.total_gb} GB total")
        for s in r.sample:
            print(f"         {s['date']}  {s['gb']:5.2f} GB  "
                  f"track {s.get('track')}/{str(s.get('direction'))[:4]}  "
                  f"pol {s['polarisation']}")

    print()
    if unreachable:
        print("NO CONCLUSION. The search did not complete cleanly, so this says nothing")
        print("about whether NISAR covers Telangana. Check connectivity and rerun.")
        print("Reporting this as 'no data' would be a false negative on a decision")
        print("that shapes the roadmap.")
        return

    if not any_data:
        print("Nothing to ingest. Either Telangana was not acquired in this window,")
        print("or the products are not yet released for it. That is a real finding:")
        print("NISAR is a roadmap item, not a this-season one, and the deck should")
        print("say so rather than implying coverage that does not exist.")
    else:
        print("Data exists. Next decisions, in order:")
        print("  1. How many acquisitions actually overlap the Kharif window?")
        print("     At a 12-day repeat, expect roughly 7-8 over June to September.")
        print("  2. Download volume against available disk and bandwidth.")
        print("  3. GCOV is preferred: already geocoded, and carries the terms for")
        print("     an L-band radar vegetation index without us geocoding anything.")
        print("  4. Extract per-polygon statistics ourselves — there is no")
        print("     statistics API, so this is the real engineering cost.")


# --------------------------------------------------------------------------
# Inspecting what ASF actually returns
# --------------------------------------------------------------------------


def inspect(start: date = date(2026, 6, 1), end: date = date(2026, 9, 30),
            bbox: Dict[str, float] = None, product: str = "GCOV",
            limit: int = 3) -> int:
    """Dump the raw property keys for a few granules.

    Size and polarisation both came back empty against the keys this module
    assumed, so rather than guessing again, print what the API really provides.
    Granule size decides whether a download is feasible at all, and polarisation
    decides whether a granule is useful — quad-pol is worth far more for cotton
    than dual-pol, and we should not be filtering on a parsed filename when the
    metadata carries it.
    """
    import json as _json

    try:
        import asf_search as asf
    except ImportError:
        print("pip install asf_search", file=sys.stderr)
        return 1

    wkt = _wkt(bbox or TELANGANA_BBOX)
    results = asf.search(dataset="NISAR", processingLevel=product,
                         intersectsWith=wkt, start=start.isoformat(),
                         end=end.isoformat(), maxResults=limit)
    props = [r.properties for r in results]
    if not props:
        print("no granules returned")
        return 1

    print(f"{len(props)} granule(s); available property keys:\n")
    for k in sorted(props[0].keys()):
        v = props[0][k]
        shown = _json.dumps(v)[:90] if not isinstance(v, str) else v[:90]
        print(f"  {k:28} {shown}")

    print("\nvalues across the sample:")
    for i, p in enumerate(props, 1):
        print(f"\n  granule {i}")
        for k in sorted(p.keys()):
            if any(w in k.lower() for w in ("byte", "size", "pol", "beam",
                                            "mode", "url", "path", "frame",
                                            "orbit", "start")):
                print(f"    {k:26} {p[k]}")
    return 0


def decode_name(name: str) -> Dict[str, str]:
    """Read the acquisition parameters out of a NISAR granule name.

    Example:
      NISAR_L2_PR_GCOV_030_084_A_011_4005_DHDH_A_20260912T001807_2

    The polarisation code is the field that matters most here. QP means quad
    polarimetric, which supports full decomposition and is the strongest input
    for separating cotton canopy from the soil between rows. DH means dual
    HH+HV, still L-band cross-pol and still better than C-band VH.
    """
    parts = name.split("_")
    out: Dict[str, str] = {"name": name}
    if len(parts) < 12:
        return out
    try:
        out.update({
            "level": parts[1],
            "product": parts[3],
            "cycle": parts[4],
            "track": parts[5],
            "direction": "ascending" if parts[6] == "A" else "descending",
            "frame": parts[7],
            "pol_code": parts[9],
            "polarimetry": ("quad" if parts[9].startswith("QP")
                            else "dual" if parts[9].startswith("DH")
                            else parts[9]),
            "acquired": parts[11][:8],
        })
    except IndexError:
        pass
    return out


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------


def feasibility(start: date = date(2026, 6, 1), end: date = date(2026, 9, 30),
                bbox: Dict[str, float] = None, mbps: float = 20.0) -> int:
    """How much data is there really, and what is the only viable way in?

    The bulk-download instinct does not survive contact with the numbers: a GCOV
    granule is 6 to 8 GB and there are tens of them over one season and one
    district. The viable route is to read only the HDF5 chunks that intersect
    each farm polygon, directly from S3, which is what "cloud optimised" buys
    and why the format was chosen.
    """
    try:
        import asf_search as asf
    except ImportError:
        print("pip install asf_search", file=sys.stderr)
        return 1

    wkt = _wkt(bbox or TELANGANA_BBOX)
    results = asf.search(dataset="NISAR", processingLevel="GCOV",
                         intersectsWith=wkt, start=start.isoformat(),
                         end=end.isoformat(), maxResults=500)
    props = [r.properties for r in results]
    if not props:
        print("no granules")
        return 1

    sizes = [product_size(p.get("bytes")) for p in props]
    total_h5 = sum(s["h5_gb"] for s in sizes)
    total_qa = sum(s["qa_stats_mb"] for s in sizes)
    total_browse = sum(s["browse_mb"] for s in sizes)

    quad = [p for p in props if len(p.get("mainBandPolarization") or []) == 4]
    tracks: Dict[str, int] = {}
    for p in props:
        key = f"{p.get('pathNumber')}{str(p.get('flightDirection') or '')[:1]}"
        tracks[key] = tracks.get(key, 0) + 1

    print(f"GCOV over the AOI, {start} to {end}\n")
    print(f"  granules              {len(props)}")
    print(f"  quad-pol among them   {len(quad)}")
    print(f"  distinct tracks       {len(tracks)}  ({', '.join(sorted(tracks))})")
    print()
    print(f"  full science products  {total_h5:8.1f} GB")
    print(f"  QA stats only          {total_qa:8.1f} MB")
    print(f"  browse images only     {total_browse:8.1f} MB")
    print()
    hours = (total_h5 * 8 * 1000) / (mbps * 3600)
    print(f"  bulk download at {mbps:.0f} Mbps: {hours:.0f} hours, plus {total_h5:.0f} GB of disk")
    print()
    print("  VERDICT")
    print("  Bulk download is not viable for a small team on a laptop, and it")
    print("  never becomes viable — this is one season over three districts.")
    print()
    print("  The workable route is chunk-level reads straight from S3. GCOV is")
    print("  chunked HDF5 in a public-facing bucket, so h5py over fsspec can")
    print("  fetch only the byte ranges covering a farm polygon. A 0.8 ha plot")
    print("  is a handful of chunks: megabytes, not gigabytes.")
    print()
    print("  What that needs, in order:")
    print("    1. Earthdata login, then temporary S3 credentials from ASF")
    print("    2. h5py + fsspec + s3fs, reading the GCOV grid without downloading")
    print("    3. Map each polygon to grid indices from the product geotransform")
    print("    4. Extract HH and HV (and VV, VH where quad-pol), compute an")
    print("       L-band RVI, and feed it in as another radar source")
    print()
    print("  Honest scope: days of work, and the products are still PROVISIONAL")
    print("  with a validated reprocessing campaign expected. This is research,")
    print("  not something to promise a client this season.")
    return 0
