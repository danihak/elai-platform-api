"""
cli.py — the ingest and fitting commands.

    python -m app.cli verify     one cheap call; proves credentials and shapes
    python -m app.cli ingest     pull real observations for every farm
    python -m app.cli fit        fit SAR to NDVI on clear-day pairs
    python -m app.cli status     what is cached right now

Run locally, once. Outputs land in data/ and are committed, so the deployed
service serves real satellite data and holds no Sentinel Hub credentials.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import date, timedelta

DATA = pathlib.Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)
OBS_FILE = DATA / "observations.json"
COEF_FILE = DATA / "coefficients.json"
HIST_FILE = DATA / "history.json"
TRAIN_FILE = DATA / "training.json"
CLIM_FILE = DATA / "climatology.json"
CONF_FILE = DATA / "conformal.json"


def _load_env() -> None:
    """Minimal .env reader so there is no python-dotenv dependency."""
    env = pathlib.Path(".env")
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cmd_verify() -> int:
    from .ingest.sentinelhub import Client, SentinelHubError, verify
    from .seed import FARMS

    try:
        client = Client.from_env()
        print("credentials found, requesting token...")
        client.token()
        print("  token OK")
        farm = FARMS[0]
        print(f"one statistics call for {farm.farm_name} ({farm.district})...")
        out = verify(client, [[x, y] for x, y in farm.polygon])
        print(f"  intervals returned : {out['intervals_returned']}")
        print(f"  intervals with data: {out['with_data']}")
        if out["first"]:
            print("  first interval:")
            print("   ", json.dumps(out["first"])[:400])
        print("\nverified. safe to run: python -m app.cli ingest")
        return 0
    except SentinelHubError as exc:
        print(f"\nFAILED\n{exc}", file=sys.stderr)
        return 1


def cmd_ingest(days: int = 120, history: bool = False) -> int:
    """history=True pulls two years and ignores the sowing date.

    The point of the longer window is the DRY season. Kharif is when optical
    fails, so Kharif contains almost no clear-day radar/optical pairs to train
    on. Rabi and the summer months are where the training data lives.
    """
    from .ingest.sentinelhub import Client, SentinelHubError, fetch_optical, fetch_radar
    from .seed import FARMS, stage_for

    try:
        client = Client.from_env()
    except SentinelHubError as exc:
        print(exc, file=sys.stderr)
        return 1

    end = date(2026, 9, 6)
    if history:
        days = 730
    bundle = {"as_of": end.isoformat(), "source": "sentinel-hub",
              "history_mode": history, "farms": {}}

    for farm in FARMS:
        poly = [[x, y] for x, y in farm.polygon]
        start = (end - timedelta(days=days)) if history else max(
            farm.sowing_date, end - timedelta(days=days))
        print(f"{farm.farm_id}  {farm.farm_name} ({farm.crop})")
        try:
            optical = fetch_optical(client, poly, start, end)
            print(f"   optical: {len(optical)} intervals, "
                  f"{sum(1 for o in optical if o['ndvi'] is not None)} with a clear read")
            radar = fetch_radar(client, poly, start, end)
            print(f"   radar  : {len(radar)} intervals")
        except SentinelHubError as exc:
            print(f"   FAILED: {exc}", file=sys.stderr)
            return 1

        radar_by_date = {r["date"]: r for r in radar}
        merged = []
        for o in optical:
            das = (date.fromisoformat(o["date"]) - farm.sowing_date).days
            row = dict(o)
            row.update({
                "das": das,
                "stage": stage_for(farm.crop, das),
                "cloudy": o["valid_pixel_fraction"] < 0.65,
            })
            row.update(radar_by_date.get(o["date"], {"sar_vv": None, "sar_vh": None,
                                                     "radar_sources": []}))
            merged.append(row)
        bundle["farms"][farm.farm_id] = merged

    target = HIST_FILE if history else OBS_FILE
    target.write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    total = sum(len(v) for v in bundle["farms"].values())
    print(f"\nwrote {target} — {total} real observations across {len(bundle['farms'])} farms")
    print("next: python -m app.cli fit")
    return 0


def cmd_fit() -> int:
    from .ingest.fit import (compare_feature_sets, fit_per_crop_stage,
                             fit_pooled_by_crop, to_estimator_table)
    from .seed import FARMS_BY_ID

    if not OBS_FILE.exists():
        print("no data/observations.json — run ingest first", file=sys.stderr)
        return 1

    pairs = []
    sources = []
    for src in (OBS_FILE, HIST_FILE):
        if src.exists():
            _collect(json.loads(src.read_text(encoding="utf-8")), pairs)
            sources.append(src.name)
    if TRAIN_FILE.exists():
        blob = json.loads(TRAIN_FILE.read_text(encoding="utf-8"))
        for aoi_id, block in blob["aois"].items():
            for r in block["observations"]:
                pairs.append({"crop": block["crop"], "stage": "unknown",
                              "sar_vh": r["sar_vh"], "sar_vv": r.get("sar_vv"),
                              "ndvi": r["ndvi"], "date": r["date"]})
        sources.append(TRAIN_FILE.name)
    print(f"sources: {', '.join(sources)}")
    print(f"{len(pairs)} clear-day radar/optical pairs\n")
    # Evidence the feature choice rather than asserting it.
    print("feature set comparison (pooled per crop, held-out RMSE / R2)")
    comp = compare_feature_sets(pairs)
    crops = sorted({c for v in comp.values() for c in v})
    print(f"  {'':12} " + "  ".join(f"{c:>22}" for c in crops))
    for name, res in comp.items():
        cells = []
        for c in crops:
            r = res.get(c)
            cells.append(f"{r['rmse']:.4f} / {r['r2']:+.3f}".rjust(22) if r else "".rjust(22))
        print(f"  {name:12} " + "  ".join(cells))
    print()

    results = fit_per_crop_stage(pairs)

    # Where a crop x stage is too thin, pool that crop across stages rather than
    # leaving it unfitted. Weaker, labelled as such, and far better than nothing.
    thin = [k for k, v in results.items() if not v.get("fitted")]
    if thin:
        pooled = fit_pooled_by_crop(pairs)
        results.update(pooled)
    for combo, r in results.items():
        if r.get("fitted"):
            coefs = "  ".join(f"{k}={v:+.4f}" for k, v in r["coefficients"].items())
            print(f"  {combo:26} n={r['n']:4}  rmse={r['rmse']:.4f}"
                  f"±{r.get('rmse_sd',0):.4f}  upper={r.get('rmse_upper',0):.4f}  "
                  f"r2={r.get('r2', 0):+.3f}  "
                  f"holdout={r.get('holdout_condition','mixed')} "
                  f"n={r['n_holdout']}x{r.get('n_splits',1)}")
            print(f"  {'':26} {coefs}")
        else:
            print(f"  {combo:26} NOT FITTED — {r['reason']}")

    table = to_estimator_table(results)

    # ---- phenology climatology and persistence decay, both from real reads ----
    from .ingest.climatology import fit_climatology, fit_persistence_decay
    from .services.fusion import validate_fusion

    clear_reads, series_by_field = [], {}
    for src in (OBS_FILE, HIST_FILE):
        if not src.exists():
            continue
        blob = json.loads(src.read_text(encoding="utf-8"))
        for farm_id, rows in blob.get("farms", {}).items():
            farm = FARMS_BY_ID.get(farm_id)
            if not farm:
                continue
            for r in rows:
                if r.get("ndvi") is None or r.get("valid_pixel_fraction", 0) < 0.8:
                    continue
                clear_reads.append({"crop": farm.crop, "date": r["date"], "ndvi": r["ndvi"]})
                series_by_field.setdefault((farm.crop, farm_id), []).append(
                    (date.fromisoformat(r["date"]), r["ndvi"]))
    if TRAIN_FILE.exists():
        blob = json.loads(TRAIN_FILE.read_text(encoding="utf-8"))
        for aoi_id, block in blob["aois"].items():
            for r in block["observations"]:
                clear_reads.append({"crop": block["crop"], "date": r["date"], "ndvi": r["ndvi"]})
                series_by_field.setdefault((block["crop"], aoi_id), []).append(
                    (date.fromisoformat(r["date"]), r["ndvi"]))

    print()
    print(f"phenology climatology from {len(clear_reads)} clear optical reads")
    clim = fit_climatology(clear_reads)
    for crop, m in clim.items():
        if m.get("fitted"):
            print(f"  {crop:8} n={m['n']:4}  residual sigma={m['residual_sigma']:.4f}  "
                  f"seasonal variance explained={m['variance_explained']:.1%}")
        else:
            print(f"  {crop:8} NOT FITTED — {m['reason']}")

    print()
    print("persistence decay, measured from clear-read pairs")
    decay = fit_persistence_decay(series_by_field)
    for crop, d in decay.items():
        if d.get("fitted"):
            print(f"  {crop:8} n={d['n']:5} pairs  decay={d['decay_per_day']:.5f} NDVI per day")
        else:
            print(f"  {crop:8} NOT FITTED — {d['reason']}")

    # ---- calibrate the fusion against held-out clear reads ----
    radar_sigma = {}
    for combo, m in table["models"].items():
        crop = combo.split(":")[0]
        if combo in table["validated"] or crop not in radar_sigma:
            radar_sigma.setdefault(crop, m["rmse"])

    truth = _fusion_truth_points(series_by_field, radar_sigma)
    print()
    print(f"fusion calibration on {len(truth)} held-out clear reads")
    calib = validate_fusion(
        truth, radar_sigma, clim,
        {c: d.get("decay_per_day", 0.015) for c, d in decay.items()},
    )
    for crop, c in calib.items():
        if c.get("validated"):
            print(f"  {crop:8} real rmse={c['rmse']:.4f}  claimed={c['mean_claimed_sigma']:.4f}  "
                  f"calibration={c['calibration']:.2f}")
            print(f"  {'':8} {c['verdict']}")
        else:
            print(f"  {crop:8} NOT VALIDATED — {c['reason']}")

    # ---- conformal prediction: turn the measured band into a guarantee ----
    from .services.conformal import (blocked_split, calibrate, enforce_measured_coverage,
                                     evaluate, to_json)
    from .services.fusion import fusion_residuals

    decay_map = {c: d.get("decay_per_day", 0.015) for c, d in decay.items()}
    residuals = fusion_residuals(truth, radar_sigma, clim, decay_map)
    cal_set, test_set = blocked_split(residuals)

    print()
    print(f"conformal calibration on {len(cal_set)} points "
          f"({len(test_set)} held back for coverage testing, split by field)")
    conformal = calibrate(cal_set)
    for crop, m in conformal.items():
        if not m.usable:
            print(f"  {crop:8} NOT CERTIFIABLE — {m.reason}")
            continue
        mults = "  ".join(f"{lvl}:x{v}" for lvl, v in m.multipliers.items())
        print(f"  {crop:8} n={m.n:4}  {mults}")

    print()
    print("achieved coverage on fields the calibration never saw")
    diags = evaluate(test_set, conformal)
    for crop, levels in diags.items():
        for lvl, d in levels.items():
            flag = "OK " if d["holds"] else "LOW"
            print(f"  {flag} {crop:8} nominal {d['nominal']:.0%}  "
                  f"PICP {d['picp']:.1%}  width {d['mpiw']:.3f}  n={d['n']}")

    conformal = enforce_measured_coverage(conformal, diags)
    withdrawn = [(c, k) for c, m in conformal.items()
                 for k, v in m.diagnostics.items() if v.get("withdrawn")]
    if withdrawn:
        print()
        print("  WITHDRAWN — measured coverage below nominal, not published:")
        for c, k in withdrawn:
            d = conformal[c].diagnostics[k]
            print(f"    {c} at {float(k):.0%}: delivered {d['measured']:.1%}")

    CONF_FILE.write_text(json.dumps(to_json(conformal), indent=1), encoding="utf-8")
    print(f"\nwrote {CONF_FILE}")

    CLIM_FILE.write_text(json.dumps(
        {"climatology": clim, "persistence_decay": decay, "fusion_calibration": calib},
        indent=1), encoding="utf-8")
    print(f"\nwrote {CLIM_FILE}")
    COEF_FILE.write_text(json.dumps({"results": results, "estimator": table}, indent=1),
                         encoding="utf-8")
    print(f"\nvalidated: {table['validated'] or 'none'}")
    if table["rejected"]:
        print("rejected :")
        for combo, why in table["rejected"].items():
            print(f"    {combo}: {why}")
    print(f"\nwrote {COEF_FILE}")
    print("Anything not validated sends its observations to Blind. That is correct.")
    return 0


def cmd_train_pull(days: int = 730) -> int:
    """Pull two years of DAILY observations over the dedicated training blocks.

    Three differences from `ingest`, and each one matters:

      P1D not P5D   Every acquisition is its own interval. Five-day buckets
                    merged distinct passes into one averaged number and made
                    optical and radar impossible to pair.

      1 km blocks   Ten thousand pixels instead of eighty-one.

      Two years     Kharif is when optical fails, so Kharif contains almost no
                    clear-day pairs. Rabi and summer are where the training
                    data actually lives.
    """
    from .ingest.sentinelhub import (Client, SentinelHubError, fetch_optical,
                                     fetch_radar, match_radar_to_optical)
    from .ingest.training_aois import TRAINING_AOIS

    try:
        client = Client.from_env()
    except SentinelHubError as exc:
        print(exc, file=sys.stderr)
        return 1

    end = date(2026, 9, 6)
    start = end - timedelta(days=days)
    bundle = {"as_of": end.isoformat(), "source": "sentinel-hub",
              "purpose": "estimator_training", "interval_days": 1, "aois": {}}

    print(f"{len(TRAINING_AOIS)} training blocks, {start} to {end}, daily intervals\n")
    total_pairs = 0
    for aoi in TRAINING_AOIS:
        poly = [[x, y] for x, y in aoi.polygon]
        try:
            optical = fetch_optical(client, poly, start, end, interval_days=1)
            radar = fetch_radar(client, poly, start, end, interval_days=1)
        except SentinelHubError as exc:
            print(f"{aoi.aoi_id} FAILED: {exc}", file=sys.stderr)
            return 1

        clear = [o for o in optical if o["ndvi"] is not None
                 and o["valid_pixel_fraction"] >= 0.80]
        merged = match_radar_to_optical(clear, radar)
        paired = [m for m in merged if m.get("sar_vh") is not None]
        total_pairs += len(paired)

        print(f"  {aoi.aoi_id} {aoi.crop:7} {len(optical):4} passes  "
              f"{len(clear):4} clear  {len(radar):4} radar  {len(paired):4} paired")
        bundle["aois"][aoi.aoi_id] = {"crop": aoi.crop, "district": aoi.district,
                                      "observations": paired}

    TRAIN_FILE.write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    print(f"\nwrote {TRAIN_FILE} — {total_pairs} clear-day radar/optical pairs")
    print("next: python -m app.cli fit")
    return 0


def cmd_nisar_check() -> int:
    """Metadata-only. Answers whether NISAR L-band exists over our districts."""
    from .ingest.nisar import check, report

    report(check())
    return 0


def cmd_nisar_feasibility() -> int:
    from .ingest.nisar import feasibility
    return feasibility()


def cmd_nisar_inspect() -> int:
    from .ingest.nisar import inspect
    return inspect()


def _fusion_truth_points(series_by_field, radar_sigma):
    """Held-out points for calibrating the fusion.

    Each clear optical read becomes a test case: hide it, rebuild the estimate
    from what was known beforehand, compare. The radar value is simulated from
    the measured radar RMSE rather than re-fetched, because what is being tested
    here is the FUSION, not the radar model — that already has its own hold-out
    score.
    """
    import random
    rng = random.Random(4471)
    out = []
    for (crop, field_id), pts in series_by_field.items():
        sigma = radar_sigma.get(crop)
        if sigma is None:
            continue
        pts = sorted(pts)
        for i in range(1, len(pts)):
            gap = (pts[i][0] - pts[i - 1][0]).days
            if gap <= 0 or gap > 60:
                continue
            truth = pts[i][1]
            out.append({
                "crop": crop,
                "field_id": field_id,
                "obs_date": pts[i][0],
                "ndvi": truth,
                "radar_ndvi": truth + rng.gauss(0, sigma),
                "last_clear_ndvi": pts[i - 1][1],
                "days_since_clear": gap,
            })
    return out


def _collect(bundle: dict, pairs: list) -> None:
    """Clear-day pairs only: a trustworthy optical NDVI and a radar reading on
    the same interval. Optical labels itself, which is why no ELAI ground truth
    is needed for this model."""
    from .seed import FARMS_BY_ID

    for farm_id, rows in bundle["farms"].items():
        farm = FARMS_BY_ID.get(farm_id)
        if not farm:
            continue
        for r in rows:
            if r.get("ndvi") is None or r.get("sar_vh") is None:
                continue
            if r.get("valid_pixel_fraction", 0) < 0.80:
                continue
            pairs.append({"crop": farm.crop, "stage": r.get("stage", "unknown"),
                          "sar_vh": r["sar_vh"], "sar_vv": r.get("sar_vv"),
                          "ndvi": r["ndvi"], "date": r["date"]})


def cmd_status() -> int:
    for f in (OBS_FILE, COEF_FILE):
        if f.exists():
            blob = json.loads(f.read_text(encoding="utf-8"))
            if f == OBS_FILE:
                n = sum(len(v) for v in blob["farms"].values())
                print(f"{f.name}: {len(blob['farms'])} farms, {n} observations, as of {blob['as_of']}")
            else:
                print(f"{f.name}: validated {blob['estimator']['validated']}")
        else:
            print(f"{f.name}: absent — service falls back to synthetic seed data")
    return 0


def main() -> int:
    _load_env()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    return {
        "verify": cmd_verify,
        "ingest": cmd_ingest,
        "history": lambda: cmd_ingest(history=True),
        "train-pull": cmd_train_pull,
        "nisar-check": cmd_nisar_check,
        "nisar-inspect": cmd_nisar_inspect,
        "nisar-feasibility": cmd_nisar_feasibility,
        "fit": cmd_fit,
        "status": cmd_status,
    }.get(cmd, lambda: (print(__doc__), 1)[1])()


if __name__ == "__main__":
    raise SystemExit(main())
