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


def cmd_landsat_test(days: int = 730) -> int:
    """Does adding Landsat actually increase CLEAR KHARIF reads?

    That is the only question worth asking. More passes is not the goal; more
    passes that see something is. Measured on the training blocks, where the
    30 m resolution is not a problem and where the Kharif sample shortage is the
    binding constraint on every model we have tried to validate.

    The test is deliberately cheap and reversible: if the extra clear Kharif
    reads are few, we drop Landsat and say so, rather than carrying a second
    sensor for a gain nobody measured.
    """
    from .ingest.sentinelhub import (Client, SentinelHubError, fetch_landsat,
                                     fetch_optical, merge_optical)
    from .ingest.training_aois import TRAINING_AOIS

    try:
        client = Client.from_env()
    except SentinelHubError as exc:
        print(exc, file=sys.stderr)
        return 1

    end = date(2026, 9, 6)
    start = end - timedelta(days=days)
    KH = {"06", "07", "08", "09", "10"}

    def kharif_clear(rows):
        return sum(1 for r in rows
                   if r["date"][5:7] in KH and r.get("valid_pixel_fraction", 0) >= 0.80)

    print(f"Landsat as a second optical source, {start} to {end}")
    print("Measured on the training blocks. Clear = at least 80% valid pixels.\n")

    tot_s2 = tot_ls = tot_merged = 0
    tot_s2_k = tot_merged_k = 0

    for aoi in TRAINING_AOIS[:4]:   # four blocks is enough to decide
        poly = [[x, y] for x, y in aoi.polygon]
        try:
            s2 = fetch_optical(client, poly, start, end, interval_days=1)
            ls = fetch_landsat(client, poly, start, end, interval_days=1)
        except SentinelHubError as exc:
            print(f"  {aoi.aoi_id} FAILED: {str(exc)[:200]}", file=sys.stderr)
            return 1

        merged = merge_optical(s2, ls)
        s2k, mk = kharif_clear(s2), kharif_clear(merged)
        tot_s2 += len(s2); tot_ls += len(ls); tot_merged += len(merged)
        tot_s2_k += s2k; tot_merged_k += mk

        gain = f"+{mk - s2k}" if mk > s2k else "0"
        print(f"  {aoi.aoi_id} {aoi.crop:7}  S2 {len(s2):4} passes / {s2k:3} clear kharif   "
              f"+LS {len(ls):4}  ->  merged {len(merged):4} / {mk:3} clear kharif  ({gain})")

    print()
    print(f"  Sentinel-2 alone   {tot_s2:5} passes, {tot_s2_k:4} clear Kharif reads")
    print(f"  With Landsat       {tot_merged:5} passes, {tot_merged_k:4} clear Kharif reads")
    if tot_s2_k:
        lift = 100 * (tot_merged_k - tot_s2_k) / tot_s2_k
        print(f"  Lift in the reads that matter: {lift:+.0f}%")
        print()
        if lift >= 25:
            print("  WORTH IT. Add Landsat to the training pull. The gain is in")
            print("  Kharif clear reads, which is the constraint on every model.")
        elif lift >= 10:
            print("  MARGINAL. Worth it for the 1 km training blocks, not for")
            print("  0.6-1.4 ha farms where 30 m pixels are only a handful.")
        else:
            print("  NOT WORTH IT. Drop Landsat and record that it was measured")
            print("  rather than assumed. Clouds block both sensors on the same")
            print("  days more often than the different orbits suggest.")
    return 0


def cmd_nisar_pull_h5(days: int = 120, farms: int = 2, granules: int = 6) -> int:
    """NISAR L-band over our farms with no GDAL and no rasterio.

    The openSEPPO route needs rasterio, rasterio needs GDAL, and GDAL's DLLs are
    blocked by endpoint security on this machine. Rather than fight a device
    policy, this removes the dependency: GCOV is geocoded HDF5, so h5py can open
    it over HTTPS and slice the window covering a farm directly. An 8 GB granule
    costs kilobytes.
    """
    from .ingest.nisar import TELANGANA_BBOX, check as nisar_search_check
    from .ingest.nisar_h5 import sample_farm
    from .seed import FARMS

    try:
        import h5py, fsspec  # noqa: F401
    except ImportError as exc:
        print(f"needs h5py and fsspec: {exc}", file=sys.stderr)
        return 1

    token = None
    tok = pathlib.Path.home() / ".cache" / "openseppo" / "earthaccess_token.json"
    if tok.exists():
        try:
            token = json.loads(tok.read_text()).get("access_token")
        except Exception:  # noqa: BLE001
            pass
    if not token:
        print("No Earthdata bearer token found.")
        print("Register at https://urs.earthdata.nasa.gov, then run:")
        print("  seppo_earthaccess_credentials -t")
        return 1

    end = date(2026, 9, 6)
    start = end - timedelta(days=days)

    # Reuse the search that already works — it needs no GDAL.
    import asf_search as asf
    wkt = (f"POLYGON(({TELANGANA_BBOX['min_lon']} {TELANGANA_BBOX['min_lat']},"
           f"{TELANGANA_BBOX['max_lon']} {TELANGANA_BBOX['min_lat']},"
           f"{TELANGANA_BBOX['max_lon']} {TELANGANA_BBOX['max_lat']},"
           f"{TELANGANA_BBOX['min_lon']} {TELANGANA_BBOX['max_lat']},"
           f"{TELANGANA_BBOX['min_lon']} {TELANGANA_BBOX['min_lat']}))")
    results = asf.search(dataset="NISAR", processingLevel="GCOV",
                         intersectsWith=wkt, start=start.isoformat(),
                         end=end.isoformat(), maxResults=granules)
    urls = [r.properties["url"] for r in results if r.properties.get("url")]
    print(f"{len(urls)} GCOV granules over the AOI, {start} to {end}\n")

    out = {"as_of": end.isoformat(), "source": "nisar-l-gcov-h5", "farms": {}}
    for farm in FARMS[:farms]:
        print(f"{farm.farm_id}  {farm.farm_name} ({farm.crop}, {farm.area_ha} ha)")
        rows = []
        for url in urls:
            name = url.rsplit("/", 1)[-1].replace(".h5", "")
            parts = name.split("_")
            acq = parts[11][:8] if len(parts) > 11 else ""
            iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}" if len(acq) == 8 else ""

            stats, note = sample_farm(url, farm.polygon, token=token)
            if stats is None:
                print(f"   {iso or name[:24]}  {note[:90]}")
                continue
            hh = stats.get("HHHH", {}).get("mean_db")
            hv = stats.get("HVHV", {}).get("mean_db")
            px = max(s.get("pixels", 0) for s in stats.values())
            ratio = f"{hv - hh:+.2f}" if hh is not None and hv is not None else "  n/a"
            print(f"   {iso}  HH {hh if hh is None else round(hh,2):>8}  "
                  f"HV {hv if hv is None else round(hv,2):>8}  "
                  f"ratio {ratio}  {px:5} px   {note[:40]}")
            rows.append({"date": iso, "sensor": "nisar_l",
                         "sar_vv": hh, "sar_vh": hv,
                         "radar_sources": ["nisar_l"], "pixels": px,
                         "granule": name})
        out["farms"][farm.farm_id] = rows
        print()

    path = DATA / "nisar.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    total = sum(len(v) for v in out["farms"].values())
    print(f"wrote {path} — {total} L-band observations")
    if total:
        px = [r["pixels"] for v in out["farms"].values() for r in v]
        med = sorted(px)[len(px) // 2]
        print(f"pixels per farm per pass: min {min(px)}, median {med}, max {max(px)}")
        print()
        if med < 10:
            print("TOO FEW PIXELS. At 20 m posting a smallholder plot is a handful")
            print("of cells, and a mean over that is not a measurement. The unit of")
            print("analysis has to move to village or mandal before L-band is used.")
        else:
            print("Enough pixels per plot to average meaningfully. Next: pull the")
            print("full season and fit L-band against the same Kharif-stratified")
            print("gate as everything else.")
    return 0


def cmd_eval_chatbot() -> int:
    """Run the golden set against the live agent and report whether it may ship.

    Krupa's 2b question is whether the explanations are actually good AND SAFE.
    Groundedness alone answers neither: an answer can be perfectly grounded and
    still tell a lender that a field nobody has seen in forty days is healthy.
    """
    from fastapi.testclient import TestClient

    from .main import app
    from .services.chatbot_eval import Severity, run_suite
    from .services.golden_set import GOLDEN

    client = TestClient(app)

    def ask(case):
        body = {"farm_id": case.farm_id, "audience": case.audience}
        if case.question:
            body["question"] = case.question
        r = client.post("/v1/agent/explain", json=body)
        return r.json() if r.status_code == 200 else {"answer": "", "refused": False}

    print(f"Chatbot evaluation — {len(GOLDEN)} golden cases\n")
    out = run_suite(GOLDEN, ask)

    for r in out["results"]:
        mark = "PASS" if r.passed else ("FATAL" if r.fatal else "FAIL")
        print(f"  {mark:5} {r.case_id}")
        for f in r.findings:
            if f.severity != Severity.MINOR:
                print(f"        {f.severity.value:5} {f.dimension:14} {f.detail}")

    print()
    print(f"  passed {out['passed']}/{out['n']}  ({out['pass_rate']:.0%})")
    print(f"  fatal findings on {out['fatal']} case(s)")
    print(f"  deterministic fallback served {out['fallback_rate']:.0%} of answers")
    if out["failures_by_dimension"]:
        print("  failures by dimension: " + ", ".join(
            f"{k} {v}" for k, v in sorted(out["failures_by_dimension"].items())))

    print()
    if out["may_ship"]:
        print("  MAY SHIP on the automated gate.")
    else:
        print("  BLOCKED:")
        for reason in out["blocking_reasons"]:
            print(f"    {reason}")

    print()
    print("  Still requires a human before release:")
    for h in out["human_review_required"]:
        print(f"    {h}")
    return 0 if out["may_ship"] else 1


def cmd_nisar_season(granules: int = 60) -> int:
    """Full available NISAR season over every farm, then RVI, health and a fit.

    NISAR L-band opened on 17 June 2026, so the available window is that date to
    now. Everything is reported in order of how strongly it can be claimed: the
    series first, then the coincidences with optical, then a model only if the
    coincidences support one.
    """
    from .ingest.nisar import TELANGANA_BBOX
    from .ingest.nisar_h5 import sample_farm
    from .ingest.nisar_season import (LBandObservation, assess_health,
                                      find_coincidences, fit_lband_to_ndvi)
    from .seed import FARMS

    tok = pathlib.Path.home() / ".cache" / "openseppo" / "earthaccess_token.json"
    token = None
    if tok.exists():
        try:
            token = json.loads(tok.read_text()).get("access_token")
        except Exception:  # noqa: BLE001
            pass
    if not token:
        print("No Earthdata token. Run: seppo_earthaccess_credentials -t", file=sys.stderr)
        return 1

    import asf_search as asf
    b = TELANGANA_BBOX
    wkt = (f"POLYGON(({b['min_lon']} {b['min_lat']},{b['max_lon']} {b['min_lat']},"
           f"{b['max_lon']} {b['max_lat']},{b['min_lon']} {b['max_lat']},"
           f"{b['min_lon']} {b['min_lat']}))")

    start, end = date(2026, 6, 17), date(2026, 9, 6)
    results = asf.search(dataset="NISAR", processingLevel="GCOV",
                         intersectsWith=wkt, start=start.isoformat(),
                         end=end.isoformat(), maxResults=granules)
    urls = [r.properties["url"] for r in results if r.properties.get("url")]
    print(f"NISAR L-band, {start} to {end}: {len(urls)} GCOV granules over the AOI")
    print("Reading each farm's window straight from S3 — nothing is downloaded.\n")

    optical = {}
    for name in ("observations.json", "history.json"):
        f = DATA / name
        if f.exists():
            for fid, rows in json.loads(f.read_text(encoding="utf-8"))["farms"].items():
                optical.setdefault(fid, []).extend(rows)

    series: Dict[str, List] = {}
    for farm in FARMS:
        rows = []
        for url in urls:
            parts = url.rsplit("/", 1)[-1].replace(".h5", "").split("_")
            acq = parts[11][:8] if len(parts) > 11 else ""
            iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}" if len(acq) == 8 else ""
            stats, _note = sample_farm(url, farm.polygon, token=token)
            if not stats:
                continue
            hh = stats.get("HHHH", {}).get("mean_db")
            hv = stats.get("HVHV", {}).get("mean_db")
            if hh is None or hv is None:
                continue
            rows.append(LBandObservation(
                farm_id=farm.farm_id, date=iso, hh_db=hh, hv_db=hv,
                pixels=max(s.get("pixels", 0) for s in stats.values()),
                granule="_".join(parts[:12]),
            ))
        series[farm.farm_id] = sorted(rows, key=lambda r: r.date)
        got = len(rows)
        print(f"  {farm.farm_id} {farm.crop:7} {got:3} usable of {len(urls)} granules"
              f"   ({100*got/max(len(urls),1):.0f}% on-swath with valid backscatter)")

    print()
    print("L-band vegetation index over the season")
    print("  RVI = 4*HV/(HH+HV) in linear power. Rises as a canopy gains volume")
    print("  structure, falls at senescence, and is measured through cloud.\n")
    for fid, rows in series.items():
        if not rows:
            continue
        strip = "  ".join(f"{r.date[5:]}:{r.rvi:.3f}" for r in rows if r.rvi is not None)
        print(f"  {fid}  {strip}")

    print()
    print("Crop health, from the series alone")
    print("  Direction and peer deviation. Not a yield and not a stress percentage —")
    print("  those need calibration against ground truth that does not exist yet.\n")
    for farm in FARMS:
        h = assess_health(farm.farm_id, series.get(farm.farm_id, []), series)
        if h is None:
            print(f"  {farm.farm_id}  no usable L-band reads")
            continue
        dev = f"{h.deviation:+.3f}" if h.deviation is not None else "n/a"
        print(f"  {farm.farm_id} {farm.crop:7} RVI {h.rvi:.3f}  vs peers {dev}  "
              f"-> {h.verdict}")
        print(f"  {'':12} {h.basis}")

    print()
    print("Coincidences with clear optical — the only rows that can calibrate L-band")
    all_pairs = []
    for farm in FARMS:
        pairs = find_coincidences(series.get(farm.farm_id, []),
                                  optical.get(farm.farm_id, []))
        all_pairs.extend(pairs)
        print(f"  {farm.farm_id}  {len(pairs)} pairs")

    print()
    fit = fit_lband_to_ndvi(all_pairs)
    if fit["fitted"]:
        print(f"L-band to NDVI: rmse {fit['rmse']}±{fit['rmse_sd']}  "
              f"upper {fit['rmse_upper']}  r2 {fit['r2']:+.3f}  "
              f"n={fit['n']} over {fit['n_splits']} splits")
        print("  " + "  ".join(f"{k}={v:+.4f}" for k, v in fit["coefficients"].items()))
        verdict = "PASSES" if fit["rmse_upper"] <= 0.12 else "FAILS"
        print(f"  Against the 0.12 ceiling on the upper bound: {verdict}")
    else:
        print(f"NO FIT: {fit['reason']}")
        print()
        print("  That is the honest result, not a failure. The series and the health")
        print("  reading above stand on their own — they are measurements. A model")
        print("  mapping L-band to NDVI needs a Kharif with more coincident clear")
        print("  optical days than this one had.")

    out = {
        "as_of": end.isoformat(), "source": "nisar-l-gcov",
        "granules_searched": len(urls),
        "farms": {fid: [{"date": r.date, "hh_db": r.hh_db, "hv_db": r.hv_db,
                         "ratio_db": r.ratio_db, "rvi": r.rvi, "pixels": r.pixels}
                        for r in rows] for fid, rows in series.items()},
        "coincidences": all_pairs,
        "fit": fit,
    }
    path = DATA / "nisar_season.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


def cmd_nisar_env() -> int:
    """What is installed, and is there an Earthdata login."""
    from .ingest.nisar_ingest import check_environment

    env = check_environment()
    print("NISAR ingest environment\n")
    for k, v in env.items():
        if k == "ready":
            continue
        print(f"  {'OK ' if v else '-- '} {k}")
    print()
    if env["ready"]:
        print("Ready. Next: python -m app.cli nisar-pull")
    else:
        print("Not ready. Install:")
        print("  pip install openseppo rasterio xarray h5py earthaccess")
        if not env["netrc_has_earthdata"]:
            print()
            print("And add an Earthdata login. Register free at")
            print("  https://urs.earthdata.nasa.gov")
            print("then run: seppo_earthaccess_credentials -t")
    return 0 if env["ready"] else 1


def cmd_nisar_pull(days: int = 120, farms: int = 2) -> int:
    """Subset L-band over a couple of farms and report what came back.

    Deliberately two farms, not six. The question is whether the pipeline
    produces usable backscatter over a smallholder plot at all — 20 m posting
    against a 0.8 ha field is a handful of pixels, and if that handful is too
    few the whole approach needs a different unit of analysis, which is worth
    finding out before pulling twenty granules.
    """
    from .ingest.nisar_ingest import check_environment, ingest_farm
    from .seed import FARMS

    env = check_environment()
    if not env["ready"]:
        print("Environment not ready. Run: python -m app.cli nisar-env", file=sys.stderr)
        return 1

    end = date(2026, 9, 6)
    start = end - timedelta(days=days)
    out = {"as_of": end.isoformat(), "source": "nisar-l-gcov", "farms": {}}

    for farm in FARMS[:farms]:
        print(f"\n{farm.farm_id}  {farm.farm_name} ({farm.crop})")
        rows, notes = ingest_farm(farm.farm_id, farm.polygon, start, end)
        for n in notes:
            print(f"   {n}")
        for r in rows:
            ratio = f"{r.ratio_db:+.2f}" if r.ratio_db is not None else "  n/a"
            print(f"   {r.date}  HH {r.hh_db:7.2f}  HV {r.hv_db:7.2f}  "
                  f"ratio {ratio}  {r.pixels:5} px")
        out["farms"][farm.farm_id] = [r.as_pair_row() for r in rows]

    path = DATA / "nisar.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    total = sum(len(v) for v in out["farms"].values())
    print(f"\nwrote {path} — {total} L-band observations")
    if total == 0:
        print("Nothing came back. That is a result, not a failure: report it.")
    else:
        px = [r["pixels"] for v in out["farms"].values() for r in v]
        print(f"pixels per farm per pass: min {min(px)}, median "
              f"{sorted(px)[len(px)//2]}, max {max(px)}")
        print("A mean over very few pixels is not the same measurement as a mean")
        print("over many. If these counts are single digits, the unit of analysis")
        print("has to change before the data is used.")
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
        "landsat-test": cmd_landsat_test,
        "nisar-env": cmd_nisar_env,
        "nisar-pull": cmd_nisar_pull,
        "nisar-pull-h5": cmd_nisar_pull_h5,
        "eval-chatbot": cmd_eval_chatbot,
        "nisar-season": cmd_nisar_season,
        "nisar-check": cmd_nisar_check,
        "nisar-inspect": cmd_nisar_inspect,
        "nisar-feasibility": cmd_nisar_feasibility,
        "fit": cmd_fit,
        "status": cmd_status,
    }.get(cmd, lambda: (print(__doc__), 1)[1])()


if __name__ == "__main__":
    raise SystemExit(main())
