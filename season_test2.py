import json, pathlib, random, statistics
from app.seed import FARMS_BY_ID
from app.ingest.fit import _prepare, _fit_ols, _predict, _rmse, _r2, FEATURES

K = {"06", "07", "08", "09", "10"}
pairs = []
for n in ("training.json", "history.json", "observations.json"):
    p = pathlib.Path("data") / n
    if not p.exists():
        continue
    b = json.loads(p.read_text(encoding="utf-8"))
    g = ([(v["crop"], v["observations"]) for v in b["aois"].values()] if "aois" in b
         else [(FARMS_BY_ID[k].crop, v) for k, v in b["farms"].items() if k in FARMS_BY_ID])
    for crop, rows in g:
        for r in rows:
            if r.get("ndvi") is None or r.get("sar_vh") is None:
                continue
            if r.get("valid_pixel_fraction", 0) < 0.80:
                continue
            pairs.append({"crop": crop, "stage": "x", "sar_vh": r["sar_vh"],
                          "sar_vv": r.get("sar_vv"), "ndvi": r["ndvi"],
                          "season": "kharif" if r["date"][5:7] in K else "other"})

print("30 random splits, mean RMSE on a Kharif hold-out\n")
for crop in ("maize", "cotton"):
    kh = _prepare([p for p in pairs if p["crop"] == crop and p["season"] == "kharif"])
    ot = _prepare([p for p in pairs if p["crop"] == crop and p["season"] != "kharif"])
    res = {"kharif only": [], "non-kharif only": [], "all seasons": []}
    for seed in range(30):
        rng = random.Random(seed)
        k = kh[:]
        rng.shuffle(k)
        cut = int(len(k) * 0.6)
        tr, te = k[:cut], k[cut:]
        if len(te) < 10:
            continue
        for label, train in (("kharif only", tr), ("non-kharif only", ot), ("all seasons", tr + ot)):
            co = _fit_ols(train, FEATURES)
            if co is None:
                continue
            pr = [(_predict(co, r["features"], FEATURES), r["ndvi"]) for r in te]
            res[label].append(_rmse(pr))
    print(crop, " kharif n=%d" % len(kh))
    for label, v in res.items():
        if v:
            print("    %-16s rmse %.4f  sd %.4f  (n=%d splits)"
                  % (label, statistics.mean(v), statistics.pstdev(v), len(v)))
    print()