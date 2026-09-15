import json, pathlib, random
from app.seed import FARMS_BY_ID
from app.ingest.fit import _prepare, _fit_ols, _predict, _rmse, _r2, FEATURES

K = {"06", "07", "08", "09", "10"}

def s(d):
    return "kharif" if d[5:7] in K else "other"

pairs = []
for n in ("training.json", "history.json", "observations.json"):
    p = pathlib.Path("data") / n
    if not p.exists():
        continue
    b = json.loads(p.read_text(encoding="utf-8"))
    if "aois" in b:
        g = [(v["crop"], v["observations"]) for v in b["aois"].values()]
    else:
        g = [(FARMS_BY_ID[k].crop, v) for k, v in b["farms"].items() if k in FARMS_BY_ID]
    for crop, rows in g:
        for r in rows:
            if r.get("ndvi") is None or r.get("sar_vh") is None:
                continue
            if r.get("valid_pixel_fraction", 0) < 0.80:
                continue
            pairs.append({"crop": crop, "stage": "x", "sar_vh": r["sar_vh"],
                          "sar_vv": r.get("sar_vv"), "ndvi": r["ndvi"],
                          "season": s(r["date"])})

rng = random.Random(7)
print("Trained on WHAT, tested on a KHARIF hold-out\n")

for crop in ("maize", "cotton"):
    kh = _prepare([p for p in pairs if p["crop"] == crop and p["season"] == "kharif"])
    ot = _prepare([p for p in pairs if p["crop"] == crop and p["season"] != "kharif"])
    rng.shuffle(kh)
    cut = int(len(kh) * 0.6)
    tr_k, te_k = kh[:cut], kh[cut:]
    if len(te_k) < 10:
        print(crop, "too few Kharif points")
        continue
    print("%s  kharif n=%d (train %d / test %d)  other n=%d"
          % (crop, len(kh), len(tr_k), len(te_k), len(ot)))
    for label, train in (("kharif only", tr_k),
                         ("non-kharif only", ot),
                         ("all seasons", tr_k + ot)):
        co = _fit_ols(train, FEATURES)
        if co is None:
            continue
        pr = [(_predict(co, r["features"], FEATURES), r["ndvi"]) for r in te_k]
        print("    %-16s trained on %4d  ->  kharif rmse %.4f  r2 %+0.3f"
              % (label, len(train), _rmse(pr), _r2(pr)))
    print()