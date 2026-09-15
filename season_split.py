import json, pathlib
from collections import Counter
from app.seed import FARMS_BY_ID
K = {"06","07","08","09","10"}
R = {"11","12","01","02","03"}
def s(d):
    m = d[5:7]
    return "kharif" if m in K else "rabi" if m in R else "other"
bc = {}
for n in ("training.json","history.json","observations.json"):
    p = pathlib.Path("data")/n
    if not p.exists(): continue
    b = json.loads(p.read_text(encoding="utf-8"))
    if "aois" in b:
        g = [(v["crop"], v["observations"]) for v in b["aois"].values()]
    else:
        g = [(FARMS_BY_ID[k].crop, v) for k,v in b["farms"].items() if k in FARMS_BY_ID]
    for crop, rows in g:
        for r in rows:
            if r.get("ndvi") is None or r.get("sar_vh") is None: continue
            if r.get("valid_pixel_fraction",0) < 0.80: continue
            bc.setdefault(crop, Counter())[s(r["date"])] += 1
for crop, c in sorted(bc.items()):
    t = sum(c.values())
    print(crop)
    for k in ("kharif","rabi","other"):
        print("   %-7s %5d  %5.1f%%" % (k, c[k], 100*c[k]/t))
    print("   total   %5d" % t)
