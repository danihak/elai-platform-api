import json, pathlib
from datetime import date
from collections import defaultdict

blob = json.loads(pathlib.Path("data/history.json").read_text(encoding="utf-8"))
print("Two-year season strips  (# = clear >=80%,  + = partial 50-80%,  . = unusable)\n")

for farm_id, rows in blob["farms"].items():
    by_month = defaultdict(list)
    for r in sorted(rows, key=lambda x: x["date"]):
        vf = r.get("valid_pixel_fraction", 0)
        ch = "#" if vf >= 0.80 else "+" if vf >= 0.50 else "."
        by_month[r["date"][:7]].append(ch)

    months = sorted(by_month)
    strip = "".join("".join(by_month[m]) for m in months)
    clear = strip.count("#")
    print(f"  {farm_id}")
    print(f"    {strip}")

    # Kharif window only, across both years
    kh = [c for m in months if m[5:7] in ("06","07","08","09") for c in by_month[m]]
    rb = [c for m in months if m[5:7] in ("11","12","01","02","03") for c in by_month[m]]
    def pct(s, ch="#"):
        return f"{100*s.count(ch)/len(s):.0f}%" if s else "n/a"
    print(f"    all {clear}/{len(strip)} clear  |  Kharif {pct(kh)} clear  |  Rabi {pct(rb)} clear")
    print()

print("months present:", ", ".join(sorted({r['date'][:7] for rows in blob['farms'].values() for r in rows})))
