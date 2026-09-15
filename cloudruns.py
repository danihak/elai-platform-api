import json, pathlib, statistics
from datetime import date
from app.seed import FARMS_BY_ID

K = {"06", "07", "08", "09", "10"}
blob = json.loads(pathlib.Path("data/history.json").read_text(encoding="utf-8"))

runs_all, runs_kharif = [], []
for fid, rows in blob["farms"].items():
    rows = sorted(rows, key=lambda r: r["date"])
    run_start = None
    for r in rows:
        clear = r.get("valid_pixel_fraction", 0) >= 0.80
        d = date.fromisoformat(r["date"])
        if not clear:
            if run_start is None:
                run_start = d
        else:
            if run_start is not None:
                length = (d - run_start).days
                runs_all.append(length)
                if r["date"][5:7] in K:
                    runs_kharif.append(length)
                run_start = None

def report(name, runs):
    if not runs:
        print(name, "no runs"); return
    runs = sorted(runs)
    print(f"{name}: {len(runs)} cloudy runs")
    print(f"   median {statistics.median(runs):.0f} days   mean {statistics.mean(runs):.1f}")
    print(f"   share longer than 3 days: {100*sum(1 for r in runs if r>3)/len(runs):.0f}%")
    print(f"   share longer than 7 days: {100*sum(1 for r in runs if r>7)/len(runs):.0f}%")
    print(f"   longest {max(runs)} days")

report("All year", runs_all)
print()
report("Kharif  ", runs_kharif)
print()
print("Landsat passes ~2-3 days offset from Sentinel-2.")
print("It only helps on cloudy runs SHORTER than that offset.")
print("If most Kharif runs exceed 3 days, a second optical sensor cannot help.")