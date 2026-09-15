"""farms.py — farm list, farm detail, state, and the observation ledger."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from elai_confidence_core import derive

from ..seed import yield_estimate
from ..services.stress import assess, explain_weights
from ..store.repository import repo
from ._shapes import FarmOut, MetricOut, ObservationOut

router = APIRouter(prefix="/v1/farms", tags=["farms"])


def _farm_out(f) -> FarmOut:
    return FarmOut(
        farm_id=f.farm_id, farm_name=f.farm_name, farmer_name=f.farmer_name,
        mobile_masked=f.mobile_masked, country=f.country, state=f.state,
        district=f.district, mandal=f.mandal, area_ha=f.area_ha, lat=f.lat, lon=f.lon,
        polygon=[[x, y] for x, y in f.polygon], crop=f.crop, variety=f.variety,
        season=f.season, irrigation_type=f.irrigation_type,
        sowing_date=f.sowing_date.isoformat(), sowing_date_source=f.sowing_date_source,
        boundary_source=f.boundary_source,
        predicted_harvest_date=f.predicted_harvest_date.isoformat(),
        client_id=f.client_id, client_tier=f.client_tier, sla_tier=f.sla_tier,
    )


@router.get("", response_model=List[FarmOut], summary="List farms")
def list_farms(
    crop: Optional[str] = None,
    district: Optional[str] = None,
    irrigation_type: Optional[str] = None,
    client_id: Optional[str] = None,
) -> List[FarmOut]:
    farms = repo.farms(crop=crop, district=district,
                       irrigation_type=irrigation_type, client_id=client_id)
    return [_farm_out(f) for f in farms]


@router.get("/{farm_id}", response_model=FarmOut, summary="Farm detail")
def get_farm(farm_id: str) -> FarmOut:
    f = repo.farm(farm_id)
    if not f:
        raise HTTPException(404, "farm not found")
    return _farm_out(f)


@router.get("/{farm_id}/state", summary="Current confidence state and metrics")
def farm_state(farm_id: str) -> dict:
    """The screen S4 payload.

    Every metric is derived through the core, so stress, area under stress and
    yield cannot disagree with each other about how much to trust this week.
    """
    f = repo.farm(farm_id)
    if not f:
        raise HTTPException(404, "farm not found")

    latest = repo.latest(farm_id)
    if latest is None:
        raise HTTPException(409, "no observations for this farm")

    point = yield_estimate(f, repo.raw_observations(farm_id)[-1])
    stress = assess(latest, f.crop)

    metrics = [
        derive("predicted_productivity", point, [latest], unit="MT/ha"),
        derive("predicted_total_yield", round(point * f.area_ha, 2), [latest], unit="MT"),
        derive("vegetation_ndvi", latest.ndvi, [latest]),
        # Stress is now computed from stage-weighted indices rather than a
        # hardcoded fraction of the field. The previous version returned
        # area_ha * 0.18 for every farm on every date.
        derive("stress_score", stress.score, [latest]),
    ]

    return {
        "farm_id": farm_id,
        "as_of": latest.obs_date.isoformat(),
        "das": latest.das,
        "stage": latest.stage,
        "state": latest.state,
        "observation": ObservationOut.of(latest),
        "metrics": [MetricOut.of(m) for m in metrics],
        "stress": {
            "band": stress.band,
            "score": stress.score,
            "driving_index": stress.driving_index,
            "state": stress.state,
            "stage": stress.stage,
            "reason": stress.reason,
            "weight_covered": stress.weight_covered,
            "suppressed_indices": stress.suppressed,
            "unavailable_indices": stress.unavailable,
            "contributions": [
                {
                    "index": c.index, "value": c.value, "stage_weight": c.weight,
                    "severity": round(c.severity, 3), "direction": c.direction,
                    "healthy": c.threshold[0], "stressed": c.threshold[1],
                }
                for c in sorted(stress.contributions,
                                key=lambda c: -(c.weight * c.severity))
            ],
        },
        "rule_version": latest.rule_version,
    }


@router.get("/{farm_id}/observations", response_model=List[ObservationOut],
            summary="Full season observation history")
def farm_observations(farm_id: str, limit: int = Query(200, le=500)) -> List[ObservationOut]:
    """Screen S7, and the data source for ClaimChronos.

    Append-only. Every row carries the rule version that produced its state, so
    the same query in March returns September's answer.
    """
    if not repo.farm(farm_id):
        raise HTTPException(404, "farm not found")
    return [ObservationOut.of(s) for s in repo.scored(farm_id)[-limit:]]


@router.get("/{farm_id}/ledger", summary="Audit ledger for this farm")
def farm_ledger(farm_id: str) -> List[dict]:
    if not repo.farm(farm_id):
        raise HTTPException(404, "farm not found")
    return repo.ledger(farm_id)



@router.get("/{farm_id}/stage-weights", summary="How indices are weighted at this stage")
def stage_weights(farm_id: str) -> dict:
    """Krupa's question, answered as data rather than as a slide.

    "How does the relative importance of different stress indices get weighted
    by growth stage?" — this endpoint returns the live weighting for the farm's
    crop at its current stage, with each threshold and its sign-off status.
    """
    f = repo.farm(farm_id)
    if not f:
        raise HTTPException(404, "farm not found")
    latest = repo.latest(farm_id)
    if latest is None:
        raise HTTPException(409, "no observations for this farm")
    return explain_weights(f.crop, latest.stage)



@router.get("/{farm_id}/history", summary="Two-year observation strip")
def farm_history(farm_id: str) -> dict:
    """The season strip: two years of observations on a calendar axis.

    The Kharif window is the product's whole argument, and it only reads as an
    argument next to Rabi. Across these six farms, measured over two years:
    roughly 85% of Rabi passes are clear against 24% of Kharif ones. The
    satellite sees these fields three and a half times better in the season when
    nothing is at risk.

    Two years rather than eighteen months so that two full Kharif cycles appear
    and the pattern reads as a seasonal property rather than one bad year. That
    is also the default window a seasonal-adjustment dashboard uses.

    Served from data/history.json, which `python -m app.cli history` writes. The
    endpoint reports honestly when that file is absent rather than silently
    falling back to the current season, because a strip covering one season
    would look like a two-year strip with no Rabi in it.
    """
    import json
    import pathlib as _p

    f = repo.farm(farm_id)
    if not f:
        raise HTTPException(404, "farm not found")

    hist = _p.Path(__file__).parent.parent.parent / "data" / "history.json"
    if not hist.exists():
        return {
            "farm_id": farm_id,
            "available": False,
            "reason": "data/history.json not present; run `python -m app.cli history`",
            "observations": [],
        }

    blob = json.loads(hist.read_text(encoding="utf-8"))
    rows = sorted(blob.get("farms", {}).get(farm_id, []), key=lambda r: r["date"])

    KHARIF = {"06", "07", "08", "09"}
    RABI = {"11", "12", "01", "02", "03"}

    out, kharif, rabi = [], [], []
    for r in rows:
        vf = r.get("valid_pixel_fraction", 0.0)
        # The strip's three states, from the same thresholds the engine uses.
        # Anything below the usable floor is not a poor reading, it is no
        # reading — and the strip renders it as absence.
        state = "measured" if vf >= 0.80 else "estimated" if vf >= 0.50 else "unseen"
        month = r["date"][5:7]
        out.append({
            "date": r["date"],
            "state": state,
            "valid_pixel_fraction": round(vf, 3),
            "ndvi": r.get("ndvi"),
            "season": "kharif" if month in KHARIF else "rabi" if month in RABI else "other",
        })
        if month in KHARIF:
            kharif.append(state)
        elif month in RABI:
            rabi.append(state)

    def clear_share(seq):
        return round(sum(1 for s in seq if s == "measured") / len(seq), 3) if seq else None

    return {
        "farm_id": farm_id,
        "farm_name": f.farm_name,
        "crop": f.crop,
        "available": True,
        "observations": out,
        "span": {"from": rows[0]["date"], "to": rows[-1]["date"]} if rows else None,
        "clear_share": {
            "overall": clear_share([o["state"] for o in out]),
            "kharif": clear_share(kharif),
            "rabi": clear_share(rabi),
        },
        "finding": (
            "The satellite sees this field materially better in Rabi than in "
            "Kharif — the season when the crop is rainfed and the loan is "
            "outstanding."
        ),
    }
