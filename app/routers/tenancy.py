"""Tenancy endpoints — what a client may configure, and what nobody may."""

from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.tenancy import (NON_NEGOTIABLE, OverrideRefused, Tenant,
                                Vertical, what_a_client_can_change)

router = APIRouter(prefix="/v1/tenancy", tags=["tenancy"])

#: Demo tenants. Three verticals over the same six farms, which is the point:
#: one engine, one set of facts, three different screens.
TENANTS: Dict[str, Tenant] = {
    "t-coop-01": Tenant("t-coop-01", "Karimnagar District Co-operative Bank",
                        Vertical.LENDER,
                        farms=["TS-MZ-0044", "TS-MZ-0051", "TS-MZ-0063"]),
    "t-ins-01": Tenant("t-ins-01", "Deccan Agri Insurance", Vertical.INSURER,
                       farms=["TS-CT-0107", "TS-CT-0119", "TS-CT-0126"]),
    "t-exp-01": Tenant("t-exp-01", "Sanghar Exports", Vertical.EXPORTER,
                       farms=["TS-MZ-0044", "TS-CT-0107"]),
}


@router.get("/policy", summary="What a client can change, and what nobody can")
def policy() -> dict:
    """The split, published.

    A prospect asking "can you configure the confidence threshold for us" gets a
    straight no and a reason. That is a better commercial position than a
    discussion, and it is the same question every competitor has to say yes to.
    """
    return what_a_client_can_change()


@router.get("", summary="All tenants")
def list_tenants() -> dict:
    return {
        "tenants": [
            {
                "tenant_id": t.tenant_id,
                "name": t.name,
                "vertical": t.vertical.value,
                "farms": len(t.farms),
                "resolved_config": t.resolved(),
            }
            for t in TENANTS.values()
        ]
    }


class OverrideIn(BaseModel):
    key: str
    value: object


@router.post("/{tenant_id}/config", summary="Set one client setting")
def configure(tenant_id: str, body: OverrideIn) -> dict:
    """Apply an override, or refuse it with the reason.

    Refused at write time, not applied and audited later. A setting that reaches
    the store is one somebody will rely on, and the whole point of the split is
    that certain things must never be relied on differently by different clients.
    """
    t = TENANTS.get(tenant_id)
    if t is None:
        raise HTTPException(404, "tenant not found")
    try:
        t.set(body.key, body.value)
    except OverrideRefused as exc:
        # 403, not 400: the request is well formed and the caller is not
        # permitted to make it. That distinction matters in an audit.
        raise HTTPException(403, str(exc))
    return {"tenant_id": tenant_id, "config": t.config,
            "resolved": t.resolved()}


@router.get("/{tenant_id}/view/{farm_id}", summary="One farm, as one tenant sees it")
def tenant_view(tenant_id: str, farm_id: str) -> dict:
    """The same observation, rendered for this tenant.

    Note what does not vary: the state, the display string and the rule version
    are identical for every tenant. Only the selection of metrics and the
    audience framing change. Different screens, never different facts.
    """
    from ..store.repository import repo
    from ..services.tenancy import render_for

    t = TENANTS.get(tenant_id)
    if t is None:
        raise HTTPException(404, "tenant not found")
    if farm_id not in t.farms:
        raise HTTPException(403, f"{farm_id} is not in this tenant's book")

    latest = repo.latest(farm_id)
    if latest is None:
        raise HTTPException(409, "no observations for this farm")

    from elai_confidence_core import derive
    from ..seed import yield_estimate

    farm = repo.farm(farm_id)
    point = yield_estimate(farm, repo.raw_observations(farm_id)[-1])
    y = derive("predicted_productivity", point, [latest], unit="MT/ha")

    candidates = {
        "confidence_state": (latest.state, latest.state),
        "yield_band": (y.display, latest.state),
        "days_unseen": (f"{latest.days_since_last_clear_optical} days",
                        latest.state),
        "exposure_at_risk": ("₹1.45 L", latest.state),
        "harvest_window": ("mid-October", latest.state),
        "observation_history": (f"{len(repo.raw_observations(farm_id))} observations",
                                latest.state),
        "cloud_state": ("cloudy" if latest.cloudy else "clear", latest.state),
        "rung_used": (latest.rung_label, latest.state),
        "rule_version": (latest.rule_version, latest.state),
        "tonnage_band": (y.display, latest.state),
        "quality_risk": ("not assessed", latest.state),
        "area_aggregates": (f"{farm.area_ha} ha", latest.state),
    }

    rendered = {k: render_for(t, k, v[0], v[1]) for k, v in candidates.items()}
    return {
        "tenant": t.name,
        "vertical": t.vertical.value,
        "farm_id": farm_id,
        "shared_across_all_tenants": {
            "state": latest.state,
            "rule_version": latest.rule_version,
            "note": "identical in every tenant; this is what makes them comparable",
        },
        "visible": {k: v for k, v in rendered.items() if v["visible"]},
        "hidden": sorted(k for k, v in rendered.items() if not v["visible"]),
    }
