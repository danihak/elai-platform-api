"""
tools.py — the tool surface the model is allowed to use.

This module is the safety boundary. The model cannot read the database, cannot
browse, and cannot recall anything about a farm from training. Every fact in an
answer must have entered through one of these functions, and every one of them
returns data straight out of the confidence core.

Consequence: a hallucinated NDVI value is structurally impossible, because the
model was never in a position to produce one. It phrases an evidence chain it
was handed. It does not decide what the evidence is.

Guardrails live here rather than in a prompt. A prompt can be argued with; a
code path cannot.
"""

from __future__ import annotations

from typing import Any, Dict, List

from elai_confidence_core import BLIND, DEGRADED, VERIFIED

from ..store.repository import repo

# --------------------------------------------------------------------------
# Package of Practices — the only agronomic content the model may cite.
# Deliberately thin, deliberately non-prescriptive, and it never names a
# chemical or a dose. Replace with the tenant's own PoP reference in production.
# --------------------------------------------------------------------------

POP_REFERENCE: Dict[str, Dict[str, str]] = {
    "maize": {
        "vegetative": "Check for uneven stand and yellowing of lower leaves. Weed competition is the usual cause at this stage.",
        "flowering": "Water stress at tasselling and silking costs more yield than at any other stage. Check soil moisture at root depth.",
        "grain_fill": "Leaf yellowing from the bottom upward is normal late in grain fill. Yellowing from the top is not.",
        "maturation": "Grain moisture and black layer formation determine harvest timing.",
    },
    "cotton": {
        "vegetative": "Look for uneven emergence and early sucking-pest damage on the youngest leaves.",
        "flowering": "Square and boll shedding rises sharply under water stress. Inspect the top third of the plant.",
        "grain_fill": "Boll development is sensitive to both water stress and waterlogging.",
    },
}


def tool_definitions() -> List[Dict[str, Any]]:
    """The schema handed to the model. Five tools, no more."""
    farm_id = {
        "type": "object",
        "properties": {"farm_id": {"type": "string", "description": "Farm identifier, e.g. TS-MZ-0044"}},
        "required": ["farm_id"],
    }
    return [
        {
            "name": "get_farm_context",
            "description": (
                "Who and where. Farm name, farmer name, crop, variety, area, district, "
                "sowing date, days after sowing, current growth stage, irrigation type."
            ),
            "input_schema": farm_id,
        },
        {
            "name": "get_latest_observation",
            "description": (
                "The most recent satellite observation and its confidence state. Returns "
                "the indices, the confidence state (Verified, Degraded or Blind), which "
                "fallback rung produced it, the error band if estimated, the cloud "
                "condition, and the rule version. This is the primary evidence for any "
                "explanation of why a field is flagged."
            ),
            "input_schema": farm_id,
        },
        {
            "name": "get_observation_history",
            "description": (
                "Recent observations for this farm so you can say what CHANGED, not just "
                "what the current value is. Use this to compare this week against previous "
                "weeks, and to find when the field was last seen clearly."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "farm_id": {"type": "string"},
                    "last_n": {"type": "integer", "description": "How many recent observations, max 12"},
                },
                "required": ["farm_id"],
            },
        },
        {
            "name": "get_cloud_state",
            "description": (
                "Cloud and coverage picture for this farm: how many days since a clear "
                "optical pass, blind days this season against the SLA budget, which radar "
                "sources are contributing, and whether the SLA is in breach."
            ),
            "input_schema": farm_id,
        },
        {
            "name": "get_pop_reference",
            "description": (
                "Package of Practices guidance for this crop at this growth stage. "
                "Observational only — what to look for in the field. It contains no "
                "chemical names and no application rates, by design."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string"},
                    "stage": {"type": "string"},
                },
                "required": ["crop", "stage"],
            },
        },
    ]


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def _not_found(farm_id: str) -> Dict[str, Any]:
    return {"error": "farm not found", "farm_id": farm_id}


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one tool call. Every return value is data from the core."""
    farm_id = args.get("farm_id", "")

    if name == "get_farm_context":
        f = repo.farm(farm_id)
        if not f:
            return _not_found(farm_id)
        latest = repo.latest(farm_id)
        return {
            "farm_id": f.farm_id, "farm_name": f.farm_name, "farmer_name": f.farmer_name,
            "crop": f.crop, "variety": f.variety, "area_ha": f.area_ha,
            "district": f.district, "mandal": f.mandal, "state": f.state,
            "season": f.season, "irrigation_type": f.irrigation_type,
            "sowing_date": f.sowing_date.isoformat(),
            "sowing_date_source": f.sowing_date_source,
            "days_after_sowing": latest.das if latest else None,
            "growth_stage": latest.stage if latest else None,
        }

    if name == "get_latest_observation":
        latest = repo.latest(farm_id)
        if latest is None:
            return _not_found(farm_id)
        return {
            "obs_date": latest.obs_date.isoformat(),
            "days_after_sowing": latest.das,
            "growth_stage": latest.stage,
            "confidence_state": latest.state,
            "state_meaning": {
                VERIFIED: "Measured directly from a clear optical pass.",
                DEGRADED: "Estimated, because optical was unusable. The error band is stated.",
                BLIND: "Not observed, and no estimate is trustworthy enough to show a value.",
            }[latest.state],
            "ndvi": latest.ndvi,
            "ndvi_error_band": latest.ndvi_uncertainty,
            "indices": latest.indices,
            "suppressed_indices": latest.suppressed_indices,
            "fallback_rung": latest.rung_label,
            "data_source": latest.source,
            "cloudy": latest.cloudy,
            "valid_pixel_percent": round(latest.valid_pixel_fraction * 100),
            "days_since_last_clear_optical": latest.days_since_last_clear_optical,
            "last_clear_date": latest.last_verified_date.isoformat() if latest.last_verified_date else None,
            "last_clear_das": latest.last_verified_das,
            "engine_reason": latest.reason,
            "rule_version": latest.rule_version,
        }

    if name == "get_observation_history":
        series = repo.scored(farm_id)
        if not series:
            return _not_found(farm_id)
        n = min(int(args.get("last_n", 6)), 12)
        return {
            "farm_id": farm_id,
            "observations": [
                {
                    "obs_date": s.obs_date.isoformat(), "das": s.das, "stage": s.stage,
                    "confidence_state": s.state, "ndvi": s.ndvi, "cloudy": s.cloudy,
                    "fallback_rung": s.rung_label,
                }
                for s in series[-n:]
            ],
        }

    if name == "get_cloud_state":
        from ..routers.coverage import _farm_coverage
        cov = _farm_coverage(farm_id)
        if cov is None:
            return _not_found(farm_id)
        latest = repo.latest(farm_id)
        return {
            "days_since_last_clear_optical": cov["days_since_last_valid"],
            "blind_days_this_season": cov["blind_days_this_season"],
            "blind_days_budget": cov["blind_days_budget"],
            "sla_tier": cov["sla_tier"],
            "sla_in_breach": cov["in_breach"],
            "radar_sources_used": list(
                repo.raw_observations(farm_id)[-1].radar_sources
            ) if repo.raw_observations(farm_id) else [],
            "current_state": latest.state if latest else None,
        }

    if name == "get_pop_reference":
        crop = str(args.get("crop", "")).lower()
        stage = str(args.get("stage", "")).lower()
        text = POP_REFERENCE.get(crop, {}).get(stage)
        return {
            "crop": crop, "stage": stage,
            "guidance": text or "No Package of Practices entry for this crop and stage.",
            "constraint": (
                "Observational guidance only. This reference contains no chemical names "
                "and no application rates. Do not supply either from any other source."
            ),
        }

    return {"error": f"unknown tool: {name}"}
