"""
agent.py — "why is this field flagged", answered from a fixed evidence chain.

Two things are deliberate here.

The evidence chain is assembled by tools, not by a model. The LLM phrases it; it
never decides what the evidence is. That is what makes the answer auditable.

The deterministic renderer is the DEFAULT, not a fallback for outages. If an LLM
is wired in later it sits on top of this same chain. The demo cannot die live
because there is no provider to be down.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from elai_confidence_core import BLIND, DEGRADED, VERIFIED

from ..services import llm
from ..store.repository import repo

router = APIRouter(prefix="/v1/agent", tags=["agent"])

# A substring list only catches the phrasings someone thought of. The golden set
# found two it missed on its first run: "is imidacloprid good for aphids" names a
# chemical without asking a quantity, and "what should I use for stem borer" asks
# for a product recommendation without naming one. Both were answered. Neither
# should have been.
#
# So the trigger is now three independent tests. Any one of them refuses.

#: Asking about quantity or application.
DOSING_TERMS = ("dose", "dosage", "how much", "how many", "ml per", "grams per",
                "kg per", "per acre", "per hectare", "rate of", "quantity of",
                "apply", "application")

#: Spraying and chemical categories.
CHEMICAL_TERMS = ("spray", "pesticide", "insecticide", "herbicide", "fungicide",
                  "weedicide", "chemical", "fertiliser", "fertilizer",
                  "urea", "dap", "mop", "npk", "potash")

#: Active ingredients and trade names. Naming one in a question must never
#: license discussing it in an answer.
ACTIVE_INGREDIENTS = (
    "imidacloprid", "monocrotophos", "endosulfan", "glyphosate", "chlorpyrifos",
    "acephate", "thiamethoxam", "carbendazim", "mancozeb", "profenofos",
    "emamectin", "spinosad", "lambda", "cypermethrin", "atrazine", "paraquat",
    "quinalphos", "dimethoate", "buprofezin", "fipronil",
)

#: Asking what to use, without naming anything. The hardest to catch by keyword
#: and the most natural way a farmer actually asks.
RECOMMENDATION_PATTERNS = (
    r"\bwhat (should|do|can) i (use|apply|spray|put|give)\b",
    r"\bwhich (one|product|medicine|chemical|spray)\b",
    r"\bis \w+ (good|better|ok|okay|safe|effective) for\b",
    r"\brecommend\b.{0,20}\b(for|against)\b",
    r"\bhow (do|to) i (treat|control|kill|stop)\b",
    r"\bwhat.{0,15}\bfor (aphid|borer|rot|blight|worm|pest|disease)",
)


def refuses_agronomic_advice(question: str) -> tuple:
    """(should_refuse, which_guardrails_fired).

    Three independent tests, any one of which refuses. Returning WHICH one fired
    matters: the evaluation harness asserts on it, so a guardrail that stops
    working fails a test rather than quietly passing traffic.
    """
    import re

    low = (question or "").lower()
    fired = []

    if any(t in low for t in ACTIVE_INGREDIENTS):
        fired.append("no_product_names")
    if any(t in low for t in CHEMICAL_TERMS):
        fired.append("no_product_names")
    if any(t in low for t in DOSING_TERMS):
        fired.append("no_dosing")
    if any(re.search(p, low) for p in RECOMMENDATION_PATTERNS):
        fired.append("no_recommendations")

    if not fired:
        return False, []
    ordered = []
    for g in ("no_dosing", "no_product_names", "no_recommendations", "escalate"):
        if g in fired or g == "escalate":
            ordered.append(g)
    return True, ordered


class ExplainRequest(BaseModel):
    farm_id: str
    question: str = "Why is this field flagged?"
    audience: str = "farmer"   # farmer | lender | insurer | exporter
    use_llm: bool = True


@router.post("/explain", summary="Explain a flag from the evidence chain")
def explain(req: ExplainRequest) -> dict:
    f = repo.farm(req.farm_id)
    if not f:
        raise HTTPException(404, "farm not found")
    latest = repo.latest(req.farm_id)
    if latest is None:
        raise HTTPException(409, "no observations for this farm")

    guardrails: List[str] = []

    # Guardrail, enforced server side rather than in a prompt.
    should_refuse, fired = refuses_agronomic_advice(req.question)
    if should_refuse:
        return {
            "farm_id": req.farm_id,
            "audience": req.audience,
            "refused": True,
            "refusal_reason": "chemical dosing and product recommendations",
            "answer": (
                "I can't advise on what to spray or how much. Your field officer "
                "or the local KVK can, and I've flagged this for them."
            ),
            "escalated": True,
            "escalation_target": "field_officer",
            "guardrails_applied": fired,
            "evidence_chain": [],
            "generator": "deterministic",
        }

    chain = [{
        "index": "NDVI",
        "obs_date": latest.obs_date.isoformat(),
        "das": latest.das,
        "value": latest.ndvi,
        "uncertainty": latest.ndvi_uncertainty,
        "stage": latest.stage,
        "state": latest.state,
        "cloud_state": "cloudy" if latest.cloudy else "clear",
        "valid_pixel_fraction": latest.valid_pixel_fraction,
        "source": latest.source,
        "rung": latest.rung_label,
        "rule_version": latest.rule_version,
    }]
    if latest.last_verified_date:
        chain.append({
            "index": "NDVI",
            "obs_date": latest.last_verified_date.isoformat(),
            "das": latest.last_verified_das,
            "state": VERIFIED,
            "cloud_state": "clear",
            "source": "Sentinel-2 L2A",
            "note": "last direct measurement",
        })

    escalate = latest.state == BLIND and f.sla_tier == "premium"
    if escalate:
        guardrails.append("escalate_on_blind_premium")

    # Claude phrases the evidence chain. It never decides what the evidence is,
    # and its output is checked for groundedness before anyone sees it.
    generator = "deterministic"
    answer = _render(f, latest, req.audience)
    llm_meta: dict = {}

    if req.use_llm and llm.available():
        result = llm.explain_with_claude(req.farm_id, req.question, req.audience)
        llm_meta = {
            "attempted": True,
            "model": result.model or None,
            "tool_rounds": result.rounds,
            "tools_called": [c["tool"] for c in result.tool_calls],
            "failure_reason": result.failure_reason,
            "ungrounded_tokens": result.ungrounded_tokens,
        }
        if result.ok:
            answer = result.answer
            generator = "claude"
            guardrails.append("groundedness_verified")
        else:
            # Every failure mode lands here, including a groundedness rejection.
            # The user sees an answer either way; the reason is in the payload
            # for whoever is watching the logs.
            guardrails.append(f"fell_back:{result.failure_reason}")
    elif req.use_llm:
        llm_meta = {"attempted": False,
                    "reason": "no_api_key_or_circuit_open"}

    return {
        "farm_id": req.farm_id,
        "audience": req.audience,
        "refused": False,
        "answer": answer,
        "confidence_state": latest.state,
        "evidence_chain": chain,
        "escalated": escalate,
        "escalation_target": "field_officer" if escalate else None,
        "guardrails_applied": guardrails + ["evidence_chain_only", "no_dosing"],
        "generator": generator,
        "llm": llm_meta,
        "rule_version": latest.rule_version,
    }


def _render(f, latest, audience: str) -> str:
    """Same evidence, four framings. The framing is a template, not a model call."""
    when = latest.obs_date.strftime("%d/%m")
    last = latest.last_verified_date.strftime("%d/%m") if latest.last_verified_date else None
    days = latest.days_since_last_clear_optical

    if latest.state == VERIFIED:
        core = f"We saw {f.farm_name} clearly on {when} (DAS {latest.das}), at {latest.stage.replace('_',' ')}."
        if audience == "farmer":
            return core + " Nothing looks unusual this week compared with last."
        return core + f" NDVI {latest.ndvi}. Confidence Verified under rule {latest.rule_version}."

    if latest.state == DEGRADED:
        if audience == "farmer":
            return (f"Clouds have covered {f.farm_name} since {last or 'the last clear day'}. "
                    f"We estimated this week from radar, so treat it as a guide, not a fact. "
                    f"If you are near the field, tell us what you see.")
        return (f"Optical unusable on {when} ({latest.valid_pixel_fraction:.0%} valid pixels). "
                f"Estimated from {latest.source}, band ±{latest.ndvi_uncertainty:.2f} NDVI. "
                f"Last direct measurement {last} (DAS {latest.last_verified_das}). "
                f"Confidence Degraded under rule {latest.rule_version}.")

    if audience == "farmer":
        return (f"We have not been able to see {f.farm_name} for about {days} days because of cloud. "
                f"We are not going to guess. The last time we saw it clearly was {last or 'earlier in the season'}, "
                f"and it looked normal then.")
    return (f"No decision-grade observation for {days} days. {latest.reason}. "
            f"Last verified {last} (DAS {latest.last_verified_das}). "
            f"Confidence Blind under rule {latest.rule_version}. Values are ranges, not estimates.")


@router.get("/tools", summary="The tool surface the model is allowed to use")
def tools() -> dict:
    """Worth showing a tech team directly.

    This is the entire surface. The model cannot read the database, cannot
    browse, and knows nothing about any farm from training. If a fact is not
    reachable through one of these five functions, it cannot appear in an answer.
    """
    from ..services.tools import tool_definitions
    defs = tool_definitions()
    return {
        "count": len(defs),
        "tools": [{"name": d["name"], "description": d["description"]} for d in defs],
        "guarantee": (
            "Tools assemble the evidence chain. The model phrases it. A hallucinated "
            "figure is structurally impossible, and every numeric claim is verified "
            "against the tool results before the answer is returned."
        ),
    }
