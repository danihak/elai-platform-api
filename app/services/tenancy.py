"""
tenancy.py — what a client may configure, and what they may never.

KRUPA'S LINE. "We serve multiple clients." Until now that was a `client_id`
string on a farm and nothing else: every client saw the same thresholds, the same
SLA, the same rendering of the same number.

THE QUESTION THIS ANSWERS, AND IT IS A PRODUCT QUESTION NOT AN ENGINEERING ONE.
A lender, an insurer and an exporter buy different things from the same
observation. The lender needs to know what to hold; the insurer needs a defensible
chronology; the exporter needs tonnage by week. They should see different
screens. They must not see different FACTS.

So configuration splits in two, and the split is the whole design:

  NEGOTIABLE      what a client is shown, how often, at what tier, in what
                  language, against what SLA, at what cost. Commercial terms.

  NON-NEGOTIABLE  the thresholds, the confidence states, the precision policy,
                  the ceiling. If a client could soften these, "Verified" would
                  mean something different per contract and the word would be
                  worth nothing.

A tenant that could buy a looser definition of Verified is a tenant that has
bought the right to be lied to, and every other tenant pays for it — because the
only thing holding the product together is that a state means the same thing
everywhere.

WHAT MAKES THIS DIFFERENT FROM A FEATURE FLAG SYSTEM. Every override is checked
against that split at write time and refused with a reason, rather than being
applied and audited later. The refusal is the product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class Tier(str, Enum):
    BASIC = "basic"
    ASSURED = "assured"
    PREMIUM = "premium"


class Vertical(str, Enum):
    LENDER = "lender"
    INSURER = "insurer"
    EXPORTER = "exporter"
    GOVERNMENT = "government"


#: What a client may set. All commercial, none of it changes what a number means.
NEGOTIABLE: Set[str] = {
    "sla_tier", "blind_days_budget", "revisit_target_days",
    "escalation_enabled", "escalation_budget_inr", "paid_tasking_enabled",
    "default_audience", "languages", "channels", "alert_thresholds_notify_only",
    "report_cadence", "retention_days", "webhook_url", "data_residency",
    "visible_metrics", "branding", "contact_emails",
}

#: What no client may set, at any price, and the reason in each case. The reason
#: is stored with the rule because a refusal without one reads as an obstruction
#: rather than a position.
NON_NEGOTIABLE: Dict[str, str] = {
    "decision_grade_ceiling":
        "the ceiling defines what may be shown at all; a client who could raise "
        "it would be buying the right to see numbers we know are unreliable",
    "confidence_states":
        "Verified, Degraded and Blind must mean the same thing in every tenant, "
        "or they mean nothing in any of them",
    "precision_policy":
        "how many decimals a state permits is the mechanism that stops a blind "
        "field printing a point value; it is the product",
    "stage_index_weights":
        "which index matters at which growth stage is agronomy, not commerce",
    "stress_thresholds":
        "a client-specific stress threshold would make 'stressed' a negotiated "
        "term rather than a measured one",
    "conformal_levels":
        "a coverage guarantee is only a guarantee because it was measured; a "
        "client cannot select a level that was not certified for their crop",
    "rule_version":
        "the rule version is the audit anchor; a tenant-specific version would "
        "make two clients' records incomparable",
    "estimator_version":
        "every tenant runs the same estimator or the comparison between them is "
        "meaningless",
    "validation_gate":
        "a model either cleared the gate or it did not; that is not a per-client "
        "fact",
}


#: Sensible defaults per vertical. These are STARTING POINTS a client may then
#: negotiate within the negotiable set — not a second way to change meaning.
VERTICAL_DEFAULTS: Dict[Vertical, Dict[str, object]] = {
    Vertical.LENDER: {
        "default_audience": "lender",
        "visible_metrics": ["confidence_state", "yield_band", "days_unseen",
                            "exposure_at_risk", "harvest_window"],
        "report_cadence": "weekly",
        "sla_tier": Tier.ASSURED.value,
        "blind_days_budget": 18,
    },
    Vertical.INSURER: {
        "default_audience": "insurer",
        "visible_metrics": ["confidence_state", "observation_history",
                            "cloud_state", "rung_used", "rule_version"],
        "report_cadence": "per_claim",
        "sla_tier": Tier.PREMIUM.value,
        "blind_days_budget": 10,
        "retention_days": 2555,   # seven years; claims are disputed late
    },
    Vertical.EXPORTER: {
        "default_audience": "exporter",
        "visible_metrics": ["harvest_window", "tonnage_band", "confidence_state",
                            "quality_risk"],
        "report_cadence": "weekly",
        "sla_tier": Tier.ASSURED.value,
    },
    Vertical.GOVERNMENT: {
        "default_audience": "insurer",
        "visible_metrics": ["confidence_state", "observation_history",
                            "area_aggregates"],
        "report_cadence": "monthly",
        "sla_tier": Tier.BASIC.value,
        "data_residency": "in_country",
    },
}


class OverrideRefused(ValueError):
    """Raised when a tenant tries to configure something that is not theirs."""


@dataclass
class Tenant:
    tenant_id: str
    name: str
    vertical: Vertical
    config: Dict[str, object] = field(default_factory=dict)
    farms: List[str] = field(default_factory=list)

    def resolved(self) -> Dict[str, object]:
        """Vertical defaults, then this tenant's negotiated overrides."""
        out = dict(VERTICAL_DEFAULTS.get(self.vertical, {}))
        out.update(self.config)
        return out

    def set(self, key: str, value: object) -> "Tenant":
        """Apply one override, or refuse it with the reason.

        Refused at write time rather than applied and audited afterwards. A
        setting that reaches the store is a setting someone will rely on, and
        the point of the split is that certain things can never be relied on
        differently by different clients.
        """
        if key in NON_NEGOTIABLE:
            raise OverrideRefused(
                f"{key} cannot be set per client. {NON_NEGOTIABLE[key]}")
        if key not in NEGOTIABLE:
            raise OverrideRefused(
                f"{key} is not a recognised setting. Configurable: "
                f"{', '.join(sorted(NEGOTIABLE))}")
        self.config[key] = value
        return self

    def may_see(self, metric: str) -> bool:
        visible = self.resolved().get("visible_metrics") or []
        return metric in visible


def what_a_client_can_change() -> Dict[str, object]:
    """The split, exposed. Sales needs it as much as engineering does.

    A prospect asking "can you configure the confidence threshold for us" gets a
    straight no and a reason, which is a better commercial position than a
    discussion — it is the same answer every competitor would have to say yes to.
    """
    return {
        "negotiable": sorted(NEGOTIABLE),
        "non_negotiable": NON_NEGOTIABLE,
        "principle": (
            "Clients configure what they are shown, how often, at what tier and "
            "at what cost. Nobody configures what a number means. A tenant that "
            "could buy a looser definition of Verified has bought the right to "
            "be lied to, and every other tenant pays for it."
        ),
        "what_is_shared": [
            "the confidence engine and its rule version",
            "the estimator and its validation gate",
            "the precision policy",
            "the stage-weighted stress model",
            "the conformal calibration, per crop",
        ],
        "what_is_per_tenant": [
            "which metrics appear on screen",
            "the SLA tier and blind-day budget",
            "whether paid escalation is available and at what budget",
            "languages, channels and report cadence",
            "retention and data residency",
        ],
    }


def render_for(tenant: Tenant, metric: str, display: str, state: str) -> Dict[str, object]:
    """The same finding, rendered for one tenant.

    Note what is NOT tenant-specific: `display` arrives already formatted by the
    precision policy and is passed through unchanged. A tenant may decide whether
    to show a metric; it may not decide how a number is written.
    """
    if not tenant.may_see(metric):
        return {"visible": False,
                "reason": f"{metric} is not in this tenant's visible set"}
    return {
        "visible": True,
        "metric": metric,
        "display": display,          # verbatim, always
        "state": state,
        "audience": tenant.resolved().get("default_audience"),
    }
