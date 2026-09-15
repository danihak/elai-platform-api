"""
kisanvoice.py — the farmer-facing surface, and the loop it exists to close.

TWO OF KRUPA'S LINES MEET HERE. Part 1 asks how a model reaches "the hands of
farmers"; Part 2a asks how confidence, "or its absence", is communicated to
clients AND farmers. Everything built so far speaks to institutions. Nothing has
reached a farmer.

WHAT THIS IS NOT. It is not the advisory chatbot every competitor already ships.
Farmer.Chat, Cropwise Grower and Kisan e-Mitra answer "how do I grow cotton" from
a document corpus, and they answer it well. None of them explains the satellite
flag on YOUR plot, and — the part that matters — nothing a farmer tells them ever
reaches a model.

THE ASYMMETRY THIS FIXES. Krupa said most farmers work on gut feeling. That is
not an obstacle to route around, it is the only source of ground truth that
exists at scale. A farmer standing in the field knows in two seconds what a
satellite cannot resolve in two weeks. So every message ends with a question
answerable in one tap, and the answer writes back as a label.

THE ECONOMICS, STATED PLAINLY. Free to the farmer, paid for by the institution,
because the institution is the one who gets a validated model out of it. Under
the RBI Kisan Credit Card Directions a lender may finance remote-sensing advisory
within the drawing limit, which is what makes "free to the farmer" a business
model rather than charity.

WHAT A FARMER NEVER SEES. No index name, no RMSE, no confidence percentage, no
model version. Confidence is carried by what the message CLAIMS, not by a label
attached to a claim. "We could not see your field" is a confidence statement in
plain language, and it is the one a farmer can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional


class Language(str, Enum):
    TELUGU = "te"
    HINDI = "hi"
    ENGLISH = "en"


class Channel(str, Enum):
    WHATSAPP_TEXT = "whatsapp_text"
    WHATSAPP_VOICE = "whatsapp_voice"
    SMS = "sms"
    IVR = "ivr"


#: What a farmer can be asked in one tap. Each maps to a label the model can
#: actually use, which is the test for whether a question earns its place.
@dataclass
class Prompt:
    prompt_id: str
    question: Dict[str, str]           # language -> text
    options: List[Dict[str, str]]      # each: {id, te, hi, en}
    writes: str                        # the label this produces
    why: str                           # why this question is worth asking


TWO_TAP_PROMPTS: Dict[str, Prompt] = {

    "canopy_check": Prompt(
        prompt_id="canopy_check",
        question={
            "te": "మీ పొలంలో పైరు ఇప్పుడు ఎలా ఉంది?",
            "hi": "आपके खेत में फसल अभी कैसी है?",
            "en": "How does the crop in your field look right now?",
        },
        options=[
            {"id": "good", "te": "బాగుంది", "hi": "अच्छी है", "en": "Looks fine"},
            {"id": "patchy", "te": "కొంత భాగం తక్కువ", "hi": "कुछ हिस्सा कमज़ोर",
             "en": "Part of it is weaker"},
            {"id": "bad", "te": "బాగాలేదు", "hi": "खराब है", "en": "Not good"},
        ],
        writes="canopy_condition",
        why=("Resolves a Degraded estimate into a label. A farmer answers this "
             "in two seconds from the edge of the field; the satellite cannot "
             "answer it at all for another eleven days."),
    ),

    "sowing_confirm": Prompt(
        prompt_id="sowing_confirm",
        question={
            "te": "మీరు విత్తనం వేసిన తేదీ ఇదేనా?",
            "hi": "क्या आपने इसी तारीख को बुवाई की थी?",
            "en": "Is this the date you sowed?",
        },
        options=[
            {"id": "yes", "te": "అవును", "hi": "हाँ", "en": "Yes"},
            {"id": "earlier", "te": "ముందుగా", "hi": "पहले", "en": "Earlier"},
            {"id": "later", "te": "తరువాత", "hi": "बाद में", "en": "Later"},
        ],
        writes="sowing_date_confirmed",
        why=("Sowing date sets days-after-sowing, which sets the growth stage, "
             "which sets every stage-weighted threshold in the system. It is the "
             "least glamorous input and the one most models get wrong."),
    ),

    "harvest_timing": Prompt(
        prompt_id="harvest_timing",
        question={
            "te": "ఎప్పుడు కోత కోస్తారు?",
            "hi": "कटाई कब करेंगे?",
            "en": "When will you harvest?",
        },
        options=[
            {"id": "lt2w", "te": "రెండు వారాల లోపు", "hi": "दो हफ़्ते में",
             "en": "Within two weeks"},
            {"id": "2to4w", "te": "రెండు నుండి నాలుగు వారాలు",
             "hi": "दो से चार हफ़्ते", "en": "Two to four weeks"},
            {"id": "gt4w", "te": "నెల తరువాత", "hi": "एक महीने बाद",
             "en": "More than a month"},
        ],
        writes="expected_harvest_window",
        why=("The lender's collection calendar and the exporter's commit window "
             "both hang off this, and the farmer knows it before any model does."),
    ),

    "actual_yield": Prompt(
        prompt_id="actual_yield",
        question={
            "te": "ఈ సారి ఎంత దిగుబడి వచ్చింది?",
            "hi": "इस बार कितनी पैदावार हुई?",
            "en": "How much did you harvest this time?",
        },
        options=[
            {"id": "more", "te": "గత సారి కంటే ఎక్కువ", "hi": "पिछली बार से ज़्यादा",
             "en": "More than last time"},
            {"id": "same", "te": "అంతే", "hi": "उतनी ही", "en": "About the same"},
            {"id": "less", "te": "తక్కువ", "hi": "कम", "en": "Less"},
        ],
        writes="actual_yield_direction",
        why=("The closing of the loop. Every yield figure the platform has ever "
             "shown was a prediction against nothing — Actual Productivity reads "
             "N/A on every farm. This is the first real answer."),
    ),
}


# --------------------------------------------------------------------------
# What a farmer is told
# --------------------------------------------------------------------------

#: One message per confidence state. Confidence is carried by what the message
#: CLAIMS, never by a label or a percentage attached to a claim.
STATE_MESSAGE: Dict[str, Dict[str, str]] = {
    "Verified": {
        "te": "ఈ రోజు ఉపగ్రహం మీ పొలాన్ని స్పష్టంగా చూసింది.",
        "hi": "आज उपग्रह ने आपका खेत साफ़ देखा है।",
        "en": "The satellite saw your field clearly today.",
    },
    "Degraded": {
        "te": "మేఘాల వల్ల పొలం కనిపించలేదు. రాడార్ ద్వారా అంచనా వేశాం — "
              "ఇది కచ్చితమైనది కాదు.",
        "hi": "बादलों की वजह से खेत दिखा नहीं। रडार से अंदाज़ा लगाया है — "
              "यह पक्की बात नहीं है।",
        "en": "Clouds covered your field. We estimated using radar, so treat "
              "this as a guide, not a fact.",
    },
    "Blind": {
        "te": "చాలా రోజులుగా మీ పొలం మాకు కనిపించలేదు. ఇప్పుడు మేము ఏమీ "
              "చెప్పలేము.",
        "hi": "कई दिनों से आपका खेत हमें दिखा नहीं है। अभी हम कुछ कह नहीं सकते।",
        "en": "We have not been able to see your field for some days. We cannot "
              "tell you anything right now.",
    },
}

#: The closing line on every message. It is not politeness — it is the request
#: that turns a broadcast into a measurement.
CLOSING: Dict[str, str] = {
    "te": "మీరు చూసినది మాకు చెప్పండి.",
    "hi": "आप जो देख रहे हैं वह हमें बताइए।",
    "en": "Tell us what you see.",
}

#: Never sent to a farmer, in any language. Not because farmers cannot handle
#: technical language — because none of it helps them decide anything, and a
#: voice note containing it is worse than one that does not.
NEVER_IN_A_FARMER_MESSAGE = (
    "ndvi", "rmse", "conformal", "estimator", "sentinel", "radar vegetation "
    "index", "confidence interval", "rule_version", "p-value", "model",
)


@dataclass
class FarmerMessage:
    farm_id: str
    farmer_name: str
    language: Language
    channel: Channel
    body: str
    prompt: Optional[Prompt]
    state: str
    send_reason: str
    audit: Dict[str, object] = field(default_factory=dict)

    def leaks_jargon(self) -> List[str]:
        low = self.body.lower()
        return [t for t in NEVER_IN_A_FARMER_MESSAGE if t in low]


def compose(
    *,
    farm_id: str,
    farmer_name: str,
    state: str,
    language: Language,
    days_unseen: int,
    last_seen: Optional[str],
    prompt_id: str = "canopy_check",
    channel: Channel = Channel.WHATSAPP_VOICE,
    rule_version: str = "",
) -> FarmerMessage:
    """Build the message a farmer actually receives.

    Short enough to be a voice note, specific enough to be checkable, and it
    always ends by asking. A message that tells a farmer something and asks for
    nothing back is a broadcast, and broadcasts do not close loops.
    """
    lang = language.value
    lines = [STATE_MESSAGE.get(state, STATE_MESSAGE["Blind"])[lang]]

    if state == "Blind" and last_seen:
        when = {
            "te": f"చివరిసారి {last_seen} నాడు చూశాం.",
            "hi": f"आखिरी बार {last_seen} को देखा था।",
            "en": f"We last saw it clearly on {last_seen}.",
        }[lang]
        lines.append(when)

    prompt = TWO_TAP_PROMPTS.get(prompt_id)
    if prompt:
        lines.append(prompt.question[lang])

    lines.append(CLOSING[lang])

    return FarmerMessage(
        farm_id=farm_id, farmer_name=farmer_name, language=language,
        channel=channel, body=" ".join(lines), prompt=prompt, state=state,
        send_reason=f"state={state}, unseen={days_unseen}d",
        audit={"rule_version": rule_version, "days_unseen": days_unseen,
               "last_seen": last_seen, "prompt": prompt_id},
    )


# --------------------------------------------------------------------------
# What comes back
# --------------------------------------------------------------------------


@dataclass
class FarmerReply:
    farm_id: str
    prompt_id: str
    option_id: str
    received: str
    language: Language
    free_text: str = ""
    photo_url: Optional[str] = None
    gps: Optional[Dict[str, float]] = None


#: What a reply is worth. A farmer standing in the field is a better instrument
#: than a radar estimate for the question "is the canopy patchy", and a worse
#: one for "what is the NDVI". Both facts are encoded here.
REPLY_CONFIDENCE = {
    "canopy_condition": 0.85,
    "sowing_date_confirmed": 0.95,
    "expected_harvest_window": 0.80,
    "actual_yield_direction": 0.90,
}


def ingest_reply(reply: FarmerReply, current_state: str) -> Dict[str, object]:
    """Turn a tap into a labelled observation, and say what it changes.

    THE RULE THAT MATTERS. A farmer reply is ground truth about the FIELD. It is
    not a licence to upgrade a satellite observation. If the satellite could not
    see the field, it still could not see the field — what changes is that we now
    have an independent label for that date, which is worth more than the
    observation would have been.

    Upgrading the satellite state on the strength of a farmer's tap would be
    laundering one measurement into another, which is the exact move this system
    exists to refuse.
    """
    prompt = TWO_TAP_PROMPTS.get(reply.prompt_id)
    if prompt is None:
        return {"accepted": False, "reason": f"unknown prompt {reply.prompt_id}"}

    valid = {o["id"] for o in prompt.options}
    if reply.option_id not in valid:
        return {"accepted": False,
                "reason": f"{reply.option_id} is not an option for {reply.prompt_id}"}

    label = {
        "farm_id": reply.farm_id,
        "label": prompt.writes,
        "value": reply.option_id,
        "observed_at": reply.received,
        "source": "farmer_reply",
        "confidence": REPLY_CONFIDENCE.get(prompt.writes, 0.7),
        "has_photo": reply.photo_url is not None,
        "has_gps": reply.gps is not None,
    }

    # A photo with GPS inside the polygon is a materially stronger label than a
    # tap alone, and is the only farmer input that can support a disputed claim.
    if reply.photo_url and reply.gps:
        label["confidence"] = min(0.98, label["confidence"] + 0.1)
        label["verifiable"] = True

    disagrees = (
        reply.prompt_id == "canopy_check"
        and reply.option_id in ("patchy", "bad")
        and current_state in ("Verified", "Degraded")
    )

    return {
        "accepted": True,
        "label": label,
        "satellite_state_unchanged": current_state,
        "note": ("A farmer reply is ground truth about the field, not about what "
                 "the satellite saw. The observation's state is untouched."),
        "farmer_disagrees_with_model": disagrees,
        "action": ("flag for field verification: the farmer reports a problem the "
                   "model did not" if disagrees else "recorded as a training label"),
    }


def should_message(state: str, days_unseen: int, last_messaged_days: int) -> tuple:
    """Who gets a message this week, and why.

    Messaging every farmer every week trains them to ignore it. The rule is that
    a message is sent when we have either something worth saying or something
    worth asking — and being unable to see a field is worth saying.
    """
    if last_messaged_days < 5:
        return False, "messaged within the last five days"
    if state == "Blind":
        return True, "we cannot see this field and the farmer should know that"
    if state == "Degraded" and days_unseen >= 10:
        return True, "estimating for over a week; a two-tap reply is worth more "\
                     "than another estimate"
    if state == "Verified":
        return False, "a clear reading needs no confirmation"
    return False, "nothing worth saying or asking this week"
