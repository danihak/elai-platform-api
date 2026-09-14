"""
llm.py â€” Claude tool-use, with the output verified before anyone sees it.

Three deliberate choices.

**Raw HTTP, not the SDK.** The Argos build lost time to `anthropic==0.39.0`
breaking against httpx 0.28. The Messages API is a stable JSON contract; the SDK
is another version to pin. One fewer moving part.

**Groundedness is checked in code, not trusted.** Every number the model writes
must appear in the evidence the tools returned. If it does not, the answer is
rejected and the deterministic renderer serves instead. This is the mechanism
behind the evaluation answer: faithfulness is not a judgement call here, it is a
set membership test on numeric tokens.

**Failure is invisible to the user.** No API key, provider down, circuit open,
groundedness failed â€” all four produce the deterministic evidence chain. The
explain endpoint has no error state. A live demo cannot die on stage.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .tools import run_tool, tool_definitions

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = os.getenv("ELAI_LLM_MODEL", "claude-sonnet-4-6")
MAX_TOOL_ROUNDS = 5
TIMEOUT_S = float(os.getenv("ELAI_LLM_TIMEOUT", "25"))

SYSTEM = """You explain satellite crop observations to people who will act on them.

ABSOLUTE RULES

1. Every fact you state must come from a tool result in this conversation. You
   have no other knowledge of this farm. If a tool did not return it, you do not
   know it.
2. Never invent, estimate, round or interpolate a number. Use figures exactly as
   the tools returned them.
3. Never recommend a chemical, a product name, a dose or an application rate. If
   asked, say you cannot advise on that and that the field officer or the local
   KVK can.
4. Never tell a farmer what to do. Describe what was observed and let them
   decide. "Here is what we see, you decide."
5. Confidence is not decoration. If the state is Degraded, say the number is an
   estimate. If Blind, say plainly that the field was not seen and do not give a
   value at all.

METHOD

Call get_latest_observation first, always. Call get_observation_history when the
question is about what changed. Call get_cloud_state when the state is Degraded
or Blind, so you can say how long the field has been unseen. Call
get_pop_reference only when pointing at something the person can check in the
field themselves.

STYLE

Answer in three or four sentences. No headings, no bullet points, no preamble.
Name the specific field and a specific date. Prefer something the person can
verify cheaply and soon over something abstract.
"""

AUDIENCE_NOTE = {
    "farmer": (
        "You are speaking to a smallholder farmer who will hear this as a voice note. "
        "Short sentences. No index names, no percentages, no technical terms. Refer to "
        "the field by name and to time as 'since last week' or a plain date. End by "
        "inviting them to tell you what they can see."
    ),
    "lender": (
        "You are speaking to a credit officer. They need to know how much to trust this "
        "and what it means for repayment timing. Cite the observation date, the state and "
        "the rule version, because they may have to justify it to a branch manager."
    ),
    "insurer": (
        "You are speaking to a claims assessor. Be precise and chronological. Cite dates, "
        "states, cloud conditions and the fallback used. This may end up in a claims file."
    ),
    "exporter": (
        "You are speaking to a procurement manager. Frame the observation in terms of what "
        "it means for harvest timing and expected volume, and how confident that is."
    ),
}

REFUSED_PATTERNS = re.compile(
    r"\b(dose|dosage|dosing|spray|spraying|pesticide|insecticide|herbicide|fungicide|"
    r"urea|dap|npk|ml per|grams? per|kg per|litres? per|how much (should|do) i)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------


@dataclass
class Circuit:
    """Three consecutive failures opens it for 60 seconds.

    Without this, a provider outage turns every explain request into a 25-second
    wait before falling back. With it, the first three pay that cost and the rest
    fall back instantly.
    """

    failures: int = 0
    opened_at: float = 0.0
    threshold: int = 3
    cool_off_s: float = 60.0

    @property
    def is_open(self) -> bool:
        if self.failures < self.threshold:
            return False
        if time.time() - self.opened_at > self.cool_off_s:
            self.failures = 0
            return False
        return True

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.time()

    def record_success(self) -> None:
        self.failures = 0


CIRCUIT = Circuit()


# --------------------------------------------------------------------------
# Groundedness
# --------------------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# A number immediately followed by a unit is a CLAIM ABOUT DATA and must be
# grounded regardless of size. Without this, "yield is about 7 MT/ha" passed the
# check purely because 7 is a small integer â€” the exact class of invented figure
# a lender would act on.
_UNIT_BOUND = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent|mt/ha|mt |ha\b|hectare|tonnes?|kg|days?|weeks?|ndvi)",
    re.IGNORECASE,
)

# Bare small integers in ordinary prose ("two weeks ago" written as "2") are not
# data claims. Kept deliberately short.
_ALLOWED_BARE = {"0", "1", "2", "3"}


def _numeric_tokens(blob: Any) -> set:
    out = set()
    for m in _NUMBER.finditer(json.dumps(blob, default=str)):
        tok = m.group(0)
        out.add(tok)
        if tok.endswith(".0"):
            out.add(tok[:-2])
        # A figure the tools gave as 0.62 may legitimately be spoken as 62%.
        try:
            val = float(tok)
            if 0 < val < 1:
                out.add(str(round(val * 100)))
                out.add(f"{val * 100:.0f}")
            out.add(f"{val:.0f}")
            out.add(f"{val:.1f}")
        except ValueError:
            pass
    return out


def check_grounded(answer: str, evidence: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Every number in the answer must appear in what the tools returned.

    This is the automated half of the evaluation rubric. It does not judge whether
    an explanation is *useful* â€” an agronomist panel does that â€” but it makes an
    invented figure impossible to ship, which is the failure that actually hurts
    a farmer or a lender.
    """
    from_tools = _numeric_tokens(evidence)
    allowed = from_tools | _ALLOWED_BARE
    ungrounded = []

    # Anything attached to a unit is checked against the tools only.
    unit_bound = {m.group(1) for m in _UNIT_BOUND.finditer(answer)}
    for tok in unit_bound:
        if tok not in from_tools:
            ungrounded.append(tok)

    for m in _NUMBER.finditer(answer):
        tok = m.group(0)
        if tok in unit_bound or tok in allowed:
            continue
        ungrounded.append(tok)

    return (not ungrounded), sorted(set(ungrounded))


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------


@dataclass
class LlmResult:
    ok: bool
    answer: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    rounds: int = 0
    failure_reason: Optional[str] = None
    ungrounded_tokens: List[str] = field(default_factory=list)


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY")) and not CIRCUIT.is_open


def explain_with_claude(farm_id: str, question: str, audience: str) -> LlmResult:
    """Run the tool-use loop. Never raises â€” failure returns ok=False."""

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return LlmResult(ok=False, failure_reason="no_api_key")
    if CIRCUIT.is_open:
        return LlmResult(ok=False, failure_reason="circuit_open")

    # Guardrail before a token is spent. Cheaper, and not negotiable by the model.
    if REFUSED_PATTERNS.search(question):
        return LlmResult(ok=False, failure_reason="refused_topic")

    messages: List[Dict[str, Any]] = [{
        "role": "user",
        "content": (
            f"{AUDIENCE_NOTE.get(audience, AUDIENCE_NOTE['lender'])}\n\n"
            f"Farm: {farm_id}\nQuestion: {question}"
        ),
    }]
    tool_calls: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    headers = {
        "x-api-key": key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }

    try:
        with httpx.Client(timeout=TIMEOUT_S) as client:
            for rnd in range(1, MAX_TOOL_ROUNDS + 1):
                resp = client.post(API_URL, headers=headers, json={
                    "model": DEFAULT_MODEL,
                    "max_tokens": 1000,
                    "system": SYSTEM,
                    "tools": tool_definitions(),
                    "messages": messages,
                })
                if resp.status_code != 200:
                    CIRCUIT.record_failure()
                    return LlmResult(ok=False, failure_reason=f"http_{resp.status_code}:{resp.text[:400]}", rounds=rnd)

                data = resp.json()
                blocks = data.get("content", [])
                messages.append({"role": "assistant", "content": blocks})

                if data.get("stop_reason") != "tool_use":
                    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                    grounded, bad = check_grounded(text, evidence)
                    CIRCUIT.record_success()
                    if not grounded:
                        return LlmResult(
                            ok=False, failure_reason="ungrounded", ungrounded_tokens=bad,
                            tool_calls=tool_calls, evidence=evidence,
                            model=DEFAULT_MODEL, rounds=rnd,
                        )
                    return LlmResult(
                        ok=True, answer=text, tool_calls=tool_calls, evidence=evidence,
                        model=DEFAULT_MODEL, rounds=rnd,
                    )

                results = []
                for b in blocks:
                    if b.get("type") != "tool_use":
                        continue
                    out = run_tool(b["name"], b.get("input", {}))
                    tool_calls.append({"tool": b["name"], "input": b.get("input", {})})
                    evidence.append({"tool": b["name"], "result": out})
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": b["id"],
                        "content": json.dumps(out, default=str),
                    })
                messages.append({"role": "user", "content": results})

        CIRCUIT.record_failure()
        return LlmResult(ok=False, failure_reason="max_rounds", tool_calls=tool_calls,
                         evidence=evidence, rounds=MAX_TOOL_ROUNDS)

    except Exception as exc:  # noqa: BLE001 â€” the endpoint must never raise
        CIRCUIT.record_failure()
        return LlmResult(ok=False, failure_reason=f"exception:{type(exc).__name__}")


