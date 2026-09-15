"""KisanVoice — the farmer surface, and the loop it closes."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.kisanvoice import (Channel, FarmerReply, Language,
                                     NEVER_IN_A_FARMER_MESSAGE, TWO_TAP_PROMPTS,
                                     compose, ingest_reply, should_message)


def msg(**kw):
    base = dict(farm_id="TS-MZ-0044", farmer_name="K. Ramulu", state="Degraded",
                language=Language.TELUGU, days_unseen=11, last_seen="22 June")
    base.update(kw)
    return compose(**base)


# ---------------------------------------------------------- what a farmer sees

def test_no_jargon_reaches_a_farmer_in_any_language():
    """Not because farmers cannot handle it — because none of it helps them
    decide anything."""
    for lang in Language:
        for state in ("Verified", "Degraded", "Blind"):
            m = msg(language=lang, state=state)
            assert m.leaks_jargon() == [], f"{lang} {state}: {m.leaks_jargon()}"


def test_every_message_asks_for_something_back():
    """A message that tells a farmer something and asks nothing is a broadcast,
    and broadcasts do not close loops."""
    for lang in Language:
        m = msg(language=lang)
        from app.services.kisanvoice import CLOSING
        assert CLOSING[lang.value] in m.body
        assert m.prompt is not None


def test_a_blind_message_says_we_cannot_tell_you_anything():
    """The hardest sentence to send and the most important one."""
    m = msg(state="Blind", language=Language.ENGLISH)
    assert "cannot tell you anything" in m.body
    assert "22 June" in m.body


def test_a_degraded_message_says_guide_not_fact():
    m = msg(state="Degraded", language=Language.ENGLISH)
    assert "guide, not a fact" in m.body


def test_messages_are_short_enough_to_be_a_voice_note():
    for lang in Language:
        for state in ("Verified", "Degraded", "Blind"):
            m = msg(language=lang, state=state)
            assert len(m.body.split()) < 60


# ------------------------------------------------------------ what comes back

def test_a_farmer_reply_never_upgrades_the_satellite_state():
    """The rule that matters.

    A farmer standing in the field is ground truth about the FIELD. If the
    satellite could not see it, it still could not see it. Upgrading the
    observation on the strength of a tap would launder one measurement into
    another, which is the exact move this system exists to refuse.
    """
    out = ingest_reply(
        FarmerReply(farm_id="TS-MZ-0044", prompt_id="canopy_check",
                    option_id="good", received="2026-09-15",
                    language=Language.TELUGU),
        current_state="Blind")
    assert out["accepted"]
    assert out["satellite_state_unchanged"] == "Blind"
    assert "not about what the satellite saw" in out["note"]


def test_a_reply_becomes_a_labelled_training_row():
    out = ingest_reply(
        FarmerReply(farm_id="TS-MZ-0044", prompt_id="actual_yield",
                    option_id="less", received="2026-10-20",
                    language=Language.HINDI),
        current_state="Degraded")
    assert out["label"]["label"] == "actual_yield_direction"
    assert out["label"]["source"] == "farmer_reply"


def test_a_photo_with_gps_is_worth_more_than_a_tap():
    plain = ingest_reply(
        FarmerReply("TS-MZ-0044", "canopy_check", "patchy", "2026-09-15",
                    Language.TELUGU), "Degraded")
    with_photo = ingest_reply(
        FarmerReply("TS-MZ-0044", "canopy_check", "patchy", "2026-09-15",
                    Language.TELUGU, photo_url="s3://x.jpg",
                    gps={"lat": 18.51, "lon": 79.09}), "Degraded")
    assert with_photo["label"]["confidence"] > plain["label"]["confidence"]
    assert with_photo["label"]["verifiable"] is True


def test_disagreement_is_surfaced_not_averaged_away():
    """A farmer reporting a problem the model missed is the single most
    valuable message in the system."""
    out = ingest_reply(
        FarmerReply("TS-MZ-0044", "canopy_check", "bad", "2026-09-15",
                    Language.TELUGU), "Verified")
    assert out["farmer_disagrees_with_model"] is True
    assert "field verification" in out["action"]


def test_an_invalid_option_is_refused():
    out = ingest_reply(
        FarmerReply("TS-MZ-0044", "canopy_check", "maybe", "2026-09-15",
                    Language.TELUGU), "Degraded")
    assert not out["accepted"]


# ------------------------------------------------------------ who gets a message

def test_a_blind_field_always_earns_a_message():
    send, why = should_message("Blind", days_unseen=30, last_messaged_days=7)
    assert send and "should know" in why


def test_a_clear_reading_earns_no_message():
    send, why = should_message("Verified", days_unseen=1, last_messaged_days=30)
    assert not send


def test_nobody_is_messaged_twice_in_five_days():
    """Messaging every farmer every week trains them to ignore it."""
    send, why = should_message("Blind", days_unseen=30, last_messaged_days=2)
    assert not send and "five days" in why


# ------------------------------------------------------------ the prompts

def test_every_prompt_writes_a_label_and_justifies_itself():
    """A question that produces no usable label does not earn a farmer's time."""
    for p in TWO_TAP_PROMPTS.values():
        assert p.writes
        assert len(p.why) > 40
        assert len(p.options) <= 3, "more than three options is not two taps"
        for lang in ("te", "hi", "en"):
            assert p.question[lang]
            assert all(o[lang] for o in p.options)
