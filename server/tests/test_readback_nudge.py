"""Silence after the read-back gets a nudge, once.

A phone test sat through 33 seconds of it: the bot had read the details back
and was waiting for a yes, the caller had not realised a reply was wanted, and
nothing in the bot would make it ask.

The nudge itself fires from an idle-timer handler inside run_bot, which needs a
live pipeline — evals/readback_nudge_audio.yaml covers that end. What is
checked here is the wording, the wait, and the bookkeeping that keeps it to one
nudge per read-back.
"""

from types import SimpleNamespace

from bot_phone import (
    CLOSING_SILENCE_SECS,
    READBACK_NUDGE_LINE,
    READBACK_SILENCE_SECS,
    _confirm,
    is_callback_offer,
    is_closing_utterance,
)


def _flow(**state):
    return SimpleNamespace(state=dict(state))


# --- the line ---------------------------------------------------------------


def test_the_nudge_is_a_short_question():
    assert READBACK_NUDGE_LINE == "よろしいでしょうか。"


def test_the_nudge_does_not_trip_the_other_detectors():
    """It is spoken as a TTSSpeakFrame, so it comes back round as an assistant
    turn and passes under both of the handlers that watch for fixed lines."""
    assert not is_closing_utterance(READBACK_NUDGE_LINE)
    assert not is_callback_offer(READBACK_NUDGE_LINE)


# --- the wait ---------------------------------------------------------------


def test_the_wait_leaves_room_to_think():
    """Short enough that the line does not feel dead, long enough to read a
    number back off a screen before answering."""
    assert 5.0 <= READBACK_SILENCE_SECS <= 15.0


def test_the_closing_hangs_up_sooner_than_the_nudge_waits():
    """Both run off the same timer. If the nudge's wait were the shorter of the
    two it would be the one to fire after the closing line."""
    assert CLOSING_SILENCE_SECS < READBACK_SILENCE_SECS


# --- one nudge per read-back ------------------------------------------------


def test_reaching_the_read_back_arms_a_nudge():
    flow = _flow(name="佐藤")

    _confirm(flow, phone_number="08012345678")

    assert flow.state["readback_nudged"] is False


def test_a_second_read_back_arms_another():
    """A corrected number is read back again, and the caller can miss their cue
    the second time just as easily."""
    flow = _flow(name="佐藤")
    _confirm(flow, phone_number="08012345678")
    flow.state["readback_nudged"] = True  # as the handler would leave it

    # 違うと言われ、聞き直して、もう一度復唱する
    flow.state["phone_number"] = ""
    _confirm(flow, phone_number="09087654321")

    assert flow.state["readback_nudged"] is False


def test_a_number_still_coming_does_not_arm_one():
    """The bot is not waiting on an answer yet — it is waiting on more digits."""
    flow = _flow(name="佐藤")

    node = _confirm(flow, phone_number="080")

    assert node["name"] == "number_fragment"
    assert "readback_nudged" not in flow.state


def test_a_number_sent_back_does_not_arm_one():
    flow = _flow(name="佐藤")

    node = _confirm(flow, phone_number="0801234567")

    assert node["name"] == "number_retry"
    assert "readback_nudged" not in flow.state
