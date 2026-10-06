"""Where the callback flow goes once a number has been heard.

The decision is made in code rather than by the LLM, which called three digits a
complete number and, when told to wait, worded the wait itself and talked over
the caller. Three outcomes, one per state the number can be in.
"""

from types import SimpleNamespace

import pytest

from bot_phone import _confirm


def _flow(**state):
    """A stand-in for the flow manager: _confirm only reads and writes state."""
    return SimpleNamespace(state=dict(state))


@pytest.mark.parametrize(
    ("phone_number", "why"),
    [
        ("080", "読み上げ始めたばかり"),
        ("0801234", "携帯で7桁、あと4桁"),
        ("03123456", "固定電話で8桁、あと2桁"),
        ("", "まだ数字を聞き取れていない"),
    ],
)
def test_a_number_still_coming_waits(phone_number, why):
    node = _confirm(_flow(name="佐藤"), phone_number=phone_number)

    assert node["name"] == "number_fragment", why
    assert node["pre_actions"][0]["text"] == "はい。"
    assert node["respond_immediately"] is False


@pytest.mark.parametrize(
    ("phone_number", "why"),
    [
        ("0801234567", "携帯なのに10桁"),
        ("031234567", "固定電話なのに9桁"),
        ("080123456789", "携帯なのに12桁"),
    ],
)
def test_a_number_that_came_out_wrong_is_asked_for_again(phone_number, why):
    node = _confirm(_flow(name="佐藤"), phone_number=phone_number)

    assert node["name"] == "number_retry", why
    assert "もう一度最初から" in node["pre_actions"][0]["text"]
    assert node["respond_immediately"] is False


@pytest.mark.parametrize("phone_number", ["08012345678", "0312345678"])
def test_a_complete_number_is_read_back(phone_number):
    node = _confirm(_flow(name="佐藤祐希"), phone_number=phone_number)

    assert node["name"] == "callback_confirm"
    assert node["pre_actions"][0]["text"].startswith("佐藤祐希様、お電話番号は")


def test_what_was_heard_is_remembered_across_the_pieces():
    """Each piece replaces the number; the name heard earlier stays."""
    flow = _flow(name="佐藤")

    _confirm(flow, phone_number="080")
    _confirm(flow, phone_number="08012345678")

    # The two the caller gave, rather than the whole dict: the flow keeps other
    # bookkeeping in there too, and this test is about the number and the name.
    assert flow.state["name"] == "佐藤"
    assert flow.state["phone_number"] == "08012345678"


# --- a number read out in pieces --------------------------------------------
#
# The eval cannot check this: its only handle on a turn is the arguments the
# model passed, and the whole point is that those vary — sometimes the new
# piece, sometimes everything so far. What matters is where the flow ends up.


def _read_out(pieces: list[str], **state) -> str:
    """Feed *pieces* to _confirm one at a time; return the number it assembled."""
    flow = _flow(name="佐藤", **state)
    for piece in pieces:
        _confirm(flow, phone_number=piece)
    return flow.state["phone_number"]


def test_pieces_accumulate():
    """The model hands over each new piece as it is heard."""
    assert _read_out(["080", "1234", "5678"]) == "08012345678"


def test_a_repeated_whole_number_does_not_double_up():
    """The same caller, the same digits, the other thing the model does."""
    assert _read_out(["080", "0801234", "08012345678"]) == "08012345678"


def test_the_two_habits_mix():
    """Whichever it does on any given turn, the number comes out the same."""
    assert _read_out(["080", "0801234", "5678"]) == "08012345678"


def test_a_rejected_number_is_not_merged_into_the_next_one():
    """A wrong digit count clears the slate, so the retry starts from nothing."""
    flow = _flow(name="佐藤")
    # Ten digits behind a mobile prefix: wrong, and asked for again.
    assert _confirm(flow, phone_number="0801234567")["name"] == "number_retry"
    assert flow.state["phone_number"] == ""

    _confirm(flow, phone_number="080")
    _confirm(flow, phone_number="12345678")

    assert flow.state["phone_number"] == "08012345678"
