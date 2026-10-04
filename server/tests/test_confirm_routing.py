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

    assert flow.state == {"name": "佐藤", "phone_number": "08012345678"}
