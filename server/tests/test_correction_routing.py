"""Getting out of a wrong read-back, and the name the caller keeps correcting.

Both come from call tests. The caller was read back a wrong name and a wrong
number; when they said the name was wrong while the bot was re-taking the
number, the node they were in had no function for a name, so the model called
the one it had — with the number that had just been turned down — and the bot
read the same wrong number out again.

The name is the other half. It arrives in pieces, it arrives again when the
caller corrects it, and nothing in the value says which — so an attempt to join
the pieces turned each correction into a longer name (佐藤小林 →
佐藤小林本木 → 佐藤小林本木も汽) until the caller gave up. A name is replaced,
and the read-back is what catches a piece the model left out.

Everything here is decided in code, so it is checked here rather than in an
eval: the model only has to pass on what it heard.
"""

import asyncio
from types import SimpleNamespace

import pytest

import bot_phone
from bot_phone import (
    _confirm,
    confirm_name,
    callback_collect_node,
    correct_name,
    correct_phone_number,
    name_correction_node,
    number_retry_node,
    phone_correction_node,
    submit_name,
)


def _flow(node="callback_collect", **state):
    """A stand-in for the flow manager: these functions read state and the node."""
    return SimpleNamespace(state=dict(state), current_node=node)


def _names(node):
    return [f.__name__ for f in node["functions"]]


# --- the name a caller gives, and gives again ------------------------------


def test_the_name_is_taken_as_the_model_passes_it():
    flow = _flow()

    asyncio.run(submit_name(flow, "佐藤祐希"))

    assert flow.state["name"] == "佐藤祐希"


def test_submitting_a_name_stays_in_the_node():
    """Nothing to transition to: the number has still to be asked for."""
    result, next_node = asyncio.run(submit_name(_flow(), "佐藤"))

    assert next_node is None
    assert result == {"name": "佐藤"}


def test_a_name_given_again_while_collecting_replaces_the_first():
    """The caller's own correction, mid-collection.

    「佐藤」「あ、間違えました。小林」 — joining the two made it 「佐藤小林」,
    and the bot read that back (call test 2026-10-08 11:55).
    """
    flow = _flow()

    asyncio.run(submit_name(flow, "佐藤"))
    asyncio.run(submit_name(flow, "小林"))

    assert flow.state["name"] == "小林"


def test_a_name_submitted_after_the_read_back_is_a_correction():
    """The model reaches for submit_name instead of correct_name.

    It answered 「名前が違ってて、小林本木です」 by calling submit_name, so the
    correction never went through a correction node — and with joining, each
    attempt to fix the name made it longer: 佐藤小林 → 佐藤小林本木 →
    佐藤小林本木も汽. The caller hung up.
    """
    flow = _flow(node="callback_confirm", name="佐藤小林", phone_number="07015162121")

    _, next_node = asyncio.run(submit_name(flow, "小林本木"))

    assert flow.state["name"] == "小林本木"
    assert next_node["name"] == "callback_confirm"
    assert next_node["pre_actions"][0]["text"].startswith("小林本木様、")


def test_repeated_attempts_at_one_name_do_not_pile_up():
    """Whichever function the model uses, the name stays the length of a name."""
    flow = _flow(node="callback_confirm", name="小林本木", phone_number="07011111212")

    for heard in ["林本木", "小林元木", "小早市本木"]:
        asyncio.run(submit_name(flow, heard))
        asyncio.run(confirm_name(flow, heard))

    assert flow.state["name"] == "小早市本木"


def test_the_name_in_state_is_replaced_by_the_read_back_call():
    """confirm_callback carries a name of its own, and it is the one that wins."""
    flow = _flow()
    asyncio.run(submit_name(flow, "小林"))

    node = _confirm(flow, name="小林本木", phone_number="08012345678")

    assert flow.state["name"] == "小林本木"
    assert node["pre_actions"][0]["text"].startswith("小林本木様、")


def test_a_re_stated_name_replaces_the_one_held():
    """Pieces are joined while collecting; a correction is the whole name."""
    flow = _flow(name="林本木", phone_number="08012345678")

    asyncio.run(confirm_name(flow, "小林元木"))

    assert flow.state["name"] == "小林元木"


def test_collecting_can_take_a_name_on_its_own():
    assert "submit_name" in _names(callback_collect_node())


# --- a value the caller has turned down --------------------------------------


def test_a_rejected_number_is_not_read_back_again():
    """The call test: the number turned down came straight back as a read-back.

    Nothing new was heard, so the call stays where it is rather than being sent
    anywhere — which is also what keeps a correction called in the same turn
    from being overridden.
    """
    flow = _flow(name="本木", phone_number="07011111152")
    asyncio.run(correct_phone_number(flow))

    assert _confirm(flow, phone_number="07011111152") is None
    assert flow.state["phone_number"] == ""


def test_a_rejected_number_is_recognised_however_it_is_written():
    flow = _flow(name="本木", phone_number="07011111152")
    asyncio.run(correct_phone_number(flow))

    assert _confirm(flow, phone_number="070-1111-1152") is None


def test_a_different_number_is_accepted_after_a_rejection():
    flow = _flow(name="本木", phone_number="07011111152")
    asyncio.run(correct_phone_number(flow))

    node = _confirm(flow, phone_number="07015151212")

    assert node["name"] == "callback_confirm"
    assert flow.state["phone_number"] == "07015151212"


def test_a_rejected_name_is_not_read_back_again():
    flow = _flow(name="本木", phone_number="08012345678")
    asyncio.run(correct_name(flow))

    assert _confirm(flow, name="本木") is None
    assert flow.state["name"] == ""


def test_a_different_name_is_accepted_after_a_rejection():
    flow = _flow(name="本木", phone_number="08012345678")
    asyncio.run(correct_name(flow))

    node = _confirm(flow, name="小林本木")

    assert node["name"] == "callback_confirm"
    assert flow.state["name"] == "小林本木"


def test_a_rejection_is_forgotten_once_the_detail_is_fixed():
    """Only the latest rejection stands: a caller may correct back again."""
    flow = _flow(name="佐藤", phone_number="08012345678")
    asyncio.run(correct_phone_number(flow))
    _confirm(flow, phone_number="0312345678")
    assert flow.state["rejected_phone_number"] == ""

    # "no, the first one was right after all"
    asyncio.run(correct_phone_number(flow))

    assert _confirm(flow, phone_number="08012345678")["name"] == "callback_confirm"


def test_two_corrections_in_one_turn_land_on_the_name():
    """Told both details were wrong, the model called two functions at once.

    It asked to re-take the name and, in the same turn, passed the number that
    had just been turned down. Flows keeps one pending transition, so the
    second call decided where the call went: into the number's correction,
    which carries no function for a name — and the name the caller gave next
    had nowhere to go (eval run 2026-10-06 11:24, 1/3).
    """
    flow = _flow(name="本木", phone_number="07011111152")
    asyncio.run(correct_phone_number(flow))

    # The model's two calls, in the order the eval log recorded them.
    _, after_correction = asyncio.run(correct_name(flow))
    stale = _confirm(flow, phone_number="07011111152")

    assert after_correction["name"] == "name_correction"
    assert stale is None


@pytest.mark.parametrize("offered", ["", "   "])
def test_an_empty_value_goes_nowhere(offered):
    """The model's other way of acknowledging the detail it was not asked for.

    Rather than leave confirm_phone_number alone it called it with nothing in
    it, and an empty number used to read as "the name was fixed, now ask for
    the number" — which sent the call to the number's re-ask and undid the
    correction requested in the same turn (eval run 2026-10-06 11:28, 2/3).
    """
    flow = _flow(name="本木", phone_number="07011111152")
    asyncio.run(correct_phone_number(flow))
    _, after_correction = asyncio.run(correct_name(flow))

    assert _confirm(flow, phone_number=offered) is None
    assert after_correction["name"] == "name_correction"


def test_correcting_the_name_does_not_touch_the_number():
    flow = _flow(name="本木", phone_number="08012345678")
    asyncio.run(correct_name(flow))

    assert flow.state["phone_number"] == "08012345678"


# --- getting out of the wrong correction -------------------------------------


@pytest.mark.parametrize(
    ("node", "own", "other"),
    [
        (phone_correction_node(), "confirm_phone_number", "correct_name"),
        (name_correction_node(), "confirm_name", "correct_phone_number"),
        (number_retry_node(), "confirm_phone_number", "correct_name"),
    ],
)
def test_a_correction_can_hand_over_to_the_other_detail(node, own, other):
    """Being re-asked for one detail is when people mention the other one."""
    assert _names(node) == [own, other]


ALL_TOOLS = [
    "start_callback",
    "submit_name",
    "confirm_callback",
    "confirm_name",
    "confirm_phone_number",
    "correct_name",
    "correct_phone_number",
    "record_callback",
]


@pytest.mark.parametrize(
    "build",
    [
        bot_phone.reception_node,
        bot_phone.reception_offered_node,
        bot_phone.callback_collect_node,
        bot_phone.name_correction_node,
        bot_phone.phone_correction_node,
        bot_phone.number_fragment_node,
        bot_phone.number_retry_node,
        bot_phone.closing_node,
    ],
)
def test_a_node_only_names_functions_it_carries(build):
    """The instructions and the function list have to agree.

    The procedures are shared between nodes, so a sentence added for one node
    reaches the others: the hand-off instruction went into the text every
    re-asking node shares, and number_fragment — which has no correction
    function — was left being told to call one.
    """
    node = build()
    named = {tool for tool in ALL_TOOLS if tool in node["role_message"]}

    assert named <= set(_names(node))


def test_the_name_is_read_back_with_the_number_still_held():
    """Fixing the name alone keeps the number that was never questioned."""
    flow = _flow(name="本木", phone_number="08012345678")
    asyncio.run(correct_name(flow))

    node = _confirm(flow, name="小林本木")

    assert node["pre_actions"][0]["text"].startswith("小林本木様、お電話番号は")
    assert "ハチ" in node["pre_actions"][0]["text"]


def test_fixing_the_name_after_the_number_was_cleared_asks_for_the_number():
    """Both were wrong: with no number to read back, ask for it."""
    flow = _flow(name="本木", phone_number="07011111152")
    asyncio.run(correct_phone_number(flow))
    asyncio.run(correct_name(flow))

    node = _confirm(flow, name="小林本木")

    assert node["name"] == "number_retry"
    assert flow.state["name"] == "小林本木"
