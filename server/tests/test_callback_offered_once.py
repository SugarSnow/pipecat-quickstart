"""The callback is offered once a call, and "once" is decided in code.

It used to be a rule in the system instruction — "一度だけ", with the line
itself quoted two lines above it — and the model broke it about one turn in
five. Measured over eight runs apiece either side of an unrelated change, the
scenario passed 6/8 and 7/8.

The instruction can only ask. Moving the bot to a node whose text never
mentions the line takes the wording away, which is what these pin.
"""

from bot_phone import (
    CALLBACK_OFFER_LINE,
    CLOSING_LINE,
    GREETING_LINE,
    OUT_OF_SCOPE_LINE,
    UNKNOWN_LINE,
    is_callback_offer,
    reception_node,
    reception_offered_node,
    reception_task,
)


# --- recognising the offer ---------------------------------------------------


def test_the_whole_offer_counts():
    assert is_callback_offer(CALLBACK_OFFER_LINE)
    assert is_callback_offer(OUT_OF_SCOPE_LINE + CALLBACK_OFFER_LINE)


def test_an_offer_cut_off_before_the_question_does_not_count():
    """Interrupted speech reaches the context truncated, and a caller who never
    heard the question was never really offered anything."""
    cut = "申し訳ございませんが、営業時間、休診日、場所のご案内以外はお答えいたしかねます。担当者から折り返し"

    assert not is_callback_offer(cut)


def test_the_bots_other_fixed_lines_do_not_count():
    """A false match here would swap the instructions out mid-read-back."""
    for line in (GREETING_LINE, CLOSING_LINE, OUT_OF_SCOPE_LINE, UNKNOWN_LINE):
        assert not is_callback_offer(line), line


# --- what each form of the node is allowed to say ---------------------------


def test_the_offer_is_available_before_it_is_made():
    task = reception_task(offered=False)

    assert CALLBACK_OFFER_LINE in task


def test_the_refusals_stop_leading_into_the_offer():
    """The clause that actually drove the bot to offer again is what goes.

    Each refusal used to be followed by "続けて「<the offer>」とそのまま伝えて
    ください". Removing that is the change; what is left of the line lives in
    one sentence about repeating it, so the bot can still say it back to a
    caller who asks what was said.
    """
    before = reception_task(offered=False)
    after = reception_task(offered=True)

    assert before.count(CALLBACK_OFFER_LINE) == 2, "once per refusal"
    assert after.count(CALLBACK_OFFER_LINE) == 1, "only the repeat rule"
    assert "続けて" in before
    assert "続けて" not in after


def test_the_offer_can_still_be_repeated_on_request():
    """Taking the wording away entirely cost the bot its 言い直し: asked what
    had been said, it read back the refusal and dropped the offer."""
    after = reception_task(offered=True)

    assert "聞き返され" in after
    assert CALLBACK_OFFER_LINE in after


def test_the_offered_form_never_volunteers_it():
    after = reception_task(offered=True)

    assert "こちらから持ちかけてはいけません" in after


def test_both_forms_still_refuse_the_same_way():
    """Only the offer is taken away; the refusals are the scenarios' business."""
    for offered in (False, True):
        task = reception_task(offered=offered)
        assert OUT_OF_SCOPE_LINE in task
        assert UNKNOWN_LINE in task
        # The caller can still ask for a callback themselves.
        assert "start_callback" in task


# --- the node the bot moves to ----------------------------------------------


def test_the_offered_node_says_nothing_on_arrival():
    """The bot has just finished speaking the offer; it must not start again."""
    node = reception_offered_node()

    assert node["respond_immediately"] is False
    assert "pre_actions" not in node


def test_the_offered_node_does_not_greet_again():
    node = reception_offered_node()

    assert GREETING_LINE not in node["role_message"]


def test_the_offered_node_keeps_start_callback():
    """A caller who says yes to the offer still has to be able to get through."""
    before = {f.__name__ for f in reception_node()["functions"]}

    assert {f.__name__ for f in reception_offered_node()["functions"]} == before
