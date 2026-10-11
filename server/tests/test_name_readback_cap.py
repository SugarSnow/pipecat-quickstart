"""The name is read back twice, and then it becomes a person's job.

From the call that ended the worst: the bot had the name written correctly all
along — it said so when asked — but it could not pronounce it, because the
reading is gone the moment the speech-to-text settles on kanji. 「本木」 reads
もとき or ほんぎ and nothing downstream can tell which. The caller said "名前が
違います" five times and hung up on 「一旦大丈夫です。失礼します」.

Reading it back a third time cannot work, so it is not attempted. What the call
can still settle is the number; the name goes to whoever makes the callback,
with a mark on the record so they know to ask. The caller is told an AI is
answering, which is also said in the greeting.

The count is kept in code. The model has no tally across turns, and on that
call it simply kept asking.
"""

import asyncio

import pytest
from conftest import make_flow

from bot_phone import (
    MAX_NAME_READBACKS,
    _confirm,
    NAME_DEFERRED_LINE,
    SPELL_NAME_LINE,
    confirm_name,
    confirm_number_only,
    correct_name,
    correct_phone_number,
    name_deferred_node,
    phone_correction_node,
    name_spellout_node,
    record_callback,
    spell_name,
    submit_name,
)
from callback_store import read_callback_requests


def _flow(node="callback_confirm", **state):
    return make_flow(node, name="小林本木", phone_number="08011112222", **state)


def _names(node):
    return [f.__name__ for f in node["functions"]]


# --- the cap ----------------------------------------------------------------


def test_the_first_correction_asks_again():
    flow = _flow()

    _, node = asyncio.run(correct_name(flow))

    assert node["name"] == "name_correction"
    assert NAME_DEFERRED_LINE not in flow.worker.spoken


def test_the_second_correction_hands_the_name_over():
    flow = _flow()

    asyncio.run(correct_name(flow))
    asyncio.run(confirm_name(flow, "林本木"))
    _, node = asyncio.run(correct_name(flow))

    assert node["name"] == "name_deferred"
    assert flow.worker.spoken[-1] == NAME_DEFERRED_LINE


def test_the_name_is_read_back_exactly_twice():
    """The whole exchange, counted from the first read-back to the hand-over."""
    flow = make_flow("callback_collect")

    asyncio.run(submit_name(flow, "小林本木"))
    asyncio.run(_confirm(flow, phone_number="08011112222"))  # read-back 1
    asyncio.run(correct_name(flow))
    asyncio.run(confirm_name(flow, "林本木"))  # read-back 2
    asyncio.run(correct_name(flow))  # no third attempt

    read_backs = [line for line in flow.worker.spoken if "様、お電話番号は" in line]
    assert len(read_backs) == MAX_NAME_READBACKS
    assert flow.worker.spoken.count(NAME_DEFERRED_LINE) == 1


def test_the_route_the_model_actually_takes_is_capped_too():
    """submit_name after a read-back is a correction, and counts as one."""
    flow = _flow()

    asyncio.run(submit_name(flow, "林本木"))
    _, node = asyncio.run(submit_name(flow, "小林元木"))

    assert node["name"] == "name_deferred"
    assert flow.worker.spoken[-1] == NAME_DEFERRED_LINE


def test_the_two_routes_share_one_count():
    """A caller correcting twice gets handed over, whichever way the model goes."""
    flow = _flow()

    asyncio.run(correct_name(flow))
    _, node = asyncio.run(submit_name(flow, "小林元木"))

    assert node["name"] == "name_deferred"


def test_the_latest_attempt_is_kept():
    """Unconfirmed is not unknown: the closest anyone got is worth keeping."""
    flow = _flow()

    asyncio.run(submit_name(flow, "林本木"))
    asyncio.run(submit_name(flow, "小林元木"))

    assert flow.state["name"] == "小林元木"


def test_the_number_has_no_such_cap():
    """A digit wrong means nobody can call back, so the number is asked for
    until it adds up. The digit count is what bounds that instead."""
    flow = _flow()

    for _ in range(4):
        _, node = asyncio.run(correct_phone_number(flow))

    assert node["name"] == phone_correction_node()["name"]


# --- what the caller is offered next ----------------------------------------


def test_the_handover_offers_the_number_or_spelling_it_out():
    assert _names(name_deferred_node()) == ["confirm_number_only", "spell_name"]


def test_the_handover_waits_rather_than_talking():
    """The line has just been spoken; nothing more until the caller answers."""
    assert name_deferred_node()["respond_immediately"] is False
    assert "pre_actions" not in name_deferred_node()


def test_agreeing_reads_back_the_number_alone():
    flow = _flow(node="name_deferred")

    _, node = asyncio.run(confirm_number_only(flow))

    assert node["name"] == "callback_confirm"
    assert flow.worker.spoken[-1].startswith("お電話番号は")
    assert "様" not in flow.worker.spoken[-1]


def test_agreeing_marks_the_name_unconfirmed():
    flow = _flow(node="name_deferred")

    asyncio.run(confirm_number_only(flow))

    assert flow.state["name_unconfirmed"] is True


def test_the_record_carries_the_mark(tmp_path, monkeypatch):
    monkeypatch.setenv("CALLBACK_LOG_PATH", str(tmp_path / "callbacks.jsonl"))
    flow = _flow(node="callback_confirm", name_unconfirmed=True)

    asyncio.run(record_callback(flow))

    stored = read_callback_requests(tmp_path / "callbacks.jsonl")
    assert stored[0]["name"] == "小林本木"
    assert stored[0]["name_unconfirmed"] is True


def test_a_confirmed_name_is_not_marked(tmp_path, monkeypatch):
    monkeypatch.setenv("CALLBACK_LOG_PATH", str(tmp_path / "callbacks.jsonl"))

    asyncio.run(record_callback(_flow()))

    assert read_callback_requests(tmp_path / "callbacks.jsonl")[0]["name_unconfirmed"] is False


# --- spelling it out, for the caller who would rather settle it now ----------


def test_spelling_it_out_starts_from_nothing():
    """The name just turned down must not be merged into what is spelled."""
    flow = _flow(node="name_deferred")

    _, node = asyncio.run(spell_name(flow))

    assert node["name"] == "name_spellout"
    assert flow.state["name"] == ""
    assert flow.worker.spoken[-1] == SPELL_NAME_LINE


def test_the_spelling_node_asks_for_kana_as_heard():
    """Isolated syllables are the one input the speech-to-text leaves as kana."""
    node = name_spellout_node()

    assert _names(node) == ["confirm_name"]
    assert "漢字に直してはいけません" in node["role_message"]
    assert node["respond_immediately"] is False


@pytest.mark.parametrize("build", [name_deferred_node, name_spellout_node])
def test_the_new_nodes_only_name_functions_they_carry(build):
    node = build()
    mentioned = {
        tool
        for tool in ("confirm_number_only", "spell_name", "confirm_name", "record_callback")
        if tool in node["role_message"]
    }
    assert mentioned <= set(_names(node))
