"""This clinic's own details, kept out of the code.

Name, hours, address, and the words its callers say are all particular to one
clinic. Holding them in config/tenant.json is what lets a second clinic be a
second file rather than a second bot — so what these check is that the bot
really reads from the file, not that the file says what it says today.
"""

import json

import pytest

from bot_phone import CLINIC_INFO, KEYTERMS, reception_task
from tenant_config import CONFIG_PATH, clinic_info_text, keyterms


# --- the file ---------------------------------------------------------------


def test_the_file_parses():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert "clinic_info" in config
    assert "keyterms" in config


def test_the_clinic_details_cover_what_the_bot_may_answer():
    """営業時間・休診日・場所・アクセス are the four topics it is allowed to
    discuss; a missing one turns into "お答えいたしかねます" on a real call."""
    info = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["clinic_info"]

    for topic in ("クリニック名", "営業時間", "休診日", "場所", "アクセス"):
        assert topic in info, topic
        assert info[topic].strip(), topic


# --- rendering into the prompt ----------------------------------------------


def test_details_render_one_per_line():
    rendered = clinic_info_text(
        {"clinic_info": {"クリニック名": "みどり歯科", "休診日": "水曜"}}
    )

    assert rendered == "クリニック名:みどり歯科\n休診日:水曜\n"


def test_no_spaces_creep_into_the_rendering():
    """Cartesia reads a half-width space in a Japanese line as a long pause,
    and the bot quotes these values back verbatim."""
    assert " " not in CLINIC_INFO


def test_the_prompt_is_built_from_the_file():
    """The system instruction has to carry whatever the file says, or the bot
    answers from the wrong clinic's details."""
    for value in json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["clinic_info"].values():
        assert value in reception_task(offered=False), value


# --- the keyterms -----------------------------------------------------------


def test_keyterms_are_a_list_of_words():
    assert KEYTERMS
    assert all(isinstance(term, str) and term.strip() for term in KEYTERMS)


def test_the_clinic_name_is_boosted():
    """The one proper noun every caller says, and the one the 8 kHz line
    mangles most ("さくら歯科" comes back as "桜市科")."""
    name = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["clinic_info"]["クリニック名"]

    assert name in KEYTERMS


def test_keyterms_come_from_the_file():
    assert keyterms({"keyterms": ["歯医者", "予約"]}) == ["歯医者", "予約"]


@pytest.mark.parametrize("term", ["予約", "折り返し"])
def test_the_words_the_call_turns_on_are_boosted(term):
    """Mishearing either of these sends the conversation down the wrong branch."""
    assert term in KEYTERMS
