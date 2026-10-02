"""The phone bot's own utterances survive Cartesia's Japanese word-timestamp path.

``test_cartesia_ja_space_corruption.py`` shows that a half-width space inside a
Japanese sentence corrupts the stored assistant message. The bot avoids that by
never producing one: ``CLINIC_INFO`` is written without spaces, and
``SYSTEM_INSTRUCTION`` forbids them mid-sentence.

These tests hold both ends of that: the constants really are space-free, and the
sentences the bot is expected to speak round-trip under every Cartesia message
boundary.
"""

import pytest
from cartesia_history_sim import failing_chunkings, history_for, isolates_final_mark

from bot_phone import CLINIC_INFO, SYSTEM_INSTRUCTION

# Answers the bot gives from CLINIC_INFO, phrased as the prompt asks for them
# (one or two spoken sentences, no symbols).
CLINIC_ANSWERS = [
    "診療時間は平日が午前9時から午後6時まで、土曜が午前9時から午後1時までです。",
    "休診日は日曜と祝日です。",
    "場所は東京都調布市小島町1丁目2番3号、さくらビル2階です。",
    "京王線、調布駅中央口から徒歩5分です。",
]

# The callback flow: the name and the phone number read back one digit at a time.
CALLBACK_CONFIRMATIONS = [
    "お名前をお願いできますか。",
    "お電話番号をお願いできますか。",
    "佐藤祐希様、お電話番号はゼロ、ハチ、ゼロ、イチ、ニ、サン、ヨン、ゴ、ロク、ナナ、ハチ"
    "でよろしいでしょうか。",
    "失礼いたしました、もう一度お電話番号をお願いできますか。",
    "失礼いたします。",
]


def test_clinic_info_has_no_half_width_space():
    """A space here would reach the TTS verbatim, since the prompt embeds it."""
    assert " " not in CLINIC_INFO


def test_system_instruction_forbids_mid_sentence_spaces():
    assert "文の途中に半角スペースを入れないでください" in SYSTEM_INSTRUCTION
    assert "姓と名の間にスペースを入れず" in SYSTEM_INSTRUCTION


@pytest.mark.parametrize("sentence", CLINIC_ANSWERS + CALLBACK_CONFIRMATIONS)
def test_sentence_survives_every_cartesia_chunking(sentence):
    """However Cartesia groups the characters, the history matches what was sent.

    The one exception is a last message holding nothing but the closing ``。``,
    which doubles that mark — a distinct quirk, pinned down on its own below.
    """
    unsafe = [
        cuts for cuts in failing_chunkings(sentence) if not isolates_final_mark(sentence, cuts)
    ]

    assert unsafe == []


@pytest.mark.parametrize("sentence", CLINIC_ANSWERS + CALLBACK_CONFIRMATIONS)
def test_per_character_timestamps_only_double_the_closing_mark(sentence):
    """The other extreme: Cartesia reports every character in its own message.

    Every word still lands where it belongs, so the text is intact; only the
    sentence-final mark is repeated.
    """
    cuts = tuple(range(1, len(sentence)))

    assert history_for(sentence, cuts) == sentence + sentence[-1]


def test_closing_mark_alone_in_the_last_message_is_doubled():
    """A second, milder framework bug, unrelated to spaces and not fixed by the prompt.

    Harmless to the spoken audio (this is the history only), but it is how a
    doubled ``。。`` can appear in the context — and, once there, in what the LLM
    writes next.
    """
    assert history_for("失礼いたします。", (7,)) == "失礼いたします。。"


def test_a_name_written_with_a_space_would_still_break():
    """The prompt rule earns its place: the same name with a space is unsafe.

    Kept as the counter-example to the cases above — if this ever starts passing,
    the framework has been fixed and the prompt rule can be relaxed.
    """
    assert failing_chunkings("佐藤 祐希様、承知いたしました。")
