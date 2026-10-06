"""Runs of 。 are collapsed before the TTS ever sees them.

Pipecat doubles the sentence-final 。 when it writes a turn into the history;
the LLM reads that back, copies it, and the run grows. The call logs show the
end of that road: "はい。。" went to the TTS 13 times and "。。" on its own 4
times — a request to speak nothing but a full stop.
"""

import asyncio

import pytest
from pipecat.frames.frames import (
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.tests.utils import run_test

from text_cleanup import CollapseRepeatedPunctuation


def _spoken(pieces: list) -> list[str]:
    """Play *pieces* through one processor; return the text that got through."""
    frames = [LLMTextFrame(p) if isinstance(p, str) else p for p in pieces]

    async def go():
        received, _ = await run_test(
            CollapseRepeatedPunctuation(),
            frames_to_send=frames,
            expected_down_frames=None,
        )
        return received

    return [f.text for f in asyncio.run(go()) if isinstance(f, LLMTextFrame)]


# --- within one piece -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("はい。。", "はい。"),
        ("はい。。。。", "はい。"),
        ("お名前をお伺いしてもよろしいでしょうか。。", "お名前をお伺いしてもよろしいでしょうか。"),
        # Untouched when there is nothing to collapse.
        ("はい。", "はい。"),
        ("営業時間は午後6時までです。", "営業時間は午後6時までです。"),
        # Other sentence-final marks, each collapsed on its own.
        ("本当ですか？？", "本当ですか？"),
    ],
)
def test_a_run_inside_one_piece(text, expected):
    assert _spoken([text]) == [expected]


# --- across the pieces the LLM streams --------------------------------------


def test_a_run_split_across_pieces():
    """The LLM streams tokens, so "はい。。。。" arrives in parts."""
    assert _spoken(["はい。", "。", "。", "。"]) == ["はい。"]


def test_a_bare_full_stop_never_reaches_the_tts():
    """This is the "。。" request from the logs: a turn to speak punctuation."""
    assert _spoken(["はい。", "。。"]) == ["はい。"]


def test_the_next_sentence_still_gets_through():
    assert _spoken(["はい。", "。", "承知しました。"]) == ["はい。", "承知しました。"]


# --- where a run ends -------------------------------------------------------


def test_a_new_turn_starts_the_run_over():
    """Two turns that each open with 。 are two sentences, not a run."""
    pieces = ["はい。", LLMFullResponseStartFrame(), "。承知しました。"]

    assert _spoken(pieces) == ["はい。", "。承知しました。"]



# --- everything else passes through -----------------------------------------


def test_other_frames_are_untouched():
    """A tts_say line is a TTSSpeakFrame and must not be rewritten or dropped."""
    text = "佐藤様、お電話番号は。。でよろしいでしょうか。"

    async def go():
        received, _ = await run_test(
            CollapseRepeatedPunctuation(),
            frames_to_send=[TTSSpeakFrame(text)],
            expected_down_frames=None,
        )
        return received

    (received,) = [f for f in asyncio.run(go()) if isinstance(f, TTSSpeakFrame)]
    assert received.text == text
