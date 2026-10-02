"""The call ends when the bot says so: the caller is muted from then on.

Two halves, verified where each can be:

- which turn ends the call — :func:`is_closing_utterance`, checked directly;
- what muting does to the caller's next words — checked by running a real
  ``LLMUserAggregator`` with the bot's own mute strategy.

The rest of the behavior (the hang-up a few seconds later) depends on the bot
producing audio, so it is exercised by an audio-mode eval or a real call, not
from here. See evals/closing_no_reply_after_farewell.yaml.
"""

import asyncio

import pytest
from pipecat.frames.frames import (
    LLMContextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import run_test
from pipecat.utils.time import time_now_iso8601

from bot_phone import CLOSING_LINE, FAREWELL_LINE, ClosingUserMuteStrategy, is_closing_utterance


def test_closing_line_ends_the_call():
    assert is_closing_utterance(CLOSING_LINE)
    assert is_closing_utterance(FAREWELL_LINE)


@pytest.mark.parametrize(
    "text",
    [
        # The apology for a misheard number: one syllable apart from the closing
        # line, and in the middle of the callback flow.
        "失礼いたしました、もう一度お電話番号をお願いできますか。",
        "お電話番号はゼロ、ハチ、ゼロでよろしいでしょうか。",
        "営業時間は平日は午前9時から午後6時までです。",
        "",
    ],
)
def test_other_turns_do_not_end_the_call(text):
    assert not is_closing_utterance(text)


async def _run(text: str, closing: bool) -> tuple[list[str], bool]:
    strategy = ClosingUserMuteStrategy()
    strategy.closing = closing
    context = LLMContext()
    user_aggregator, _ = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_mute_strategies=[strategy]),
    )
    # One complete caller turn: speech starts, the transcript arrives, speech ends.
    frames_to_send = [
        UserStartedSpeakingFrame(),
        TranscriptionFrame(text, "caller", time_now_iso8601()),
        UserStoppedSpeakingFrame(),
    ]
    received_down, _ = await run_test(
        user_aggregator,
        frames_to_send=frames_to_send,
        expected_down_frames=None,
    )
    messages = [m["content"] for m in context.get_messages() if m.get("role") == "user"]
    # LLMContextFrame is what runs the LLM, so its absence is the absence of a reply.
    ran_llm = any(isinstance(f, LLMContextFrame) for f in received_down)
    return messages, ran_llm


def _caller_says(text: str, *, closing: bool) -> tuple[list[str], bool]:
    """Run one caller turn through the aggregator; return context + whether the LLM ran."""
    return asyncio.run(_run(text, closing))


def test_caller_is_heard_during_the_call():
    messages, ran_llm = _caller_says("駐車場ってありますか", closing=False)

    assert messages == ["駐車場ってありますか"]
    assert ran_llm


def test_caller_is_muted_after_the_closing():
    """The parting "失礼します" must not reach the LLM and start another turn."""
    messages, ran_llm = _caller_says("あ、はい、失礼します", closing=True)

    assert messages == []
    assert not ran_llm
