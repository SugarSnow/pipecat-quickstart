"""The opening greeting: fixed wording, and nothing can talk over it.

A phone test lost a greeting entirely — the LLM was still composing it when the
caller spoke, the VAD broadcast an interruption, and the bot never named the
clinic. Both halves of the fix are checked here:

- the wording, which is now a constant rather than something the LLM writes;
- the mute that holds from the moment the line connects until the greeting has
  been spoken in full.

That the line really survives a caller talking over it needs the bot to produce
audio, so it is exercised by evals/greeting_not_interrupted_audio.yaml rather
than from here.
"""

import asyncio

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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

from bot_phone import GREETING_LINE, OpeningUserMuteStrategy, reception_node


def test_greeting_wording():
    assert GREETING_LINE == (
        "お電話ありがとうございます。さくら歯科クリニックです。ご用件をお聞かせいただけますか。"
    )


def test_greeting_is_speakable():
    # Cartesia reads a half-width space inside a Japanese line as a pause long
    # enough to sound like a different sentence; the rest is the voice-safety
    # rule the whole bot follows.
    assert " " not in GREETING_LINE
    assert not any(c in GREETING_LINE for c in "*-•#`[]()")


def test_reception_node_speaks_the_greeting_and_waits():
    node = reception_node()

    assert node["pre_actions"] == [{"type": "tts_say", "text": GREETING_LINE}]
    # Without this the LLM would answer on top of the line just queued.
    assert node["respond_immediately"] is False


def test_llm_is_not_asked_to_greet():
    """The prompt must not invite a second greeting on top of the fixed one."""
    role_message = reception_node()["role_message"]

    assert GREETING_LINE not in role_message
    assert "はじめの挨拶はこちらで読み上げ済みです" in role_message


async def _run(bot_frames: list) -> tuple[list[str], bool]:
    """Play *bot_frames*, then one complete caller turn, through the aggregator."""
    context = LLMContext()
    user_aggregator, _ = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_mute_strategies=[OpeningUserMuteStrategy()]
        ),
    )
    frames_to_send = [
        *bot_frames,
        UserStartedSpeakingFrame(),
        TranscriptionFrame("もしもし、すみません", "caller", time_now_iso8601()),
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


def test_caller_is_muted_before_the_greeting_starts():
    """The window FirstSpeechUserMuteStrategy leaves open: connect to first speech."""
    messages, ran_llm = asyncio.run(_run([]))

    assert messages == []
    assert not ran_llm


def test_caller_is_muted_while_the_greeting_plays():
    messages, ran_llm = asyncio.run(_run([BotStartedSpeakingFrame()]))

    assert messages == []
    assert not ran_llm


def test_caller_is_heard_once_the_greeting_ends():
    messages, ran_llm = asyncio.run(
        _run([BotStartedSpeakingFrame(), BotStoppedSpeakingFrame()])
    )

    assert messages == ["もしもし、すみません"]
    assert ran_llm


def test_mute_releases_if_the_bot_never_speaks():
    """A TTS outage must not leave the caller talking into a line that cannot hear."""
    strategy = OpeningUserMuteStrategy(timeout=0.0)

    async def run() -> list[bool]:
        # The clock starts at the first frame, so the first call is still muted
        # and the second is past a zero-length timeout.
        return [
            await strategy.process_frame(UserStartedSpeakingFrame()),
            await strategy.process_frame(UserStartedSpeakingFrame()),
        ]

    assert asyncio.run(run()) == [True, False]


def test_mute_does_not_come_back_for_later_turns():
    strategy = OpeningUserMuteStrategy()

    async def run() -> bool:
        await strategy.process_frame(BotStoppedSpeakingFrame())
        # A later bot turn starts speaking again; barge-in must still work.
        await strategy.process_frame(BotStartedSpeakingFrame())
        return await strategy.process_frame(UserStartedSpeakingFrame())

    assert asyncio.run(run()) is False
