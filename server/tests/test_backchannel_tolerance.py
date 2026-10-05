"""A noise or an "はい" over the bot must not cut the bot off.

From a phone test: the refusal and the callback offer were interrupted by a
0.2s sound, so "いかがなさいますか" never reached the caller, who then asked
what had been said and got "はい。" back.

The strategy is exercised directly here — feed it frames, see whether it starts
a turn — because a turn start is what broadcasts an interruption, and neither
mode of the eval harness can produce the overlapping audio this is about. The
call test is what confirms it feels right.
"""

import asyncio

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.utils.time import time_now_iso8601

from bot_phone import BACKCHANNEL_MAX_CHARS, BackchannelToleranceUserTurnStartStrategy


def _transcript(text: str, interim: bool = False):
    cls = InterimTranscriptionFrame if interim else TranscriptionFrame
    return cls(text, "caller", time_now_iso8601())


def _run(frames: list) -> list[bool]:
    """Play *frames* through a fresh strategy; return which ones started a turn."""
    strategy = BackchannelToleranceUserTurnStartStrategy()
    started: list[bool] = []

    async def go():
        for frame in frames:
            result = await strategy.process_frame(frame)
            started.append(result == ProcessFrameResult.STOP)

    asyncio.run(go())
    return started


def _started_any(frames: list) -> bool:
    return any(_run(frames))


# --- the line is quiet: nothing changes -------------------------------------


def test_vad_starts_a_turn_when_the_bot_is_silent():
    """The read-back is answered with "はい", so a short reply must still land."""
    assert _started_any([VADUserStartedSpeakingFrame()])


def test_short_answer_is_heard_when_the_bot_is_silent():
    assert _started_any([VADUserStartedSpeakingFrame(), _transcript("はい")])


# --- over the bot: the threshold applies ------------------------------------


@pytest.mark.parametrize("text", ["はい", "ええ", "うん", "なるほど", "そうですか"])
def test_backchannel_does_not_interrupt(text):
    assert not _started_any(
        [
            BotStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _transcript(text),
        ]
    )


@pytest.mark.parametrize(
    "text",
    [
        "ちょっと待ってください",
        "もう一度お願いします",
        "何て言いましたか",
        "途中で切れました",
    ],
)
def test_a_real_interruption_still_interrupts(text):
    assert _started_any(
        [
            BotStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _transcript(text),
        ]
    )


def test_vad_alone_does_not_interrupt_the_bot():
    """The 0.2s noise from the phone test: VAD fires, no transcript follows."""
    assert not _started_any([BotStartedSpeakingFrame(), VADUserStartedSpeakingFrame()])


def test_interim_transcripts_count():
    """Waiting for the final transcript would make barge-in feel sluggish."""
    assert _started_any(
        [
            BotStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _transcript("ちょっと待ってください", interim=True),
        ]
    )


def test_spaces_do_not_count_as_characters():
    """Deepgram spaces a surname from a given name; that is not extra speech."""
    spoken = "あ" * (BACKCHANNEL_MAX_CHARS - 1)
    padded = " ".join(spoken)  # long enough to pass only if the spaces count

    assert len(padded) >= BACKCHANNEL_MAX_CHARS
    assert not _started_any(
        [BotStartedSpeakingFrame(), VADUserStartedSpeakingFrame(), _transcript(padded)]
    )


def test_one_more_character_is_an_interruption():
    """The other side of the threshold, so the boundary itself is pinned."""
    assert _started_any(
        [
            BotStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _transcript("あ" * BACKCHANNEL_MAX_CHARS),
        ]
    )


# --- the handover at the end of the bot's turn ------------------------------


def test_a_caller_still_talking_gets_their_turn_when_the_bot_stops():
    """Speech that began as a backchannel must not be left without a turn."""
    assert _started_any(
        [
            BotStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _transcript("はい"),
            BotStoppedSpeakingFrame(),
        ]
    )


def test_a_finished_backchannel_is_dropped():
    """It ended while the bot was still talking, so there is nothing to answer."""
    assert not _started_any(
        [
            BotStartedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _transcript("はい"),
            VADUserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
        ]
    )


def test_the_threshold_only_applies_over_the_bot():
    """Once the bot has stopped, a two-character answer starts a turn again."""
    assert _started_any(
        [
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _transcript("はい"),
        ]
    )


# --- once a turn is running, stay out of the way ----------------------------


def test_nothing_is_discarded_once_the_turn_has_started():
    """trigger_reset_aggregation() mid-turn would throw away the turn's words."""
    strategy = BackchannelToleranceUserTurnStartStrategy()
    resets: list[object] = []
    strategy.add_event_handler("on_reset_aggregation", lambda s: resets.append(s))

    async def go():
        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        # Long enough to interrupt; the controller then announces the turn.
        await strategy.process_frame(_transcript("ちょっと待ってください"))
        await strategy.handle_user_turn_started()
        # A short fragment arriving inside the turn must be left alone.
        await strategy.process_frame(_transcript("はい", interim=True))

    asyncio.run(go())

    assert resets == []


def test_the_next_turn_is_judged_again():
    strategy = BackchannelToleranceUserTurnStartStrategy()

    async def go() -> bool:
        await strategy.process_frame(BotStartedSpeakingFrame())
        await strategy.handle_user_turn_started()
        await strategy.handle_user_turn_stopped()
        # Back to watching: a backchannel over the bot is a backchannel again.
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        result = await strategy.process_frame(_transcript("はい"))
        return result == ProcessFrameResult.STOP

    assert asyncio.run(go()) is False
