"""The eval stand-in for the TTS: no network, but a turn that still looks spoken.

A text-mode scenario reads the LLM's words, never the bot's voice, so
synthesizing them is pure cost — and once the read-back and the greeting became
tts_say lines, a full suite run was enough to exhaust the Cartesia free tier and
start failing on HTTP 402.

What matters is that the swap is invisible to everything else in the pipeline,
which is what these check.
"""

import asyncio

import pytest
from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.runner.types import RunnerArguments

from bot_phone import _wants_silent_tts
from silent_tts import SILENCE_MS, SilentTTSService


def _collect(service: SilentTTSService, text: str = "お電話ありがとうございます。") -> list:
    async def go():
        return [f async for f in service.run_tts(text, "ctx-1")]

    return asyncio.run(go())


def _service(sample_rate: int = 8000) -> SilentTTSService:
    service = SilentTTSService()
    # Normally set when the pipeline starts; these tests drive run_tts directly.
    service._sample_rate = sample_rate
    return service


def test_it_produces_audio():
    """Returning nothing reads as a broken provider: TTSService reports an error
    for every silent context and eventually stops giving the service work."""
    frames = _collect(_service())

    assert frames
    assert all(isinstance(f, TTSAudioRawFrame) for f in frames)


def test_the_audio_is_silence_at_the_pipeline_sample_rate():
    (frame,) = _collect(_service(sample_rate=8000))

    assert frame.sample_rate == 8000
    assert frame.num_channels == 1
    # 16-bit mono, so two bytes a sample, and every one of them zero.
    assert len(frame.audio) == 2 * int(8000 * SILENCE_MS / 1000)
    assert set(frame.audio) == {0}


def test_it_is_short():
    """A suite run waits on this once per spoken line, so it has to stay small."""
    assert SILENCE_MS <= 50


def test_settings_are_declared():
    """Left unset, TTSService logs every unsupported field as an error at startup."""
    service = SilentTTSService()

    assert service._settings.model is None
    assert service._settings.voice is None
    assert service._settings.language is None


# --- choosing between this and the real thing -------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"silent_tts": True},
    ],
)
def test_the_runner_body_can_ask_for_silence(body):
    assert _wants_silent_tts(RunnerArguments(body=body))


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"silent_tts": False},
        {"something_else": True},
        "not a dict",
    ],
)
def test_everything_else_gets_the_real_voice(body):
    """A real call has no runner body at all, and must never be silent."""
    assert not _wants_silent_tts(RunnerArguments(body=body))
