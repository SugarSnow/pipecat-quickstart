"""A TTS service that speaks silence and talks to nobody.

Text-mode eval scenarios never listen to the bot — the harness reads the LLM's
text, not its voice — but the bot still synthesized every fixed line it speaks
with ``tts_say``: the greeting, the read-back, the acknowledgements. That is
real Cartesia traffic for audio nothing plays, and a full suite run burned
through the free tier until connections started failing with HTTP 402.

Swapping this in keeps everything about the pipeline except the network call.
It emits a brief silence per utterance rather than nothing at all, so the
pipeline sees an ordinary spoken turn: the started/text/stopped frames arrive,
the line lands in the LLM context exactly as it would otherwise, and
BotStartedSpeakingFrame / BotStoppedSpeakingFrame bracket it — which the bot's
own mute strategies are built on. Returning no audio instead reads, to
:class:`TTSService`, as a provider that accepts requests and says nothing: it
reports an error per utterance and eventually stops giving the service work.
"""

from collections.abc import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

# Enough to count as audio, short enough that a suite run doesn't wait on it.
# Nothing reads these samples — only that some arrived.
SILENCE_MS = 10


class SilentTTSService(TTSService):
    """Speaks a few milliseconds of silence in place of synthesizing anything."""

    def __init__(self, **kwargs):
        """Initialize the service.

        Args:
            **kwargs: Passed to :class:`TTSService`.
        """
        # There is no model, voice or language here, and TTSService wants every
        # settings field accounted for — left unset it logs each one as an error
        # at startup. None is how a service says "not supported".
        kwargs.setdefault("settings", TTSSettings(model=None, voice=None, language=None))
        super().__init__(**kwargs)

    def can_generate_metrics(self) -> bool:
        """Whether this service reports TTFB and processing metrics.

        Returns:
            False: there is no synthesis here to measure, and a time-to-first-byte
            of zero would only skew the numbers a real run is compared against.
        """
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Produce silence for *text*.

        Args:
            text: The text that would have been spoken.
            context_id: The context this utterance belongs to.

        Yields:
            One frame of silence, at the pipeline's own sample rate.
        """
        logger.debug(f"{self}: not speaking {text!r}")
        await self.start_ttfb_metrics()
        # 16-bit mono, which is what the rest of the pipeline carries.
        samples = int(self.sample_rate * SILENCE_MS / 1000)
        yield TTSAudioRawFrame(
            audio=b"\x00\x00" * samples,
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=context_id,
        )
