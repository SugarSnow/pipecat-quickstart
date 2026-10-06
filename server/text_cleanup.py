"""Tidying the LLM's text on its way to the TTS.

Pipecat doubles the sentence-final 。 when it writes a turn into the
conversation history — harmless in itself, since the audio is already spoken by
then. The LLM reads that history back on the next turn, copies the doubling,
and the run grows: "はい。。" became "はい。。。。" over a few turns, and the
sentence aggregator then handed the TTS requests consisting of nothing but
punctuation. Across the call logs to date: "はい。。" 13 times, "。。" on its
own 4 times.

Collapsing has to happen before the TTS aggregates text into sentences. A
``text_transforms`` entry on the service runs per aggregated chunk, by which
point "はい。。。。" has already been split into "はい。", "。", "。" — nothing
left for it to collapse.
"""

import re

from loguru import logger
from pipecat.frames.frames import Frame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Every Japanese sentence-final mark the bot's own lines use. Collapsed per
# character, so "。。！！" keeps one of each rather than becoming one mark.
_RUN = re.compile(r"([。！？])\1+")


class CollapseRepeatedPunctuation(FrameProcessor):
    """Reduces runs of sentence-final punctuation in LLM text to one mark.

    The LLM streams a turn in pieces, and a run can straddle them ("はい。" then
    "。"), so the last character of what went out is remembered and a piece that
    opens by repeating it has that repeat trimmed. A piece left with nothing in
    it is dropped rather than forwarded, which is what keeps a bare "。" from
    reaching the TTS as a request of its own.
    """

    def __init__(self, **kwargs):
        """Initialize the processor.

        Args:
            **kwargs: Passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._last_char = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Collapse punctuation runs in LLM text, passing everything else through.

        Args:
            frame: The frame to process.
            direction: The direction the frame is travelling.
        """
        await super().process_frame(frame, direction)

        # A new turn starts the run over. This covers an interrupted turn too:
        # whatever the LLM says next opens with its own start frame, and no text
        # reaches here in between.
        if isinstance(frame, LLMFullResponseStartFrame):
            self._last_char = ""
        elif isinstance(frame, LLMTextFrame):
            text = self._collapse(frame.text)
            if not text:
                # Nothing left to say. Forwarding it would reach the TTS as a
                # request to speak a full stop.
                logger.debug(f"Dropping repeated punctuation: {frame.text!r}")
                return
            frame.text = text

        await self.push_frame(frame, direction)

    def _collapse(self, text: str) -> str:
        """Collapse runs inside *text*, and any run it continues from before."""
        if not text:
            return text

        collapsed = _RUN.sub(r"\1", text)
        # The run may have started in an earlier piece.
        while collapsed and collapsed[0] == self._last_char:
            collapsed = collapsed[1:]

        if collapsed:
            self._last_char = collapsed[-1]
        return collapsed
