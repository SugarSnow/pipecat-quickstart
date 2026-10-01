"""Replay Cartesia's Japanese word-timestamp path and report what reaches the LLM context.

``CartesiaTTSService`` is constructed with ``push_text_frames=False``, so the
assistant's conversation history is **not** the text handed to the TTS. It is
rebuilt from the word timestamps Cartesia streams back, matched character by
character against the text that was sent:

    sent text --> Cartesia --> timestamp messages
                                    |
              CartesiaTTSService._normalize_word_timestamps   (ja: joins each
                                    |                          message's chars
                                    v                          into one token)
              AggregatedFrameSequencer.process_word           (matches the token
                                    |                          against the sent
                                    v                          text)
              TTSTextFrame(s) --> assistant aggregator --> history

The helpers here drive that same chain with the real framework classes, so a
test can ask one question: *does the history come back equal to what was sent?*

The only thing simulated is Cartesia itself — how it groups characters into
timestamp messages. That grouping is not observable from our side and varies run
to run, which is why the tests sweep every plausible grouping rather than
assuming one.
"""

import asyncio
from collections.abc import Iterator
from itertools import combinations

from pipecat.frames.frames import AggregatedTextFrame, AggregationType, TTSTextFrame
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text

_CONTEXT_ID = "test-context"


def _ja_service() -> CartesiaTTSService:
    """A service stub carrying only what ``_normalize_word_timestamps`` reads.

    Constructing the real service would open a websocket and demand an API key;
    the normalizer only looks at ``_settings.language``.
    """
    service = object.__new__(CartesiaTTSService)
    service._settings = CartesiaTTSService.Settings(language="ja")
    return service


def cartesia_ja_tokens(spoken_text: str, cuts: tuple[int, ...] = ()) -> list[str]:
    """Return the tokens Cartesia's ja path yields for *spoken_text*.

    Args:
        spoken_text: The text sent to Cartesia for synthesis.
        cuts: Character offsets where one timestamp message ends and the next
            begins. ``()`` means Cartesia reported the whole text in one message.

    Returns:
        One token per timestamp message, as ``_normalize_word_timestamps``
        produces them (it joins each message's characters into a single token).
    """
    service = _ja_service()
    tokens: list[str] = []
    for start, end in zip((0, *cuts), (*cuts, len(spoken_text))):
        chars = list(spoken_text[start:end])
        if not chars:
            continue
        starts = [float(start + index) for index in range(len(chars))]
        tokens.extend(token for token, _ in service._normalize_word_timestamps(chars, starts))
    return tokens


async def _build_history(llm_text: str, spoken_text: str, tokens: list[str]) -> str:
    sequencer = AggregatedFrameSequencer(name="test", streaming=False)
    frame = AggregatedTextFrame(llm_text, AggregationType.SENTENCE)
    await sequencer.register_spoken(frame, _CONTEXT_ID, spoken_text, append_to_context=True)

    text_frames: list[TTSTextFrame] = []
    pts = 0
    for token in tokens:
        pts += 1
        text_frames += [
            f
            for f in sequencer.process_word(
                token, pts, _CONTEXT_ID, includes_inter_frame_spaces=True
            )
            if isinstance(f, TTSTextFrame)
        ]
    # End of the audio context: whatever the word timestamps never accounted for
    # is emitted here, which is where the duplicated tail comes from.
    text_frames += [
        f for f in sequencer.force_complete(_CONTEXT_ID, pts) if isinstance(f, TTSTextFrame)
    ]

    # What LLMAssistantAggregator._handle_text does with each frame it receives.
    return concatenate_aggregated_text(
        [
            TextPartForConcatenation(
                f.raw_text or f.text, includes_inter_part_spaces=f.includes_inter_frame_spaces
            )
            for f in text_frames
        ]
    )


def history_for_tokens(llm_text: str, tokens: list[str], spoken_text: str | None = None) -> str:
    """Return the assistant history for an explicit token stream.

    For replaying a stream taken from a real call, where Cartesia's own choices —
    which characters it reports at all, and how it groups them — are known rather
    than swept over.
    """
    return asyncio.run(
        _build_history(llm_text, llm_text if spoken_text is None else spoken_text, tokens)
    )


def history_for(llm_text: str, cuts: tuple[int, ...] = (), spoken_text: str | None = None) -> str:
    """Return the assistant history produced for one LLM sentence.

    Args:
        llm_text: The sentence the LLM wrote.
        cuts: How Cartesia grouped the characters into timestamp messages.
        spoken_text: The text actually sent to the TTS, when a text transformer
            rewrote it. Defaults to *llm_text* (no transform), which is how the
            bot runs.
    """
    spoken = llm_text if spoken_text is None else spoken_text
    return asyncio.run(_build_history(llm_text, spoken, cartesia_ja_tokens(spoken, cuts)))


def every_chunking(text: str, max_cuts: int = 2) -> Iterator[tuple[int, ...]]:
    """Yield every way of splitting *text* into up to ``max_cuts + 1`` messages.

    Cartesia's message boundaries are outside our control, so a text is only
    safe if *all* of them round-trip. Two cuts is enough to expose the failure:
    one message boundary on each side of a space already breaks it.
    """
    positions = range(1, len(text))
    for count in range(max_cuts + 1):
        yield from combinations(positions, count)


def failing_chunkings(text: str, max_cuts: int = 2) -> list[tuple[int, ...]]:
    """Return the chunkings of *text* whose history differs from *text* itself."""
    return [cuts for cuts in every_chunking(text, max_cuts) if history_for(text, cuts) != text]


def isolates_final_mark(text: str, cuts: tuple[int, ...]) -> bool:
    """Whether *cuts* leave the sentence's closing punctuation alone in the last message.

    A separate framework quirk from the space bug: once every letter has been
    spoken the slot is already complete, so a trailing ``。`` arriving on its own
    matches nothing and is emitted as passthrough *in addition to* the one the
    tracker already accounted for, doubling it in the history.
    """
    tokens = cartesia_ja_tokens(text, cuts)
    return bool(tokens) and not any(char.isalnum() for char in tokens[-1])
