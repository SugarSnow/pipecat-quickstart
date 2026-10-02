"""Why a half-width space inside a Japanese sentence corrupts the conversation history.

This is the failure observed in a phone call: the text sent to Cartesia was fine,
but the assistant message stored in the LLM context came back duplicated and
interleaved, alongside the warning

    aggregated_frame_sequencer:process_word - Word '...' not recognised by any
    slot, emitting as passthrough

These tests pin the mechanism down so a future framework upgrade tells us if it
changes. The companion file (``test_cartesia_ja_history_roundtrip.py``) checks
that what the bot is prompted to say today is safe.
"""

from cartesia_history_sim import (
    cartesia_ja_tokens,
    failing_chunkings,
    history_for,
    history_for_tokens,
    isolates_final_mark,
)

# The sentence the bot sent to the TTS during the failed call.
SENT_IN_FAILED_CALL = (
    "佐藤祐希様、お電話番号は「ゼロゼロゼロゼロ ゼロゼロ」でよろしいでしょうか。。"
)

# What ended up in the conversation history instead.
STORED_IN_FAILED_CALL = (
    "佐藤祐希様、お電話番号は「ゼロゼロゼロゼロゼロゼロゼでよろしいでしょうか。。"
    "ロゼロゼロゼロ ゼロゼロ」でよろしいでしょうか。。"
)


def test_cartesia_drops_half_width_spaces_from_ja_tokens():
    """The root cause: the ja normalizer strips each character, so a space vanishes."""
    tokens = cartesia_ja_tokens("番号は「ゼロ ゼロ」で")

    assert tokens == ["番号は「ゼロゼロ」で"]
    assert " " not in "".join(tokens)


def test_space_makes_the_history_diverge_from_the_sent_text():
    """With a space in the sentence, some Cartesia message boundaries corrupt the history."""
    broken = failing_chunkings(SENT_IN_FAILED_CALL)

    assert broken, "expected at least one chunking to corrupt the history"


def test_reproduces_the_history_recorded_during_the_call():
    """The exact stored string, from the token stream the call's warnings imply.

    Cartesia reported ``…お電話番号は``, then the digits with the space gone, then
    a stray ``ゼ``, then the tail. The first token past the space no longer matches
    what is left to speak, so it is emitted as passthrough while the tracker stays
    put, and ``force_complete`` later appends everything it still considers unspoken.
    """
    # Taken as Cartesia reported them: the space is gone, and so are the quote
    # marks, which it never reports because they are not spoken.
    tokens = [
        "佐藤祐希様、お電話番号は",
        "ゼロゼロゼロゼロゼロゼロ",
        "ゼ",
        "でよろしいでしょうか。。",
    ]

    assert history_for_tokens(SENT_IN_FAILED_CALL, tokens) == STORED_IN_FAILED_CALL


def test_removing_the_space_makes_every_chunking_safe():
    """The fix: no space in the sentence, so no boundary can desynchronise the match.

    Chunkings that leave the closing ``。`` alone in the last message are excluded:
    they double that mark through an unrelated path, covered in
    ``test_cartesia_ja_history_roundtrip.py``.
    """
    without_space = SENT_IN_FAILED_CALL.replace(" ", "")

    unsafe = [
        cuts
        for cuts in failing_chunkings(without_space)
        if not isolates_final_mark(without_space, cuts)
    ]

    assert unsafe == []


def test_stripping_the_space_for_tts_only_does_not_help():
    """Why a text transformer was rejected: it desynchronises the channels instead.

    Removing the space only from what is sent to the TTS leaves the LLM text and
    the spoken text different lengths, so the tracker can no longer map a spoken
    position back onto the text it records, and the history breaks anyway.
    """
    spoken = SENT_IN_FAILED_CALL.replace(" ", "")
    cuts = (12, 24)

    assert history_for(SENT_IN_FAILED_CALL, cuts, spoken_text=spoken) != SENT_IN_FAILED_CALL
