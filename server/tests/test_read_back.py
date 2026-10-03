"""The sentence the bot speaks to confirm a callback request.

The bot says this itself rather than leaving it to the LLM, so it is worth
pinning: the wording is what the caller hears on every call, and the digits are
read one at a time because that is what survives a phone line.
"""

import pytest

from bot_phone import read_back_line, spell_out_digits


def test_the_name_comes_first_then_the_number():
    assert read_back_line("佐藤祐希", "08012345678") == (
        "佐藤祐希様、お電話番号はゼロ、ハチ、ゼロ、イチ、ニ、サン、ヨン、ゴ、ロク、ナナ、ハチ、"
        "でよろしいでしょうか。"
    )


def test_every_digit_has_a_reading():
    assert (
        spell_out_digits("0123456789") == "ゼロ、イチ、ニ、サン、ヨン、ゴ、ロク、ナナ、ハチ、キュウ"
    )


@pytest.mark.parametrize(
    "given",
    ["080-1234-5678", "080 1234 5678", "（080）12345678", "08012345678です"],
)
def test_anything_that_is_not_a_digit_is_dropped(given):
    """This text goes straight to the TTS: a stray character would be read aloud."""
    assert spell_out_digits(given) == "ゼロ、ハチ、ゼロ、イチ、ニ、サン、ヨン、ゴ、ロク、ナナ、ハチ"


def test_no_half_width_space_reaches_the_tts():
    """A space mid-sentence desynchronises Cartesia's Japanese word timestamps."""
    assert " " not in read_back_line("佐藤祐希", "080 1234 5678")


def test_a_missing_number_still_reads_as_a_sentence():
    """Better an odd question than a crash while the caller is on the line."""
    assert read_back_line("佐藤", "") == "佐藤様、お電話番号は、でよろしいでしょうか。"
