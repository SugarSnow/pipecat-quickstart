"""What is written down when a caller asks to be called back."""

import json
from datetime import datetime

import pytest

from callback_store import (
    expected_digit_count,
    has_expected_digit_count,
    merge_phone_number,
    normalize_phone_number,
    read_callback_requests,
    save_callback_request,
)


def test_request_is_written_as_one_json_line(tmp_path):
    log = tmp_path / "callbacks.jsonl"

    record = save_callback_request(
        "佐藤祐希", "08012345678", path=log, now=datetime(2026, 10, 2, 14, 30, 5)
    )

    assert record == {
        "requested_at": "2026-10-02T14:30:05",
        "name": "佐藤祐希",
        "phone_number": "08012345678",
    }
    assert json.loads(log.read_text(encoding="utf-8")) == record


def test_requests_accumulate(tmp_path):
    log = tmp_path / "callbacks.jsonl"

    save_callback_request("佐藤祐希", "08012345678", path=log)
    save_callback_request("田中太郎", "09087654321", path=log)

    stored = read_callback_requests(log)
    assert [r["name"] for r in stored] == ["佐藤祐希", "田中太郎"]


def test_the_directory_is_created(tmp_path):
    """The bot should not have to be started in a prepared directory."""
    log = tmp_path / "nested" / "dir" / "callbacks.jsonl"

    save_callback_request("佐藤祐希", "08012345678", path=log)

    assert read_callback_requests(log)


def test_no_log_yet_reads_as_empty(tmp_path):
    assert read_callback_requests(tmp_path / "missing.jsonl") == []


def test_japanese_survives_the_round_trip(tmp_path):
    """Written as UTF-8, not escape sequences: a person may open this file."""
    log = tmp_path / "callbacks.jsonl"

    save_callback_request("佐藤祐希", "08012345678", path=log)

    assert "佐藤祐希" in log.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("080-1234-5678", "08012345678"),
        ("080 1234 5678", "08012345678"),
        ("０８０12345678", "０８０12345678"),  # full-width digits are left visible
        (" 08012345678 ", "08012345678"),
        ("080ー1234ー5678", "08012345678"),  # the katakana prolonged sound mark
    ],
)
def test_grouping_characters_are_removed(given, expected):
    assert normalize_phone_number(given) == expected


def test_an_undialable_value_is_kept_as_it_is():
    """Mangling a bad value into digits would hide that it was bad."""
    assert normalize_phone_number("ゼロハチゼロ") == "ゼロハチゼロ"


@pytest.mark.parametrize(
    ("phone_number", "expected"),
    [
        ("08012345678", 11),  # 携帯
        ("07012345678", 11),
        ("09012345678", 11),
        ("0312345678", 10),  # 固定電話
        ("0612345678", 10),
        ("0501234567", 10),  # IP電話
        ("0120123456", 10),  # フリーダイヤル
    ],
)
def test_how_many_digits_a_prefix_calls_for(phone_number, expected):
    assert expected_digit_count(phone_number) == expected


@pytest.mark.parametrize(
    "phone_number",
    ["08012345678", "09012345678", "0312345678", "0120123456", "080-1234-5678"],
)
def test_a_complete_number_passes(phone_number):
    assert has_expected_digit_count(phone_number)


@pytest.mark.parametrize(
    ("phone_number", "why"),
    [
        ("0801234567", "携帯なのに10桁"),
        ("080123456789", "携帯なのに12桁"),
        ("031234567", "固定電話なのに9桁"),
        ("03123456789", "固定電話なのに11桁"),
        ("", "何も聞き取れていない"),
    ],
)
def test_a_number_that_cannot_be_right_is_rejected(phone_number, why):
    assert not has_expected_digit_count(phone_number), why


def test_the_count_is_of_digits_not_characters():
    """The caller's grouping is theirs; only the digits decide."""
    assert has_expected_digit_count("080 1234 5678")
    assert has_expected_digit_count("（03）1234-5678")


# --- assembling a number read out in pieces ---------------------------------
#
# The model passes sometimes the new piece and sometimes everything so far; the
# caller said the same thing either way, so both have to land the same.


@pytest.mark.parametrize(
    ("collected", "heard", "expected"),
    [
        # Nothing yet: whatever arrives is the number so far.
        ("", "080", "080"),
        # The new piece only.
        ("080", "1234", "0801234"),
        ("0801234", "5678", "08012345678"),
        # Everything so far, repeated.
        ("080", "0801234", "0801234"),
        ("0801234", "08012345678", "08012345678"),
        # The same piece twice — a stutter, not more digits.
        ("080", "080", "080"),
        # Nothing new at all.
        ("08012345678", "", "08012345678"),
        # Non-digits are dropped on the way in, as everywhere else.
        ("080", "1234の", "0801234"),
    ],
)
def test_merge_phone_number(collected, heard, expected):
    assert merge_phone_number(collected, heard) == expected
