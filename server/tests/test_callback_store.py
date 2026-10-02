"""What is written down when a caller asks to be called back."""

import json
from datetime import datetime

import pytest

from callback_store import (
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
