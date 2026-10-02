"""What is kept of a call once it is over."""

import json
from datetime import datetime

from call_record_store import (
    read_call_records,
    save_call_record,
    transcript_from_messages,
)

MESSAGES = [
    {"role": "developer", "content": "さくら歯科クリニックと名乗って短く挨拶してください。"},
    {"role": "assistant", "content": "さくら歯科クリニックでございます。"},
    {"role": "user", "content": "何時までやってますか"},
    {"role": "assistant", "content": "平日は午後6時まででございます。"},
]


def test_a_call_is_written_as_one_json_line(tmp_path):
    log = tmp_path / "calls.jsonl"

    record = save_call_record(
        MESSAGES,
        started_at=datetime(2026, 10, 2, 18, 50, 0),
        ended_at=datetime(2026, 10, 2, 18, 52, 30),
        session_id="abc123",
        path=log,
    )

    assert record["started_at"] == "2026-10-02T18:50:00"
    assert record["ended_at"] == "2026-10-02T18:52:30"
    assert record["session_id"] == "abc123"
    assert json.loads(log.read_text(encoding="utf-8")) == record


def test_the_whole_conversation_is_kept_in_order(tmp_path):
    log = tmp_path / "calls.jsonl"

    record = save_call_record(MESSAGES, started_at=datetime.now(), path=log)

    assert [m["role"] for m in record["messages"]] == [
        "developer",
        "assistant",
        "user",
        "assistant",
    ]
    assert record["messages"][2]["content"] == "何時までやってますか"


def test_calls_accumulate(tmp_path):
    log = tmp_path / "calls.jsonl"

    save_call_record(MESSAGES, started_at=datetime.now(), path=log)
    save_call_record(MESSAGES[:2], started_at=datetime.now(), path=log)

    assert [len(r["messages"]) for r in read_call_records(log)] == [4, 2]


def test_the_directory_is_created(tmp_path):
    log = tmp_path / "nested" / "calls.jsonl"

    save_call_record(MESSAGES, started_at=datetime.now(), path=log)

    assert read_call_records(log)


def test_no_log_yet_reads_as_empty(tmp_path):
    assert read_call_records(tmp_path / "missing.jsonl") == []


def test_japanese_survives_the_round_trip(tmp_path):
    """Written as UTF-8, not escape sequences: a person may open this file."""
    log = tmp_path / "calls.jsonl"

    save_call_record(MESSAGES, started_at=datetime.now(), path=log)

    assert "何時までやってますか" in log.read_text(encoding="utf-8")


def test_a_call_with_nothing_said_is_still_recorded(tmp_path):
    """A caller who hangs up during the greeting is a call that happened."""
    log = tmp_path / "calls.jsonl"

    record = save_call_record([], started_at=datetime.now(), path=log)

    assert record["messages"] == []
    assert len(read_call_records(log)) == 1


def test_message_parts_are_reduced_to_their_text():
    """A multimodal message is kept as the text it carries, not as wire format."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "こんにちは"}]},
        {"role": "user", "content": [{"text": "続きです"}]},
    ]

    assert transcript_from_messages(messages) == [
        {"role": "user", "content": "こんにちは"},
        {"role": "user", "content": "続きです"},
    ]


def test_an_unexpected_content_shape_is_not_dropped():
    """Better a stringified oddity in the record than a silently empty turn."""
    assert transcript_from_messages([{"role": "user", "content": 42}]) == [
        {"role": "user", "content": "42"}
    ]
