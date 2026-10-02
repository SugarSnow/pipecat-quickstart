"""What was said on a call, kept once the call is over.

One JSON object per line, like the callback log: when the call started and
ended, and every message the LLM saw, in order. Written from
``on_pipeline_finished``, which runs whichever way the call ended — the caller
hanging up or the bot doing so itself.

The path comes from ``$CALL_LOG_PATH`` when set, so an eval run can write
somewhere other than the file real calls go to.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("data/calls.jsonl")


def call_log_path() -> Path:
    """Return the file call records are appended to."""
    configured = os.environ.get("CALL_LOG_PATH")
    return Path(configured) if configured else DEFAULT_PATH


def _as_text(content: Any) -> str:
    """Return a message's content as text.

    Content is usually a string. A list of parts (what a multimodal message
    looks like) is reduced to the text it carries, so a record stays readable
    rather than reproducing the provider's wire format.

    Args:
        content: The message content, as the context holds it.

    Returns:
        The text of the content.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text", "input_text")
        ]
        return "".join(parts)
    return str(content)


def transcript_from_messages(messages: list[dict]) -> list[dict]:
    """Return ``messages`` reduced to the role and text of each turn.

    Args:
        messages: The context's messages.

    Returns:
        One ``{"role", "content"}`` entry per message, in order.
    """
    return [
        {"role": str(m.get("role", "")), "content": _as_text(m.get("content", ""))}
        for m in messages
    ]


def save_call_record(
    messages: list[dict],
    *,
    started_at: datetime,
    ended_at: datetime | None = None,
    session_id: str | None = None,
    path: Path | str | None = None,
) -> dict:
    """Append one call's record to the log and return what was written.

    Args:
        messages: The context's messages, in order.
        started_at: When the call started.
        ended_at: When the call ended. Defaults to the current local time.
        session_id: The runner's session id, when there is one.
        path: Where to append. Defaults to :func:`call_log_path`.

    Returns:
        The stored record: ``started_at``, ``ended_at``, ``session_id`` and
        ``messages``.
    """
    record = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": (ended_at or datetime.now()).isoformat(timespec="seconds"),
        "session_id": session_id,
        "messages": transcript_from_messages(messages),
    }

    target = Path(path) if path is not None else call_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def read_call_records(path: Path | str | None = None) -> list[dict]:
    """Return every call record in the log, oldest first.

    Args:
        path: Where to read from. Defaults to :func:`call_log_path`.

    Returns:
        The stored records; empty when the log does not exist yet.
    """
    target = Path(path) if path is not None else call_log_path()
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
