"""Where a callback request goes once the caller has given their details.

One JSON object per line, appended as each call ends. A line is small and
self-contained, so the file can be read by the admin console later, tailed while
a call is in progress, or opened in an editor when something looks wrong —
without any of them having to agree on a schema first.

The path comes from ``$CALLBACK_LOG_PATH`` when set, so an eval run can write
somewhere other than the file real calls go to.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path("data/callbacks.jsonl")


def callback_log_path() -> Path:
    """Return the file callback requests are appended to."""
    configured = os.environ.get("CALLBACK_LOG_PATH")
    return Path(configured) if configured else DEFAULT_PATH


def normalize_phone_number(phone_number: str) -> str:
    """Return *phone_number* as something that can be dialled.

    Spaces and the characters people put between groups of digits are removed;
    anything else is left alone. A value that arrives in a form nobody can dial
    should stay visible in the record rather than be quietly reduced to the
    digits it happens to contain.

    Args:
        phone_number: The number as the caller gave it.

    Returns:
        The number with grouping characters removed.
    """
    return re.sub(r"[\s\-‐-―ー−()（）]", "", phone_number.strip())


# Mobile numbers are 11 digits and start 070, 080 or 090; everything else a
# caller is likely to give — landline, IP, freephone — is 10.
_MOBILE_PREFIXES = ("070", "080", "090")
_MOBILE_DIGITS = 11
_OTHER_DIGITS = 10


def expected_digit_count(phone_number: str) -> int:
    """Return how many digits *phone_number* should have.

    Args:
        phone_number: The number, as the caller gave it.

    Returns:
        11 for a mobile prefix, 10 otherwise.
    """
    digits = re.sub(r"\D", "", phone_number)
    return _MOBILE_DIGITS if digits.startswith(_MOBILE_PREFIXES) else _OTHER_DIGITS


def has_expected_digit_count(phone_number: str) -> bool:
    """Whether *phone_number* has as many digits as its prefix calls for.

    Counting digits is left to code rather than to the LLM, which read a
    complete 10-digit landline as too short and accepted a 10-digit number
    beginning 080 as complete.

    Args:
        phone_number: The number, as the caller gave it.

    Returns:
        True when the digit count matches :func:`expected_digit_count`.
    """
    digits = re.sub(r"\D", "", phone_number)
    return len(digits) == expected_digit_count(phone_number)


def save_callback_request(
    name: str,
    phone_number: str,
    *,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> dict:
    """Append one callback request to the log and return what was written.

    Args:
        name: The caller's name, as read back and confirmed.
        phone_number: The caller's phone number, as read back and confirmed.
        path: Where to append. Defaults to :func:`callback_log_path`.
        now: Timestamp to record. Defaults to the current local time.

    Returns:
        The stored record: ``requested_at``, ``name`` and ``phone_number``.
    """
    record = {
        "requested_at": (now or datetime.now()).isoformat(timespec="seconds"),
        "name": name.strip(),
        "phone_number": normalize_phone_number(phone_number),
    }

    target = Path(path) if path is not None else callback_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def read_callback_requests(path: Path | str | None = None) -> list[dict]:
    """Return every request in the log, oldest first.

    Args:
        path: Where to read from. Defaults to :func:`callback_log_path`.

    Returns:
        The stored records; empty when the log does not exist yet.
    """
    target = Path(path) if path is not None else callback_log_path()
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
