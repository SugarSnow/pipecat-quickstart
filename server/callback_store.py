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


def phone_digits(phone_number: str) -> str:
    """Return just the digits of *phone_number*.

    What "the same number" means has to be decided somewhere, and it is not
    string equality: the model hands over "080-1234-5678" and "08012345678" for
    the same number on different turns.

    Args:
        phone_number: The number in any form.

    Returns:
        The digits, in order, with everything else removed.
    """
    return re.sub(r"\D", "", phone_number)


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
    digits = phone_digits(phone_number)
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
    digits = phone_digits(phone_number)
    return len(digits) == expected_digit_count(phone_number)


# A number more than two digits short is still being read out; one or two digits
# short is a number that was finished and came out wrong.
_FRAGMENT_MARGIN = 2


def is_partial_phone_number(phone_number: str) -> bool:
    """Whether *phone_number* looks like a number still being read out.

    Someone reading a number off a screen says it in pieces, and asking them to
    start over after three digits is never right. Where exactly the line falls
    is a judgement, so it is made here rather than left to the LLM, which called
    three digits complete.

    Args:
        phone_number: The digits heard so far.

    Returns:
        True when more digits are expected than a slip would account for.
    """
    digits = phone_digits(phone_number)
    return len(digits) <= expected_digit_count(phone_number) - _FRAGMENT_MARGIN


def merge_phone_number(collected: str, heard: str) -> str:
    """Combine the digits collected so far with the next thing heard.

    Someone reading a number off a screen says it in pieces, and the model is
    free to hand over either the new piece ("5678") or everything so far
    ("08012345678") — the prompt asks for one, the conversation suggests the
    other, and which one arrives varies run to run. Both mean the same thing to
    the caller, so both are made to mean the same thing here.

    Args:
        collected: Digits already gathered for this number.
        heard: What the model just passed in.

    Returns:
        The number as it stands after this piece.
    """
    collected = phone_digits(collected)
    heard = phone_digits(heard)

    if not collected:
        return heard
    # The whole number again, not just the new part.
    if heard.startswith(collected):
        return heard
    # Nothing new in it; keep what we have rather than going backwards.
    if collected.startswith(heard):
        return collected
    return collected + heard


def merge_name(collected: str, heard: str) -> str:
    """Combine the name collected so far with the next thing heard.

    A caller gives their name the way they say it out loud — "小林" and then
    "本木です" — and each piece arrives as its own turn. The model then hands
    over sometimes the new piece and sometimes the whole name, exactly as it
    does with a phone number, and in one call test it passed only the second
    piece: the surname was lost and the read-back was wrong. Joining the pieces
    here makes either choice mean the same thing.

    Unlike digits, a name can be extended at the front (surname heard last), so
    a piece that contains what we already have — at either end — replaces it.

    Args:
        collected: The name as it stands for this attempt.
        heard: What the model just passed in.

    Returns:
        The name after this piece.
    """
    collected = collected.strip()
    heard = heard.strip()

    if not collected:
        return heard
    if not heard:
        return collected
    # The whole name, not just the new piece.
    if heard.startswith(collected) or heard.endswith(collected):
        return heard
    # Nothing new in it; keep what we have rather than going backwards.
    if collected.startswith(heard) or collected.endswith(heard):
        return collected
    return collected + heard


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
