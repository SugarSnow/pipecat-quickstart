"""Where this clinic's own details live.

Everything in ``config/tenant.json`` is particular to one clinic: its name and
hours, and the words its callers actually say. Keeping them out of the code is
what lets a second clinic be a second file rather than a second bot.

The file is read once at import. It is configuration, not state — a change to
it is a restart, the same as a change to the code would be.
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config" / "tenant.json"


def _load() -> dict:
    """Read the tenant file.

    Returns:
        The parsed configuration.
    """
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def clinic_info_text(config: dict | None = None) -> str:
    """Render the clinic's details as the lines the system instruction carries.

    One "key:value" per line, which is how the prompt has always held them —
    compact enough not to crowd the instruction, and plain enough that the LLM
    quotes the values back rather than paraphrasing them.

    Args:
        config: Parsed configuration; read from disk when omitted.

    Returns:
        The block of text to embed in the prompt, ending in a newline.
    """
    info = (config or _load())["clinic_info"]
    return "".join(f"{key}:{value}\n" for key, value in info.items())


def keyterms(config: dict | None = None) -> list[str]:
    """The words to boost in the speech-to-text.

    Args:
        config: Parsed configuration; read from disk when omitted.

    Returns:
        The keyterm list, in file order.
    """
    return list((config or _load())["keyterms"])


TENANT = _load()
