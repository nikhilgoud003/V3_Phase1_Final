"""Discovery-time validity for schema-free field typing.

Used when a never-seen JSON path is typed as judge/firm/party. Mirrors the
spirit of ``name_validity``: reject non-name strings *before* they become
mentions — do not rely on post-hoc entity deletion.

Deterministic rules only (no LLM). Safe to call from orchestration without
changing Tier0–3 matching or YAML identity rules.
"""

from __future__ import annotations

import re
from typing import Iterable

# Leaf keys / path suffixes that hold case metadata or counsel dumps — never
# person/org name fields, regardless of sample text.
NON_ENTITY_PATH_LEAFS = frozenset(
    {
        "raw_info",
        "member_case_key",
        "lead_case_key",
        "related_case_key",
        "pacer_case_id",
        "case_key",
        "mdl_id_source",
        "scraper_labels",
        "case_flags",
        "download_url",
        "pdf_url",
        "nature_suit",
        "nature_of_suit",
        "case_status",
        "cause",
        "jurisdiction",
        "jury_demand",
        "city",  # division label, not a party
        "source",
    }
)

NON_ENTITY_PATH_SUBSTR = (
    ".raw_info",
    "member_case_key",
    "lead_case_key",
    "related_case_key",
)

# PACER counsel HTML / status boilerplate (not a litigant name).
_HTML_TAG_RE = re.compile(r"(?i)<\s*/?\s*(?:br|i|b|em|strong|p|span|div|a)\b[^>]*>")
_COUNSEL_STATUS_RE = re.compile(
    r"(?i)\b("
    r"lead\s+attorney|attorney\s+to\s+be\s+noticed|see\s+above\s+for\s+address|"
    r"terminated\s*:|pro\s+hac\s+vice|notice\s+of\s+appearance"
    r")\b"
)

# Internal case-tracking / UCID-like codes (azd;;2:15-md-02641).
_UCID_KEY_RE = re.compile(
    r"(?i)^\s*[a-z]{2,5}\s*;;\s*\d{1,2}\s*:\s*\d{2}\s*-\s*(?:cv|cr|mc|md|mj)\s*-\s*\d+\s*$"
)
_DOCKET_ID_RE = re.compile(
    r"(?i)^\s*\d{1,2}\s*:\s*\d{2}\s*-\s*(?:cv|cr|mc|md|mj)\s*-\s*\d+\s*$"
)


def path_is_non_entity(path_pattern: str) -> tuple[bool, str | None]:
    """True when this field path cannot hold a judge/firm/party name."""
    pat = (path_pattern or "").strip()
    if not pat:
        return True, "empty_path"
    leaf = pat.split(".")[-1].replace("[]", "")
    if leaf in NON_ENTITY_PATH_LEAFS:
        return True, f"non_entity_leaf:{leaf}"
    low = pat.lower()
    for sub in NON_ENTITY_PATH_SUBSTR:
        if sub in low:
            return True, f"non_entity_path:{sub.strip('.')}"
    return False, None


def value_is_discovery_junk(raw: str) -> tuple[bool, str | None]:
    """True when a string must not be typed as an entity name."""
    s = (raw or "").strip()
    if not s:
        return True, "empty"

    if _HTML_TAG_RE.search(s):
        return True, "html_markup"

    if _COUNSEL_STATUS_RE.search(s):
        return True, "counsel_status_boilerplate"

    # Compact HTML-stripped form still often starts with see-above address cue
    low = re.sub(r"\s+", " ", s.lower())
    if low.startswith("(see above") or low.startswith("see above for address"):
        return True, "see_above_address_boilerplate"

    if _UCID_KEY_RE.match(s) or _DOCKET_ID_RE.match(s):
        return True, "case_tracking_code"

    return False, None


def samples_are_all_junk(samples: Iterable[str]) -> tuple[bool, str | None]:
    reasons: list[str] = []
    n = 0
    for s in samples:
        n += 1
        junk, reason = value_is_discovery_junk(s)
        if not junk:
            return False, None
        if reason:
            reasons.append(reason)
    if n == 0:
        return True, "no_samples"
    # Dominant reason
    return True, reasons[0] if reasons else "all_samples_junk"


def force_type_other(path_pattern: str, samples: list[str]) -> tuple[bool, str | None]:
    """Decide whether field typing must return ``other`` (skip entity emit)."""
    bad_path, why = path_is_non_entity(path_pattern)
    if bad_path:
        return True, why
    all_junk, why2 = samples_are_all_junk(samples)
    if all_junk:
        return True, why2
    return False, None
