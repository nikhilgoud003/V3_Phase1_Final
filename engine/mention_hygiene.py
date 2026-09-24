"""Mention hygiene: quarantine honorific/fragment noise; same-UCID for low-info.

A4: when ``low_info_within_ucid_only`` is true, single common first-name
mentions (mary, william, mark, …) stay in the pool but may ONLY merge
within their own UCID at every tier.
"""

from __future__ import annotations

import re
from typing import Any

from engine.common_first_names import COMMON_FIRST_NAMES

HONORIFIC_ONLY = {
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
    "esq",
    "hon",
    "honorable",
    "judge",
    "magistrate",
    "chief",
    "justice",
    "the",
}

# Truncation / garbage tokens often left by NER
FRAGMENT_LITERALS = {
    "without f",
    "will b",
    "consent o",
    "assignment",
    "under",
    "as t",
    "in t",
    "for f",
}


def is_honorific_only(normalized_name: str) -> bool:
    nn = (normalized_name or "").strip().lower()
    if not nn:
        return True
    toks = nn.split()
    if len(toks) == 1 and toks[0] in HONORIFIC_ONLY:
        return True
    if nn in HONORIFIC_ONLY:
        return True
    return False


def is_fragment_literal(normalized_name: str) -> bool:
    nn = (normalized_name or "").strip().lower()
    return nn in FRAGMENT_LITERALS or bool(re.fullmatch(r"[a-z]\b", nn))


def is_common_first_name_only(mention: dict) -> bool:
    nn = (mention.get("normalized_name") or "").strip().lower()
    toks = nn.split()
    return len(toks) == 1 and toks[0] in COMMON_FIRST_NAMES


def is_low_information(mention: dict, common_surnames: set[str] | None = None) -> bool:
    """Single-token, initial-heavy, honorific-only, or known fragment."""
    from engine.normalize import initial_token_ratio, tokens

    nn = mention.get("normalized_name") or ""
    if is_honorific_only(nn) or is_fragment_literal(nn):
        return True
    toks = tokens(nn)
    if len(toks) == 1:
        return True
    if initial_token_ratio(nn) >= 0.5:
        return True
    sur = mention.get("surname") or ""
    if common_surnames and sur in common_surnames and len(toks) <= 2:
        return True
    return False


def classify_mention(
    mention: dict,
    common_surnames: set[str] | None = None,
    *,
    low_info_within_ucid_only: bool = True,
) -> str:
    """Return: keep | quarantine | same_ucid_only."""
    nn = mention.get("normalized_name") or ""
    if is_honorific_only(nn) or is_fragment_literal(nn):
        return "quarantine"
    if low_info_within_ucid_only and is_common_first_name_only(mention):
        return "same_ucid_only"
    if is_low_information(mention, common_surnames):
        return "same_ucid_only"
    return "keep"


def apply_hygiene(
    mentions: list[dict],
    common_surnames: set[str] | None = None,
    *,
    cfg: dict | None = None,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Split mentions into cascade pool vs quarantine; tag same_ucid_only.

    Quarantined mentions are removed from the cascade pool entirely.
    same_ucid_only mentions stay but are flagged for tier enforcement.
    """
    mh = (cfg or {}).get("mention_hygiene") or {}
    low_info_flag = bool(mh.get("low_info_within_ucid_only", True))

    kept: list[dict] = []
    quarantined: list[dict] = []
    counts = {"keep": 0, "same_ucid_only": 0, "quarantine": 0}
    for m in mentions:
        label = classify_mention(
            m, common_surnames, low_info_within_ucid_only=low_info_flag
        )
        counts[label] += 1
        if label == "quarantine":
            q = dict(m)
            q["hygiene_action"] = "quarantine"
            quarantined.append(q)
            continue
        out = dict(m)
        if label == "same_ucid_only":
            out["hygiene_scope"] = "same_ucid"
            out["hygiene_action"] = "same_ucid_only"
            if is_common_first_name_only(m):
                out["low_info_first_name"] = True
        else:
            out["hygiene_scope"] = "global"
            out["hygiene_action"] = "keep"
        kept.append(out)
    return kept, quarantined, counts


def pair_allowed(a: dict, b: dict) -> bool:
    """Gate Tier2/Tier3 pairing (never Tier0 exact keys).

    Short single-token mentions are owned by within-UCID anchoring and are not
    eligible for similarity pairing at all. Same-UCID permission is not enough
    for them: one case can list a district judge and a referred magistrate, so
    "walter" vs "walsh" would otherwise be proposed as a pair.
    """
    if a.get("anchor_managed") or b.get("anchor_managed"):
        return False

    scopes = {a.get("hygiene_scope"), b.get("hygiene_scope")}
    if "same_ucid" in scopes:
        return (a.get("ucid") or "") == (b.get("ucid") or "") and bool(a.get("ucid"))
    if a.get("low_info_first_name") or b.get("low_info_first_name"):
        return (a.get("ucid") or "") == (b.get("ucid") or "") and bool(a.get("ucid"))
    return True
