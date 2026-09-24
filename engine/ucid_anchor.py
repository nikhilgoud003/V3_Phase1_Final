"""Within-UCID anchoring for short mentions.

A docket line often names a judge with one word — ``walter``, ``wimes``,
``william``. The old A4 rule let any such token merge with anything inside its
own UCID, which is wrong the moment a case has two judges: CACD 8:16-cv-00800
lists district judge **John F. Walter** and referred magistrate **Patrick J.
Walsh**, so "walter" and "walsh" were fused into one entity. Worse, tokens like
``walter``, ``william`` and ``robert`` are both given names and surnames, so
classifying them as "first-name-only" mishandled exactly this case.

The rule here is anchoring rather than permission: a short mention is resolved
against the *full-name* judges of its own UCID.

- Exactly one anchor matches on surname or first name → attach to that anchor
  (deterministic, method ``ucid_anchor``).
- Zero or several anchors match → **abstain**. The mention stays a singleton.

Short mentions never enter Tier2/Tier3 pairing at all, so ``walsh`` vs
``walter`` is never proposed to the LLM.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

MAX_ANCHOR_EDIT = 1


def _tokens(name: str | None) -> list[str]:
    return [t for t in (name or "").strip().lower().split() if t]


def _alpha(tok: str) -> bool:
    return tok.replace("-", "").replace("'", "").isalpha()


def is_short_mention(mention: dict) -> bool:
    """A mention carrying one name token: no given name + surname pair."""
    toks = _tokens(mention.get("normalized_name"))
    if len(toks) != 1:
        return False
    return _alpha(toks[0]) and len(toks[0]) >= 2


def is_anchor_mention(mention: dict) -> bool:
    """A full name: a given name (or initial) plus a surname we can key on."""
    toks = _tokens(mention.get("normalized_name"))
    if len(toks) < 2:
        return False
    return _alpha(toks[-1]) and len(toks[-1]) >= 2


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _close(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if abs(len(a) - len(b)) > MAX_ANCHOR_EDIT:
        return False
    return _levenshtein(a, b) <= MAX_ANCHOR_EDIT


def anchor_keys(mention: dict, *, protected: set[str] | frozenset[str] = frozenset(),
                noise: set[str] | frozenset[str] = frozenset()) -> dict[str, str]:
    """The surname and first name a short token may legitimately match."""
    from engine.name_compat import resolve_surname

    toks = _tokens(mention.get("normalized_name"))
    surname = resolve_surname(
        mention.get("normalized_name"), protected=protected, noise=noise
    ) or (toks[-1] if toks else "")
    first = toks[0] if toks else ""
    return {"surname": surname, "first": first}


def resolve_short_mentions(
    mentions: list[dict],
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Attach each short mention to the unique matching anchor in its UCID.

    Returns the merge pairs to apply plus per-mention outcomes. Nothing is
    unioned here; the caller owns the union-find and the journal.
    """
    prof = (cfg or {}).get("_token_profile") or {}
    protected = frozenset(prof.get("protected") or ())
    noise = frozenset(prof.get("noise") or ())

    anchors_by_ucid: dict[str, list[dict]] = defaultdict(list)
    shorts: list[dict] = []
    for m in mentions:
        if is_short_mention(m):
            shorts.append(m)
        elif is_anchor_mention(m):
            anchors_by_ucid[m.get("ucid") or ""].append(m)

    keys: dict[str, dict[str, str]] = {}
    for group in anchors_by_ucid.values():
        for a in group:
            keys[a["mention_id"]] = anchor_keys(a, protected=protected, noise=noise)

    merges: list[dict] = []
    outcomes: dict[str, dict] = {}
    stats = {
        "short_mentions": len(shorts),
        "attached": 0,
        "abstained_no_anchor": 0,
        "abstained_ambiguous": 0,
    }

    for s in shorts:
        tok = _tokens(s.get("normalized_name"))[0]
        ucid = s.get("ucid") or ""
        candidates = anchors_by_ucid.get(ucid) or []

        matched: list[tuple[dict, str]] = []
        for a in candidates:
            k = keys[a["mention_id"]]
            if _close(tok, k["surname"]):
                matched.append((a, "surname"))
            elif _close(tok, k["first"]):
                matched.append((a, "first_name"))

        # Several mentions of one judge are one anchor for this purpose.
        distinct = {a.get("normalized_name") for a, _ in matched}
        if len(distinct) == 1:
            anchor, field = matched[0]
            merges.append(
                {
                    "short_id": s["mention_id"],
                    "anchor_id": anchor["mention_id"],
                    "short_name": s.get("normalized_name"),
                    "anchor_name": anchor.get("normalized_name"),
                    "ucid": ucid,
                    "matched_on": field,
                }
            )
            outcomes[s["mention_id"]] = {"status": "attached", "matched_on": field}
            stats["attached"] += 1
        elif not matched:
            outcomes[s["mention_id"]] = {
                "status": "abstain",
                "reason": "no_anchor_in_ucid",
                "n_anchors": len(candidates),
            }
            stats["abstained_no_anchor"] += 1
        else:
            outcomes[s["mention_id"]] = {
                "status": "abstain",
                "reason": "ambiguous_anchor",
                "candidates": sorted(distinct),
            }
            stats["abstained_ambiguous"] += 1

    return {"merges": merges, "outcomes": outcomes, "stats": stats}


def mark_anchor_outcomes(mentions: list[dict], outcomes: dict[str, dict]) -> None:
    """Tag mentions so the tiers can refuse to pair short mentions."""
    for m in mentions:
        if is_short_mention(m):
            o = outcomes.get(m["mention_id"]) or {"status": "abstain"}
            m["anchor_status"] = o.get("status")
            m["anchor_managed"] = True
            if o.get("reason"):
                m["anchor_abstain_reason"] = o["reason"]
