"""Judge-name typo duplicates (config: tier0 rule of type judge_name_typo).

Same court + same last name + first name 1-2 letters off = same judge
("rondald g morgan" -> "ronald g morgan", "igancio torteya" -> "ignacio
torteya"). Trailing junk words ("bond") and date tokens are ignored, so
"ronald g morgan bond" -> "ronald g morgan". Guards:
  - the shorter first name has at least min_first_len letters;
  - both first names start with the same letter (same_first_letter);
  - middle names agree (equal or initial) when both names have them;
  - every name it fits in its court + last name is a form of one judge;
  - two different FJC ids never join;
  - two different generational suffixes never join (Jr. vs Sr., II vs III).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from engine.normalize import GENERATIONAL_SUFFIXES

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def typo_rule(cfg: dict) -> dict | None:
    for r in (cfg.get("tier0") or {}).get("rules") or []:
        if r.get("type") == "judge_name_typo" and r.get("enabled", True):
            return r
    return None


def clean_judge_tokens(normalized: str, rule: dict) -> list[str]:
    """Drop trailing junk words / dates and generational suffixes."""
    junk = {str(w).lower() for w in rule.get("trailing_junk_words") or []}
    if rule.get("trailing_months", True):
        junk |= _MONTHS
    toks = (normalized or "").split()
    changed = True
    while changed and toks:
        changed = False
        t = toks[-1].rstrip(".").lower()
        if t in junk or any(c.isdigit() for c in t) or t in GENERATIONAL_SUFFIXES:
            toks.pop()
            changed = True
    while toks and toks[0].rstrip(".").lower() in GENERATIONAL_SUFFIXES:
        toks.pop(0)
    return toks


def judge_suffix(normalized: str) -> str:
    """Generational suffix of a judge name ("jr", "iii"), or ""."""
    toks = [t.rstrip(".").lower() for t in (normalized or "").split()]
    return next((t for t in reversed(toks) if t in GENERATIONAL_SUFFIXES), "")


def _osa(a: str, b: str, cap: int) -> int:
    """Edit distance where swapping two neighbouring letters counts as one."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _middles_ok(a: list[str], b: list[str]) -> bool:
    if not a or not b:
        return True
    if len(a) != len(b):
        return False
    return all(x == y or (len(x) == 1 and y[0] == x) or (len(y) == 1 and x[0] == y) for x, y in zip(a, b))


def typo_fit(a: list[str], b: list[str], rule: dict) -> bool:
    """True when cleaned names a and b are the same judge under the typo rule."""
    if len(a) < 2 or len(b) < 2 or a[-1] != b[-1]:
        return False
    if a == b:
        return True  # only trailing junk / suffix differed
    fa, fb = a[0], b[0]
    max_edits = int(rule.get("max_first_edits", 2))
    if fa == fb or min(len(fa), len(fb)) < int(rule.get("min_first_len", 5)):
        return False
    if rule.get("same_first_letter", True) and fa[0] != fb[0]:
        return False
    if _osa(fa, fb, max_edits) > max_edits:
        return False
    return _middles_ok(a[1:-1], b[1:-1])


def _same_judge(x: tuple, y: tuple, rule: dict) -> bool:
    if x[-1] != y[-1] or not _middles_ok(list(x[1:-1]), list(y[1:-1])):
        return False
    return x[0] == y[0] or typo_fit([x[0], x[-1]], [y[0], y[-1]], rule)


def judge_typo_merges(mentions: list[dict], rule: dict) -> list[dict[str, Any]]:
    """Return merges [{a, b, name_a, name_b, court}] for one Tier0 pass.

    Names with different FJC ids or different generational suffixes
    (Jr. vs Sr., II vs III) never join.
    """
    groups: dict[tuple, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for m in mentions:
        toks = clean_judge_tokens(m.get("normalized_name") or "", rule)
        if len(toks) >= 2 and m.get("court"):
            groups[(m["court"], toks[-1])][tuple(toks)].append(m)

    def nids(ms: list[dict]) -> set[str]:
        return {str(m["fjc_nid"]) for m in ms if m.get("fjc_nid")}

    def compatible(a: dict, b: dict) -> bool:
        sa, sb = judge_suffix(a.get("normalized_name") or ""), judge_suffix(b.get("normalized_name") or "")
        if sa and sb and sa != sb:
            return False
        na, nb = nids([a]), nids([b])
        return not (na and nb and not na & nb)

    out: list[dict[str, Any]] = []
    for (court, _), names in groups.items():
        keys = list(names)
        for k in keys:
            # identical cleaned names join each other (junk tail only)
            ms = names[k]
            for m in ms[1:]:
                if m.get("normalized_name") != ms[0].get("normalized_name") and compatible(ms[0], m):
                    out.append(dict(a=ms[0]["mention_id"], b=m["mention_id"], name_a=ms[0]["normalized_name"],
                                    name_b=m["normalized_name"], court=court, how="junk_tail"))
            fits = [
                o for o in keys
                if o != k and typo_fit(list(k), list(o), rule)
                and all(compatible(x, y) for x in ms for y in names[o])
            ]
            # Several fits are fine only when they are forms of one judge
            # ("ronald g morgan" / "ronal g morgan").
            if not fits or not all(_same_judge(x, y, rule) for x in fits for y in fits if x < y):
                continue
            if len(set().union(*(nids(names[o]) for o in fits))) > 1:
                continue
            o = max(fits, key=lambda x: len(names[x]))
            out.append(dict(a=names[o][0]["mention_id"], b=ms[0]["mention_id"], name_a=" ".join(o),
                            name_b=" ".join(k), court=court, how="first_name_typo"))
    return out
