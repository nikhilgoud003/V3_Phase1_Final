"""Repair mention names that carry docket prose past the surname.

The name gate could already *see* through trailing prose, but the mention kept
its polluted name on disk, so ``juan r sanchez sentencing`` and
``juan r sanchez`` remained two different names in every downstream artifact —
and any honest last-token audit of a final entity reported two surnames for one
judge. Repairing the stored name instead makes the pool self-consistent: Tier0
exact keys match more often, and the acceptance metric needs no exemptions.

Runs as a second pass over the pool, because the corpus token profile can only
be learned once every mention is known.
"""

from __future__ import annotations

from typing import Any


def _tokens(name: str | None) -> list[str]:
    return [t for t in (name or "").strip().lower().split() if t]


def _alpha(tok: str) -> bool:
    return tok.replace("-", "").replace("'", "").isalpha()


def repair_name(
    name: str | None,
    *,
    protected: frozenset[str] | set[str],
    noise: frozenset[str] | set[str],
    words: frozenset[str] | set[str] = frozenset(),
    fjc_surnames: frozenset[str] | set[str] = frozenset(),
) -> str:
    """Trim trailing prose, keeping the name through its real surname."""
    from engine.name_compat import GENERATIONAL_SUFFIXES

    toks = _tokens(name)
    if len(toks) < 2:
        return " ".join(toks)

    # A free-standing dash separates the judge from an annotation
    # ("claire v eagan - expediting"). Hyphenated surnames carry no spaces, so
    # this cannot touch "wilder-doomes".
    if "-" in toks[1:]:
        toks = toks[: toks.index("-", 1)]
        if len(toks) < 2:
            return " ".join(toks)

    # Keep generational suffixes attached; they belong to the person.
    tail_suffixes: list[str] = []
    while len(toks) > 1 and toks[-1] in GENERATIONAL_SUFFIXES:
        tail_suffixes.insert(0, toks.pop())

    while len(toks) > 1:
        t = toks[-1]
        if t in protected:
            break
        drop = (
            not _alpha(t)
            or len(t) < 2
            or t in noise
        )
        if not drop:
            break
        toks.pop()

    # A surname glued to a long prose word: "stengelheld", "baylsontelephone".
    # These forms can themselves enter the protected set (they look like a
    # clean surname), so we still try to decompose them. Remainder must look
    # like English of ≥4 letters and the stem ≥5, so "westmore" and
    # "beckerman" stay intact.
    _GLUE_TAILS = frozenset(
        {
            "held",
            "telephone",
            "violation",
            "initial",
            "website",
            "pretrial",
            "sentencing",
            "expediting",
            "granting",
        }
    )
    if toks:
        last = toks[-1]
        # Never amputate a surname that appears on the FJC roster.
        if last in fjc_surnames:
            pass
        elif _alpha(last) and len(last) > 8:
            for cut in range(len(last) - 4, 4, -1):
                head, rest = last[:cut], last[cut:]
                if (
                    head in protected
                    and len(head) >= 5
                    and len(rest) >= 4
                    and (
                        rest in noise
                        or rest in words
                        or rest in _GLUE_TAILS
                    )
                    and rest not in protected
                ):
                    toks[-1] = head
                    break
                # The glued form itself may be protected; still prefer the
                # shorter real surname when the remainder is known glue.
                if (
                    last in protected
                    and head in protected
                    and len(head) >= 5
                    and (rest in _GLUE_TAILS or rest in words)
                    and rest not in protected
                ):
                    toks[-1] = head
                    break

    # Trailing lone initials ("johnston s") and dangling hyphens ("wilder-")
    # carry no surname of their own.
    if toks:
        toks[-1] = toks[-1].rstrip("-")
    while len(toks) > 1 and len(toks[-1]) < 2:
        toks.pop()

    return " ".join(t for t in toks + tail_suffixes if t)


def repair_mention_names(mentions: list[dict], cfg: dict | None = None) -> dict[str, Any]:
    """Rewrite polluted ``normalized_name`` values in place; report the changes."""
    from engine.normalize import surname as surname_of
    from engine.normalize import tokens as tokens_of
    from engine.tiers import build_token_profile_for_run, resolve_path

    from engine.name_validity import load_english_wordlist, load_fjc_name_sets, strip_courtroom_suffix

    profile = build_token_profile_for_run(mentions, cfg or {})
    protected = profile["protected"]
    noise = profile.get("repair_noise") or profile["noise"]
    words = load_english_wordlist(
        ((cfg or {}).get("name_validity") or {}).get("english_wordlist_path")
    ) - protected

    fjc_surnames: set[str] = set()
    try:
        nrm = (cfg or {}).get("normalization") or {}
        fjc_path = resolve_path(cfg or {}, "data/judges_fjc.csv")
        fjc_surnames, _ = load_fjc_name_sets(
            fjc_path, nrm.get("strip_honorifics") or [], nrm.get("strip_chars") or ""
        )
    except Exception:
        fjc_surnames = set()

    changes: dict[tuple[str, str], int] = {}
    n_changed = 0
    for m in mentions:
        raw = m.get("normalized_name") or ""
        fixed = repair_name(
            raw,
            protected=protected,
            noise=noise,
            words=words,
            fjc_surnames=fjc_surnames,
        )
        # Removing prose can expose a courtroom code that was not last before
        # ("… mansfield bbb certificate" → "… mansfield bbb").
        fixed = strip_courtroom_suffix(fixed)
        if fixed and fixed != raw.strip().lower():
            m["name_repair_from"] = raw
            m["normalized_name"] = fixed
            m["surname"] = surname_of(fixed)
            m["token_count"] = len(tokens_of(fixed))
            changes[(raw, fixed)] = changes.get((raw, fixed), 0) + 1
            n_changed += 1

    return {
        "mentions_repaired": n_changed,
        "distinct_repairs": len(changes),
        "examples": sorted(
            ({"from": a, "to": b, "count": c} for (a, b), c in changes.items()),
            key=lambda r: -r["count"],
        )[:40],
        "token_profile": {"protected": len(protected), "noise": len(noise)},
    }
