"""Name normalization helpers (config-driven honorific stripping)."""

from __future__ import annotations

import re
from typing import Iterable


_WS = re.compile(r"\s+")
GENERATIONAL_SUFFIXES = frozenset(
    {"jr", "sr", "ii", "iii", "iv", "2nd", "3rd", "4th", "junior", "senior", "esq", "esquire"}
)


def strip_honorifics(text: str, honorifics: Iterable[str]) -> str:
    t = text.strip()
    # longest first
    ordered = sorted({h.lower().rstrip(".") for h in honorifics}, key=len, reverse=True)
    lower = t.lower()
    changed = True
    while changed:
        changed = False
        for h in ordered:
            if lower == h:
                return ""
            prefix = h + " "
            if lower.startswith(prefix):
                t = t[len(h) :].lstrip(" .")
                lower = t.lower()
                changed = True
                break
            # trailing honorific rare but strip "Name, Judge"
            suffix = " " + h
            if lower.endswith(suffix):
                t = t[: -len(h)].rstrip(" ,.")
                lower = t.lower()
                changed = True
                break
    return t


def strip_name_prefixes(text: str, prefixes: Iterable[str] | None) -> str:
    """Strip leading jurisdiction tags like 'US' / 'U.S.' from PACER header strings."""
    if not text or not prefixes:
        return text
    t = text.strip()
    ordered = sorted({p.lower().strip() for p in prefixes if p}, key=len, reverse=True)
    lower = t.lower()
    changed = True
    while changed:
        changed = False
        for p in ordered:
            if lower == p:
                return ""
            # "us clay", "u.s. clay", "u.s clay"
            for cand in (p + " ", p + "."):
                if lower.startswith(cand):
                    t = t[len(cand) :].lstrip(" .")
                    lower = t.lower()
                    changed = True
                    break
            if changed:
                break
    return t


def strip_corp_suffixes(text: str, suffixes: Iterable[str] | None) -> str:
    """Strip trailing corporate suffixes (llp, llc, …) for matching keys."""
    if not text or not suffixes:
        return text
    t = text.strip()
    ordered = sorted({s.lower().strip().rstrip(".") for s in suffixes if s}, key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        lower = t.lower()
        for s in ordered:
            for cand in (f" {s}", f" {s}."):
                if lower.endswith(cand):
                    t = t[: -len(cand)].rstrip(" ,.")
                    changed = True
                    break
            if changed:
                break
    return t


def corp_core_name(
    normalized: str,
    *,
    corp_suffixes: Iterable[str] | None = None,
    max_tokens: int | None = None,
    hyphen_to_space: bool = True,
) -> str:
    """Corporate litigant core key: hyphen→space, collapse, optional suffix strip.

    Used for Tier0/Tier1 blocking keys (config-driven). Not written to mentions
    unless extract chooses to; tiers compute on demand from normalized_name.
    """
    if not normalized:
        return ""
    t = str(normalized).lower().strip()
    if hyphen_to_space:
        t = re.sub(r"[-–—]", " ", t)
    t = _WS.sub(" ", t).strip()
    if corp_suffixes:
        t = strip_corp_suffixes(t, corp_suffixes)
        t = _WS.sub(" ", t).strip()
    toks = [tok for tok in t.split() if tok]
    if not toks:
        return ""
    if max_tokens is not None and max_tokens > 0:
        toks = toks[: int(max_tokens)]
    return " ".join(toks)


def apply_replace_tokens(text: str, mapping: dict[str, str] | None) -> str:
    if not text or not mapping:
        return text
    t = text
    for a, b in mapping.items():
        t = t.replace(str(a), str(b))
    return t


def apply_replace_regex(text: str, rules: Iterable[dict] | None) -> str:
    """Apply ordered regex substitutions from config (institutional abbreviations, etc.)."""
    if not text or not rules:
        return text
    t = text
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        pat = rule.get("pattern")
        if not pat:
            continue
        repl = rule.get("replacement", "")
        try:
            t = re.sub(pat, str(repl), t)
        except re.error:
            continue
    return t


def strip_procedural_office_prefixes(text: str, patterns: Iterable[str] | None = None) -> str:
    """Strip leading PACER procedural/admin prefixes from office_name strings.

    General rule family (not a list of firm names): leading text of the form
    ``(counsel|attorney) not admitted to … bar/court`` and close variants.
    Applied repeatedly until stable so stacked junk collapses.
    """
    if not text:
        return text
    default = (
        # "COUNSEL NOT ADMITTED TO USDC-NJ BAR\nBEASLEY ALLEN…"
        r"(?is)^\s*(?:counsel|attorney|attorneys)\s+not\s+admitted\s+to\b.*?\b(?:bar|court)\b[\s:,;\-]*",
        r"(?is)^\s*not\s+admitted\s+to\b(?:\s+the)?(?:\s+practice\s+of)?(?:\s+this)?\s*\b(?:bar|court)\b[\s:,;\-]*",
        r"(?is)^\s*(?:counsel|attorney)\s+to\s+be\s+noticed\b[\s:,;\-]*",
    )
    pats = list(patterns) if patterns is not None else list(default)
    t = str(text)
    changed = True
    while changed:
        changed = False
        for pat in pats:
            try:
                newt = re.sub(pat, "", t, count=1)
            except re.error:
                continue
            if newt != t:
                t = newt
                changed = True
                break
    return t.strip()


def apply_delete_chars(text: str, chars: str | None) -> str:
    """Remove characters entirely (no space), e.g. apostrophes so Attorney's → Attorneys."""
    if not text or not chars:
        return text
    trans = str.maketrans("", "", str(chars))
    return text.translate(trans)


def split_trailing_location(
    raw: str,
    *,
    separators: Iterable[str] | None = None,
    remainder_patterns: Iterable[str] | None = None,
) -> tuple[str, str | None]:
    """Split a generic trailing location fragment off a name.

    Uses the rightmost configured separator whose remainder matches any
    location regex. Does not encode firm names. Returns (core, location_or_none).
    """
    if not raw or not separators or not remainder_patterns:
        return raw, None
    compiled: list[re.Pattern[str]] = []
    for pat in remainder_patterns:
        try:
            compiled.append(re.compile(pat))
        except re.error:
            continue
    if not compiled:
        return raw, None
    seps = [s for s in separators if s]
    best_core, best_loc, best_pos = raw, None, None
    for sep in seps:
        pos = raw.rfind(sep)
        if pos < 0:
            continue
        core = raw[:pos].rstrip(" ,;/-")
        loc = raw[pos + len(sep) :].strip(" ,;/-")
        if not core or not loc:
            continue
        if not any(c.search(loc) for c in compiled):
            continue
        # Prefer leftmost separator so "Name\nBuilding - Suite N" peels the
        # whole address block, not only the suite after an inner dash.
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_core, best_loc = core, loc
    return best_core, best_loc


def strip_trailing_address_contamination(
    raw: str,
    *,
    patterns: Iterable[str] | None = None,
    junk_chars: str | None = None,
    min_core_tokens: int = 2,
) -> tuple[str, str | None]:
    """Peel trailing suite/room/street/courthouse fragments without a dash separator.

    Config-driven; does not hardcode firm names. Keeps a core with at least
    ``min_core_tokens`` whitespace tokens. Returns (core, stripped_fragment).
    """
    if not raw:
        return raw, None
    t = str(raw).strip()
    fragments: list[str] = []

    # Trailing junk punctuation (e.g. FLM*) — not part of the office name.
    if junk_chars:
        while t and t[-1] in junk_chars:
            fragments.append(t[-1])
            t = t[:-1].rstrip()

    compiled: list[re.Pattern[str]] = []
    for pat in patterns or []:
        try:
            compiled.append(re.compile(pat))
        except re.error:
            continue

    changed = True
    while changed and compiled:
        changed = False
        for cre in compiled:
            m = cre.search(t)
            if not m or m.end() != len(t):
                continue
            frag = t[m.start() :].strip(" ,;/-")
            core = t[: m.start()].rstrip(" ,;/-")
            core_toks = [x for x in core.split() if x]
            if len(core_toks) < min_core_tokens or not frag:
                continue
            fragments.append(frag)
            t = core
            changed = True
            break

    if not fragments:
        return raw, None
    # Fragments were peeled outermost-last; reverse for reading order.
    loc = " ".join(reversed(fragments)).strip()
    return t, loc or None


def normalize_name(
    raw: str,
    *,
    honorifics: Iterable[str],
    strip_chars: str = ".,;:\"'()[]{}",
    lowercase: bool = True,
    collapse_whitespace: bool = True,
    name_prefixes: Iterable[str] | None = None,
    corp_suffixes: Iterable[str] | None = None,
    replace_tokens: dict[str, str] | None = None,
    replace_regex: Iterable[dict] | None = None,
    delete_chars: str | None = None,
    procedural_prefix_patterns: Iterable[str] | None = None,
    strip_procedural_prefixes: bool = False,
) -> str:
    if not raw:
        return ""
    t = strip_honorifics(str(raw), honorifics)
    if strip_procedural_prefixes or procedural_prefix_patterns is not None:
        t = strip_procedural_office_prefixes(t, procedural_prefix_patterns)
    t = strip_name_prefixes(t, name_prefixes)
    t = apply_replace_tokens(t, replace_tokens)
    t = apply_replace_regex(t, replace_regex)
    t = apply_delete_chars(t, delete_chars)
    if strip_chars:
        trans = str.maketrans({c: " " for c in strip_chars})
        t = t.translate(trans)
    if collapse_whitespace:
        t = _WS.sub(" ", t).strip()
    if lowercase:
        t = t.lower()
    t = strip_corp_suffixes(t, corp_suffixes)
    if collapse_whitespace:
        t = _WS.sub(" ", t).strip()
    return t


def presentable_name(raw: str, honorifics: Iterable[str]) -> str:
    t = strip_honorifics(str(raw or ""), honorifics)
    t = _WS.sub(" ", t).strip(" .,")
    return t.title() if t else ""


def tokens(normalized: str) -> list[str]:
    return [tok for tok in (normalized or "").split() if tok]


def generational_core_name(normalized: str) -> str:
    """Drop optional leading/trailing generational suffixes for merge keys.

    ``joseph c wilkinson jr`` and ``joseph c wilkinson`` → ``joseph c wilkinson``.
    ``sr robert c brack`` → ``robert c brack``.
    """
    toks = list(tokens(normalized or ""))
    while toks and toks[0].rstrip(".").lower() in GENERATIONAL_SUFFIXES:
        toks.pop(0)
    while toks and toks[-1].rstrip(".").lower() in GENERATIONAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def surname(normalized: str) -> str:
    """Last token after stripping trailing generational suffixes (Jr., II, …)."""
    toks = list(tokens(normalized or ""))
    while toks and toks[-1].rstrip(".").lower() in GENERATIONAL_SUFFIXES:
        toks.pop()
    return toks[-1] if toks else ""


# USPS state/territory codes used when parsing office addresses.
_US_STATE_CODES = frozenset(
    {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
        "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
        "PR", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "VI", "WA", "WV", "WI",
        "WY", "GU", "MP", "AS",
    }
)
_ADDR_STATE_RE = re.compile(
    r",\s*([A-Za-z]{2})(?:\s+\d{5}(?:-\d{4})?)?\s*$"
)
_ADDR_CITY_STATE_RE = re.compile(
    r"([A-Za-z][A-Za-z .'\-]{1,40}),\s*([A-Za-z]{2})(?:\s+\d{5}(?:-\d{4})?)?\s*$"
)


def parse_office_address_geo(address: str | None) -> dict[str, str | None]:
    """Parse office_state / office_city from a PACER counsel address string.

    Scans non-empty lines from the bottom for ``City, ST ZIP`` (skipping
    PACER junk like ``NA`` / ``**NA**``). Never invents a district from the
    case court.
    """
    if not address or not str(address).strip():
        return {"office_state": None, "office_city": None}
    lines = [ln.strip() for ln in str(address).replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        lines = [str(address).strip()]

    def _clean_line(ln: str) -> str:
        ln = re.sub(r"\*+NA\*+", " ", ln, flags=re.I)
        ln = re.sub(r"(^|\s)NA(\s|$)", " ", ln, flags=re.I)
        return re.sub(r"\s+", " ", ln).strip(" ,")

    for raw in reversed(lines):
        last = _clean_line(raw)
        if not last or last.upper() in {"NA", "N/A", "NONE"}:
            continue
        m = _ADDR_CITY_STATE_RE.search(last)
        if m:
            st = m.group(2).upper()
            if st in _US_STATE_CODES:
                city = m.group(1).strip(" ,")
                return {"office_state": st, "office_city": city or None}
        m2 = _ADDR_STATE_RE.search(last)
        if m2:
            st = m2.group(1).upper()
            if st in _US_STATE_CODES:
                return {"office_state": st, "office_city": None}
    return {"office_state": None, "office_city": None}


def surname_block_keys(
    normalized: str,
    *,
    compound_surnames: Iterable[str] | None = None,
) -> list[str]:
    """Surname keys for Tier1 blocking: last token plus compound extras.

    Emits last-1 and last-2 when the trailing pair looks like a compound
    surname (config list, hyphenated last token, or two non-given tokens
    of length ≥3). Blocking only — does not merge.
    """
    from engine.common_first_names import COMMON_FIRST_NAMES

    toks = list(tokens(normalized or ""))
    while toks and toks[-1].rstrip(".").lower() in GENERATIONAL_SUFFIXES:
        toks.pop()
    if not toks:
        return []
    keys: list[str] = []

    def _add(k: str) -> None:
        k = (k or "").strip().lower()
        if k and k not in keys:
            keys.append(k)

    last = toks[-1].rstrip(".").lower()
    _add(last)
    if "-" in last:
        for part in last.split("-"):
            part = part.strip()
            if len(part) >= 2 and part.replace("'", "").isalpha():
                _add(part)

    compound_set = {
        " ".join(str(c).lower().split())
        for c in (compound_surnames or [])
        if c and str(c).strip()
    }
    if len(toks) >= 2:
        a = toks[-2].rstrip(".").lower()
        b = toks[-1].rstrip(".").lower()
        joined = f"{a} {b}"
        listed = joined in compound_set
        heuristic = (
            len(a) >= 3
            and len(b) >= 3
            and a.replace("-", "").replace("'", "").isalpha()
            and b.replace("-", "").replace("'", "").isalpha()
            and a not in COMMON_FIRST_NAMES
            and b not in COMMON_FIRST_NAMES
            and a not in GENERATIONAL_SUFFIXES
            and b not in GENERATIONAL_SUFFIXES
        )
        if listed or heuristic:
            _add(a)
            _add(joined)
    return keys


def first_last_initials(normalized: str) -> str:
    toks = tokens(normalized)
    if not toks:
        return ""
    if len(toks) == 1:
        return toks[0][:1]
    return f"{toks[0][:1]}|{toks[-1][:1]}"


def is_initial_token(tok: str) -> bool:
    t = tok.replace(".", "")
    return len(t) == 1 and t.isalpha()


def initial_token_ratio(normalized: str) -> float:
    toks = tokens(normalized)
    if not toks:
        return 1.0
    return sum(1 for t in toks if is_initial_token(t)) / len(toks)


def year_from_date(date_str: str | None) -> int | None:
    if not date_str:
        return None
    m = re.search(r"(19|20)\d{2}", str(date_str))
    return int(m.group(0)) if m else None
