"""Cascade name-compatibility gate (surname / low-info first-name policy).

Before Tier2 auto-merge or Tier3 LLM: surnames must be equal OR within
edit distance 1 OR one side is a single-token first-name-only mention
(A4). When surnames match, full given names must also agree (blocks
same-surname homonyms such as Guillermo Garcia vs Marina Garcia). Otherwise
→ deterministic NO_MATCH, method ``name_gate``.

Shared FJC NID always allows (same person, name variants).

Surname resolution
------------------
The gate cannot simply read the last whitespace token. Roughly a third of
mentions arrive with procedural text, a courtroom code, a date or a ``jr``
suffix glued on by extraction, and taking the last token then compares noise
instead of the surname. That fails in both directions: ``carlson 5/17/2016``
vs ``carlson 5/24/2016`` gets rejected, while ``… initial appearance`` vs
``… initial appearance`` gets waved through. ``resolve_surname`` walks back
past trailing non-name tokens to the real surname, and the comparison also
tolerates a surname with a dictionary word glued to it (``stengelheld``).
"""

from __future__ import annotations

import re
from typing import Any

GENERATIONAL_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "esq", "esquire"})
_VOWELS = frozenset("aeiouy")
_TOKEN_RE = re.compile(r"[^a-z0-9'-]+")


def _tokens(name: str | None) -> list[str]:
    return [t for t in _TOKEN_RE.split((name or "").lower()) if t]


def _is_courtroom_code(tok: str) -> bool:
    """Short all-consonant abbreviation such as ``bsb``, ``sff``, ``ftr``."""
    return 2 <= len(tok) <= 4 and tok.isalpha() and not (set(tok) & _VOWELS)


def _english(tok: str, words: frozenset[str] | set[str]) -> bool:
    """Dictionary lookup that tolerates inflection (``advises`` → ``advise``)."""
    if not words or not tok.isalpha():
        return False
    if tok in words:
        return True
    for suf, repl in (("s", ""), ("es", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3 and tok[: -len(suf)] + repl in words:
            return True
    return False


def build_token_profile(
    mentions: list[dict],
    fjc_surnames: set[str] | frozenset[str] | None = None,
    *,
    min_occurrences: int = 2,
) -> dict[str, frozenset[str]]:
    """Learn which tokens are surnames and which are docket prose, from the pool.

    A system dictionary cannot make this call: ``/usr/share/dict/words`` lists
    ``moore``, ``sandra`` and ``michael`` as words, while missing ``website``.
    The corpus itself is a better witness.

    - A token is a **surname** when it ends a cleanly shaped name, i.e. every
      token before it is an initial or a common given name (``michael north``,
      ``barbara l major``). Prose never reaches that slot, because something
      name-like always sits between the given name and the prose.
    - A token is **noise** when it co-occurs with several *different* judges'
      surnames (``sentencing`` follows Sanchez and Kauffman; ``web`` sits beside
      Denlow, Johnston and Finnegan) and never ends a cleanly shaped name. A
      genuine surname keeps company with its own given names, not with a crowd
      of other surnames.
    """
    from collections import defaultdict

    from engine.common_first_names import COMMON_FIRST_NAMES

    fjc_surnames = set(fjc_surnames or ())

    def given_like(t: str) -> bool:
        return len(t) == 1 or t in COMMON_FIRST_NAMES

    names = [_tokens(m.get("normalized_name")) for m in mentions]
    clean_support: dict[str, set[str]] = defaultdict(set)
    for toks in names:
        if len(toks) >= 2 and toks[-1].isalpha() and all(given_like(t) for t in toks[:-1]):
            clean_support[toks[-1]].add(" ".join(toks))
    surnames = fjc_surnames | set(clean_support)

    # Only distinctive surnames count as company. Tokens like ``james`` or
    # ``lee`` are both surnames and given names, so counting them would make
    # ``gardner`` in "james knoll gardner" look like prose.
    # Weigh the two signals: prose keeps company with several other judges'
    # surnames and rarely sits in a clean name slot, so ``ellis website`` alone
    # cannot make ``website`` a surname, while ``moore`` survives on its many
    # clean forms. Position matters — ``lee`` trails only its own given names
    # but leads a crowd of surnames, so only trailing company can condemn it.
    def score(anchors: set[str]) -> tuple[set[str], set[str]]:
        company: dict[str, set[str]] = defaultdict(set)
        trailing: dict[str, set[str]] = defaultdict(set)
        for toks in names:
            present = anchors.intersection(toks)
            for t in set(toks):
                company[t] |= present - {t}
            if toks:
                trailing[toks[-1]] |= present - {toks[-1]}

        def prose(counts: dict[str, set[str]]) -> set[str]:
            return {
                tok
                for tok, seen in counts.items()
                if len(seen) >= min_occurrences and len(seen) > len(clean_support.get(tok, ()))
            }

        return prose(company), prose(trailing)

    # Anchors are unambiguous surnames. The FJC roster counts too, otherwise
    # prose is self-concealing: "interpreter deborah berry" is not a clean name
    # precisely because "interpreter" is in it, so "berry" would never vouch
    # against it. Given names are excluded so "james knoll gardner" cannot make
    # "gardner" look like prose.
    prose_any, trailing_prose = score((set(clean_support) | fjc_surnames) - COMMON_FIRST_NAMES)

    # Prose that only ever leads ("interpreter …", "ftr …"). It is never
    # stripped — the resolver walks back from the end — but it must not be
    # mistaken for shared surname evidence between two different judges.
    # A token that stands alone as an entire mention ("denlow") is somebody's
    # name, so it is never stripped as mid-name filler — unlike "web" or
    # "interpreter", which only ever appear propping up other words.
    standalone = {toks[0] for toks in names if len(toks) == 1}
    strippable = trailing_prose | (prose_any - set(clean_support) - standalone)
    leads: dict[str, set[str]] = defaultdict(set)
    for toks in names:
        # Only name-shaped finals testify. "hurd advises" / "hurd hears
        # argument" is one judge trailed by prose, not prose leading two
        # judges, so a lead whose finals are themselves prose proves nothing.
        if len(toks) > 1 and toks[-1] not in strippable:
            leads[toks[0]].add(toks[-1])
    leading_prose = {
        tok
        for tok, finals in leads.items()
        if len(finals) >= min_occurrences and not clean_support.get(tok)
    }

    # Strippable: prose in the trailing slot, plus tokens that are prose
    # wherever they appear and never end a clean name ("web", "initial",
    # "interpreter"). Without the second group the walk-back stalls in the
    # middle of "denlow s web page" once "page" is gone.
    noise = strippable | GENERATIONAL_SUFFIXES
    return {
        "protected": frozenset(surnames - noise),
        "noise": frozenset(noise),
        "prose": frozenset(prose_any | noise | leading_prose),
        # Rewriting a stored name is less forgiving than gating a pair, so the
        # repair set is the trailing signal plus mid-name filler that the FJC
        # roster does not vouch for. That strips "web" and "interpreter" while
        # sparing "daniel" in "wiley y daniel"; "page" trails many judges, so
        # the trailing signal condemns it despite being on the roster.
        "repair_noise": frozenset(
            trailing_prose | GENERATIONAL_SUFFIXES | (noise - fjc_surnames)
        ),
    }


def resolve_surname(
    name: str | None,
    *,
    words: frozenset[str] | set[str] | None = None,
    protected: frozenset[str] | set[str] | None = None,
    noise: frozenset[str] | set[str] | None = None,
) -> str:
    """Last token that plausibly is a surname.

    Drops trailing generational suffixes, non-alphabetic junk (dates, docket
    fragments), single letters, courtroom codes, and tokens the corpus profile
    marked as noise. Never strips the only remaining token.
    """
    protected = protected or frozenset()
    noise = noise or frozenset()
    toks = _tokens(name)
    while len(toks) > 1:
        t = toks[-1]
        if t in protected:
            break
        drop = (
            t in GENERATIONAL_SUFFIXES
            or not t.replace("-", "").replace("'", "").isalpha()
            or len(t) < 2
            or _is_courtroom_code(t)
            or t in noise
        )
        if not drop:
            break
        toks.pop()
    return toks[-1] if toks else ""


def _glued_variant(
    sa: str,
    sb: str,
    words: frozenset[str] | set[str],
    protected: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """True when the two surnames are one name with junk glued onto it.

    Covers ``stengel``/``stengelheld``, ``wells``/``wellsinitial``,
    ``wilder``/``wilder-doomes`` and ``baylsontelephone``/``baylsonviolation``.
    The glued remainder must look like a word rather than a name ending, so
    ``smith``/``smithson`` stays incompatible.
    """
    short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    if not short:
        return False

    if long_.startswith(short):
        rest = long_[len(short) :]
        if rest.startswith("-") and len(rest) > 1:
            return True
        if len(rest) >= 4 and (_english(rest, words) or short in protected):
            return True

    # Neither contains the other, but both extend a shared real stem.
    common = 0
    for ca, cb in zip(sa, sb):
        if ca != cb:
            break
        common += 1
    if common >= 5:
        ra, rb = sa[common:], sb[common:]
        if ra and rb and _english(ra, words) and _english(rb, words):
            return True
    return False


def shared_distinctive_token(
    a: str | None,
    b: str | None,
    noise: frozenset[str] | set[str],
) -> str | None:
    """A token both names carry that could only be a shared surname.

    Excludes given names and corpus noise, so ``jeffrey cole`` and
    ``jeffrey cummings`` share nothing, while ``finnegan s web page`` and
    ``finnegan s proposed`` share ``finnegan``.
    """
    from engine.common_first_names import COMMON_FIRST_NAMES

    ta, tb = set(_tokens(a)), set(_tokens(b))
    for t in sorted(ta & tb, key=len, reverse=True):
        if len(t) < 3 or not t.replace("-", "").replace("'", "").isalpha():
            continue
        if t in COMMON_FIRST_NAMES or t in GENERATIONAL_SUFFIXES or t in noise:
            continue
        return t
    return None


def _levenshtein(a: str, b: str) -> int:
    a, b = a or "", b or ""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > 1 and min(len(a), len(b)) > 0:
        # quick reject when clearly >1 (still compute for short strings)
        pass
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _primary_given_token(
    name: str | None,
    *,
    protected: frozenset[str] | set[str] | None = None,
    noise: frozenset[str] | set[str] | None = None,
) -> str | None:
    """First token used as given-name evidence when name-shaped."""
    toks = _tokens(name)
    if not toks:
        return None
    t = toks[0]
    if t in GENERATIONAL_SUFFIXES:
        return None
    rs = resolve_surname(name, protected=protected or frozenset(), noise=noise or frozenset())
    if len(toks) >= 3:
        return t if len(t) > 1 else None
    if len(toks) == 2 and toks[-1] == rs and len(t) > 1:
        if protected and t in protected:
            return None
        return t
    return None


def truncated_span_missing_given(
    a_name: str | None,
    b_name: str | None,
    *,
    protected: frozenset[str] | set[str] | None = None,
    noise: frozenset[str] | set[str] | None = None,
) -> tuple[bool, str]:
    """Block merges when one side is a truncated span lacking the other's given name."""

    def is_truncated_span(name: str | None) -> bool:
        toks = _tokens(name)
        if len(toks) != 2:
            return False
        rs = resolve_surname(name, protected=protected or frozenset(), noise=noise or frozenset())
        return toks[-1] == rs and bool(protected and toks[0] in protected)

    ga = _primary_given_token(a_name, protected=protected, noise=noise)
    gb = _primary_given_token(b_name, protected=protected, noise=noise)
    ta, tb = set(_tokens(a_name)), set(_tokens(b_name))
    if ga and not gb and is_truncated_span(b_name):
        if ta.issubset(tb):
            return False, ""
        if ga not in tb:
            return True, f"truncated_span_missing_given:{ga}"
    if gb and not ga and is_truncated_span(a_name):
        if ta.issubset(tb):
            return False, ""
        if gb not in ta:
            return True, f"truncated_span_missing_given:{gb}"
    return False, ""


def given_names_incompatible(
    a_name: str | None,
    b_name: str | None,
    *,
    protected: frozenset[str] | set[str] | None = None,
    noise: frozenset[str] | set[str] | None = None,
) -> tuple[bool, str]:
    """True when both sides carry full given names that clearly disagree.

    Blocks same-surname homonyms (Guillermo Garcia vs Marina Garcia) while
    allowing initial expansions, nickname variants (steven/steve), and
    truncated spans (garcia marmolejo vs marina garcia marmolejo).
    """
    ga = _primary_given_token(a_name, protected=protected, noise=noise)
    gb = _primary_given_token(b_name, protected=protected, noise=noise)
    if not ga or not gb:
        return False, ""
    if len(ga) <= 1 or len(gb) <= 1:
        return False, ""
    if ga == gb:
        return False, ""
    if ga.startswith(gb) or gb.startswith(ga):
        return False, ""
    ta, tb = set(_tokens(a_name)), set(_tokens(b_name))
    ra = resolve_surname(a_name, protected=protected or frozenset(), noise=noise or frozenset())
    rb = resolve_surname(b_name, protected=protected or frozenset(), noise=noise or frozenset())
    if ga in tb - {(_tokens(b_name) or [""])[0], rb}:
        return False, ""
    if gb in ta - {(_tokens(a_name) or [""])[0], ra}:
        return False, ""
    return True, f"incompatible_given:{ga}|{gb}"


def is_first_name_only_mention(mention: dict) -> bool:
    """Single-token mention tagged as low-info / first-name-only (A4)."""
    nn = (mention.get("normalized_name") or "").strip()
    toks = nn.split()
    if len(toks) != 1:
        return False
    if mention.get("low_info_first_name") or mention.get("hygiene_scope") == "same_ucid":
        return True
    # Fallback for mentions carrying no hygiene metadata. A bare surname
    # ("smith") is not low-information and must still face the surname check.
    from engine.common_first_names import COMMON_FIRST_NAMES

    return toks[0] in COMMON_FIRST_NAMES


def surnames_compatible(
    sa: str,
    sb: str,
    *,
    max_edit: int = 1,
    words: frozenset[str] | set[str] | None = None,
    protected: frozenset[str] | set[str] | None = None,
) -> bool:
    sa = (sa or "").strip().lower()
    sb = (sb or "").strip().lower()
    if not sa or not sb:
        # Missing surname — do not hard-reject (let other rails handle)
        return True
    if sa == sb:
        return True
    if _levenshtein(sa, sb) <= max_edit:
        return True
    return _glued_variant(sa, sb, words or frozenset(), protected or frozenset())


def _firm_content_tokens(name: str | None) -> list[str]:
    stop = frozenset(
        {
            "and",
            "of",
            "the",
            "for",
            "to",
            "a",
            "an",
            "in",
            "at",
            "llp",
            "llc",
            "pc",
            "pa",
            "ltd",
            "inc",
            "pllc",
            "lpa",
            "law",
            "firm",
            "office",
            "offices",
            "associates",
            "group",
            "p",
            "c",
        }
    )
    return [
        t
        for t in _tokens(name)
        if len(t) >= 2 and t not in stop and t not in GENERATIONAL_SUFFIXES
    ]


def _firm_content_bigrams(name: str | None) -> set[str]:
    toks = _firm_content_tokens(name)
    return {f"{toks[i]} {toks[i + 1]}" for i in range(len(toks) - 1)}


def _norm_phone(phone: str | None) -> str | None:
    digits = re.sub(r"\D+", "", phone or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) >= 7 else None


def _norm_domain(domain: str | None) -> str | None:
    d = (domain or "").strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d or None


def _norm_address(address: str | None) -> str | None:
    a = re.sub(r"\s+", " ", (address or "").strip().lower())
    a = re.sub(r"[^\w\s]", "", a)
    return a if len(a) >= 8 else None


def firm_contacts_all_differ(a: dict, b: dict) -> bool:
    """True when domain, phone, and address each disagree.

    A present value vs missing counts as disagreement. Both-missing on any
    one field means we cannot claim all three differ → False.
    """

    def differs(va: str | None, vb: str | None) -> bool | None:
        if not va and not vb:
            return None
        if not va or not vb:
            return True
        return va != vb

    diffs = [
        differs(_norm_domain(a.get("domain")), _norm_domain(b.get("domain"))),
        differs(_norm_phone(a.get("phone")), _norm_phone(b.get("phone"))),
        differs(_norm_address(a.get("address")), _norm_address(b.get("address"))),
    ]
    if any(d is None for d in diffs):
        return False
    return all(bool(d) for d in diffs)


def firm_surname_only_contacts_block(a: dict, b: dict, *, cfg: dict | None = None) -> tuple[bool, str]:
    """Class 3 firms rail: weak name overlap + disagreeing contacts → refuse.

    When domain, phone, and address all differ, a single shared partner/surname
    token is not firm identity. Require either exact normalized equality or at
    least one shared content bigram (e.g. ``reed smith``, ``nelson mullins``).
    """
    cfg = cfg or {}
    if cfg.get("entity_type") != "firm":
        return False, ""
    gate = cfg.get("name_compat") or {}
    if gate.get("refuse_surname_only_when_contacts_differ", True) is False:
        return False, ""
    if not firm_contacts_all_differ(a, b):
        return False, ""
    na = (a.get("normalized_name") or "").strip().lower()
    nb = (b.get("normalized_name") or "").strip().lower()
    if not na or not nb:
        return False, ""
    if na == nb:
        return False, ""
    if _firm_content_bigrams(na) & _firm_content_bigrams(nb):
        return False, ""
    shared = set(_firm_content_tokens(na)) & set(_firm_content_tokens(nb))
    token = sorted(shared, key=len, reverse=True)[0] if shared else ""
    return True, f"surname_only_contacts_differ:{token or 'none'}"


def all_initial_tokens(name: str | None) -> list[str] | None:
    """Return tokens when every token is a single-letter initial; else None."""
    from engine.normalize import is_initial_token, tokens as norm_tokens

    toks = norm_tokens(name or "")
    if len(toks) < 2:
        return None
    if all(is_initial_token(t) for t in toks):
        return toks
    return None


def person_initials_expand_compatible(a: dict, b: dict) -> bool:
    """True when an all-initials person name is a subsequence of the other's initials.

    ``C. M.`` fits ``Corey Mitchell``; ``I. M.`` does not. Extra middle initials
    on the short side (``H. R. M.`` vs ``Humberto Martinez``) do not fit.
    """
    from engine.normalize import tokens as norm_tokens

    ia = all_initial_tokens(a.get("normalized_name"))
    ib = all_initial_tokens(b.get("normalized_name"))
    if ia and ib:
        return ia[0] == ib[0] and ia[-1] == ib[-1]
    if ia and not ib:
        full = [t[0] for t in norm_tokens(b.get("normalized_name") or "") if t]
        return _initials_fit(ia, full)
    if ib and not ia:
        full = [t[0] for t in norm_tokens(a.get("normalized_name") or "") if t]
        return _initials_fit(ib, full)
    return False


def _initials_fit(short: list[str], full: list[str]) -> bool:
    if not short or not full:
        return False
    if short[0] != full[0] or short[-1] != full[-1]:
        return False
    fi = 0
    for ch in short:
        found = False
        while fi < len(full):
            if full[fi] == ch:
                found = True
                fi += 1
                break
            fi += 1
        if not found:
            return False
    return True


def all_initials_incompatible(a: dict, b: dict, *, cfg: dict | None = None) -> tuple[bool, str]:
    """Bare-initial person names with disagreeing first/last initials → incompatible.

    Config: ``name_compat.all_initials_must_agree`` plus optional
    ``all_initials_office_classes``. Off by default (judges/firms unchanged).
    """
    cfg = cfg or {}
    gate = cfg.get("name_compat") or {}
    if not gate.get("all_initials_must_agree"):
        return False, ""
    classes = {str(x) for x in (gate.get("all_initials_office_classes") or [])}
    if classes:
        ca, cb = a.get("office_class") or "", b.get("office_class") or ""
        if ca not in classes and cb not in classes:
            return False, ""
    ia = all_initial_tokens(a.get("normalized_name"))
    ib = all_initial_tokens(b.get("normalized_name"))
    if not ia or not ib:
        return False, ""
    if ia == ib:
        return False, ""
    if ia[0] != ib[0] or ia[-1] != ib[-1]:
        return True, f"incompatible_initials:{''.join(ia)}|{''.join(ib)}"
    return False, ""


_PUNCT_SPLIT_RE = re.compile(r"[-–—/.,'’]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_PUNCT_CONNECTORS = frozenset({"and", "of", "the", "for"})


def _punct_fold_spaced(name: str | None) -> str:
    """Hyphen/slash/period/ampersand → spaces. Apostrophe drops, not a token break."""
    t = (name or "").lower().replace("&", " ")
    t = t.replace("'", "").replace("’", "")
    t = _PUNCT_SPLIT_RE.sub(" ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _punct_fold_tokens(name: str | None, suffixes: list[str] | None) -> list[str]:
    from engine.normalize import strip_corp_suffixes

    spaced = _punct_fold_spaced(name)
    if suffixes:
        spaced = strip_corp_suffixes(spaced, suffixes)
        spaced = re.sub(r"\s+", " ", spaced).strip()
    return [tok for tok in spaced.split() if tok and tok not in _PUNCT_CONNECTORS]


def _punct_fold_collapsed_and(name: str | None, suffixes: list[str] | None) -> str:
    """Space-free key that keeps ``and`` so ``a c and s`` equals ``acands``."""
    from engine.normalize import strip_corp_suffixes

    t = (name or "").lower().replace("&", "and")
    t = t.replace("'", "").replace("’", "")
    t = _PUNCT_SPLIT_RE.sub(" ", t)
    t = _NON_ALNUM_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if suffixes:
        t = strip_corp_suffixes(t, suffixes)
        t = re.sub(r"\s+", " ", t).strip()
    return t.replace(" ", "")


def punct_fold_compatible(a: dict, b: dict, *, cfg: dict | None = None) -> tuple[bool, str]:
    """Org-name punctuation/spacing variants are the same legal name.

    Surname resolution treats ``harbison-walker`` and ``walker`` as different
    surnames once a trailing industry word is noise. That is a person-gate
    artifact, not a corporate identity signal. Gated off unless
    ``name_compat.punct_fold`` is set (parties only). Does not apply to
    person or placeholder classes, so Sr/Jr, USA/Doe, and Group-token
    remainders stay on their existing rails.
    """
    cfg = cfg or {}
    gate = cfg.get("name_compat") or {}
    if not gate.get("punct_fold"):
        return False, ""
    classes = {str(x) for x in (gate.get("punct_fold_office_classes") or [])}
    if not classes:
        return False, ""
    ca, cb = a.get("office_class") or "", b.get("office_class") or ""
    if ca not in classes or cb not in classes:
        return False, ""

    suffixes = list((cfg.get("normalization") or {}).get("strip_corp_suffixes") or [])
    ta = _punct_fold_tokens(a.get("normalized_name") or a.get("surname"), suffixes)
    tb = _punct_fold_tokens(b.get("normalized_name") or b.get("surname"), suffixes)
    if not ta or not tb:
        return False, ""
    if ta == tb:
        return True, "punct_fold_tokens"

    ja, jb = "".join(ta), "".join(tb)
    min_len = int(gate.get("punct_fold_min_collapsed_len", 6))
    if ja and ja == jb and len(ja) >= min_len:
        return True, "punct_fold_collapsed"

    ca_ = _punct_fold_collapsed_and(a.get("normalized_name") or a.get("surname"), suffixes)
    cb_ = _punct_fold_collapsed_and(b.get("normalized_name") or b.get("surname"), suffixes)
    if ca_ and ca_ == cb_ and len(ca_) >= min_len:
        return True, "punct_fold_collapsed"
    if _initialism_glue(ta, tb):
        return True, "punct_fold_initialism"
    return False, ""


def _initialism_glue(ta: list[str], tb: list[str]) -> bool:
    """``c s r`` / ``csr`` and ``u s a`` / ``usa``. Short, so not the general collapse."""
    def glued(short: list[str], long_: list[str]) -> bool:
        if len(short) != 1 or len(short[0]) < 2:
            return False
        if not long_ or any(len(t) != 1 for t in long_):
            return False
        return "".join(long_) == short[0]

    return glued(ta, tb) or glued(tb, ta)


def names_compatible(a: dict, b: dict, *, cfg: dict | None = None) -> tuple[bool, str]:
    """Return (ok, reason). ok=False → caller must NO_MATCH via name_gate."""
    cfg = cfg or {}
    gate = cfg.get("name_compat") or {}
    if gate.get("enabled", True) is False:
        return True, "disabled"

    # Shared FJC NID → always compatible
    na, nb = a.get("fjc_nid"), b.get("fjc_nid")
    if na and nb and str(na) == str(nb):
        return True, "shared_fjc_nid"

    bad_init, init_reason = all_initials_incompatible(a, b, cfg=cfg)
    if bad_init:
        return False, init_reason

    # A4: first-name-only may only attach same-UCID (enforced elsewhere);
    # for surname check, treat as compatible so UCID-local merges can proceed.
    if is_first_name_only_mention(a) or is_first_name_only_mention(b):
        return True, "first_name_only_side"

    folded, fold_reason = punct_fold_compatible(a, b, cfg=cfg)
    if folded:
        return True, fold_reason

    words = _gate_wordlist(gate)
    protected, noise = _gate_profile(cfg)
    na_name = a.get("normalized_name") or a.get("surname")
    nb_name = b.get("normalized_name") or b.get("surname")
    sa = resolve_surname(na_name, protected=protected, noise=noise)
    sb = resolve_surname(nb_name, protected=protected, noise=noise)
    max_edit = int(gate.get("max_surname_edit_distance", 1))
    if surnames_compatible(sa, sb, max_edit=max_edit, words=words, protected=protected):
        bad_given, given_reason = given_names_incompatible(
            na_name, nb_name, protected=protected, noise=noise
        )
        if bad_given:
            return False, given_reason
        bad_trunc, trunc_reason = truncated_span_missing_given(
            na_name, nb_name, protected=protected, noise=noise
        )
        if bad_trunc:
            return False, trunc_reason
        blocked, block_reason = firm_surname_only_contacts_block(a, b, cfg=cfg)
        if blocked:
            return False, block_reason
        return True, "surname_ok"

    # Trailing noise can leave the resolved surnames disagreeing even though
    # both names carry the same distinctive token (finnegan s web page /
    # finnegan s proposed). A shared non-noise, non-given-name token is
    # surname evidence in its own right.
    shared = shared_distinctive_token(na_name, nb_name, _gate_prose(cfg))
    if shared:
        bad_given, given_reason = given_names_incompatible(
            na_name, nb_name, protected=protected, noise=noise
        )
        if bad_given:
            return False, given_reason
        bad_trunc, trunc_reason = truncated_span_missing_given(
            na_name, nb_name, protected=protected, noise=noise
        )
        if bad_trunc:
            return False, trunc_reason
        blocked, block_reason = firm_surname_only_contacts_block(a, b, cfg=cfg)
        if blocked:
            return False, block_reason
        return True, f"shared_surname_token:{shared}"

    return False, f"cross_surname:{sa}|{sb}"


def _gate_wordlist(gate: dict) -> frozenset[str]:
    """Only used for the glued-surname test (stengel / stengelheld)."""
    if gate.get("resolve_surname", True) is False:
        return frozenset()
    from engine.name_validity import load_english_wordlist

    return load_english_wordlist()


def _gate_profile(cfg: dict) -> tuple[frozenset[str], frozenset[str]]:
    """Corpus token profile, populated per-run by the cascade."""
    prof = cfg.get("_token_profile") or {}
    return frozenset(prof.get("protected") or ()), frozenset(prof.get("noise") or ())


def _gate_prose(cfg: dict) -> frozenset[str]:
    """Tokens that are docket prose anywhere in a name, not just trailing."""
    prof = cfg.get("_token_profile") or {}
    return frozenset(prof.get("prose") or prof.get("noise") or ())


def _name_gate_signal(reason: str) -> str:
    if reason.startswith("surname_only_contacts_differ"):
        return "surname_only_contacts_differ"
    if reason.startswith("cross_surname"):
        return "cross_surname"
    if reason.startswith("incompatible_initials"):
        return "incompatible_initials"
    return "incompatible_given"


def name_gate_decision_row(
    a: dict,
    b: dict,
    *,
    cfg: dict,
    reason: str,
    sim: float | None = None,
    extra_evidence: dict | None = None,
) -> dict[str, Any]:
    """Journal row for deterministic name_gate NO_MATCH."""
    protected, noise = _gate_profile(cfg)
    ev: dict[str, Any] = {
        "name_gate_reason": reason,
        "surname_a": a.get("surname"),
        "surname_b": b.get("surname"),
        "resolved_surname_a": resolve_surname(a.get("normalized_name"), protected=protected, noise=noise),
        "resolved_surname_b": resolve_surname(b.get("normalized_name"), protected=protected, noise=noise),
        "normalized_a": a.get("normalized_name"),
        "normalized_b": b.get("normalized_name"),
    }
    if sim is not None:
        ev["embedding_similarity"] = sim
    if extra_evidence:
        ev.update(extra_evidence)
    return {
        "entity_type": cfg.get("entity_type"),
        "decision": "NO_MATCH",
        "confidence": 100,
        "method": "name_gate",
        "rationale": (
            f"Name-compatibility gate: incompatible names ({reason}). "
            "LLM not called."
        ),
        "signals": [
            "name_gate",
            _name_gate_signal(reason),
        ],
        "evidence": ev,
        "config_version": cfg.get("version"),
    }
