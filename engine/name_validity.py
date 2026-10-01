"""Config-driven name-validity gate for extracted judge mentions.

Principled rules (Phase A2/A3):
- Reject when ALL tokens are common-English dictionary words UNLESS the
  surname is in the FJC surname list OR appears in the corpus inside a
  multi-token judge name (rescue).
- Leading single-letter initial is VALID when followed by ≥1 multi-letter
  token (``j thomas marten`` survives; ``jr mcguire`` does not).
- Strip trailing non-name / procedural tokens; quarantine with reasons —
  never silent drop.
"""

from __future__ import annotations

import csv
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.run_cache import file_memo

DEFAULT_GENERATIONAL_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "esq", "esquire"}

# Trailing fragments often glued by NER after a real name
DEFAULT_TRAILING_NON_NAME = {
    "added", "is", "so", "caused", "and", "in", "the", "a", "an", "of", "to",
    "for", "on", "at", "by", "with", "from", "as", "or", "that", "this",
    "defendant", "plaintiff", "counsel", "attorney", "court", "order",
    "motion", "case", "matter", "regarding", "concerning", "pursuant",
}

DEFAULT_PROCEDURAL_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "from", "with", "as", "is", "be", "are", "was", "were", "this", "that",
    "these", "those", "it", "its", "no", "not", "all", "any", "per", "via",
    "re", "vs", "v", "eu", "assignment", "consent", "pursuant", "jurisdiction",
    "motion", "motions", "order", "orders", "notice", "notices", "referral",
    "reassignment", "designation", "transfer", "form", "forms", "packet",
    "questionnaire", "standing", "minute", "minutes", "entry", "entries",
    "case", "cases", "matter", "matters", "above", "undersigned", "presiding",
    "panel", "bench", "court", "courts", "district", "magistrate", "judge",
    "judges", "honorable", "hon", "signed", "referred", "refer", "assigned",
    "assign", "unassigned", "filed", "filing", "entered", "enter", "calendar",
    "docket", "cause", "action", "party", "parties", "plaintiff", "defendant",
    "counsel", "attorney", "attorneys", "clerk", "status", "hearing",
    "conference", "settlement", "discovery", "complaint", "answer", "brief",
    "memorandum", "opinion", "judgment", "sentence", "without", "prejudice",
    "leave", "proceed", "proceeding", "proceedings", "regarding", "concerning",
    "under", "upon", "within", "thereafter", "herein", "thereof", "therein",
    "hereof", "jury", "room", "courtroom", "chamber", "chambers", "jkl", "who",
    "will", "shall", "has", "have", "findings", "recommendation", "text",
    "longer", "available", "dismissed", "vacated", "withdrawn", "present",
    "before", "after", "signs", "terminating", "terminate", "update", "updated",
    "updating", "set", "respect", "because", "only", "rules", "miscellaneous",
    "granted", "denied", "so", "added", "caused", "accepts",
    # Docket NER / procedural conference boilerplate (research_dev gaps)
    "telephonic", "scheduling", "notify", "notifies", "participates", "disposition",
    "calendar", "accept", "issues", "conference", "conferenced", "status", "indicated",
    "memos", "sentencing", "vjdistrict", "none", "cri", "criminal", "either",
    "designated", "adopted", "initials", "usca", "cvrecovery", "crstatistics",
    "executive", "committee", "debt", "duty", "recovery", "statistics", "longer",
    "generally", "conducts", "should", "either", "letter", "jurisdition", "recommended",
    "ruling", "administrative", "issued", "whose", "parties", "call", "jeg",
    # NOTE: do NOT put "page"/"web" here — kills real names like Denise Page Hood.
    # Website boilerplate is handled by phrase patterns + trailing-only lexemes.
}

# Strip ONLY from the end (never cut mid-name). Class B / Class A glue.
DEFAULT_TRAILING_LEXEMES = {
    "plea", "modified", "trial", "rules", "dtd", "see", "added", "attached",
    "pre-trial", "pretrial", "certificate", "spanish", "copies", "finding",
    "staff", "requirements", "pointers", "practice", "operating", "procedures",
    "signature", "dismissing", "corrected", "memorandum", "recently", "management",
    "civil", "ps", "advises", "advise", "decision", "decisions", "schedules",
    "schedule", "individual", "located", "found", "requires", "compliance",
    "full", "practices", "retaining", "website", "webpage", "web", "page",
    "further", "until", "continued", "needed", "interpreter", "county",
    # Docket NER tails: "Stephen Smith Appearances", "Olvera recused",
    # "dos Santos Dispositive", "Olvera Deft remanded"
    "appearances", "appearance", "recused", "dispositive", "remanded", "deft",
}

# research_dev _judge_extract.py — full-string noise + legal-document token overlap
DEFAULT_NOISE_FULL_NAMES = {
    "on", "in", "by", "for", "set", "at", "cc", "re", "with", "of", "matters",
}

DEFAULT_LEGAL_DOCUMENT_TERMS = {
    w.lower()
    for w in {
        "Hearing", "Appearance", "Courtroom", "Conference", "Entry", "Order", "Defender",
        "Affidavit", "Jury", "Sheet", "Annotation", "Release", "Document",
        "Information", "Pretrial", "Bail", "Motion", "Detention", "Warrant", "Felony",
        "Summons", "Judgment", "Decree", "Complaint", "Proceeding", "Report", "Brief",
        "Minute", "Defendant", "Plaintiff", "Proposed", "Filing",
    }
}

# research_dev extract_header_judges.py droppers (+ telephonic/update variants)
DEFAULT_HEADER_DROP_PHRASES = {
    "ebc",
    "executive committee",
    "unassigned",
    "magistrate judge",
    "magistrate judge unassigned magistrate",
    "judge unassigned judge",
    "judge unassigned",
    "cvb judge",
    "debt-magistrate",
    "mia duty magistrate",
    "cvrecovery case crstatistics unassigned",
    "civ",
    "mdl",
    "telephonic scheduling",
    "telephonic status conference",
    "telephonic scheduling conference",
    "update in case",
    "unassigned disposition",
    "unassigned - cri",
    "none unassigned vjdistrict",
    "who participates",
    "who participates in the",
    "will notify parties",
    "will notify parties of",
    "calendar call",
    "sentencing memos",
    "status hearing set",
    "whose name is indicated",
    "whose name",
    "has set this case",
    "granted",
    "schedules page",
}

_HEADER_CAUSE_RE = re.compile(r"\s*cause:\s*", flags=re.I)
_HEADER_REFERRED_RE = re.compile(r"\s*referred\s*(?:to)?\s*", flags=re.I)
_HEADER_DESIGNATED_RE = re.compile(r"^designated\s*", flags=re.I)
_HEADER_DASH_MJ_RE = re.compile(r"-mj\s*$", flags=re.I)
_DATE_OR_CODE_RE = re.compile(r"(?i)^(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|blm\d+)$")
_CASE_NUMBER_RE = re.compile(
    r"(?i)\b\d:\d{2}-(?:cv|cr)-\d+|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
)
_PAREN_CODE_CUT_RE = re.compile(r"\.\(|\)\(")
_BARE_PAREN_TAIL_RE = re.compile(r"\)[A-Za-z*\[]")
_POSSESSIVE_CONSTRUCTION_RE = re.compile(
    r"(?i)\b([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z]\.?){0,3}\s+[A-Za-z][A-Za-z'\-]*)['']s\s+\S+"
)
_POSSESSIVE_SIMPLE_RE = re.compile(r"(?i)\b([A-Za-z][A-Za-z'\-]*)['']s\s+\S+")
_INTERPRETER_RE = re.compile(r"(?i)^interpreter\b")
_FRAGMENT_RE = re.compile(
    r"(?i)^(is\s+)?continued\s+until(\s+further)?$|"
    r"^guilty\s+plea$|^take\s+plea$|^without\s+further$|^because\s+one$|"
    r"^who\s+may$|^larimer\s+county\b"
)
_PROCEDURAL_PREFIX_RE = re.compile(
    r"^(?:telephonic|update|unassigned|calendar|status|sentencing|whose|will|"
    r"has|granted|accept|issues|notify|participates|jeg|none|disposition|"
    r"schedules|website|webpage)\b",
    flags=re.I,
)
_WEB_BOILERPLATE_RE = re.compile(
    r"(?i)\b(?:web\s*page|website|webpage|individual\s+rules|advises?|"
    r"trial\s+rules|practice\s+pointers|operating\s+procedures|"
    r"case\s+management|corrected\s+memorandum)\b"
)


@lru_cache(maxsize=1)
def load_english_wordlist(path: str | None = None) -> frozenset[str]:
    """Load common-English wordlist (default: /usr/share/dict/words)."""
    candidates = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get("TIER_V3_ENGLISH_WORDLIST")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/usr/share/dict/words"),
            Path("/usr/share/dict/web2"),
        ]
    )
    for p in candidates:
        if p.exists():
            words = set()
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip().lower()
                if w.isalpha() and len(w) >= 2:
                    words.add(w)
            return frozenset(words)
    # Minimal fallback so tests still run without system dict
    return frozenset(
        {
            "set", "respect", "because", "only", "rules", "miscellaneous",
            "granted", "denied", "consent", "assignment", "motion", "order",
            "judge", "court", "the", "and", "of", "to", "in", "on", "for",
            "from", "with", "that", "this", "findings", "upon", "under",
            "added", "caused", "so", "is", "are", "was", "were", "will",
            "shall", "has", "have", "who", "signs", "text", "present",
        }
    )


@file_memo
def load_fjc_name_sets(
    fjc_csv: Path,
    honorifics: list[str],
    strip_chars: str,
) -> tuple[set[str], set[str]]:
    from engine.normalize import normalize_name, surname, tokens

    surnames: set[str] = set()
    full: set[str] = set()
    if not fjc_csv.exists():
        return surnames, full
    with fjc_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parts = [
                (row.get("First Name") or "").strip(),
                (row.get("Middle Name") or "").strip(),
                (row.get("Last Name") or "").strip(),
            ]
            raw = " ".join(p for p in parts if p)
            if not raw:
                continue
            nn = normalize_name(raw, honorifics=honorifics, strip_chars=strip_chars)
            if not nn:
                continue
            toks = tokens(nn)
            if len(toks) >= 2:
                full.add(nn)
            sur = surname(nn)
            if sur and len(sur) > 1:
                surnames.add(sur)
    return surnames, full


def build_corpus_name_rescue(normalized_names: list[str]) -> tuple[set[str], set[str]]:
    """Surnames / any tokens that appear inside multi-token names in the pool."""
    multi_surnames: set[str] = set()
    multi_tokens: set[str] = set()
    for nn in normalized_names:
        toks = [t for t in (nn or "").lower().split() if t]
        # drop generationals for rescue matching
        core = [t for t in toks if t not in DEFAULT_GENERATIONAL_SUFFIXES]
        if len(core) < 2:
            continue
        multi_surnames.add(core[-1])
        multi_tokens.update(core)
    return multi_surnames, multi_tokens


def strip_courtroom_suffix(normalized_name: str) -> str:
    """Drop court-reporter and courtroom-code tails from a judge name.

    Dockets append recording labels and courtroom deputy initials to the judge:
    ``timothy j sullivan ftr - singletary``, ``bridget s bade bsb``,
    ``g r smith sff``. Left in place, the last token becomes the "surname" and
    the same judge looks like two people — or two judges who share a court
    reporter look like one.
    """
    toks = (normalized_name or "").strip().lower().split()
    if not toks:
        return ""

    # "... ftr - singletary" / "... ftr singletary": everything from the
    # recording marker onward belongs to the reporter, not the judge.
    for marker in ("ftr", "fjc-ftr", "etr", "dr"):
        if marker in toks[1:]:
            i = toks.index(marker, 1)
            toks = toks[:i]
            break
    while toks and toks[-1] in {"-", "--"}:
        toks.pop()
    if not toks:
        return ""

    # Trailing courtroom code that is just the initials of the preceding name:
    # "bridget s bade bsb" -> b,s,b; "g r smith sff" keeps only its own letters.
    while len(toks) >= 3:
        tail = toks[-1]
        if not (2 <= len(tail) <= 3 and tail.isalpha()):
            break
        head = toks[:-1]
        initials = "".join(t[0] for t in head if t and t[0].isalpha())
        if tail == initials[-len(tail):] or tail == initials[: len(tail)]:
            toks.pop()
            continue
        # Same letter repeated ("sff", "bbb") is a code, never a surname.
        if len(set(tail)) == 1 or (len(tail) == 3 and tail[1] == tail[2]):
            surname_initial = head[-1][0] if head and head[-1] else ""
            if tail[0] == surname_initial:
                toks.pop()
                continue
        break
    return " ".join(toks)


def preclean_header_junk(raw_name: str) -> str:
    """Truncate header/docket glue; return '' when the span must be rejected."""
    s = (raw_name or "").strip()
    if not s:
        return ""
    # Case-number / docket-id spans are never person names
    if _CASE_NUMBER_RE.search(s) and not re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{2,}", s):
        return ""
    if _INTERPRETER_RE.match(s):
        return ""
    # Possessive constructions ("Wexler's Trial Rules") → reject (not strip-to-surname)
    if _POSSESSIVE_SIMPLE_RE.search(s) or _POSSESSIVE_CONSTRUCTION_RE.search(s):
        return ""
    cause = _HEADER_CAUSE_RE.search(s)
    if cause:
        s = s[: cause.start()].strip()
    referred = _HEADER_REFERRED_RE.search(s)
    if referred:
        s = s[: referred.start()].strip()
    designated = _HEADER_DESIGNATED_RE.search(s)
    if designated:
        s = s[designated.end() :].strip()
    dash_mj = _HEADER_DASH_MJ_RE.search(s)
    if dash_mj:
        s = s[: dash_mj.start()].strip()
    # Cut courtroom/deputy codes: Name.(de  Name)(rls  Name.(Cdomadi
    cut = _PAREN_CODE_CUT_RE.search(s)
    if cut:
        s = s[: cut.start()].strip()
    bare = _BARE_PAREN_TAIL_RE.search(s)
    if bare:
        s = s[: bare.start()].strip()
    # Also cut at first bare '(' after letters (INTROCASO(er)
    m_paren = re.search(r"(?<=[A-Za-z])\(", s)
    if m_paren:
        s = s[: m_paren.start()].strip()
    web = _WEB_BOILERPLATE_RE.search(s)
    if web:
        # phrase after a name → keep only the name prefix; pure boilerplate → empty
        head = s[: web.start()].strip()
        s = head if head else ""
    s = re.sub(r"(?i)([a-z][a-z\-]+)'s?\s*$", r"\1", s)
    if ":" in s:
        head, tail = s.split(":", 1)
        tail_l = tail.strip().lower()
        if any(
            tok in tail_l
            for tok in ("sentencing", "disposition", "conference", "scheduling", "order")
        ):
            s = head.strip()
    s = re.sub(r"\s{2,}", " ", s).strip(" .,;:'\"*")
    if _FRAGMENT_RE.match(s.lower()):
        return ""
    return s


def _merged_drop_phrases(cfg: dict | None) -> set[str]:
    nv = (cfg or {}).get("name_validity") or {}
    out = set(DEFAULT_HEADER_DROP_PHRASES)
    out.update(w.lower().strip() for w in (nv.get("header_drop_phrases") or []) if w)
    return out


def _merged_legal_terms(cfg: dict | None) -> set[str]:
    nv = (cfg or {}).get("name_validity") or {}
    out = set(DEFAULT_LEGAL_DOCUMENT_TERMS)
    out.update(w.lower().strip() for w in (nv.get("legal_document_terms") or []) if w)
    return out


def strip_trailing_procedural(
    normalized_name: str,
    stopwords: set[str],
    *,
    trailing_non_name: set[str] | None = None,
) -> str:
    """Strip leading honorifics/procedural + cut at first non-initial stop token.

    Also strip trailing non-name tokens (``... added``, ``... is so``, …) and
    courtroom/reporter tails (``... ftr - singletary``, ``... bsb``).
    """
    trailing_non_name = trailing_non_name or DEFAULT_TRAILING_NON_NAME
    # Possessive residue from "Gilliam's docket" → "gilliam's"
    nn = re.sub(r"(?i)\b([a-z][a-z'\-]+)'s\b", r"\1", (normalized_name or "").strip())
    toks = strip_courtroom_suffix(nn).split()
    if not toks:
        return ""
    leading = stopwords | {
        "us", "u.s", "u.s.", "united", "states", "chief", "senior", "acting",
        "honorable", "hon", "hon.",
    }
    while toks and toks[0] in leading:
        toks.pop(0)
    if not toks:
        return ""
    out: list[str] = [toks[0]]
    for i, t in enumerate(toks[1:], start=1):
        if t in DEFAULT_GENERATIONAL_SUFFIXES:
            out.append(t)
            continue
        if t in stopwords or t in trailing_non_name:
            if i == 1 and len(t) == 1:
                out.append(t)
                continue
            break
        out.append(t)
    # Strip trailing non-name / stopwords (keep generationals)
    while out and out[-1] not in DEFAULT_GENERATIONAL_SUFFIXES and (
        out[-1] in trailing_non_name or out[-1] in stopwords
    ):
        out.pop()
    return strip_trailing_lexemes(" ".join(out))


def _plausible_surname_token(t: str) -> bool:
    """Surname-like token; allow hyphen / apostrophe (Aenlle-Rocha, O'Connell)."""
    tt = t.rstrip(".").lower()
    if tt in DEFAULT_GENERATIONAL_SUFFIXES:
        return False
    core = tt.replace("'", "").replace("-", "")
    return len(core) >= 2 and core.isalpha()


def strip_trailing_lexemes(normalized_name: str, extra: set[str] | None = None) -> str:
    """Remove trailing procedural lexemes only (Denise Page Hood keeps mid-name 'page')."""
    lex = set(DEFAULT_TRAILING_LEXEMES)
    if extra:
        lex.update(extra)
    toks = (normalized_name or "").split()
    while toks and toks[-1].lower().rstrip(".") not in DEFAULT_GENERATIONAL_SUFFIXES:
        t = toks[-1].lower().rstrip(".")
        if t in lex or t in DEFAULT_TRAILING_NON_NAME:
            toks.pop()
            continue
        if (
            len(toks) >= 3
            and 2 <= len(t) <= 3
            and t.isalpha()
            and t not in {"jr", "sr", "ii", "iii", "iv"}
            and _plausible_surname_token(toks[-2])
        ):
            toks.pop()
            continue
        if (
            len(toks) >= 3
            and len(t) >= 5
            and t.isalpha()
            and _plausible_surname_token(toks[-2])
            and (
                t.endswith(("adi", "domadi"))
                or t in {"singletary", "pomerantz", "siegert", "mckenzie", "noel"}
            )
        ):
            toks.pop()
            continue
        break
    return " ".join(toks)

def classify_name_validity(
    normalized_name: str,
    *,
    stopwords: set[str] | None = None,
    fjc_surnames: set[str] | None = None,
    fjc_full_names: set[str] | None = None,
    corpus_surnames: set[str] | None = None,
    corpus_tokens: set[str] | None = None,
    english_words: frozenset[str] | set[str] | None = None,
    reject_single_char_second_token: bool = True,
    wordlist_path: str | None = None,
    min_name_tokens: int = 2,
    header_drop_phrases: set[str] | None = None,
    legal_document_terms: set[str] | None = None,
    noise_full_names: set[str] | None = None,
) -> tuple[bool, list[str]]:
    stopwords = stopwords or DEFAULT_PROCEDURAL_STOPWORDS
    fjc_surnames = fjc_surnames or set()
    fjc_full_names = fjc_full_names or set()
    corpus_surnames = corpus_surnames or set()
    corpus_tokens = corpus_tokens or set()
    header_drop_phrases = header_drop_phrases or set(DEFAULT_HEADER_DROP_PHRASES)
    legal_document_terms = legal_document_terms or set(DEFAULT_LEGAL_DOCUMENT_TERMS)
    noise_full_names = noise_full_names or set(DEFAULT_NOISE_FULL_NAMES)
    if english_words is None:
        english_words = load_english_wordlist(wordlist_path)

    raw_nn = preclean_header_junk(normalized_name or "").lower()
    if not raw_nn:
        return False, ["empty_name"]

    if raw_nn in noise_full_names or raw_nn in header_drop_phrases:
        return False, ["header_drop_phrase"]

    nn = strip_trailing_procedural(raw_nn, stopwords)
    reasons: list[str] = []
    if nn != raw_nn:
        reasons.append("stripped_trailing_procedural")
    if not nn:
        if raw_nn in DEFAULT_GENERATIONAL_SUFFIXES or raw_nn in {"jr", "sr"}:
            return False, ["generational_only"] + reasons
        return False, ["procedural_stopword_full_name", "empty_after_strip"] + reasons

    if nn in noise_full_names or nn in header_drop_phrases:
        return False, ["header_drop_phrase"] + reasons
    if _PROCEDURAL_PREFIX_RE.match(nn):
        return False, ["procedural_prefix"] + reasons
    if _DATE_OR_CODE_RE.match(nn) or any(ch.isdigit() for ch in nn):
        return False, ["date_or_numeric_token"] + reasons

    toks = nn.split()
    core = list(toks)
    while core and core[-1] in DEFAULT_GENERATIONAL_SUFFIXES:
        core.pop()
    if not core:
        return False, ["generational_only"] + reasons

    # A3 / A4: generational used as given name ("jr mcguire") —
    # but "Sr Robert C Brack" is honorific-before-name (strip, keep).
    while (
        len(core) >= 3
        and core[0] in DEFAULT_GENERATIONAL_SUFFIXES
        and _plausible_surname_token(core[-1])
    ):
        core.pop(0)
        reasons.append("stripped_leading_generational_honorific")
    if core and core[0] in DEFAULT_GENERATIONAL_SUFFIXES:
        reasons.append("generational_as_given_name")
        return False, reasons

    if nn in stopwords or (len(toks) == 1 and toks[0] in stopwords):
        reasons.append("procedural_stopword_full_name")
    if len(toks) == 1 and toks[0] in DEFAULT_GENERATIONAL_SUFFIXES:
        reasons.append("generational_only")
    if len(toks) == 1 and len(toks[0]) <= 2:
        reasons.append("too_short_single_token")
    if len(core) < min_name_tokens:
        # research_dev: always require ≥2 tokens (no single-token FJC exemption)
        reasons.append("too_few_name_tokens")
    legal_hits = {t.rstrip(".") for t in core if t.rstrip(".") in legal_document_terms}
    if legal_hits:
        reasons.append("legal_document_term:" + ",".join(sorted(legal_hits)))
    # A courtroom/deputy code standing in as a whole name ("jpp", "bsb", "sff").
    # No vowel means no name; the FJC roster gets the final say.
    if len(toks) == 1 and 2 <= len(toks[0]) <= 4 and toks[0].isalpha():
        tok = toks[0]
        if not set(tok) & set("aeiouy") and tok not in fjc_surnames:
            reasons.append("initials_only_token")
    if reject_single_char_second_token and len(core) == 2 and len(core[1].rstrip(".")) == 1:
        reasons.append("single_char_second_token")
    if len(core) == 2 and core[0] in stopwords and len(core[1].rstrip(".")) <= 2:
        reasons.append("procedural_word_plus_fragment")

    # A3: single-letter token handling
    # - Leading initial + ≥1 multi-letter token → VALID
    # - One or more middle initials (Sarah A. L. Merriam; Fernando L. …) → VALID
    # - O'/D' particle before surname (beverly reid o connell) → VALID
    # - Otherwise → single_char_token_non_middle
    for i, t in enumerate(core):
        tt = t.rstrip(".")
        if len(tt) != 1 or not tt.isalpha():
            continue
        has_given_before = any(len(x.rstrip(".")) > 1 for x in core[:i])
        all_prior_initials = i > 0 and all(
            len(x.rstrip(".")) == 1 and x.rstrip(".").isalpha() for x in core[:i]
        )
        has_surname_after = any(_plausible_surname_token(x) for x in core[i + 1 :])
        if i == 0 and has_surname_after:
            continue
        if (has_given_before or all_prior_initials) and has_surname_after:
            continue
        if (
            has_surname_after
            and i + 1 < len(core)
            and _plausible_surname_token(core[i + 1])
            and tt in {"o", "d", "l"}
            and (has_given_before or all_prior_initials or i == 0)
        ):
            continue
        reasons.append("single_char_token_non_middle")
        break

    bad_toks = [t for t in core if t in stopwords]
    if bad_toks:
        # Leading single-letter is never a stopword hit we care about
        flagged = [t for t in bad_toks if not (len(t) == 1 and t.isalpha())]
        if flagged:
            reasons.append("procedural_stopword_token:" + ",".join(sorted(set(flagged))))
    sur_raw = core[-1]
    sur = sur_raw.replace("'", "").replace("-", "")
    if sur_raw in stopwords or sur in stopwords:
        reasons.append("procedural_surname")
    if core[0] in stopwords and len(core) <= 2 and not (
        len(core[0]) == 1 and core[0].isalpha()
    ):
        reasons.append("procedural_given_name")

    # A2 dictionary gate — research_dev keeps First(+Middle)+Last even when
    # tokens are English words ("Michael North"). Only hard-reject when the
    # span is NOT a well-formed person-name skeleton.
    leading_initial_structure = (
        len(core) >= 2
        and len(core[0].rstrip(".")) == 1
        and core[0].rstrip(".").isalpha()
        and any(_plausible_surname_token(x) for x in core[1:])
    )
    well_formed_person = (
        len(core) >= 2
        and _plausible_surname_token(core[-1])
        and core[0] not in stopwords
        and not any(t.rstrip(".").lower() in legal_document_terms for t in core)
        and not any(t in stopwords and len(t) > 1 for t in core[:-1])
    )
    alpha_core = []
    for t in core:
        tt = t.rstrip(".").replace("'", "").replace("-", "")
        if tt.isalpha():
            alpha_core.append(tt)
    if alpha_core and not leading_initial_structure and not well_formed_person:
        multi = [t for t in alpha_core if len(t) > 1]
        if multi and all(t in english_words for t in multi):
            rescued = False
            if sur in fjc_surnames or sur in corpus_surnames or sur_raw in fjc_surnames:
                rescued = True
                reasons.append("dict_all_tokens_rescued_surname")
            elif any(t in corpus_tokens for t in multi) and len(multi) == 1:
                rescued = True
                reasons.append("dict_all_tokens_rescued_corpus")
            elif nn in fjc_full_names:
                rescued = True
                reasons.append("dict_all_tokens_rescued_fjc_full")
            if not rescued:
                reasons.append("all_tokens_english_dictionary")
    elif leading_initial_structure:
        reasons.append("leading_initial_name_structure")
    elif well_formed_person:
        reasons.append("well_formed_person_name")

    hard = [r for r in reasons if r not in {
        "stripped_trailing_procedural",
        "stripped_leading_generational_honorific",
        "dict_all_tokens_rescued_surname",
        "dict_all_tokens_rescued_corpus",
        "dict_all_tokens_rescued_fjc_full",
        "leading_initial_name_structure",
        "well_formed_person_name",
    }]
    if not hard:
        return True, reasons

    hard_blockers = {
        "single_char_second_token",
        "single_char_token_non_middle",
        "procedural_word_plus_fragment",
        "generational_only",
        "generational_as_given_name",
        "procedural_stopword_full_name",
        "too_short_single_token",
        "too_few_name_tokens",
        "procedural_given_name",
        "all_tokens_english_dictionary",
        "header_drop_phrase",
        "procedural_prefix",
        "date_or_numeric_token",
    }
    if any(r.startswith("legal_document_term:") for r in hard):
        return False, reasons
    if any(r in hard_blockers for r in hard):
        return False, reasons
    if "procedural_surname" in hard and sur_raw not in fjc_surnames and sur not in fjc_surnames and sur_raw not in corpus_surnames and sur not in corpus_surnames:
        return False, reasons

    if nn in fjc_full_names and len(core) >= 2:
        return True, ["whitelisted_fjc_full_name"] + reasons
    if (sur_raw in fjc_surnames or sur in fjc_surnames) and len(core) >= 2 and (
        core[0] not in stopwords or (len(core[0]) == 1 and core[0].isalpha())
    ):
        return True, ["whitelisted_fjc_surname"] + reasons
    if (
        len(core) >= 3
        and sur_raw not in stopwords
        and (core[0] not in stopwords or (len(core[0]) == 1 and core[0].isalpha()))
        and all(t in {"a", "v"} or t not in stopwords or len(t) == 1 for t in core[1:-1])
    ):
        return True, ["allowed_middle_initial_token"] + reasons

    token_flags = [r for r in hard if r.startswith("procedural_stopword_token:")]
    other_hard = [r for r in hard if not r.startswith("procedural_stopword_token:")]
    if token_flags and not other_hard:
        flagged = set()
        for r in token_flags:
            flagged.update(r.split(":", 1)[1].split(","))
        if flagged.issubset({"a", "v"}) and len(core) >= 3:
            return True, ["allowed_middle_initial_token"] + reasons

    return False, reasons


def gate_mention(
    mention: dict,
    *,
    cfg: dict,
    fjc_surnames: set[str],
    fjc_full_names: set[str],
    corpus_surnames: set[str] | None = None,
    corpus_tokens: set[str] | None = None,
) -> tuple[dict | None, dict | None]:
    nv = cfg.get("name_validity") or {}
    if not nv.get("enabled", False):
        return mention, None

    stopwords = set(DEFAULT_PROCEDURAL_STOPWORDS)
    stopwords.update(w.lower() for w in (nv.get("procedural_stopwords") or []))
    raw_display = (mention.get("raw_name") or mention.get("normalized_name") or "").strip()
    raw_nn = mention.get("normalized_name") or ""

    # Prefer cleaning the raw display string (preserves .(code glue before normalize)
    pre = preclean_header_junk(raw_display)
    if not pre:
        reasons = ["deterministic_reject"]
        low = raw_display.lower()
        if _POSSESSIVE_SIMPLE_RE.search(raw_display) or _POSSESSIVE_CONSTRUCTION_RE.search(raw_display):
            reasons = ["possessive_construction"]
        elif _INTERPRETER_RE.match(raw_display):
            reasons = ["interpreter_prefix"]
        elif _FRAGMENT_RE.match(low):
            reasons = ["sentence_fragment"]
        elif _CASE_NUMBER_RE.search(raw_display):
            reasons = ["case_number_pattern"]
        elif _WEB_BOILERPLATE_RE.search(raw_display):
            reasons = ["web_boilerplate"]
        q = dict(mention)
        q["name_validity"] = "quarantine"
        q["name_validity_reasons"] = reasons
        q["quarantine_reason"] = reasons
        q["name_validity_method"] = "deterministic"
        return None, q

    from engine.normalize import normalize_name, surname, tokens

    honorifics = (cfg.get("normalization") or {}).get("strip_honorifics") or []
    strip_chars = (cfg.get("normalization") or {}).get("strip_chars") or ".,;:\"()[]{}"
    cleaned = normalize_name(pre, honorifics=honorifics, strip_chars=strip_chars)
    cleaned = strip_trailing_procedural(cleaned, stopwords)
    cleaned = strip_trailing_lexemes(cleaned)

    ok, reasons = classify_name_validity(
        cleaned or raw_nn,
        stopwords=stopwords,
        fjc_surnames=fjc_surnames,
        fjc_full_names=fjc_full_names,
        corpus_surnames=corpus_surnames or set(),
        corpus_tokens=corpus_tokens or set(),
        reject_single_char_second_token=bool(nv.get("reject_single_char_second_token", True)),
        wordlist_path=nv.get("english_wordlist_path"),
        min_name_tokens=int(nv.get("min_name_tokens", 2)),
        header_drop_phrases=_merged_drop_phrases(cfg),
        legal_document_terms=_merged_legal_terms(cfg),
    )
    if ok and cleaned:
        out = dict(mention)
        if cleaned != raw_nn.strip().lower():
            out["normalized_name"] = cleaned
            out["name_validity_stripped_from"] = raw_nn
            out["surname"] = surname(cleaned)
            out["token_count"] = len(tokens(cleaned))
            # Keep presentable aligned with cleaned form
            out["presentable_name"] = " ".join(w.capitalize() for w in cleaned.split())
        out["name_validity"] = "kept"
        out["name_validity_reasons"] = reasons or ["deterministic_clean"]
        out["name_validity_method"] = "deterministic"
        return out, None

    q = dict(mention)
    q["name_validity"] = "quarantine"
    q["name_validity_reasons"] = reasons
    q["quarantine_reason"] = reasons
    q["name_validity_method"] = "deterministic"
    return None, q

def write_quarantine(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
