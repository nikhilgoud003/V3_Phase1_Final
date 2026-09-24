"""Docket-entry judge extraction via SpaCy NER + judge-context cues."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

# Honorific / role cues that indicate a PERSON span is judicial
_JUDGE_CUE = re.compile(
    r"(?i)\b("
    r"judge|magistrate(?:\s+judge)?|hon(?:orable|\.)?|chief\s+judge|"
    r"senior\s+judge|district\s+judge|bankruptcy\s+judge|mj"
    r")\b"
)

# Direct "Judge First Last" patterns (high precision supplement).
# Middle initial MUST include a period (A-Z\.) so "Consent Form" / "Assignment To"
# cannot truncate to "Consent F" / "Assignment T". Surnames must be 2+ letters.
_JUDGE_NAME_RE = re.compile(
    r"(?i)\b(?:hon(?:orable)?\.?\s+)?(?:(?:chief|senior|magistrate|district|bankruptcy)\s+)?"
    r"(?:judge|magistrate)\s+"
    r"([A-Z][A-Za-z'\-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z'\-]{1,}){0,3})"
)

# Procedural / form titles that appear after "Judge" / "Magistrate Judge"
_PROCEDURAL_AFTER_TITLE = re.compile(
    r"(?i)^(assignment|consent|pursuant|jurisdiction|motion|order|notice|"
    r"referral|reassignment|designation|transfer|form|packet|questionnaire|"
    r"standing|minute|entry|case|matter|above|undersigned|presiding|"
    r"panel|bench|court|district|magistrate|judges?|hon(?:orable)?|"
    r"signed|referred|re|by|to|of|the|a|an|and|or|for|in|on|at|from|"
    r"telephonic|scheduling|unassigned|update|calendar|status|sentencing|"
    r"notify|participates|disposition|accept|issues|granted|whose|will|"
    r"has|set|call|jeg|none|either|designated|adopted|initials)\b"
)


@lru_cache(maxsize=2)
def _load_nlp(model: str):
    import spacy

    return spacy.load(model)


_TRAILING_JUNK = re.compile(
    r"(?i)\s+\b("
    r"on|signed|for|held|as|to|from|by|dated|entered|cc|"
    r"and|or|is|are|was|were|has|have|will|shall|no|not|the|a|an|"
    r"in|at|of|with|who|that|this|defendant|plaintiff|counsel|"
    r"filed|filing|order|orders|hearing|conference|chambers|courtroom|"
    r"jury|room|longer|available|dismissed|vacated|withdrawn|present|"
    r"before|after|under|upon|regarding|concerning|text|standing"
    r")\b.*$"
)

# Tokens that end a person-name capture when seen after the given name
_NAME_STOP_TOKENS = {
    "and", "or", "is", "are", "was", "were", "has", "have", "will", "shall",
    "no", "not", "the", "an", "in", "at", "of", "with", "who", "that", "this",
    "defendant", "plaintiff", "counsel", "filed", "filing", "order", "orders",
    "hearing", "conference", "chambers", "courtroom", "jury", "room", "longer",
    "available", "dismissed", "vacated", "withdrawn", "present", "before",
    "after", "under", "upon", "regarding", "concerning", "text", "standing",
    "status", "minute", "entry", "form", "consent", "assignment", "pursuant",
    "jurisdiction", "motion", "signed", "referred", "assigned", "findings",
    "recommendation", "anchorage", "login", "arraignment", "detention",
    "telephonic", "scheduling", "unassigned", "update", "notify", "participates",
    "disposition", "accept", "issues", "granted", "whose", "calendar", "call",
    "jeg", "either", "designated", "adopted", "initials", "sentencing", "memos",
    # NOTE: do NOT put "page"/"web" here — mid-name cut kills Denise Page Hood.
    # Website/boilerplate tails are stripped only from the end below.
    "website", "webpage", "advises", "advise", "decision",
    "schedules", "individual", "rules", "located", "found",
    "plea", "modified", "trial", "dtd", "see", "added", "attached",
}

# Strip ONLY from the end (never mid-name).
_TRAILING_ONLY_LEXEMES = {
    "web", "page", "website", "webpage", "advises", "advise", "decision",
    "schedules", "individual", "rules", "plea", "modified", "trial", "dtd",
    "see", "added", "attached", "located", "found",
}


def _truncate_at_stop_tokens(raw: str) -> str:
    """Cut 'Deborah M. Smith and Kevin…' → 'Deborah M. Smith'."""
    toks = raw.split()
    if not toks:
        return raw
    out = [toks[0]]
    for i, t in enumerate(toks[1:], start=1):
        tl = t.lower().rstrip(".,;:")
        if tl in _NAME_STOP_TOKENS:
            # allow single-letter middle initial at position 1 (A, V, …)
            if i == 1 and len(tl) == 1:
                out.append(t)
                continue
            break
        out.append(t)
    return " ".join(out)


def _strip_trailing_only(raw: str) -> str:
    toks = (raw or "").split()
    while toks and toks[-1].lower().rstrip(".,;:") in _TRAILING_ONLY_LEXEMES:
        toks.pop()
    return " ".join(toks)


def _clean_name(raw: str) -> str:
    t = raw.strip(" .,;:")
    # Cut courtroom/deputy codes glued by NER: Name.(de  Name)(rls
    for sep in (".(", ")("):
        if sep in t:
            t = t.split(sep, 1)[0].strip()
    m_paren = re.search(r"(?<=[A-Za-z])\(", t)
    if m_paren:
        t = t[: m_paren.start()].strip()
    t = _TRAILING_JUNK.sub("", t).strip(" .,;:")
    t = _truncate_at_stop_tokens(t).strip(" .,;:")
    t = _strip_trailing_only(t).strip(" .,;:")
    # drop if looks like sentence fragment (too many lowercase function words)
    if len(t.split()) > 5:
        return ""
    return t


def _looks_like_person_name(raw: str) -> bool:
    """Reject procedural title captures before they become mentions."""
    t = (raw or "").strip()
    if not t or len(t) < 2:
        return False
    if _PROCEDURAL_AFTER_TITLE.match(t):
        return False
    toks = t.split()
    # research_dev: require ≥2 tokens for a person name
    if len(toks) < 2:
        return False
    # word + single-letter without period (truncation residue)
    if len(toks) == 2 and len(toks[1].rstrip(".")) == 1 and not toks[1].endswith("."):
        # allow only if first token looks like a real given name length AND letter has period
        return False
    return True


def extract_judges_from_docket_text(
    text: str,
    *,
    model: str = "en_core_web_sm",
    label_allowlist: tuple[str, ...] = ("PERSON",),
) -> list[dict[str, Any]]:
    """
    Return list of {raw, start, end, method} judge-like spans in one docket entry.
    Uses SpaCy PERSON entities near judicial cues, plus regex 'Judge Name' hits.
    """
    if not text or not text.strip():
        return []

    found: dict[tuple[int, int], dict] = {}

    # Regex high-precision hits
    for m in _JUDGE_NAME_RE.finditer(text):
        raw = _clean_name(m.group(1))
        if raw and _looks_like_person_name(raw) and len(raw.split()) >= 2:
            found[(m.start(1), m.start(1) + len(raw))] = {
                "raw": raw,
                "start": m.start(1),
                "end": m.start(1) + len(raw),
                "method": "regex_judge_title",
            }

    # SpaCy NER
    try:
        nlp = _load_nlp(model)
    except Exception:
        return list(found.values())

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ not in label_allowlist:
            continue
        left = max(0, ent.start_char - 30)
        cue_window = text[left : ent.start_char]
        if not _JUDGE_CUE.search(cue_window):
            continue
        raw = _clean_name(ent.text)
        if not raw or not _looks_like_person_name(raw):
            continue
        key = (ent.start_char, ent.start_char + len(raw))
        if key not in found:
            found[key] = {
                "raw": raw,
                "start": ent.start_char,
                "end": ent.start_char + len(raw),
                "method": f"spacy_{ent.label_}",
            }

    by_norm: dict[str, dict] = {}
    for item in sorted(found.values(), key=lambda x: x["start"]):
        norm = re.sub(r"\s+", " ", item["raw"]).strip().lower()
        if not norm or len(norm) < 2:
            continue
        if not _looks_like_person_name(item["raw"]):
            continue
        # Prefer longer span for same start (SpaCy full name over truncated regex)
        prev = by_norm.get(norm)
        if prev is None:
            # also supersede shorter overlapping capture with same first token start
            superseded = None
            for k, v in list(by_norm.items()):
                if v["start"] == item["start"] and len(item["raw"]) > len(v["raw"]):
                    superseded = k
                    break
            if superseded:
                del by_norm[superseded]
            by_norm[norm] = item
        elif len(item["raw"]) > len(prev["raw"]):
            by_norm[norm] = item
    return list(by_norm.values())
