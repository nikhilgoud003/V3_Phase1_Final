"""Alias and related-entity names from a party's raw_info text.

Config: parties.yaml alias_extraction. Markers are grammar phrases, not names:
  same_entity markers (fka / aka / dba): another name of the SAME entity
  related_entity markers (successor / predecessor / alter ego / c/o / ...):
    a DIFFERENT entity with a relationship to the party

An alias belongs to the most recent subject: "SUCCESSOR IN INTEREST TO X
A/K/A Y" makes Y another name of X, not of the party. A segment break in the
source (<br/>, <i>) or a ";" resets the subject to the party. Related names
always relate to the party.
"""

from __future__ import annotations

import html
import re
from typing import Any


def _compile(spec: dict) -> list[tuple[re.Pattern, str, bool]]:
    out: list[tuple[re.Pattern, str, bool]] = []
    markers = spec.get("markers") or {}
    for kind, same in (("same_entity", True), ("related_entity", False)):
        for rel, pats in (markers.get(kind) or {}).items():
            for p in pats or []:
                rx = re.compile(rf"(?<![a-z0-9/]){p}(?![a-z0-9])", re.IGNORECASE)
                out.append((rx, str(rel), same))
    return out


_CACHE: dict[int, list[tuple[re.Pattern, str, bool]]] = {}


def _segments_text(raw_info: str) -> str:
    t = html.unescape(html.unescape(raw_info or ""))
    t = re.sub(r"<[^>]+>|\n", " | ", t)
    return re.sub(r"\s+", " ", t)


def _clean(text: str, spec: dict) -> str | None:
    stop = {w.lower() for w in spec.get("stop_words") or []}
    strip = {w.lower() for w in spec.get("strip_words") or []}
    t = text.strip().strip("|").strip()
    t = re.split(r"[;()]", t, maxsplit=1)[0]
    t = t.replace('"', " ").replace("“", " ").replace("”", " ")
    # A comma followed by a lowercase word or a stop/strip word starts a
    # descriptor (", a New Jersey corporation", ", deceased", ", ETC.").
    parts = t.split(",")
    keep = [parts[0]]
    for p in parts[1:]:
        first = p.strip().split(" ")[0] if p.strip() else ""
        bare = first.strip(".").lower()
        if not first or first[:1].islower() or bare in stop or bare in strip:
            break
        keep.append(p)
    toks = ",".join(keep).split()
    for i, tok in enumerate(toks):
        if tok.strip(".,").lower() in stop:
            toks = toks[:i]
            break
    while toks and toks[0].strip(".,").lower() in strip:
        toks.pop(0)
    while toks and toks[-1].strip(".,").lower() in strip:
        toks.pop()
    name = " ".join(toks).strip(" ,.;:-")
    if sum(c.isalpha() for c in name) < 2:
        return None
    return name


def parse_raw_info(raw_info: str, spec: dict) -> list[dict[str, Any]]:
    """Return [{name, relationship, same_entity, subject}] in text order.

    subject is None for the party itself, or the related name an alias belongs to.
    """
    if not raw_info or not spec.get("enabled"):
        return []
    pats = _CACHE.get(id(spec))
    if pats is None:
        pats = _CACHE[id(spec)] = _compile(spec)
    text = _segments_text(raw_info)
    found = []
    for rx, rel, same in pats:
        for m in rx.finditer(text):
            found.append((m.start(), m.end(), rel, same))
    # Longest match wins where markers overlap.
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    marks: list[tuple[int, int, str, bool]] = []
    for f in found:
        if marks and f[0] < marks[-1][1]:
            continue
        marks.append(f)

    out: list[dict[str, Any]] = []
    last_related: tuple[str, int] | None = None
    for i, (start, end, rel, same) in enumerate(marks):
        stop_at = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        chunk = text[end:stop_at].lstrip(" |")
        chunk = chunk.split("|", 1)[0]
        name = _clean(chunk, spec)
        subject = None
        between = text[last_related[1] : start] if last_related else ""
        if same and last_related and "|" not in between and ";" not in between:
            subject = last_related[0]
        if not name:
            continue
        out.append({"name": name, "relationship": rel, "same_entity": same, "subject": subject})
        if not same:
            last_related = (name, end + len(chunk))
    return out
