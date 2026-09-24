"""Load FJC biographical directory and link mentions to NIDs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .normalize import normalize_name, tokens


def load_court_crosswalk(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fjc_suffix(row: dict[str, str]) -> str:
    return (row.get("Suffix") or "").strip().strip('"').strip(".")


def _fjc_full_name(row: dict[str, str], *, include_suffix: bool = False) -> str:
    parts = [
        (row.get("First Name") or "").strip(),
        (row.get("Middle Name") or "").strip(),
        (row.get("Last Name") or "").strip(),
    ]
    parts = [p for p in parts if p]
    if include_suffix:
        suf = _fjc_suffix(row)
        if suf:
            parts.append(suf)
    return " ".join(parts)


def fjc_directory_forms(first: str, middle: str, last: str, suffix: str = "") -> list[str]:
    """Literal FJC directory strings (exact match, not inferred aliases)."""
    core = [p for p in (first, middle, last) if p]
    if not core:
        return []
    forms = [" ".join(core)]
    if suffix:
        forms.append(" ".join(core + [suffix]))
    return forms


def fjc_alias_forms(first: str, middle: str, last: str, suffix: str = "") -> list[str]:
    """Initial / middle-dropped variants, with and without generational suffix."""
    first = (first or "").strip()
    middle = (middle or "").strip()
    last = (last or "").strip()
    suffix = (suffix or "").strip().strip(".")
    if not first or not last:
        return []
    fi = first[0]
    variants: list[str] = []
    if middle:
        mi = middle[0]
        # Do NOT emit "{fi} {last}" (e.g. "s smith") — too lossy even when unique.
        variants.extend(
            [
                f"{first} {mi} {last}",
                f"{fi} {middle} {last}",
                f"{fi} {mi} {last}",
                f"{first} {last}",
            ]
        )
    # No-middle judges: exact directory form is enough; initial+surname is unsafe.
    out: list[str] = []
    seen: set[str] = set()
    exact = set(fjc_directory_forms(first, middle, last, suffix))
    for v in variants:
        candidates = [v, f"{v} {suffix}"] if suffix else [v]
        for form in candidates:
            key = " ".join(form.split())
            if not key or key in seen or key in exact:
                continue
            seen.add(key)
            out.append(key)
    return out


def _index_add(index: dict, key, nid: str) -> None:
    index.setdefault(key, set()).add(nid)


def load_fjc_index(
    fjc_csv: Path,
    crosswalk_path: Path,
    honorifics: list[str],
    strip_chars: str,
) -> dict[str, Any]:
    """
    Returns:
      by_nid: nid -> {normalized_name, presentable, courts:set, ...}
      name_court_to_nids / name_to_nids: exact directory forms
      alias_court_to_nids / alias_to_nids: unique initial/middle aliases only
    """
    crosswalk = load_court_crosswalk(crosswalk_path)
    by_nid: dict[str, dict] = {}
    name_court_to_nids: dict[tuple[str, str], set[str]] = defaultdict(set)
    name_to_nids: dict[str, set[str]] = defaultdict(set)
    pending_alias_court: dict[tuple[str, str], set[str]] = defaultdict(set)
    pending_alias_global: dict[str, set[str]] = defaultdict(set)

    def _norm(raw: str) -> str:
        return normalize_name(raw, honorifics=honorifics, strip_chars=strip_chars)

    with fjc_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = (row.get("nid") or "").strip().strip('"')
            if not nid:
                continue
            first = (row.get("First Name") or "").strip()
            middle = (row.get("Middle Name") or "").strip()
            last = (row.get("Last Name") or "").strip()
            suffix = _fjc_suffix(row)
            raw = _fjc_full_name(row, include_suffix=False)
            if not raw:
                continue
            norm = _norm(raw)
            if not norm or len(tokens(norm)) < 2:
                continue

            courts = set()
            commissions = []
            terminations = []
            for i in range(1, 7):
                cname = (row.get(f"Court Name ({i})") or "").strip().strip('"')
                if cname and cname in crosswalk:
                    courts.add(crosswalk[cname])
                comm = (row.get(f"Commission Date ({i})") or "").strip().strip('"')
                term = (row.get(f"Termination Date ({i})") or "").strip().strip('"')
                if comm:
                    commissions.append(comm)
                if term:
                    terminations.append(term)

            if nid not in by_nid:
                by_nid[nid] = {
                    "nid": nid,
                    "normalized_name": norm,
                    "presentable": raw,
                    "first": first,
                    "middle": middle,
                    "last": last,
                    "suffix": suffix,
                    "courts": set(courts),
                    "commission_min": min(commissions) if commissions else None,
                    "termination_max": max(terminations) if terminations else None,
                }
            else:
                by_nid[nid]["courts"].update(courts)
                if commissions:
                    prev = by_nid[nid]["commission_min"]
                    by_nid[nid]["commission_min"] = min([c for c in [prev, *commissions] if c])
                if terminations:
                    prev = by_nid[nid]["termination_max"]
                    by_nid[nid]["termination_max"] = max([c for c in [prev, *terminations] if c])

            exact_norms: set[str] = set()
            for form in fjc_directory_forms(first, middle, last, suffix):
                nn = _norm(form)
                if not nn or len(tokens(nn)) < 2:
                    continue
                exact_norms.add(nn)
                name_to_nids[nn].add(nid)
                for court in by_nid[nid]["courts"]:
                    name_court_to_nids[(nn, court)].add(nid)

            for form in fjc_alias_forms(first, middle, last, suffix):
                nn = _norm(form)
                if not nn or len(tokens(nn)) < 2 or nn in exact_norms:
                    continue
                pending_alias_global[nn].add(nid)
                for court in by_nid[nid]["courts"]:
                    pending_alias_court[(nn, court)].add(nid)

    alias_court_to_nids: dict[tuple[str, str], list[str]] = {}
    alias_to_nids: dict[str, list[str]] = {}
    for key, nids in pending_alias_court.items():
        exact_here = name_court_to_nids.get(key)
        if exact_here:
            continue  # already an exact directory form in this court
        if len(nids) == 1:
            alias_court_to_nids[key] = sorted(nids)
    for nn, nids in pending_alias_global.items():
        exact_g = name_to_nids.get(nn)
        if exact_g:
            continue  # already an exact directory form globally
        if len(nids) == 1:
            alias_to_nids[nn] = sorted(nids)

    return {
        "by_nid": by_nid,
        "name_court_to_nids": {k: sorted(v) for k, v in name_court_to_nids.items()},
        "name_to_nids": {k: sorted(v) for k, v in name_to_nids.items()},
        "alias_court_to_nids": alias_court_to_nids,
        "alias_to_nids": alias_to_nids,
        "n_judges": len(by_nid),
    }


def link_mentions_to_fjc(mentions: list[dict], fjc_index: dict[str, Any]) -> dict[str, int]:
    """
    Attach fjc_nid to mentions when uniquely matched.
    Prefer (name, court) unique match; else unique global name match.
    Alias (initial/middle) hits are tagged ``fjc_match_method=fjc_alias_match``.
    Soft dates only — never reject on termination.
    """
    name_court = fjc_index["name_court_to_nids"]
    name_only = fjc_index["name_to_nids"]
    alias_court = fjc_index.get("alias_court_to_nids") or {}
    alias_only = fjc_index.get("alias_to_nids") or {}
    stats = {
        "linked_court_unique": 0,
        "linked_global_unique": 0,
        "linked_court_unique_alias": 0,
        "linked_global_unique_alias": 0,
        "ambiguous_court": 0,
        "ambiguous_global": 0,
        "unlinked": 0,
    }

    for m in mentions:
        norm = m.get("normalized_name") or ""
        court = m.get("court") or ""
        m["fjc_match_method"] = None

        nids = name_court.get((norm, court))
        if nids is not None:
            if len(nids) == 1:
                m["fjc_nid"] = nids[0]
                m["fjc_match_method"] = "fjc_exact"
                stats["linked_court_unique"] += 1
                continue
            stats["ambiguous_court"] += 1
            m["fjc_nid"] = None
            m["fjc_nid_candidates"] = nids
            continue

        nids_g = name_only.get(norm)
        if nids_g is not None and len(nids_g) == 1:
            m["fjc_nid"] = nids_g[0]
            m["fjc_match_method"] = "fjc_exact"
            stats["linked_global_unique"] += 1
            continue
        if nids_g is not None and len(nids_g) > 1:
            stats["ambiguous_global"] += 1
            m["fjc_nid"] = None
            m["fjc_nid_candidates"] = nids_g
            continue

        # Aliases are court-scoped only. Global initial/middle aliases
        # (e.g. "s smith" → Sidney Oslin Smith Jr., N.D. Ga.) are unsafe.
        nids_a = alias_court.get((norm, court))
        if nids_a is not None and len(nids_a) == 1:
            m["fjc_nid"] = nids_a[0]
            m["fjc_match_method"] = "fjc_alias_match"
            stats["linked_court_unique_alias"] += 1
            continue
        if nids_a is not None and len(nids_a) > 1:
            stats["ambiguous_court"] += 1
            m["fjc_nid"] = None
            m["fjc_nid_candidates"] = nids_a
            continue

        m["fjc_nid"] = None
        stats["unlinked"] += 1

    return stats
