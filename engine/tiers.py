"""Tier 0–3 early-exit cascade (type-agnostic, config-driven)."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
from typing import Any

from rapidfuzz import fuzz
import numpy as np

from .config_loader import ollama_endpoint, resolve_path
from .fjc import link_mentions_to_fjc, load_fjc_index
from .mention_hygiene import apply_hygiene, pair_allowed
from .ucid_anchor import mark_anchor_outcomes, resolve_short_mentions
from .name_compat import (
    name_gate_decision_row,
    names_compatible,
    person_initials_expand_compatible,
)
from .normalize import first_last_initials, initial_token_ratio, is_initial_token, surname_block_keys, tokens
from .preflight import CascadeFailedError, PreflightError, check_tier3_error_rate, preflight_ollama
from .tier3_citation import apply_citation_and_match_rails
from .provenance import DecisionJournal, append_jsonl
from .run_cache import file_memo


def prompt_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def normalize_tier3_signals(raw: Any) -> list[str]:
    """Coerce LLM signals to string tags (schema requires array of strings).

    Models sometimes mirror the evidence bundle and emit objects like
    ``{"key": "same_ucid", "value": false}`` or ``{"same_court": true}``.
    Belt-and-suspenders: extract keys / string tags before jsonschema validation.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if not isinstance(raw, list):
        return [str(raw)]
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        elif isinstance(item, dict):
            if "key" in item:
                k = str(item["key"]).strip()
                if k:
                    out.append(k)
            else:
                for k in item:
                    ks = str(k).strip()
                    if ks:
                        out.append(ks)
        elif item is not None:
            out.append(str(item))
    return out


def tier0_alias_group_id(normalized_name: str, cfg: dict) -> str | None:
    """Return Tier0 alias_group id if name is in a configured group, else None."""
    nn = " ".join((normalized_name or "").lower().split())
    if not nn:
        return None
    for grp in (cfg.get("tier0") or {}).get("alias_groups") or []:
        names = {" ".join(str(x).lower().split()) for x in (grp.get("names") or [])}
        if nn in names:
            gid = grp.get("id")
            return str(gid) if gid else None
    return None


def apply_tier0_alias_groups(
    mentions: list[dict],
    cfg: dict,
    journal: "DecisionJournal",
    uf: "UnionFind",
) -> dict:
    """Merge alias-group variants within configured scope (default: same_ucid)."""
    groups = (cfg.get("tier0") or {}).get("alias_groups") or []
    stats = {"groups": 0, "merges": 0, "rules_fired": defaultdict(int)}
    if not groups:
        return stats
    by_id = {m["mention_id"]: m for m in mentions}
    for grp in groups:
        gid = grp.get("id") or "alias"
        names = {" ".join(str(x).lower().split()) for x in (grp.get("names") or [])}
        if not names:
            continue
        scope = (grp.get("merge_scope") or "same_ucid").lower()
        conf = int(grp.get("confidence", 100))
        method = grp.get("method") or f"tier0.alias_group:{gid}"
        stats["groups"] += 1
        buckets: dict[tuple, list[str]] = defaultdict(list)
        for m in mentions:
            nn = " ".join((m.get("normalized_name") or "").lower().split())
            if nn not in names:
                continue
            if scope == "same_ucid":
                ucid = m.get("ucid")
                if not ucid:
                    continue
                key = (gid, ucid)
            elif scope == "same_court":
                court = m.get("court")
                if not court:
                    continue
                key = (gid, court)
            else:
                key = (gid,)
            buckets[key].append(m["mention_id"])
        for key, ids in buckets.items():
            if len(ids) < 2:
                continue
            ids = sorted(set(ids))
            root = ids[0]
            for other in ids[1:]:
                if uf.find(root) == uf.find(other):
                    continue
                if transfer_conflict(by_id[root], by_id[other]):
                    continue
                uf.union(root, other)
                stats["merges"] += 1
                stats["rules_fired"][f"alias_group:{gid}"] += 1
                journal.log(
                    {
                        "decision_id": f"dec_{journal.n:08d}",
                        "mention_id_a": root,
                        "mention_id_b": other,
                        "entity_type": cfg.get("entity_type"),
                        "decision": "MERGE_TIER0",
                        "confidence": conf,
                        "method": method,
                        "rationale": f"Tier0 alias_group {gid} ({scope})",
                        "signals": ["alias_group", str(gid)],
                        "evidence": {"alias_group": gid, "scope": scope, "key": list(key)},
                        "timestamp": _now(),
                        "config_version": cfg.get("version"),
                    }
                )
    return stats


def pair_in_skipped_alias_group_cross_ucid(ma: dict, mb: dict, cfg: dict) -> str | None:
    """If both mentions share a Tier0 alias_group listed in skip_alias_groups_cross_ucid
    and UCIDs differ, return the group id (pair must not go to Tier3)."""
    skip = set(
        ((cfg.get("tier3") or {}).get("routing") or {}).get("skip_alias_groups_cross_ucid")
        or []
    )
    if not skip:
        return None
    if ma.get("ucid") and mb.get("ucid") and ma.get("ucid") == mb.get("ucid"):
        return None
    ga = tier0_alias_group_id(ma.get("normalized_name") or "", cfg)
    gb = tier0_alias_group_id(mb.get("normalized_name") or "", cfg)
    if ga and ga == gb and ga in skip:
        return ga
    return None


def uf_component_size(uf: "UnionFind", x: str) -> int:
    root = uf.find(x)
    return sum(1 for y in uf.parent if uf.find(y) == root)


def large_cluster_merge_blocked(uf: "UnionFind", a: str, b: str, cfg: dict) -> bool:
    """A5: block Tier2 auto-merge when both components exceed size threshold."""
    cl = cfg.get("clustering") or {}
    if not cl.get("verify_merges", True):
        return False
    thr = int(cl.get("large_cluster_verify_min_size", 5))
    return uf_component_size(uf, a) >= thr and uf_component_size(uf, b) >= thr


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _cache_key(a: dict, b: dict) -> str:
    na, nb = sorted([a["normalized_name"], b["normalized_name"]])
    court = a.get("court") or b.get("court") or ""
    role = "|".join(sorted({a.get("role") or "", b.get("role") or ""}))
    return f"{court}|{na}|{nb}|{role}"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return dict(out)


def build_token_profile_for_run(mentions: list[dict], cfg: dict) -> dict:
    """Corpus token profile for the name_gate surname resolver (see name_compat)."""
    from engine.name_compat import build_token_profile
    from engine.name_validity import load_fjc_name_sets

    fjc_surnames: set[str] = set()
    fjc_rule = next(
        (r for r in ((cfg.get("tier0") or {}).get("rules") or []) if r.get("id") == "fjc_nid_join"),
        None,
    )
    if fjc_rule:
        ext = fjc_rule.get("external") or {}
        fjc_path = resolve_path(cfg, ext.get("path", "data/judges_fjc.csv"))
        if fjc_path.exists():
            honorifics = (cfg.get("normalization") or {}).get("strip_honorifics") or []
            strip_chars = (cfg.get("normalization") or {}).get("strip_chars") or ""
            fjc_surnames, _ = load_fjc_name_sets(fjc_path, honorifics, strip_chars)

    min_occ = int((cfg.get("name_compat") or {}).get("noise_token_min_occurrences", 2))
    return build_token_profile(mentions, fjc_surnames, min_occurrences=min_occ)


@file_memo
def load_common_surnames(path: Path) -> set[str]:
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            names.add(line)
    return names


def information_barrier(mention: dict, cfg: dict, common_surnames: set[str]) -> tuple[bool, list[str]]:
    barriers = cfg.get("information_content_barrier") or {}
    if not barriers.get("enabled", True):
        return False, []
    reasons = []
    norm = mention.get("normalized_name") or ""
    toks = tokens(norm)
    if len(toks) == 1:
        reasons.append("single_token_name")
    if initial_token_ratio(norm) >= 0.5 and len(toks) >= 2:
        reasons.append("initial_heavy")
    sur = mention.get("surname") or ""
    if sur and sur in common_surnames:
        reasons.append("very_common_surname")
    for trig in barriers.get("triggers") or []:
        if trig.get("id") != "bare_initials_person":
            continue
        classes = {str(x) for x in (trig.get("office_classes") or [])}
        oc = mention.get("office_class") or ""
        if classes and oc not in classes:
            continue
        if toks and len(toks) >= 2 and all(is_initial_token(t) for t in toks):
            reasons.append("bare_initials_person")
    return bool(reasons), reasons


def _compound_surnames_from_cfg(cfg: dict) -> list[str]:
    t1 = cfg.get("tier1") or {}
    nc = cfg.get("name_compat") or {}
    out: list[str] = []
    for src in (t1.get("compound_surnames"), nc.get("compound_surnames")):
        if src:
            out.extend(str(x) for x in src if x)
    return out


_BLOCK_TMPL = Formatter()


def _template_placeholders(tmpl: str) -> list[str]:
    names: list[str] = []
    for _, field_name, _, _ in _BLOCK_TMPL.parse(tmpl or ""):
        if not field_name:
            continue
        names.append(field_name.split("!")[0].split(":")[0])
    return names


def _first_two_tokens(normalized: str) -> str | None:
    toks = [t for t in str(normalized or "").split() if t]
    if not toks:
        return None
    return " ".join(toks[:2])


def _corp_suffixes_from_cfg(cfg: dict) -> list[str]:
    return list((cfg.get("normalization") or {}).get("strip_corp_suffixes") or [])


def _corp_core_key_cfg(cfg: dict) -> dict:
    return (cfg.get("normalization") or {}).get("corp_core_key") or {}


def corp_core_for_mention(m: dict, cfg: dict) -> str | None:
    """Config-driven corporate core key (generic; used by Tier0/Tier1 helpers)."""
    cc_cfg = _corp_core_key_cfg(cfg)
    if not cc_cfg.get("enabled"):
        return None
    from engine.normalize import corp_core_name

    suffixes = (
        _corp_suffixes_from_cfg(cfg)
        if cc_cfg.get("use_strip_corp_suffixes", True)
        else []
    )
    max_tok = cc_cfg.get("max_tokens")
    max_tokens = int(max_tok) if max_tok is not None else None
    v = corp_core_name(
        m.get("normalized_name") or "",
        corp_suffixes=suffixes,
        max_tokens=max_tokens,
        hyphen_to_space=bool(cc_cfg.get("hyphen_to_space", True)),
    )
    return v or None


def corporate_shared_prefix_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """Block corporate auto-merges that share token-1 but differ on token-2 (e.g. Owens).

    Compare suffix-stripped corp-core tokens so ``gaf corp`` vs ``gaf corporation``
    is not treated as a prefix family split.
    """
    excl = (cfg.get("identity_exclusions") or {})
    if not excl.get("corporate_shared_prefix_distinct_second_token"):
        return False
    if (ma.get("office_class") or "") != "corporate" or (mb.get("office_class") or "") != "corporate":
        return False
    ca = corp_core_for_mention(ma, cfg) or (ma.get("normalized_name") or "")
    cb = corp_core_for_mention(mb, cfg) or (mb.get("normalized_name") or "")
    ta = [t for t in str(ca).split() if t]
    tb = [t for t in str(cb).split() if t]
    if len(ta) < 2 or len(tb) < 2:
        return False
    return ta[0] == tb[0] and ta[1] != tb[1]


def _excl(cfg: dict | None) -> dict:
    return (cfg or {}).get("identity_exclusions") or {}


def _norm_name(m: dict) -> str:
    return (m.get("normalized_name") or "").strip().lower()


def _raw_name(m: dict) -> str:
    return (m.get("raw_name") or m.get("normalized_name") or "").strip().lower()


def government_jurisdiction_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """Distinct remainder after state/commonwealth/district-of prefix → conflict.

    Token-1/token-2 prefix guard misses ``state of {X}`` vs ``state of {Y}``
    (difference is token 3+). Config-driven prefixes + trailing-token strip.
    """
    spec = _excl(cfg).get("government_jurisdiction_remainder") or {}
    if not spec.get("enabled"):
        return False
    prefixes = [str(p).strip().lower() for p in (spec.get("prefixes") or []) if str(p).strip()]
    prefixes = sorted(prefixes, key=len, reverse=True)
    strip_tr = {str(t).strip().lower() for t in (spec.get("strip_trailing_tokens") or []) if str(t).strip()}

    def remainder(name: str) -> str | None:
        n = (name or "").strip().lower()
        if not n:
            return None
        for p in prefixes:
            if n == p or n.startswith(p + " "):
                rest = n[len(p) :].strip()
                toks = [t for t in rest.split() if t not in strip_tr]
                return " ".join(toks)
        return None

    ra, rb = remainder(_norm_name(ma)), remainder(_norm_name(mb))
    if ra is None or rb is None:
        return False
    return ra != rb


def _marker_hits(text: str, markers: list[str]) -> set[str]:
    n = f" {text.lower()} "
    hits: set[str] = set()
    for mk in markers:
        m = str(mk).strip().lower()
        if not m:
            continue
        if m in n or m in text.lower():
            hits.add(m)
    return hits


def _slash_join_parts(text: str) -> list[str] | None:
    """Content-bearing slash joins (``dezurik/copes-vulcan``), not ``p/o`` / ``l/l/c``."""
    raw = (text or "").strip()
    if "/" not in raw:
        return None
    parts = [re.sub(r"\s+", " ", p).strip().lower() for p in raw.split("/") if p.strip()]
    if len(parts) < 2:
        return None
    letters = [re.sub(r"[^a-z]", "", p) for p in parts]
    if all(len(x) <= 2 for x in letters):
        return None
    return parts


def division_subsidiary_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """One name has an explicit division/subsidiary/slash-suite marker the other lacks."""
    spec = _excl(cfg).get("division_subsidiary") or {}
    if not spec.get("enabled"):
        return False
    markers = [str(x) for x in (spec.get("markers") or []) if str(x).strip()]
    texts_a = f"{_norm_name(ma)} {_raw_name(ma)}"
    texts_b = f"{_norm_name(mb)} {_raw_name(mb)}"
    ha, hb = _marker_hits(texts_a, markers), _marker_hits(texts_b, markers)
    if ha != hb and (ha or hb):
        return True
    if spec.get("slash_join_distinct_entities"):
        pa = _slash_join_parts(ma.get("raw_name") or "") or _slash_join_parts(ma.get("normalized_name") or "")
        pb = _slash_join_parts(mb.get("raw_name") or "") or _slash_join_parts(mb.get("normalized_name") or "")
        if pa and pb:
            sa = {p.replace("-", " ") for p in pa}
            sb = {p.replace("-", " ") for p in pb}
            return sa != sb
        if pa and not pb:
            other = _norm_name(mb)
            return not all(p.replace("-", " ") in other.replace("-", " ") for p in pa)
        if pb and not pa:
            other = _norm_name(ma)
            return not all(p.replace("-", " ") in other.replace("-", " ") for p in pb)
    return False


def _is_placeholder_mention(m: dict, spec: dict) -> bool:
    want = str(spec.get("office_class") or "placeholder")
    if (m.get("office_class") or "") == want:
        return True
    name = _norm_name(m)
    raw = _raw_name(m)
    for pat in spec.get("name_patterns") or []:
        try:
            if re.search(pat, name) or re.search(pat, raw):
                return True
        except re.error:
            continue
    return False


def placeholder_identity_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    spec = _excl(cfg).get("placeholder") or {}
    if not spec.get("enabled"):
        return False
    if not (_is_placeholder_mention(ma, spec) or _is_placeholder_mention(mb, spec)):
        return False
    same_ucid = bool(ma.get("ucid") and ma.get("ucid") == mb.get("ucid"))
    same_name = _norm_name(ma) == _norm_name(mb) and bool(_norm_name(ma))
    if same_ucid and same_name:
        return False
    if spec.get("never_cross_ucid") and not same_ucid:
        return True
    if spec.get("require_identical_name") and not same_name:
        return True
    return False


def _role_side(m: dict, spec: dict) -> str | None:
    plaintiff = {str(x).strip().lower() for x in (spec.get("plaintiff_side") or []) if str(x).strip()}
    defendant = {str(x).strip().lower() for x in (spec.get("defendant_side") or []) if str(x).strip()}
    fields = spec.get("fields") or ["party_type", "party_role"]
    sides: set[str] = set()
    for f in fields:
        raw = (m.get(f) or "").strip().lower()
        if not raw:
            continue
        toks = [t for t in re.split(r"[^a-z]+", raw) if t]
        if raw in plaintiff or any(t in plaintiff for t in toks):
            sides.add("plaintiff")
        if raw in defendant or any(t in defendant for t in toks):
            sides.add("defendant")
    if sides == {"plaintiff"}:
        return "plaintiff"
    if sides == {"defendant"}:
        return "defendant"
    return None


def opposing_roles_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    spec = _excl(cfg).get("opposing_roles") or {}
    if not spec.get("enabled"):
        return False
    if ma.get("pacer_id") and mb.get("pacer_id") and str(ma["pacer_id"]) == str(mb["pacer_id"]):
        return False
    sa, sb = _role_side(ma, spec), _role_side(mb, spec)
    return bool(sa and sb and sa != sb)


def person_title_prefix_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """P/O (and similar) is a person-title prefix, not a company name."""
    spec = _excl(cfg).get("person_title_prefixes") or {}
    if not spec.get("enabled"):
        return False
    prefixes = [str(p).strip().lower() for p in (spec.get("prefixes") or []) if str(p).strip()]
    if not prefixes:
        return False

    def has_prefix(m: dict) -> bool:
        for text in (_raw_name(m), _norm_name(m)):
            t = text.strip()
            for p in prefixes:
                if t == p:
                    return True
                if t.startswith(p) and len(t) > len(p) and not t[len(p)].isalpha():
                    return True
        return False

    return has_prefix(ma) != has_prefix(mb)


_IDENTITY_STOP = {
    "and", "of", "the", "for", "to", "a", "an", "in", "at", "as", "by",
    "co", "inc", "corp", "llc", "ltd", "l", "p", "c", "et", "al", "no",
}


def _identity_generic_words(cfg: dict) -> set[str]:
    spec = _excl(cfg).get("generic_word_overlap") or {}
    words = {str(x).strip().lower() for x in (spec.get("words") or []) if str(x).strip()}
    suffixes = {
        str(x).strip(".").lower()
        for x in ((cfg.get("normalization") or {}).get("strip_corp_suffixes") or [])
        if str(x).strip()
    }
    return words | suffixes | set(_IDENTITY_STOP)


def _identity_tokens(name: str, cfg: dict | None = None) -> list[str]:
    n = (name or "").lower().replace("&", " ").replace("-", " ").replace("/", " ")
    n = re.sub(r"[^a-z0-9\s]", " ", n)
    return [t for t in n.split() if t and t not in _IDENTITY_STOP]


def _token_sets(name: str, cfg: dict) -> tuple[set[str], set[str], set[str]]:
    """(all, distinctive, generic) token sets for identity comparisons."""
    generic = _identity_generic_words(cfg)
    all_t = set(_identity_tokens(name, cfg))
    gen = {t for t in all_t if t in generic}
    return all_t, all_t - generic, gen


def fund_plan_type_tokens(m: dict, cfg: dict) -> set[str]:
    """Plan-type tokens present in a fund/trust/plan name (empty if not a typed fund)."""
    spec = _excl(cfg).get("fund_plan_type") or {}
    if not spec.get("enabled"):
        return set()
    markers = {str(x).strip().lower() for x in (spec.get("fund_markers") or []) if str(x).strip()}
    types = {str(x).strip().lower() for x in (spec.get("type_tokens") or []) if str(x).strip()}
    toks = set(_identity_tokens(_norm_name(m), cfg))
    if not (toks & markers):
        return set()
    return toks & types


def fund_plan_type_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """Health/welfare vs pension/retirement (etc.) funds are distinct plans.

    Round 1 IBEW/Flintkote samples were never given a general rule — name_gate
    only fired when distinctive sponsor tokens already disagreed.
    """
    type_a, type_b = fund_plan_type_tokens(ma, cfg), fund_plan_type_tokens(mb, cfg)
    if not type_a or not type_b:
        return False
    return type_a.isdisjoint(type_b)


def _generational_markers(name: str, suffixes: set[str]) -> set[str]:
    raw = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower()).split()
    found = {t for t in raw if t in suffixes}
    out: set[str] = set()
    for t in found:
        if t in {"junior", "jr"}:
            out.add("jr")
        elif t in {"senior", "sr"}:
            out.add("sr")
        else:
            out.add(t)
    return out


def generational_suffix_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """Sr/Jr distinguish parties by default — do not inherit judges' strip-merge."""
    spec = _excl(cfg).get("generational_suffix") or {}
    if not spec.get("enabled"):
        return False
    suffixes = {
        str(x).strip().lower().rstrip(".") for x in (spec.get("suffixes") or []) if str(x).strip()
    }
    if not suffixes:
        return False
    ga = _generational_markers(_norm_name(ma), suffixes) | _generational_markers(
        _raw_name(ma), suffixes
    )
    gb = _generational_markers(_norm_name(mb), suffixes) | _generational_markers(
        _raw_name(mb), suffixes
    )
    if not ga and not gb:
        return False
    if ga and gb:
        return ga != gb  # Sr vs Jr always distinct; same suffix OK
    # One-sided: block unless same UCID + same role (inconsistent labeling)
    if spec.get("allow_one_sided_if_same_ucid_and_role"):
        same_ucid = bool(ma.get("ucid") and ma.get("ucid") == mb.get("ucid"))
        ra = (ma.get("party_role") or ma.get("party_type") or "").strip().lower()
        rb = (mb.get("party_role") or mb.get("party_type") or "").strip().lower()
        if same_ucid and ra and ra == rb:
            return False
    return True


def _legal_entity_forms(name: str, forms_cfg: dict) -> set[str]:
    n = f" {(name or '').lower().replace('&', ' and ')} "
    found: set[str] = set()
    for form_id, aliases in (forms_cfg or {}).items():
        for alias in aliases or []:
            a = str(alias).strip().lower()
            if not a:
                continue
            if f" {a} " in n or re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", n):
                found.add(str(form_id))
                break
    return found


def legal_entity_form_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """L.P. / G.P. / Inc. / LLC / Trust are not interchangeable when both present."""
    spec = _excl(cfg).get("legal_entity_form") or {}
    if not spec.get("enabled"):
        return False
    forms_cfg = spec.get("forms") or {}
    if not forms_cfg:
        return False
    fa = _legal_entity_forms(_norm_name(ma), forms_cfg) | _legal_entity_forms(
        _raw_name(ma), forms_cfg
    )
    fb = _legal_entity_forms(_norm_name(mb), forms_cfg) | _legal_entity_forms(
        _raw_name(mb), forms_cfg
    )
    if not fa or not fb:
        return False
    return fa != fb


def name_core_for_corroboration(name: str, cfg: dict) -> str:
    """Suffix-stripped comparable core for parties Tier2 corroboration."""
    from engine.normalize import strip_corp_suffixes

    t = (name or "").lower().replace("&", " and ").replace("-", " ")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    aliases = {"mfg": "manufacturing", "mfr": "manufacturer"}
    toks = []
    for x in t.split():
        if not x or x in {"the", "a", "an"}:
            continue
        toks.append(aliases.get(x, x))
    t = " ".join(toks)
    suffixes = list((cfg.get("normalization") or {}).get("strip_corp_suffixes") or [])
    t = strip_corp_suffixes(t, suffixes)
    # Drop trailing national-association crumbs left after "n.a."
    parts = t.split()
    while parts and parts[-1] in {"n", "na", "association"}:
        if parts[-1] == "association" and len(parts) >= 2 and parts[-2] == "national":
            parts.pop()
            parts.pop()
            continue
        if parts[-1] in {"n", "na"}:
            parts.pop()
            continue
        break
    return " ".join(parts)


def compatible_name_core(ma: dict, mb: dict, cfg: dict) -> bool:
    """True when suffix-stripped cores represent the same name.

    Rules (all must pass):
    1. Exact match after normalization, OR
    2. fuzz.ratio >= min_core_ratio (catches spelling variants: Patterson-Kelly,
       Crown Cork, McmasterCarr typo), AND
    3. Bilateral distinctive-token coverage: each side's non-generic tokens must
       be ≥ min_core_bilateral_coverage of the other side's.  This rejects
       token_set_ratio artefacts where a short name is a subset of a longer one
       (Sony BMG ⊂ Sony BMG Music Entertainment) or two names share only a
       generic token (MBNA America Bank / Bank of America both contain "america"
       and "bank", but those are generic words).

    token_set_ratio and token_prefix are intentionally removed: both are
    one-sided tests that systematically allow subset / division-name matches
    through, which is the exact root cause of the MBNA/BofA, Sony-BMG, and
    Charter-Consolidated failures in round 4.
    """
    spec = (cfg.get("tier2") or {}).get("auto_merge_requires_corroboration") or {}
    min_ratio = float(spec.get("min_core_ratio") or 88)
    min_bilateral = float(spec.get("min_core_bilateral_coverage") or 0.80)

    ca = name_core_for_corroboration(_norm_name(ma), cfg)
    cb = name_core_for_corroboration(_norm_name(mb), cfg)
    if not ca or not cb:
        return False
    if ca == cb:
        return True

    # Bilateral distinctive-token coverage check.
    #
    # We strip generic words and check that each side's remaining "distinctive"
    # tokens are mostly present on the other side (using per-token fuzzy
    # matching so that typo/pluralisation variants like mcmaster↔mcmasters
    # still count).  This is the primary gate.
    #
    # Additionally require a character-level fuzz.ratio ≥ min_core_ratio ONLY
    # when at least one side has >1 distinctive token — this catches the case
    # where two names share one distinctive token but differ significantly
    # (e.g. "mbna america bank" vs "bank of america": token coverage is high
    # because "america" matches, but ratio=62 exposes the difference).
    # When each side has exactly 1 distinctive token (e.g. "felt products
    # manufacturing" → ["felt"]) the ratio check is skipped since the token
    # coverage alone is conclusive.
    generic = _identity_generic_words(cfg)
    ta = [t for t in ca.split() if t not in generic]
    tb = [t for t in cb.split() if t not in generic]
    if not ta or not tb:
        # Both cores reduced to only generic words — cannot confirm identity.
        return False

    _TOK_SIM = 85  # per-token fuzzy threshold

    def _tok_covered(src: list[str], tgt: list[str]) -> int:
        """Count tokens in src that have a ≥_TOK_SIM fuzzy match in tgt."""
        count = 0
        for s in src:
            if any(fuzz.ratio(s, t) >= _TOK_SIM for t in tgt):
                count += 1
        return count

    cov_a = _tok_covered(ta, tb) / len(ta)
    cov_b = _tok_covered(tb, ta) / len(tb)
    if not (cov_a >= min_bilateral and cov_b >= min_bilateral):
        return False

    # Secondary character-ratio gate: applied only when either side has >1
    # distinctive token, to reject same-single-token pairs from different orgs.
    if len(ta) > 1 or len(tb) > 1:
        if fuzz.ratio(ca, cb) < min_ratio:
            return False

    return True


def parties_tier2_lacks_corroboration(ma: dict, mb: dict, cfg: dict) -> bool:
    """True when parties Tier2 would auto-merge on embedding similarity alone.

    Requires at least one of: shared PACER id, shared contact, or compatible
    name core after entity-suffix normalization (config-driven; off for judges/firms).
    """
    spec = (cfg.get("tier2") or {}).get("auto_merge_requires_corroboration") or {}
    if not spec.get("enabled"):
        return False
    if spec.get("accept_shared_pacer_id", True):
        pa, pb = ma.get("pacer_id"), mb.get("pacer_id")
        if pa is not None and pb is not None and str(pa) == str(pb) and str(pa).strip():
            return False
    if spec.get("accept_shared_contact", True):
        from engine.name_compat import _norm_address, _norm_domain, _norm_phone

        da, db = _norm_domain(ma.get("domain")), _norm_domain(mb.get("domain"))
        if da and db and da == db and not domain_is_non_identifying(da, cfg):
            return False
        pha, phb = _norm_phone(ma.get("phone")), _norm_phone(mb.get("phone"))
        if pha and phb and pha == phb:
            return False
        aa, ab = _norm_address(ma.get("address")), _norm_address(mb.get("address"))
        if aa and ab and aa == ab:
            return False
    if spec.get("accept_compatible_name_core", True) and compatible_name_core(ma, mb, cfg):
        return False
    return True


def parent_subunit_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """Shorter name is a strict token prefix of a named sub-unit or extra brand.

    Catches City of X + City of X Police Department and Bell & + Bell & Gossett ITT
    without blocking one extra product token (Fairbanks Morse Pump, Fiberglas).
    """
    div = _excl(cfg).get("division_subsidiary") or {}
    spec = div.get("parent_prefix") or {}
    if not div.get("enabled") or not spec.get("enabled"):
        return False
    subunit = {str(x).strip().lower() for x in (spec.get("subunit_tokens") or []) if str(x).strip()}
    min_extra = int(spec.get("min_extra_distinctive_tokens") or 2)
    generic = _identity_generic_words(cfg)
    ta = _identity_tokens(_norm_name(ma), cfg)
    tb = _identity_tokens(_norm_name(mb), cfg)
    if not ta or not tb or ta == tb:
        return False
    if len(ta) < len(tb) and tb[: len(ta)] == ta:
        extra = tb[len(ta) :]
    elif len(tb) < len(ta) and ta[: len(tb)] == tb:
        extra = ta[len(tb) :]
    else:
        return False
    if any(t in subunit for t in extra):
        return True
    extra_dist = [t for t in extra if t not in generic]
    return len(extra_dist) >= min_extra


def generic_word_overlap_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """Only shared content is a generic industry word (gasket/fund/trust/industries).

    Same principle as corporate_shared_prefix_conflict, not limited to token position:
    embedding similarity plus one shared generic term is not identity.
    Near-duplicate distinctive tokens (Durabla/Durabala) count as overlap.
    """
    spec = _excl(cfg).get("generic_word_overlap") or {}
    if not spec.get("enabled"):
        return False
    _, dist_a, gen_a = _token_sets(_norm_name(ma), cfg)
    _, dist_b, gen_b = _token_sets(_norm_name(mb), cfg)
    if dist_a & dist_b or _near_duplicate_token_overlap(dist_a, dist_b):
        return False
    if not (gen_a & gen_b):
        return False
    return dist_a != dist_b and bool(dist_a or dist_b)


def _near_duplicate_token_overlap(a: set[str], b: set[str]) -> bool:
    for ta in a:
        for tb in b:
            if min(len(ta), len(tb)) >= 5 and fuzz.ratio(ta, tb) >= 85:
                return True
    return False


def person_vs_org_conflict(ma: dict, mb: dict, cfg: dict) -> bool:
    """Person-shaped vs corporate sharing only a surname/word is not identity."""
    spec = _excl(cfg).get("person_vs_org") or {}
    if not spec.get("enabled"):
        return False
    if ma.get("pacer_id") and mb.get("pacer_id") and str(ma["pacer_id"]) == str(mb["pacer_id"]):
        return False
    person_cls = {str(x) for x in (spec.get("person_classes") or ["nominal_person"])}
    org_cls = {str(x) for x in (spec.get("org_classes") or ["corporate"])}
    ca, cb = ma.get("office_class") or "", mb.get("office_class") or ""
    if ca in person_cls and cb in person_cls:
        return False
    if ca in org_cls and cb in org_cls:
        return False
    if not ((ca in person_cls and cb in org_cls) or (cb in person_cls and ca in org_cls)):
        return False
    min_shared = int(spec.get("min_shared_distinctive_tokens") or 2)
    _, dist_a, _ = _token_sets(_norm_name(ma), cfg)
    _, dist_b, _ = _token_sets(_norm_name(mb), cfg)
    return len(dist_a & dist_b) < min_shared


def identity_pair_conflict(ma: dict, mb: dict, cfg: dict) -> tuple[str, str] | None:
    """First configured identity exclusion that fires, else None.

    Returns (signal, rationale). Empty config → no-op (judges/firms unchanged).
    """
    if corporate_shared_prefix_conflict(ma, mb, cfg):
        return (
            "corporate_shared_prefix",
            "Corporate litigants share first token but differ on second "
            "(distinct corporate families; e.g. Owens-Corning vs Owens-Illinois)",
        )
    if government_jurisdiction_conflict(ma, mb, cfg):
        return (
            "government_jurisdiction",
            "Government entities share a state/commonwealth/district-of prefix "
            "but differ on the jurisdiction remainder",
        )
    if fund_plan_type_conflict(ma, mb, cfg):
        return (
            "fund_plan_type",
            "Both names are funds/trusts/plans with disjoint plan-type tokens "
            "(e.g. health/welfare vs pension/retirement)",
        )
    if generational_suffix_conflict(ma, mb, cfg):
        return (
            "generational_suffix",
            "Generational suffixes (Sr/Jr/II/…) distinguish party identity; "
            "conflicting or uncorroborated one-sided suffixes block merge",
        )
    if legal_entity_form_conflict(ma, mb, cfg):
        return (
            "legal_entity_form",
            "Legal entity forms (L.P./G.P./Inc./LLC/Trust) are not interchangeable",
        )
    if division_subsidiary_conflict(ma, mb, cfg) or parent_subunit_conflict(ma, mb, cfg):
        return (
            "division_subsidiary",
            "One name carries a division/subsidiary marker or is a named "
            "sub-unit/extra brand of the other's parent string",
        )
    if generic_word_overlap_conflict(ma, mb, cfg):
        return (
            "generic_word_overlap",
            "Names share only a generic industry/legal term with no shared "
            "distinctive tokens; embedding similarity is not a second signal",
        )
    if person_vs_org_conflict(ma, mb, cfg):
        return (
            "person_vs_org",
            "Nominal-person vs corporate sharing fewer than two distinctive "
            "tokens is not identity without a shared PACER id",
        )
    if placeholder_identity_conflict(ma, mb, cfg):
        return (
            "placeholder",
            "Anonymous Doe/placeholder parties do not merge across UCIDs "
            "or distinct placeholder strings",
        )
    if opposing_roles_conflict(ma, mb, cfg):
        return (
            "opposing_roles",
            "Plaintiff-side vs defendant-side is a hard contradiction without "
            "shared PACER party id",
        )
    if person_title_prefix_conflict(ma, mb, cfg):
        return (
            "person_title_prefix",
            "Person-title prefix (e.g. P/O) is a person-vs-company signal",
        )
    return None


def _content_tokens(name: str) -> set[str]:
    stop = {
        "and", "of", "the", "for", "to", "a", "an", "in", "at", "co", "inc",
        "corp", "llc", "ltd", "company", "corporation", "l", "p", "c",
    }
    return {t for t in tokens(name) if len(t) >= 3 and t not in stop}


def ucid_only_corroboration(ma: dict, mb: dict, cfg: dict) -> bool:
    """True when the only shared case context is UCID/court/year (weak for parties)."""
    if not ma.get("ucid") or ma.get("ucid") != mb.get("ucid"):
        return False
    if ma.get("pacer_id") and mb.get("pacer_id") and str(ma["pacer_id"]) == str(mb["pacer_id"]):
        return False
    if _norm_name(ma) and _norm_name(ma) == _norm_name(mb):
        return False
    from engine.name_compat import _norm_address, _norm_domain, _norm_phone

    da, db = _norm_domain(ma.get("domain")), _norm_domain(mb.get("domain"))
    if da and db and da == db and not domain_is_non_identifying(da, cfg):
        return False
    pa, pb = _norm_phone(ma.get("phone")), _norm_phone(mb.get("phone"))
    if pa and pb and pa == pb:
        return False
    aa, ab = _norm_address(ma.get("address")), _norm_address(mb.get("address"))
    if aa and ab and aa == ab:
        return False
    if person_initials_expand_compatible(ma, mb):
        return False
    return True


def _block_slot_value(
    m: dict,
    spec: str,
    *,
    court: str,
    year: Any,
    initials: str,
    surname: str,
    cfg: dict | None = None,
) -> str | None:
    """Resolve a key_template slot from the mention (config-generic).

    ``fields:`` may map a placeholder to a mention column or a helper
    (``first_two_tokens``, ``surname_from_normalized_name``). Empty / missing
    values return None so the strategy is skipped — never emit ``domain|``.
    """
    spec = str(spec or "").strip()
    if not spec:
        return None
    if spec == "court":
        return str(court) if court else None
    if spec == "year":
        if year is None or year == "":
            return None
        return str(year)
    if spec in {"initials", "first_last_initials"}:
        return str(initials) if initials else None
    if spec in {"surname", "surname_from_normalized_name"}:
        return str(surname) if surname else None
    if spec in {"name_prefix", "first_two_tokens"}:
        return _first_two_tokens(m.get("normalized_name") or "")
    if spec in {"corp_core", "corp_core_from_normalized_name"} and cfg:
        v = corp_core_for_mention(m, cfg)
        return v
    v = m.get(spec)
    if v is None or v == "":
        return None
    return str(v)


def build_profile_blocks(mentions: list[dict], cfg: dict) -> dict[str, list[str]]:
    """mention_id -> list of block keys.

    Substitutes any placeholder named in ``key_template`` from the mention
    (or from ``fields:`` mapping / helpers). Judges keep compound last-1 /
    last-2 iteration when ``{surname}`` is present.
    """
    strategies = (cfg.get("tier1") or {}).get("strategies") or []
    residual = (cfg.get("tier1") or {}).get("residual_bucket", "_UNBLOCKED_")
    compounds = _compound_surnames_from_cfg(cfg)
    out: dict[str, list[str]] = {}
    for m in mentions:
        keys: list[str] = []
        seen_keys: set[str] = set()
        surnames = surname_block_keys(m.get("normalized_name") or "", compound_surnames=compounds)
        if not surnames and (m.get("surname") or ""):
            surnames = [str(m["surname"]).lower()]
        initials = first_last_initials(m.get("normalized_name") or "")
        court = m.get("court") or ""
        year = m.get("year")
        for s in strategies:
            if s.get("optional") and m.get("year") is None:
                continue
            req_cls = s.get("require_office_class")
            if req_cls and (m.get("office_class") or "") != req_cls:
                continue
            tmpl = s.get("key_template", "")
            field_map = s.get("fields") or {}
            placeholders = _template_placeholders(tmpl)
            sur_iter = surnames if "{surname}" in tmpl else [m.get("surname") or ""]
            if not sur_iter:
                sur_iter = [""]
            for sur in sur_iter:
                mapping: dict[str, str] = {}
                skip = False
                for ph in placeholders:
                    spec = field_map.get(ph, ph)
                    val = _block_slot_value(
                        m,
                        spec,
                        court=court,
                        year=year,
                        initials=initials,
                        surname=sur,
                        cfg=cfg,
                    )
                    if val is None or val == "":
                        skip = True
                        break
                    mapping[ph] = val
                if skip:
                    continue
                try:
                    key = tmpl.format(**mapping)
                except Exception:
                    continue
                if not key or key.endswith("|") or "||" in key:
                    continue
                full = f"{s['id']}::{key}"
                if full not in seen_keys:
                    seen_keys.add(full)
                    keys.append(full)
        if not keys:
            keys = [residual]
        out[m["mention_id"]] = keys
    return out


def invert_blocks(mention_blocks: dict[str, list[str]]) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = defaultdict(list)
    for mid, keys in mention_blocks.items():
        for k in keys:
            blocks[k].append(mid)
    return dict(blocks)


def same_block(a_id: str, b_id: str, mention_blocks: dict[str, list[str]]) -> bool:
    return bool(set(mention_blocks.get(a_id, [])) & set(mention_blocks.get(b_id, [])))


def transfer_conflict(a: dict, b: dict) -> bool:
    an, bn = a.get("normalized_name"), b.get("normalized_name")
    if not an or not bn:
        return False
    return bn in (a.get("transfer_partners") or []) or an in (b.get("transfer_partners") or [])


def _match_forbidden(m: dict, match: dict, cfg: dict | None = None) -> bool:
    """Generic T0 skip filters from rule.match; domain lists union cfg.identity_exclusions."""
    if not match:
        return False
    excl = (cfg or {}).get("identity_exclusions") or {}

    forbid_cls = {str(x) for x in (match.get("forbid_office_class") or [])}
    if forbid_cls and (m.get("office_class") or "") in forbid_cls:
        return True

    dom = (m.get("domain") or "").strip().lower()
    forbid_dom = {str(x).strip().lower() for x in (match.get("forbid_domains") or [])}
    # One source of truth: when a rule forbids domains, always include identity_exclusions.
    if match.get("forbid_domains") is not None:
        forbid_dom |= {str(x).strip().lower() for x in (excl.get("non_identifying_domains") or [])}
    if dom and dom in forbid_dom:
        return True

    suffixes = list(match.get("forbid_domain_suffixes") or [])
    if match.get("forbid_domain_suffixes") is not None:
        suffixes = list(
            dict.fromkeys(suffixes + list(excl.get("non_identifying_domain_suffixes") or []))
        )
    for suf in suffixes:
        s = str(suf).strip().lower()
        if s and dom.endswith(s):
            return True

    nn = (m.get("normalized_name") or "").strip().lower()
    forbid_names = {str(x).strip().lower() for x in (match.get("forbid_normalized_names") or [])}
    if match.get("forbid_normalized_names") is not None:
        forbid_names |= {str(x).strip().lower() for x in (excl.get("generic_office_names") or [])}
    if nn and nn in forbid_names:
        return True
    return False


def domain_is_non_identifying(domain: str | None, cfg: dict | None = None) -> bool:
    """True when domain is shared institutional / free-mail (never firm identity)."""
    d = (domain or "").strip().lower()
    if not d:
        return True
    excl = (cfg or {}).get("identity_exclusions") or {}
    if d in {str(x).strip().lower() for x in (excl.get("non_identifying_domains") or [])}:
        return True
    for suf in excl.get("non_identifying_domain_suffixes") or []:
        s = str(suf).strip().lower()
        if s and d.endswith(s):
            return True
    return False


def _norm_office_city(city: str | None) -> str | None:
    c = re.sub(r"\s+", " ", (city or "").strip().lower())
    return c or None


def mention_office_geo(m: dict, cfg: dict | None = None) -> tuple[str | None, str | None]:
    """Address-derived (state, city). Never uses case court.

    Prefers mention.office_state / office_city; if missing, parses address
    (needed for private firms that do not store institutional geo fields).
    """
    st = (m.get("office_state") or "").strip().upper() or None
    city = _norm_office_city(m.get("office_city"))
    if (not st or not city) and (m.get("address") or "").strip():
        from engine.normalize import parse_office_address_geo

        geo = parse_office_address_geo(m.get("address"))
        st = st or ((geo.get("office_state") or "").strip().upper() or None)
        city = city or _norm_office_city(geo.get("office_city"))
    return st, city


def institutional_office_geo_conflict(
    a: dict, b: dict, cfg: dict | None = None
) -> tuple[bool, str]:
    """Institutional geo conflict: different state, or same state + different city.

    Returns (conflict, signal) where signal is ``office_state_conflict`` or
    ``office_city_conflict``. Address-derived only — never case court.
    """
    excl = (cfg or {}).get("identity_exclusions") or {}
    inst = {str(x) for x in (excl.get("institutional_office_classes") or [])}
    if not inst:
        return False, ""
    ca, cb = a.get("office_class") or "", b.get("office_class") or ""
    if ca not in inst or cb not in inst:
        return False, ""
    sa, city_a = mention_office_geo(a, cfg)
    sb, city_b = mention_office_geo(b, cfg)
    if sa and sb and sa != sb:
        return True, "office_state_conflict"
    if sa and sb and sa == sb and city_a and city_b and city_a != city_b:
        return True, "office_city_conflict"
    return False, ""


def institutional_office_state_conflict(a: dict, b: dict, cfg: dict | None = None) -> bool:
    """Backward-compatible: True for any institutional state/city geo conflict."""
    conflict, _ = institutional_office_geo_conflict(a, b, cfg)
    return conflict


def private_cross_geo_lacks_corroboration(a: dict, b: dict, cfg: dict | None = None) -> bool:
    """Class 5: private firms across state/city need shared domain/phone/address.

    Institutional classes are handled by institutional_office_geo_conflict.
    Cross-geo = different address-derived state, or same state + different city.
    Corroboration = shared *identifying* domain, shared phone, or shared address.
    """
    cfg = cfg or {}
    if cfg.get("entity_type") != "firm":
        return False
    excl = cfg.get("identity_exclusions") or {}
    inst = {str(x) for x in (excl.get("institutional_office_classes") or [])}
    ca, cb = a.get("office_class") or "", b.get("office_class") or ""
    if (ca in inst) or (cb in inst):
        return False

    sa, city_a = mention_office_geo(a, cfg)
    sb, city_b = mention_office_geo(b, cfg)
    cross = False
    if sa and sb and sa != sb:
        cross = True
    elif sa and sb and sa == sb and city_a and city_b and city_a != city_b:
        cross = True
    if not cross:
        return False

    from engine.name_compat import _norm_address, _norm_domain, _norm_phone

    da, db = _norm_domain(a.get("domain")), _norm_domain(b.get("domain"))
    if da and db and da == db and not domain_is_non_identifying(da, cfg):
        return False
    pa, pb = _norm_phone(a.get("phone")), _norm_phone(b.get("phone"))
    if pa and pb and pa == pb:
        return False
    aa, ab = _norm_address(a.get("address")), _norm_address(b.get("address"))
    if aa and ab and aa == ab:
        return False
    return True


def build_tier3_evidence(ma: dict, mb: dict, *, barrier: bool, reasons: list, cfg: dict) -> dict:
    """Evidence bundle for Tier3 — same_domain only when domain is identifying."""
    da = (ma.get("domain") or "").strip().lower()
    db = (mb.get("domain") or "").strip().lower()
    domains_equal = bool(da and db and da == db)
    non_id = domain_is_non_identifying(da, cfg) if domains_equal else False
    sa, city_a = mention_office_geo(ma, cfg)
    sb, city_b = mention_office_geo(mb, cfg)
    geo_conflict, geo_signal = institutional_office_geo_conflict(ma, mb, cfg)
    overlap = sorted(set(ma.get("co_mentions") or []) & set(mb.get("co_mentions") or []))
    overlap_total = len(overlap)
    max_overlap = int(
        ((cfg.get("tier3") or {}).get("evidence") or {}).get("co_mentions_overlap_max", 0) or 0
    )
    co_mentions_field: Any = overlap
    if max_overlap > 0 and overlap_total > max_overlap:
        co_mentions_field = overlap[:max_overlap]
    evidence: dict[str, Any] = {
        "same_court": ma.get("court") == mb.get("court"),
        "same_year": ma.get("year") == mb.get("year"),
        "same_ucid": ma.get("ucid") == mb.get("ucid"),
        "fjc_nid_a": ma.get("fjc_nid"),
        "fjc_nid_b": mb.get("fjc_nid"),
        "transfer_conflict": transfer_conflict(ma, mb),
        "information_content_barrier_triggered": barrier,
        "barrier_reasons": reasons,
        "co_mentions_overlap_count": overlap_total,
        "co_mentions_overlap": co_mentions_field,
        "office_class_a": ma.get("office_class"),
        "office_class_b": mb.get("office_class"),
        "office_state_a": sa,
        "office_state_b": sb,
        "office_city_a": city_a,
        "office_city_b": city_b,
        "same_office_state": bool(sa and sb and sa == sb),
        "same_office_city": bool(city_a and city_b and city_a == city_b),
        "office_state_conflict": bool(sa and sb and sa != sb),
        "office_city_conflict": bool(
            sa and sb and sa == sb and city_a and city_b and city_a != city_b
        ),
        "institutional_geo_conflict": geo_conflict,
        "institutional_geo_signal": geo_signal or None,
        "geo_note": (
            "OfficeState/OfficeCity are from the office ADDRESS only — "
            "never invent them from the case court field"
        ),
        "party_role_a": ma.get("party_role"),
        "party_role_b": mb.get("party_role"),
        "party_type_a": ma.get("party_type"),
        "party_type_b": mb.get("party_type"),
        "same_ucid_is_weak": bool((cfg.get("tier3") or {}).get("ucid_weak_context")),
        "ucid_only_corroboration": ucid_only_corroboration(ma, mb, cfg),
        "identity_conflict": (identity_pair_conflict(ma, mb, cfg) or (None, None))[0],
        "initials_expand_compatible": person_initials_expand_compatible(ma, mb),
    }
    if max_overlap > 0 and overlap_total > max_overlap:
        evidence["co_mentions_overlap_truncated"] = True
        evidence["co_mentions_overlap_note"] = (
            f"List capped at {max_overlap} of {overlap_total} shared co-mentions "
            "(MDL docket size). Shared UCID/co-defendants are NOT identity evidence "
            "when same_ucid_is_weak is true."
        )
    if domains_equal and not non_id:
        evidence["same_domain"] = True
        evidence["domain"] = da
    elif domains_equal and non_id:
        evidence["same_domain"] = False
        evidence["shared_non_identifying_domain"] = True
        evidence["domain"] = da
        evidence["domain_note"] = (
            "Shared institutional/free-mail domain is NOT identity evidence"
        )
    else:
        evidence["same_domain"] = False
    return evidence


def tier0_merge_groups(mentions: list[dict], cfg: dict, journal: DecisionJournal, uf: UnionFind) -> dict:
    """Apply deterministic strong keys. Returns stats."""
    rules = (cfg.get("tier0") or {}).get("rules") or []
    by_id = {m["mention_id"]: m for m in mentions}
    for m in mentions:
        uf.add(m["mention_id"])

    stats = {"rules_fired": defaultdict(int), "merges": 0, "fjc_skipped": False}

    # Config alias groups (USA / United States / …) before conjunction rules.
    alias_stats = apply_tier0_alias_groups(mentions, cfg, journal, uf)
    stats["alias_group_merges"] = alias_stats.get("merges", 0)
    for k, v in (alias_stats.get("rules_fired") or {}).items():
        stats["rules_fired"][k] += v
    stats["merges"] += int(alias_stats.get("merges", 0))

    # Index helpers
    from engine.normalize import generational_core_name as _core_name

    def _field_value(field: str, m: dict, rule: dict) -> str | int | None:
        # Same helpers as blocking (first_two_tokens, etc.) so T0 keys can
        # use derived slots without writing them onto the mention.
        if field in {"first_two_tokens", "name_prefix"}:
            return _first_two_tokens(m.get("normalized_name") or "")
        if field in {"corp_core", "corp_core_from_normalized_name"}:
            v = corp_core_for_mention(m, cfg)
            return v or None
        if field in {"first_last_initials", "initials"}:
            v = first_last_initials(m.get("normalized_name") or "")
            return v or None
        if field in {"surname_from_normalized_name"}:
            from engine.normalize import surname as _sur

            v = _sur(m.get("normalized_name") or "")
            return v or None
        v = m.get(field)
        if v is None or v == "":
            return None
        if field == "normalized_name" and rule.get("suffix_equiv_generational"):
            v = _core_name(str(v))
            if not v:
                return None
        return v

    def group_key(fields: list[str], m: dict, rule: dict) -> tuple | None:
        vals = []
        for f in fields:
            v = _field_value(f, m, rule)
            if v is None or v == "":
                return None
            vals.append(v)
        return tuple(vals)

    for rule in rules:
        rid = rule["id"]
        if rule.get("type") == "external_id_join":
            # Merge all mentions that already share the same linked FJC NID.
            # Linking happens in run_cascade before Tier0.
            ext = rule.get("external") or {}
            path = resolve_path(cfg, ext.get("path", ""))
            if not path.exists():
                stats["fjc_skipped"] = True
                continue
            buckets: dict[str, list[str]] = defaultdict(list)
            for m in mentions:
                nid = m.get("fjc_nid")
                if nid:
                    buckets[str(nid)].append(m["mention_id"])
            for nid, ids in buckets.items():
                if len(ids) < 2:
                    continue
                ids = sorted(ids)
                root = ids[0]
                for other in ids[1:]:
                    if uf.find(root) == uf.find(other):
                        continue
                    if transfer_conflict(by_id[root], by_id[other]):
                        journal.log(
                            {
                                "decision_id": f"dec_{journal.n:08d}",
                                "mention_id_a": root,
                                "mention_id_b": other,
                                "entity_type": cfg.get("entity_type"),
                                "decision": "NO_MATCH",
                                "confidence": 100,
                                "method": f"tier0.transfer_block:{rid}",
                                "rationale": "Transfer clue conflict blocks FJC NID merge",
                                "signals": ["transfer_conflict", "fjc_nid_conflict"],
                                "evidence": {"rule": rid, "fjc_nid": nid},
                                "timestamp": _now(),
                                "config_version": cfg.get("version"),
                            }
                        )
                        continue
                    # Distinct FJC NIDs should never reach this bucket; same NID → merge
                    uf.union(root, other)
                    stats["merges"] += 1
                    stats["rules_fired"][rid] += 1
                    journal.log(
                        {
                            "decision_id": f"dec_{journal.n:08d}",
                            "mention_id_a": root,
                            "mention_id_b": other,
                            "entity_type": cfg.get("entity_type"),
                            "decision": "MERGE_TIER0",
                            "confidence": int(rule.get("confidence", 100)),
                            "method": rule.get("method") or f"tier0.{rid}",
                            "rationale": f"Same FJC NID {nid}",
                            "signals": ["fjc_nid_match"],
                            "evidence": {"rule": rid, "fjc_nid": nid},
                            "timestamp": _now(),
                            "config_version": cfg.get("version"),
                        }
                    )
            continue

        if rule.get("type") != "conjunction":
            continue

        fields = rule.get("fields") or []
        require_non_null = set(rule.get("require_non_null") or [])
        min_tokens = rule.get("min_name_tokens")
        min_cc_tokens = rule.get("min_corp_core_tokens")
        buckets: dict[tuple, list[str]] = defaultdict(list)

        for m in mentions:
            if any(not m.get(f) for f in require_non_null):
                continue
            if min_cc_tokens is not None and "corp_core" in fields:
                cc = corp_core_for_mention(m, cfg)
                if not cc or len(tokens(cc)) < int(min_cc_tokens):
                    continue
            name_for_tokens = m.get("normalized_name") or ""
            if rule.get("suffix_equiv_generational"):
                name_for_tokens = _core_name(name_for_tokens)
            if min_tokens is not None and len(tokens(name_for_tokens)) < min_tokens:
                continue
            # Optional match.require filters (generic; used by firms biglaw_subclass)
            reqs = (rule.get("match") or {}).get("require") or []
            skip = False
            for req in reqs:
                if isinstance(req, str) and req.startswith("office_class_equals_"):
                    want = req[len("office_class_equals_") :]
                    if (m.get("office_class") or "") != want:
                        skip = True
                        break
                elif req == "mention_linked_nid_equal":
                    continue  # handled by external_id_join
            if skip:
                continue
            if _match_forbidden(m, rule.get("match") or {}, cfg):
                continue
            key = group_key(fields, m, rule)
            if key is None:
                continue
            buckets[key].append(m["mention_id"])

        # Merge after the full bucket index is built (was incorrectly nested inside
        # the mention loop — O(n²) duplicate journal rows; same UF outcome).
        for key, ids in buckets.items():
            if len(ids) < 2:
                continue
            ids = sorted(ids)
            root = ids[0]
            for other in ids[1:]:
                if uf.find(root) == uf.find(other):
                    continue
                na, nb = by_id[root], by_id[other]
                if (
                    na.get("fjc_nid")
                    and nb.get("fjc_nid")
                    and str(na["fjc_nid"]) != str(nb["fjc_nid"])
                ):
                    journal.log(
                        {
                            "decision_id": f"dec_{journal.n:08d}",
                            "mention_id_a": root,
                            "mention_id_b": other,
                            "entity_type": cfg.get("entity_type"),
                            "decision": "NO_MATCH",
                            "confidence": 100,
                            "method": f"tier0.fjc_nid_conflict:{rid}",
                            "rationale": "Distinct FJC NIDs block name-key merge",
                            "signals": ["fjc_nid_conflict"],
                            "evidence": {
                                "rule": rid,
                                "fjc_nid_a": na.get("fjc_nid"),
                                "fjc_nid_b": nb.get("fjc_nid"),
                            },
                            "timestamp": _now(),
                            "config_version": cfg.get("version"),
                        }
                    )
                    continue
                if transfer_conflict(na, nb):
                    journal.log(
                        {
                            "decision_id": f"dec_{journal.n:08d}",
                            "mention_id_a": root,
                            "mention_id_b": other,
                            "entity_type": cfg.get("entity_type"),
                            "decision": "NO_MATCH",
                            "confidence": 100,
                            "method": f"tier0.transfer_block:{rid}",
                            "rationale": "Transfer clue conflict blocks Tier0 merge",
                            "signals": ["transfer_conflict"],
                            "evidence": {"rule": rid, "key": list(key)},
                            "timestamp": _now(),
                            "config_version": cfg.get("version"),
                        }
                    )
                    continue
                conflict, geo_sig = institutional_office_geo_conflict(na, nb, cfg)
                if conflict:
                    sa, city_a = mention_office_geo(na, cfg)
                    sb, city_b = mention_office_geo(nb, cfg)
                    journal.log(
                        {
                            "decision_id": f"dec_{journal.n:08d}",
                            "mention_id_a": root,
                            "mention_id_b": other,
                            "entity_type": cfg.get("entity_type"),
                            "decision": "NO_MATCH",
                            "confidence": 100,
                            "method": f"tier0.{geo_sig}:{rid}",
                            "rationale": (
                                "Institutional offices with different address-derived "
                                f"{geo_sig.replace('_', ' ')} cannot merge "
                                "(case court is not office district)"
                            ),
                            "signals": [geo_sig],
                            "evidence": {
                                "rule": rid,
                                "office_state_a": sa,
                                "office_state_b": sb,
                                "office_city_a": city_a,
                                "office_city_b": city_b,
                            },
                            "timestamp": _now(),
                            "config_version": cfg.get("version"),
                        }
                    )
                    continue
                # A4 / pair_allowed is Tier2+Tier3 only — never block Tier0 exact keys
                uf.union(root, other)
                stats["merges"] += 1
                stats["rules_fired"][rid] += 1
                journal.log(
                    {
                        "decision_id": f"dec_{journal.n:08d}",
                        "mention_id_a": root,
                        "mention_id_b": other,
                        "entity_type": cfg.get("entity_type"),
                        "decision": "MERGE_TIER0",
                        "confidence": int(rule.get("confidence", 100)),
                        "method": rule.get("method") or f"tier0.{rid}",
                        "rationale": f"Deterministic strong-key merge via {rid}",
                        "signals": ["exact_name"],
                        "evidence": {"rule": rid, "key": [str(x) for x in key]},
                        "timestamp": _now(),
                        "config_version": cfg.get("version"),
                    }
                )

    stats["rules_fired"] = dict(stats["rules_fired"])
    _tier0_ucid_prefix_span_merges(mentions, by_id, uf, cfg, journal, stats)
    _tier0_ucid_truncated_span_merges(mentions, by_id, uf, cfg, journal, stats)
    return stats


def _truncated_span_name(name: str | None, *, protected: frozenset[str], noise: frozenset[str]) -> bool:
    from engine.name_compat import resolve_surname

    toks = tokens(name or "")
    if len(toks) != 2:
        return False
    rs = resolve_surname(name, protected=protected, noise=noise)
    return toks[-1] == rs and toks[0] in protected


def _tier0_ucid_truncated_span_merges(
    mentions: list[dict],
    by_id: dict[str, dict],
    uf: UnionFind,
    cfg: dict,
    journal: DecisionJournal,
    stats: dict,
) -> None:
    """Same UCID: surname-led 2-token span tokens appear inside a longer compatible name."""
    from engine.name_compat import _gate_profile, _primary_given_token, resolve_surname

    protected, noise = _gate_profile(cfg)
    by_ucid: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        u = m.get("ucid")
        if u:
            by_ucid[str(u)].append(m)

    for ucid, ms in by_ucid.items():
        if len(ms) < 2:
            continue
        for i, ma in enumerate(ms):
            for mb in ms[i + 1 :]:
                a_id, b_id = ma["mention_id"], mb["mention_id"]
                if uf.find(a_id) == uf.find(b_id):
                    continue
                na, nb = ma.get("normalized_name") or "", mb.get("normalized_name") or ""
                ta, tb = set(tokens(na)), set(tokens(nb))
                pair = None
                if _truncated_span_name(na, protected=protected, noise=noise) and ta and ta.issubset(tb) and na != nb:
                    pair = (a_id, b_id)
                elif _truncated_span_name(nb, protected=protected, noise=noise) and tb and tb.issubset(ta) and na != nb:
                    pair = (b_id, a_id)
                if not pair:
                    continue
                root, other = pair
                shorter_name = by_id[root].get("normalized_name")
                longer_name = by_id[other].get("normalized_name")
                ga = _primary_given_token(shorter_name, protected=protected, noise=noise)
                gb = _primary_given_token(longer_name, protected=protected, noise=noise)
                if ga is not None or gb is None:
                    continue
                ok_name, _ = names_compatible(by_id[root], by_id[other], cfg=cfg)
                if not ok_name or transfer_conflict(by_id[root], by_id[other]):
                    continue
                uf.union(root, other)
                stats["merges"] += 1
                stats["rules_fired"]["ucid_truncated_span"] = stats["rules_fired"].get("ucid_truncated_span", 0) + 1
                journal.log(
                    {
                        "decision_id": f"dec_{journal.n:08d}",
                        "mention_id_a": root,
                        "mention_id_b": other,
                        "entity_type": cfg.get("entity_type"),
                        "decision": "MERGE_TIER0",
                        "confidence": 100,
                        "method": "tier0.ucid_truncated_span",
                        "rationale": "Same UCID; surname-led truncated span tokens appear in longer name",
                        "signals": ["truncated_span", "same_ucid"],
                        "evidence": {"ucid": ucid, "shorter": shorter_name, "longer": longer_name},
                        "timestamp": _now(),
                        "config_version": cfg.get("version"),
                    }
                )


def _tier0_ucid_prefix_span_merges(
    mentions: list[dict],
    by_id: dict[str, dict],
    uf: UnionFind,
    cfg: dict,
    journal: DecisionJournal,
    stats: dict,
) -> None:
    """Same UCID: 2-token name is a prefix of a 3-token name (e.g. marina garcia → marina garcia marmolejo)."""
    from engine.common_first_names import COMMON_FIRST_NAMES

    min_given_len = int(((cfg.get("tier0") or {}).get("ucid_prefix_span") or {}).get("min_given_len", 5))
    by_ucid: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        u = m.get("ucid")
        if u:
            by_ucid[str(u)].append(m)

    for ucid, ms in by_ucid.items():
        if len(ms) < 2:
            continue
        for i, ma in enumerate(ms):
            for mb in ms[i + 1 :]:
                a_id, b_id = ma["mention_id"], mb["mention_id"]
                if uf.find(a_id) == uf.find(b_id):
                    continue
                na, nb = ma.get("normalized_name") or "", mb.get("normalized_name") or ""
                ta, tb = tokens(na), tokens(nb)
                pair = None
                if len(ta) == 2 and len(tb) == 3 and tb[:2] == ta and ta[0] not in COMMON_FIRST_NAMES:
                    if len(ta[0]) >= min_given_len:
                        pair = (a_id, b_id)
                elif len(tb) == 2 and len(ta) == 3 and ta[:2] == tb and tb[0] not in COMMON_FIRST_NAMES:
                    if len(tb[0]) >= min_given_len:
                        pair = (b_id, a_id)
                if not pair:
                    continue
                root, other = pair
                ok_name, _ = names_compatible(by_id[root], by_id[other], cfg=cfg)
                if not ok_name or transfer_conflict(by_id[root], by_id[other]):
                    continue
                uf.union(root, other)
                stats["merges"] += 1
                stats["rules_fired"]["ucid_prefix_span"] = stats["rules_fired"].get("ucid_prefix_span", 0) + 1
                journal.log(
                    {
                        "decision_id": f"dec_{journal.n:08d}",
                        "mention_id_a": root,
                        "mention_id_b": other,
                        "entity_type": cfg.get("entity_type"),
                        "decision": "MERGE_TIER0",
                        "confidence": 100,
                        "method": "tier0.ucid_prefix_span",
                        "rationale": "Same UCID; 2-token name is token prefix of 3-token name",
                        "signals": ["name_prefix", "same_ucid"],
                        "evidence": {"ucid": ucid, "shorter": by_id[root].get("normalized_name"), "longer": by_id[other].get("normalized_name")},
                        "timestamp": _now(),
                        "config_version": cfg.get("version"),
                    }
                )


def similarity(a: dict, b: dict) -> float:
    """Fallback profile similarity in [0,1] via rapidfuzz."""
    pa = a.get("profile") or a.get("normalized_name") or ""
    pb = b.get("profile") or b.get("normalized_name") or ""
    if not pa or not pb:
        return 0.0
    return fuzz.token_set_ratio(pa, pb) / 100.0


def build_ollama_faiss_pack(mentions: list[dict], cfg: dict) -> dict | None:
    """Embed mention profiles with Ollama nomic-embed-text and build FAISS IP index.

    Deduplicates identical compact embed texts to cut Ollama calls.
    """
    t2 = cfg.get("tier2") or {}
    model = t2.get("ollama_embed_model") or "nomic-embed-text"
    endpoint = ollama_endpoint(cfg)
    try:
        from .embeddings import (
            build_faiss_ip_index,
            compact_embed_text,
            embed_texts_ollama,
            save_embedding_pack,
        )
    except Exception as e:
        print(f"embeddings module unavailable: {e}")
        return None

    ids = [m["mention_id"] for m in mentions]
    texts = [compact_embed_text(m) for m in mentions]
    emb_dir = resolve_path(cfg, cfg["io"]["embeddings_dir"])

    # Reuse on-disk pack when mention ids match (resume / re-run without re-embed).
    try:
        from .embeddings import load_embedding_pack

        loaded = load_embedding_pack(emb_dir)
    except Exception:
        loaded = None
    if loaded is not None:
        vectors, loaded_ids, meta = loaded
        if loaded_ids == ids and int((meta or {}).get("n") or 0) in {0, len(ids)}:
            print(f"Reusing embeddings pack: n={len(ids)} dim={vectors.shape[1]} from {emb_dir}", flush=True)
            try:
                index = build_faiss_ip_index(vectors)
                id_to_row = {mid: i for i, mid in enumerate(ids)}
                return {
                    "vectors": vectors,
                    "index": index,
                    "ids": ids,
                    "id_to_row": id_to_row,
                    "backend": "ollama+faiss",
                    "model": model,
                    "reused": True,
                }
            except Exception as e:
                print(f"Reused FAISS build failed ({e}); re-embedding...", flush=True)

    # Dedup texts
    uniq_texts: list[str] = []
    text_to_urow: dict[str, int] = {}
    for t in texts:
        if t not in text_to_urow:
            text_to_urow[t] = len(uniq_texts)
            uniq_texts.append(t)

    print(f"Embedding {len(uniq_texts)} unique profiles ({len(texts)} mentions) via Ollama/{model} ...")
    try:
        uniq_vectors = embed_texts_ollama(uniq_texts, model=model, endpoint=endpoint)
        vectors = np.vstack([uniq_vectors[text_to_urow[t]] for t in texts]).astype(np.float32)
        index = build_faiss_ip_index(vectors)
    except Exception as e:
        print(f"Ollama/FAISS embed failed: {e}")
        return None

    save_embedding_pack(
        emb_dir,
        vectors,
        ids,
        {
            "model": model,
            "backend": "ollama+faiss",
            "n": len(ids),
            "n_unique": len(uniq_texts),
            "dim": int(vectors.shape[1]),
        },
    )
    id_to_row = {mid: i for i, mid in enumerate(ids)}
    return {"index": index, "vectors": vectors, "ids": ids, "id_to_row": id_to_row, "backend": "ollama+faiss"}



def tier2_candidates(
    mentions: list[dict],
    mention_blocks: dict[str, list[str]],
    blocks: dict[str, list[str]],
    uf: UnionFind,
    cfg: dict,
    embed_pack=None,
) -> list[tuple[str, str, float]]:
    """Return unresolved same-block candidate pairs with similarity."""
    t2 = cfg.get("tier2") or {}
    min_sim = float(t2.get("search", {}).get("min_similarity", 0.72))
    top_k = int(t2.get("search", {}).get("top_k", 25))
    by_id = {m["mention_id"]: m for m in mentions}
    seen = set()
    pairs = []

    use_faiss = embed_pack is not None and embed_pack.get("index") is not None
    exhaustive_cap = int(t2.get("search", {}).get("exhaustive_uniq_cap", 80))
    if use_faiss:
        import numpy as np

        vectors = embed_pack["vectors"]
        id_to_row = embed_pack["id_to_row"]
        index = embed_pack["index"]

        for block_key, mids in blocks.items():
            uniq = []
            roots_seen = set()
            for mid in mids:
                r = uf.find(mid)
                if r not in roots_seen:
                    roots_seen.add(r)
                    uniq.append(mid)
            if len(uniq) < 2:
                continue
            root_to_rep = {uf.find(m): m for m in uniq}

            # Small blocks: pairwise IP among UF reps (no global top_k miss).
            if 2 <= len(uniq) <= exhaustive_cap:
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        a, b = uniq[i], uniq[j]
                        if a not in id_to_row or b not in id_to_row:
                            continue
                        if uf.find(a) == uf.find(b):
                            continue
                        if not pair_allowed(by_id[a], by_id[b]):
                            continue
                        pk = _pair_key(a, b)
                        if pk in seen:
                            continue
                        sim = float(np.dot(vectors[id_to_row[a]], vectors[id_to_row[b]]))
                        if sim < min_sim:
                            continue
                        seen.add(pk)
                        pairs.append((a, b, sim))

            # query each against FAISS; accept neighbor if its UF root is in this block
            rows = [id_to_row[m] for m in uniq if m in id_to_row]
            if len(rows) < 2:
                continue
            q = np.ascontiguousarray(vectors[rows], dtype=np.float32)
            k = min(top_k + 1, len(embed_pack["ids"]))
            sims, idxs = index.search(q, k)
            for qi, mid_a in enumerate(uniq):
                if mid_a not in id_to_row:
                    continue
                for sim, jj in zip(sims[qi], idxs[qi]):
                    if jj < 0:
                        continue
                    mid_b = embed_pack["ids"][int(jj)]
                    if mid_b == mid_a:
                        continue
                    root_b = uf.find(mid_b)
                    if root_b not in root_to_rep:
                        continue
                    mid_b_rep = root_to_rep[root_b]
                    if mid_b_rep == mid_a or uf.find(mid_a) == root_b:
                        continue
                    if float(sim) < min_sim:
                        continue
                    if not pair_allowed(by_id[mid_a], by_id[mid_b_rep]):
                        continue
                    pk = _pair_key(mid_a, mid_b_rep)
                    if pk in seen:
                        continue
                    seen.add(pk)
                    pairs.append((mid_a, mid_b_rep, float(sim)))
    else:
        for block_key, mids in blocks.items():
            uniq = []
            roots_seen = set()
            for mid in mids:
                r = uf.find(mid)
                if r not in roots_seen:
                    roots_seen.add(r)
                    uniq.append(mid)
            if len(uniq) < 2:
                continue
            if len(uniq) > 80:
                uniq = uniq[:80]
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    if uf.find(a) == uf.find(b):
                        continue
                    if not pair_allowed(by_id[a], by_id[b]):
                        continue
                    pk = _pair_key(a, b)
                    if pk in seen:
                        continue
                    seen.add(pk)
                    if not same_block(a, b, mention_blocks):
                        continue
                    sim = similarity(by_id[a], by_id[b])
                    if sim >= min_sim:
                        pairs.append((a, b, sim))

    pairs.sort(key=lambda x: -x[2])
    return pairs


def apply_tier2_auto_merges(
    pairs: list[tuple[str, str, float]],
    by_id: dict[str, dict],
    uf: UnionFind,
    cfg: dict,
    journal: DecisionJournal,
    common_surnames: set[str],
) -> dict:
    t2 = cfg.get("tier2") or {}
    auto_min = float(t2.get("search", {}).get("auto_merge_min", 0.92))
    amb_low = float(t2.get("search", {}).get("ambiguous_low", 0.72))
    amb_high = float(t2.get("search", {}).get("ambiguous_high", 0.92))
    barrier_cfg = cfg.get("information_content_barrier") or {}
    allow_auto = (barrier_cfg.get("on_trigger") or {}).get("allow_tier2_auto_merge", False)

    stats = {"auto_merges": 0, "ambiguous": 0, "blocked_by_barrier": 0, "transfer_blocks": 0, "name_gate": 0}
    ambiguous = []

    for a, b, sim in pairs:
        if uf.find(a) == uf.find(b):
            continue
        ma, mb = by_id[a], by_id[b]
        if not pair_allowed(ma, mb):
            continue
        ok_name, name_reason = names_compatible(ma, mb, cfg=cfg)
        if not ok_name:
            stats["name_gate"] = stats.get("name_gate", 0) + 1
            row = name_gate_decision_row(ma, mb, cfg=cfg, reason=name_reason, sim=sim)
            row["decision_id"] = f"dec_{journal.n:08d}"
            row["mention_id_a"] = a
            row["mention_id_b"] = b
            row["timestamp"] = _now()
            journal.log(row)
            continue
        if transfer_conflict(ma, mb):
            stats["transfer_blocks"] += 1
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "NO_MATCH",
                    "confidence": 100,
                    "method": "tier2.transfer_conflict",
                    "rationale": "Transfer clue indicates distinct judges",
                    "signals": ["transfer_conflict"],
                    "evidence": {"embedding_similarity": sim},
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            continue
        conflict = identity_pair_conflict(ma, mb, cfg)
        if conflict:
            signal, rationale = conflict
            stats[f"{signal}_blocks"] = stats.get(f"{signal}_blocks", 0) + 1
            if signal == "corporate_shared_prefix":
                stats["corp_prefix_blocks"] = stats.get("corp_prefix_blocks", 0) + 1
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "NO_MATCH",
                    "confidence": 100,
                    "method": f"tier2.{signal}",
                    "rationale": rationale,
                    "signals": [signal],
                    "evidence": {"embedding_similarity": sim},
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            continue
        conflict, geo_sig = institutional_office_geo_conflict(ma, mb, cfg)
        if conflict:
            stats[geo_sig] = stats.get(geo_sig, 0) + 1
            sa, city_a = mention_office_geo(ma, cfg)
            sb, city_b = mention_office_geo(mb, cfg)
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "NO_MATCH",
                    "confidence": 100,
                    "method": f"tier2.{geo_sig}",
                    "rationale": (
                        "Institutional offices with different address-derived "
                        f"{geo_sig.replace('_', ' ')} are distinct "
                        "(case court is not office district)"
                    ),
                    "signals": [geo_sig],
                    "evidence": {
                        "embedding_similarity": sim,
                        "office_state_a": sa,
                        "office_state_b": sb,
                        "office_city_a": city_a,
                        "office_city_b": city_b,
                    },
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            continue

        # Class 5: private cross-state/city Tier2 auto-merge needs contact corroboration
        if private_cross_geo_lacks_corroboration(ma, mb, cfg):
            stats["corroboration_required"] = stats.get("corroboration_required", 0) + 1
            sa, city_a = mention_office_geo(ma, cfg)
            sb, city_b = mention_office_geo(mb, cfg)
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "NO_MATCH",
                    "confidence": 100,
                    "method": "tier2.corroboration_required",
                    "rationale": (
                        "Private-firm cross-state/city pair lacks shared domain, "
                        "phone, or address; name similarity alone is not sufficient "
                        "for Tier2 auto-merge"
                    ),
                    "signals": ["corroboration_required", "cross_geo_private"],
                    "evidence": {
                        "embedding_similarity": sim,
                        "office_state_a": sa,
                        "office_state_b": sb,
                        "office_city_a": city_a,
                        "office_city_b": city_b,
                    },
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            continue

        # Parties: ALL Tier2 auto-merges need a second signal beyond embedding sim
        if sim >= auto_min and parties_tier2_lacks_corroboration(ma, mb, cfg):
            stats["corroboration_required"] = stats.get("corroboration_required", 0) + 1
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "NO_MATCH",
                    "confidence": 100,
                    "method": "tier2.corroboration_required",
                    "rationale": (
                        "Embedding similarity alone is not sufficient for parties "
                        "Tier2 auto-merge; need shared PACER id, shared contact, or "
                        "compatible name core after entity-suffix normalization"
                    ),
                    "signals": ["corroboration_required"],
                    "evidence": {
                        "embedding_similarity": sim,
                        "name_core_a": name_core_for_corroboration(_norm_name(ma), cfg),
                        "name_core_b": name_core_for_corroboration(_norm_name(mb), cfg),
                    },
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            continue

        ba, ra = information_barrier(ma, cfg, common_surnames)
        bb, rb = information_barrier(mb, cfg, common_surnames)
        barrier = ba or bb

        # A5: large-cluster safety — do not Tier2-auto-fuse two big components
        if sim >= auto_min and large_cluster_merge_blocked(uf, a, b, cfg):
            stats["blocked_by_barrier"] += 1
            stats["ambiguous"] += 1
            ambiguous.append((a, b, sim, True, sorted(set(ra + rb + ["large_cluster_verify"]))))
            continue

        if sim >= auto_min and not (barrier and not allow_auto):
            uf.union(a, b)
            stats["auto_merges"] += 1
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "MERGE_TIER2",
                    "confidence": int(round(sim * 100)),
                    "method": "tier2.auto_merge",
                    "rationale": f"High profile similarity {sim:.3f} within block; barrier={barrier}",
                    "signals": ["fuzzy_name", "same_court"] if ma.get("court") == mb.get("court") else ["fuzzy_name"],
                    "evidence": {
                        "embedding_similarity": sim,
                        "barrier": barrier,
                        "barrier_reasons": sorted(set(ra + rb)),
                    },
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
        elif amb_low <= sim < amb_high:
            if barrier and (barrier_cfg.get("on_trigger") or {}).get("auto_send_to_llm") is False:
                # still eligible for LLM if same-block ambiguous — barrier only blocks auto-merge
                pass
            stats["ambiguous"] += 1
            if barrier and not allow_auto and sim >= auto_min:
                stats["blocked_by_barrier"] += 1
            ambiguous.append((a, b, sim, barrier, sorted(set(ra + rb))))
        elif sim >= auto_min and barrier and not allow_auto:
            stats["blocked_by_barrier"] += 1
            ambiguous.append((a, b, sim, True, sorted(set(ra + rb))))

    return {"stats": stats, "ambiguous": ambiguous}


def load_prompt(cfg: dict) -> str:
    path = resolve_path(cfg, (cfg.get("tier3") or {}).get("prompt_path", "prompts/tier3_adjudicate.txt"))
    return path.read_text(encoding="utf-8")


def load_output_schema(cfg: dict) -> dict:
    path = resolve_path(cfg, (cfg.get("tier3") or {}).get("json_schema_path", "prompts/tier3_output.schema.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def call_ollama_json(
    model: str,
    prompt: str,
    endpoint: str = "http://localhost:11434",
    *,
    retries: int = 5,
    retry_sleep_sec: float = 3.0,
) -> dict:
    import time
    import urllib.error
    import urllib.request

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode())
            raw = body.get("response") or "{}"
            return json.loads(raw)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as e:
            last_err = e
            if attempt + 1 >= retries:
                break
            sleep_for = retry_sleep_sec * (2**attempt)
            print(f"  ollama retry {attempt+1}/{retries} after {e} (sleep {sleep_for:.0f}s)", flush=True)
            time.sleep(sleep_for)
    assert last_err is not None
    raise last_err


def tier3_adjudicate(
    ambiguous: list[tuple],
    by_id: dict[str, dict],
    uf: UnionFind,
    cfg: dict,
    journal: DecisionJournal,
    mention_blocks: dict[str, list[str]],
    total_mentions: int,
) -> dict:
    t3 = cfg.get("tier3") or {}
    if not t3.get("enabled", True):
        return {"llm_calls": 0, "merges": 0, "skipped": len(ambiguous)}

    model = t3.get("model", "qwen2.5:7b")
    endpoint = ollama_endpoint(cfg)
    prompt_tmpl = load_prompt(cfg)
    prompt_hash = prompt_sha256(prompt_tmpl)
    schema = load_output_schema(cfg)

    try:
        import jsonschema
    except ImportError:
        jsonschema = None

    cache_path = resolve_path(cfg, (t3.get("decision_cache") or {}).get("path", "data/decisions/tier3_cache.jsonl"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                if rec.get("prompt_hash") and rec.get("prompt_hash") != prompt_hash:
                    continue
                cache[rec["cache_key"]] = rec

    # Budget: target 5-15% of mentions → approx that many pairs max
    budget = t3.get("budget") or {}
    max_frac = float(budget.get("target_mention_fraction_max", 0.15))
    max_calls = max(1, int(total_mentions * max_frac))

    routing = t3.get("routing") or {}
    require_block = routing.get("require_same_block", True)

    stats = {
        "llm_calls": 0,
        "cache_hits": 0,
        "merges": 0,
        "no_match": 0,
        "uncertain": 0,
        "errors": 0,
        "ambiguous_pairs_in": len(ambiguous),
        "pairs_skipped_already_merged": 0,
        "pairs_skipped_budget": 0,
        "pairs_skipped_not_same_block": 0,
        "model": model,
        "endpoint": endpoint,
        "uncertain_model_said": 0,
        "uncertain_threshold_converted": 0,
        "uncertain_barrier_abstain": 0,
        "uncertain_other": 0,
        "barrier_llm_overrides": 0,  # barrier-flagged MATCH allowed at high conf
        "tier3_errors_logged": 0,  # TIER3_ERROR journal rows (never counted as abstention)
        "name_gate": 0,
        "prompt_hash": prompt_hash,
        "status": "OK",
    }
    abstain_below = int(t3.get("abstain_confidence_below", 60))
    barrier_llm_min = int(t3.get("barrier_llm_min_confidence", 80))
    llm_time_sec = 0.0
    progress_every = int(t3.get("progress_log_every") or 25)
    consecutive_errors = 0
    max_error_rate = float(t3.get("max_error_rate", 0.01))
    max_consecutive_errors = int(t3.get("max_consecutive_errors", 3))

    # Prefer mid-band similarities closest to center of ambiguous range
    amb_low = float((cfg.get("tier2") or {}).get("search", {}).get("ambiguous_low", 0.72))
    amb_high = float((cfg.get("tier2") or {}).get("search", {}).get("ambiguous_high", 0.92))
    mid = (amb_low + amb_high) / 2
    ranked = sorted(ambiguous, key=lambda x: abs(x[2] - mid))
    n_pairs = len(ranked)

    def _progress(i_pair: int) -> None:
        if progress_every <= 0:
            return
        if i_pair % progress_every != 0 and i_pair != n_pairs:
            return
        adj = stats["merges"] + stats["no_match"] + stats["uncertain"] + stats.get("tier3_errors_logged", 0)
        avg = (llm_time_sec / stats["llm_calls"]) if stats["llm_calls"] else 0.0
        print(
            f"Tier3 progress: adjudicated {adj}/{n_pairs} "
            f"(llm={stats['llm_calls']} cache={stats['cache_hits']} "
            f"uncertain={stats['uncertain']} errors={stats['errors']}, "
            f"avg_sec_per_call={avg:.2f})",
            flush=True,
        )

    for pair_i, (a, b, sim, barrier, reasons) in enumerate(ranked, start=1):
        if uf.find(a) == uf.find(b):
            stats["pairs_skipped_already_merged"] += 1
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "SKIP_ALREADY_MERGED",
                    "confidence": 100,
                    "method": "tier3.transitive_skip",
                    "model": model,
                    "rationale": "Pair already in same UF component (merged earlier in cascade/Tier3)",
                    "signals": ["transitive"],
                    "evidence": {"embedding_similarity": sim, "barrier": barrier},
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            _progress(pair_i)
            continue
        if require_block and not same_block(a, b, mention_blocks):
            stats["pairs_skipped_not_same_block"] += 1
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "SKIP_NOT_SAME_BLOCK",
                    "confidence": 100,
                    "method": "tier3.routing_skip",
                    "model": model,
                    "rationale": "Pair no longer shares a Tier1 block key",
                    "signals": [],
                    "evidence": {"embedding_similarity": sim},
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            _progress(pair_i)
            continue
        if stats["llm_calls"] >= max_calls:
            ma, mb = by_id[a], by_id[b]
            ck = _cache_key(ma, mb)
            if ck not in cache:
                stats["pairs_skipped_budget"] += 1
                journal.log(
                    {
                        "decision_id": f"dec_{journal.n:08d}",
                        "mention_id_a": a,
                        "mention_id_b": b,
                        "entity_type": cfg.get("entity_type"),
                        "decision": "SKIP_BUDGET",
                        "confidence": 0,
                        "method": "tier3.budget_skip",
                        "model": model,
                        "rationale": f"Fresh LLM budget exhausted (max_calls={max_calls})",
                        "signals": ["insufficient_evidence"],
                        "evidence": {"embedding_similarity": sim, "max_calls": max_calls},
                        "cache_key": ck,
                        "timestamp": _now(),
                        "config_version": cfg.get("version"),
                    }
                )
                _progress(pair_i)
                continue

        ma, mb = by_id[a], by_id[b]
        if not pair_allowed(ma, mb):
            stats["pairs_skipped_hygiene"] = stats.get("pairs_skipped_hygiene", 0) + 1
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "NO_MATCH",
                    "confidence": 100,
                    "method": "tier3.hygiene_scope",
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "rationale": "Low-information mention restricted to same-UCID attachment",
                    "signals": ["hygiene_same_ucid"],
                    "evidence": {
                        "embedding_similarity": sim,
                        "hygiene_scope_a": ma.get("hygiene_scope"),
                        "hygiene_scope_b": mb.get("hygiene_scope"),
                    },
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            _progress(pair_i)
            continue

        # Tier0 alias_group cross-UCID: never send to LLM (e.g. USA vs USA).
        skipped_grp = pair_in_skipped_alias_group_cross_ucid(ma, mb, cfg)
        if skipped_grp:
            stats["pairs_skipped_alias_group_cross_ucid"] = (
                stats.get("pairs_skipped_alias_group_cross_ucid", 0) + 1
            )
            stats["no_match"] += 1
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "NO_MATCH",
                    "confidence": 100,
                    "method": f"tier3.alias_group_cross_ucid_skip:{skipped_grp}",
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "rationale": (
                        f"Both names in Tier0 alias_group {skipped_grp}; "
                        "cross-UCID pairs are not sent to Tier3"
                    ),
                    "signals": ["alias_group_cross_ucid", skipped_grp],
                    "evidence": {
                        "embedding_similarity": sim,
                        "alias_group": skipped_grp,
                        "ucid_a": ma.get("ucid"),
                        "ucid_b": mb.get("ucid"),
                    },
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            _progress(pair_i)
            continue

        # A1: name-compatibility gate — LLM never called on cross-surname pairs
        ok_name, name_reason = names_compatible(ma, mb, cfg=cfg)
        if not ok_name:
            stats["name_gate"] = stats.get("name_gate", 0) + 1
            stats["no_match"] += 1
            row = name_gate_decision_row(ma, mb, cfg=cfg, reason=name_reason, sim=sim)
            row["decision_id"] = f"dec_{journal.n:08d}"
            row["mention_id_a"] = a
            row["mention_id_b"] = b
            row["model"] = model
            row["prompt_hash"] = prompt_hash
            row["timestamp"] = _now()
            journal.log(row)
            _progress(pair_i)
            continue

        ident = identity_pair_conflict(ma, mb, cfg)
        if ident:
            signal, rationale = ident
            stats[f"{signal}_blocks"] = stats.get(f"{signal}_blocks", 0) + 1
            stats["no_match"] += 1
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "NO_MATCH",
                    "confidence": 100,
                    "method": f"tier3.{signal}",
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "rationale": rationale,
                    "signals": [signal],
                    "evidence": {"embedding_similarity": sim, "barrier": barrier},
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            _progress(pair_i)
            continue

        conflict, geo_sig = institutional_office_geo_conflict(ma, mb, cfg)
        if conflict:
            stats[geo_sig] = stats.get(geo_sig, 0) + 1
            stats["no_match"] += 1
            sa, city_a = mention_office_geo(ma, cfg)
            sb, city_b = mention_office_geo(mb, cfg)
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "NO_MATCH",
                    "confidence": 100,
                    "method": f"tier3.{geo_sig}",
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "rationale": (
                        "Institutional offices with different address-derived "
                        f"{geo_sig.replace('_', ' ')} are distinct "
                        "(case court is not office district)"
                    ),
                    "signals": [geo_sig],
                    "evidence": {
                        "embedding_similarity": sim,
                        "office_state_a": sa,
                        "office_state_b": sb,
                        "office_city_a": city_a,
                        "office_city_b": city_b,
                    },
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
            _progress(pair_i)
            continue

        ck = _cache_key(ma, mb)
        result = cache.get(ck)
        from_cache = result is not None

        # Even if cache says MATCH, refuse incompatible surnames (stale polluted cache)
        if from_cache and (result.get("decision") or "").upper() == "MATCH":
            ok_c, reason_c = names_compatible(ma, mb, cfg=cfg)
            if not ok_c:
                stats["name_gate"] = stats.get("name_gate", 0) + 1
                stats["no_match"] += 1
                stats["cache_hits"] += 1
                row = name_gate_decision_row(
                    ma, mb, cfg=cfg, reason=reason_c, sim=sim,
                    extra_evidence={"cache_match_overridden": True},
                )
                row["decision_id"] = f"dec_{journal.n:08d}"
                row["mention_id_a"] = a
                row["mention_id_b"] = b
                row["model"] = model
                row["prompt_hash"] = prompt_hash
                row["timestamp"] = _now()
                journal.log(row)
                _progress(pair_i)
                continue
            ident_c = identity_pair_conflict(ma, mb, cfg)
            if ident_c:
                signal, rationale = ident_c
                stats[f"{signal}_blocks"] = stats.get(f"{signal}_blocks", 0) + 1
                stats["no_match"] += 1
                stats["cache_hits"] += 1
                journal.log(
                    {
                        "decision_id": f"dec_{journal.n:08d}",
                        "mention_id_a": a,
                        "mention_id_b": b,
                        "entity_type": cfg.get("entity_type"),
                        "decision": "NO_MATCH",
                        "confidence": 100,
                        "method": f"tier3.{signal}",
                        "model": model,
                        "prompt_hash": prompt_hash,
                        "rationale": "Cache MATCH overridden: " + rationale,
                        "signals": [signal, "cache_match_overridden"],
                        "evidence": {"embedding_similarity": sim, "barrier": barrier},
                        "timestamp": _now(),
                        "config_version": cfg.get("version"),
                    }
                )
                _progress(pair_i)
                continue
            if institutional_office_geo_conflict(ma, mb, cfg)[0]:
                conflict, geo_sig = institutional_office_geo_conflict(ma, mb, cfg)
                stats[geo_sig] = stats.get(geo_sig, 0) + 1
                stats["no_match"] += 1
                stats["cache_hits"] += 1
                sa, city_a = mention_office_geo(ma, cfg)
                sb, city_b = mention_office_geo(mb, cfg)
                journal.log(
                    {
                        "decision_id": f"dec_{journal.n:08d}",
                        "mention_id_a": a,
                        "mention_id_b": b,
                        "entity_type": cfg.get("entity_type"),
                        "decision": "NO_MATCH",
                        "confidence": 100,
                        "method": f"tier3.{geo_sig}",
                        "model": model,
                        "prompt_hash": prompt_hash,
                        "rationale": (
                            "Cache MATCH overridden: institutional "
                            f"{geo_sig.replace('_', ' ')}"
                        ),
                        "signals": [geo_sig, "cache_match_overridden"],
                        "evidence": {
                            "embedding_similarity": sim,
                            "office_state_a": sa,
                            "office_state_b": sb,
                            "office_city_a": city_a,
                            "office_city_b": city_b,
                        },
                        "timestamp": _now(),
                        "config_version": cfg.get("version"),
                    }
                )
                _progress(pair_i)
                continue

        if result is None:
            evidence = build_tier3_evidence(
                ma, mb, barrier=barrier, reasons=reasons, cfg=cfg
            )
            evidence["embedding_similarity"] = sim
            block_keys = sorted(set(mention_blocks.get(a, [])) & set(mention_blocks.get(b, [])))
            prompt = (
                prompt_tmpl.replace("{{mention_a_profile}}", ma.get("profile") or ma["normalized_name"])
                .replace("{{mention_b_profile}}", mb.get("profile") or mb["normalized_name"])
                .replace("{{block_key}}", ", ".join(block_keys))
                .replace("{{embedding_similarity}}", f"{sim:.4f}")
                .replace("{{evidence_json}}", json.dumps(evidence, ensure_ascii=False))
            )
            result = None
            try:
                t_llm = time.time()
                result = call_ollama_json(model, prompt, endpoint)
                llm_time_sec += time.time() - t_llm
                # Normalize confidence if model returns float/string
                if "confidence" in result:
                    result["confidence"] = int(round(float(result["confidence"])))
                if "decision" in result:
                    result["decision"] = str(result["decision"]).upper().replace(" ", "_")
                    if result["decision"] not in {"MATCH", "NO_MATCH", "UNCERTAIN"}:
                        if result["decision"] in {"YES", "SAME", "MERGE"}:
                            result["decision"] = "MATCH"
                        elif result["decision"] in {"NO", "DIFFERENT", "DISTINCT"}:
                            result["decision"] = "NO_MATCH"
                        else:
                            result["decision"] = "UNCERTAIN"
                result["signals"] = normalize_tier3_signals(result.get("signals"))
                # Parties citation schema: coerce missing cited_evidence from signals
                if "cited_evidence" not in result or result.get("cited_evidence") is None:
                    result["cited_evidence"] = []
                if isinstance(result.get("cited_evidence"), str):
                    result["cited_evidence"] = [result["cited_evidence"]]
                if not isinstance(result.get("cited_evidence"), list):
                    result["cited_evidence"] = list(result.get("cited_evidence") or [])
                result["cited_evidence"] = [
                    str(x).strip()
                    for x in result["cited_evidence"]
                    if str(x).strip()
                ]
                if not result["cited_evidence"] and result.get("signals"):
                    result["cited_evidence"] = list(result["signals"])
                if jsonschema is not None:
                    jsonschema.validate(result, schema)
                stats["llm_calls"] += 1
            except Exception as e:
                stats["errors"] += 1
                stats["tier3_errors_logged"] += 1
                consecutive_errors += 1
                journal.log(
                    {
                        "decision_id": f"dec_{journal.n:08d}",
                        "mention_id_a": a,
                        "mention_id_b": b,
                        "entity_type": cfg.get("entity_type"),
                        "decision": "TIER3_ERROR",
                        "confidence": 0,
                        "method": f"tier3.error:{model}",
                        "model": model,
                        "rationale": f"LLM/schema failure (NOT an abstention): {e}",
                        "signals": ["tier3_error"],
                        "evidence": {
                            "embedding_similarity": sim,
                            "raw": result,
                            "error": str(e),
                        },
                        "cache_key": ck,
                        "timestamp": _now(),
                        "config_version": cfg.get("version"),
                    }
                )
                # Never write failed calls into the decision cache.
                try:
                    check_tier3_error_rate(
                        llm_calls=stats["llm_calls"],
                        errors=stats["errors"],
                        consecutive_errors=consecutive_errors,
                        max_error_rate=max_error_rate,
                        max_consecutive_errors=max_consecutive_errors,
                    )
                except CascadeFailedError as cfe:
                    stats["status"] = "FAILED"
                    stats["fail_reason"] = str(cfe)
                    print(f"CASCADE_FAILED: {cfe}", flush=True)
                    raise
                _progress(pair_i)
                continue

            consecutive_errors = 0
            append_jsonl(
                cache_path,
                {
                    "cache_key": ck,
                    **result,
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "timestamp": _now(),
                },
            )
            cache[ck] = {**result, "prompt_hash": prompt_hash}
        else:
            stats["cache_hits"] += 1
            consecutive_errors = 0

        decision = (result.get("decision") or "UNCERTAIN").upper()
        conf = int(result.get("confidence") or 0)
        rationale = result.get("rationale") or ""
        signals = list(result.get("signals") or [])
        # Older cache rows predate cited_evidence — salvage from signals.
        if not result.get("cited_evidence") and signals:
            result["cited_evidence"] = list(signals)
        shared_nid = bool(ma.get("fjc_nid") and ma.get("fjc_nid") == mb.get("fjc_nid"))
        ucid_only = ucid_only_corroboration(ma, mb, cfg)
        barrier_floor = barrier_llm_min
        extra_ucid_floor = int(t3.get("ucid_only_corroboration_barrier_min_confidence") or 0)
        if barrier and ucid_only and extra_ucid_floor:
            barrier_floor = max(barrier_floor, extra_ucid_floor)

        # ------------------------------------------------------------------
        # Citation verification (+ optional asymmetric MATCH bar).
        # Rebuild the same evidence shape used in the prompt when possible.
        # ------------------------------------------------------------------
        citation_meta: dict = {}
        if (t3.get("citation_verification") or {}).get("enabled") or (
            t3.get("match_requires_discriminating_fact") or {}
        ).get("enabled"):
            ev_for_cite = build_tier3_evidence(
                ma, mb, barrier=barrier, reasons=reasons, cfg=cfg
            )
            ev_for_cite["embedding_similarity"] = sim
            decision, rationale, signals, citation_meta = apply_citation_and_match_rails(
                decision,
                rationale,
                signals,
                result,
                ma,
                mb,
                ev_for_cite,
                cfg,
            )
            if citation_meta.get("citation_verification", {}).get("forced_uncertain"):
                stats["citation_hallucination_abstain"] = (
                    stats.get("citation_hallucination_abstain", 0) + 1
                )
            if citation_meta.get("match_bar_forced_uncertain"):
                stats["match_bar_abstain"] = stats.get("match_bar_abstain", 0) + 1

        # ------------------------------------------------------------------
        # Abstention rails (ordered):
        # 1) Non-barrier pairs (and any NO_MATCH): conf < abstain_confidence_below
        #    → UNCERTAIN (threshold conversion).
        # 2) Barrier-flagged MATCH without shared FJC NID:
        #    allow merge only if conf >= barrier floor; else abstain.
        #    UCID-only corroboration uses a stricter floor when configured
        #    (parties: shared case number is co-litigant, not identity).
        # 3) Model-said UNCERTAIN passes through.
        # ------------------------------------------------------------------
        abstained = False
        abstain_cause = None

        if decision == "UNCERTAIN":
            if citation_meta.get("citation_verification", {}).get("forced_uncertain"):
                abstain_cause = "citation_hallucination"
                abstained = True
            elif citation_meta.get("match_bar_forced_uncertain"):
                abstain_cause = "no_discriminating_fact"
                abstained = True
            else:
                abstain_cause = "model_said_uncertain"

        # Threshold rail — non-barrier pairs, or any NO_MATCH
        if decision in {"MATCH", "NO_MATCH"} and conf < abstain_below:
            if decision == "MATCH" and barrier and not shared_nid:
                # Barrier MATCH uses the barrier confidence floor below, not 60.
                pass
            else:
                decision = "UNCERTAIN"
                abstained = True
                abstain_cause = "threshold_converted"
                signals = sorted(set(signals + ["insufficient_evidence", "low_confidence_abstain"]))
                rationale = f"[abstain conf<{abstain_below}] " + rationale

        # UCID-only MATCH without shared content tokens → co-litigant, not identity
        if (
            decision == "MATCH"
            and ucid_only
            and t3.get("ucid_only_match_requires_name_overlap")
            and not (_content_tokens(_norm_name(ma)) & _content_tokens(_norm_name(mb)))
        ):
            decision = "NO_MATCH"
            signals = sorted(set(signals + ["ucid_only_no_name_overlap"]))
            rationale = (
                "[ucid-only: no shared content token; co-litigant not identity] " + rationale
            )

        if (
            decision == "MATCH"
            and ucid_only
            and t3.get("ucid_only_match_requires_same_corp_core")
            and (ma.get("office_class") or "") == "corporate"
            and (mb.get("office_class") or "") == "corporate"
        ):
            ca, cb = corp_core_for_mention(ma, cfg), corp_core_for_mention(mb, cfg)
            if ca and cb and ca != cb:
                decision = "NO_MATCH"
                signals = sorted(set(signals + ["ucid_only_corp_core_mismatch"]))
                rationale = (
                    f"[ucid-only: distinct corp_core {ca!r} vs {cb!r}] " + rationale
                )

        # Barrier rail — Tier3 may overrule at high confidence
        if decision == "MATCH" and barrier and not shared_nid:
            if conf >= barrier_floor:
                stats["barrier_llm_overrides"] += 1
                signals = sorted(set(signals + ["barrier_overruled_by_llm"]))
                rationale = (
                    f"[barrier overruled: MATCH conf={conf}>={barrier_floor}] " + rationale
                )
            else:
                decision = "UNCERTAIN"
                abstained = True
                abstain_cause = "barrier_abstain"
                signals = sorted(set(signals + ["low_information_name", "insufficient_evidence", "barrier_below_llm_min"]))
                if ucid_only and extra_ucid_floor and barrier_floor > barrier_llm_min:
                    signals = sorted(set(signals + ["ucid_only_weak_context"]))
                rationale = (
                    f"[abstain barrier: MATCH conf={conf}<{barrier_floor}] " + rationale
                )

        # A5: fusing two large clusters requires LLM verification — this Tier3
        # call IS that verification; require barrier_llm_min confidence floor.
        large_fuse = large_cluster_merge_blocked(uf, a, b, cfg)
        if decision == "MATCH" and large_fuse and not shared_nid and conf < barrier_llm_min:
            decision = "UNCERTAIN"
            abstained = True
            abstain_cause = "large_cluster_verify_abstain"
            signals = sorted(set(signals + ["large_cluster_verify", "insufficient_evidence"]))
            rationale = (
                f"[abstain large-cluster fuse: conf={conf}<{barrier_llm_min}] " + rationale
            )

        if decision == "MATCH":
            uf.union(a, b)
            stats["merges"] += 1
            out_decision = "MERGE_TIER3"
        elif decision == "NO_MATCH":
            stats["no_match"] += 1
            out_decision = "NO_MATCH"
        else:
            stats["uncertain"] += 1
            out_decision = "UNCERTAIN"
            if abstain_cause == "model_said_uncertain":
                stats["uncertain_model_said"] += 1
            elif abstain_cause == "threshold_converted":
                stats["uncertain_threshold_converted"] += 1
            elif abstain_cause == "barrier_abstain":
                stats["uncertain_barrier_abstain"] += 1
            elif abstain_cause == "citation_hallucination":
                stats["uncertain_citation_hallucination"] = (
                    stats.get("uncertain_citation_hallucination", 0) + 1
                )
            elif abstain_cause == "no_discriminating_fact":
                stats["uncertain_no_discriminating_fact"] = (
                    stats.get("uncertain_no_discriminating_fact", 0) + 1
                )
            else:
                stats["uncertain_other"] += 1
                abstain_cause = abstain_cause or "other"

        journal.log(
            {
                "decision_id": f"dec_{journal.n:08d}",
                "mention_id_a": a,
                "mention_id_b": b,
                "entity_type": cfg.get("entity_type"),
                "decision": out_decision,
                "confidence": conf,
                "method": f"tier3.ollama.{model}",
                "model": model,
                "prompt_hash": prompt_hash,
                "rationale": rationale,
                "signals": signals,
                "evidence": {
                    "embedding_similarity": sim,
                    "barrier": barrier,
                    "barrier_reasons": reasons,
                    "shared_fjc_nid": shared_nid,
                    "abstained": abstained,
                    "abstain_cause": abstain_cause,
                    "cache_hit": from_cache,
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "large_cluster_fuse": large_fuse,
                    "barrier_llm_min_confidence": barrier_llm_min,
                    "abstain_confidence_below": abstain_below,
                    "citation_verification": citation_meta.get("citation_verification"),
                    "discriminating_facts_present": citation_meta.get(
                        "discriminating_facts_present"
                    ),
                    "discriminating_facts_accepted": citation_meta.get(
                        "discriminating_facts_accepted"
                    ),
                    "cited_evidence": result.get("cited_evidence"),
                },
                "cache_key": ck,
                "timestamp": _now(),
                "config_version": cfg.get("version"),
            }
        )
        _progress(pair_i)

    adjudicated = stats["merges"] + stats["no_match"] + stats["uncertain"]
    accounted = (
        adjudicated
        + stats["pairs_skipped_already_merged"]
        + stats["pairs_skipped_budget"]
        + stats["pairs_skipped_not_same_block"]
        + stats["errors"]
    )
    stats["adjudicated_pairs"] = adjudicated
    stats["accounted_pairs"] = accounted
    stats["unaccounted_pairs"] = stats["ambiguous_pairs_in"] - accounted
    stats["max_calls"] = max_calls
    stats["abstain_confidence_below"] = abstain_below
    stats["barrier_llm_min_confidence"] = barrier_llm_min
    stats["pct_ambiguous_pairs_adjudicated"] = round(
        100.0 * adjudicated / max(1, stats["ambiguous_pairs_in"]), 2
    )
    stats["pct_ambiguous_pairs_accounted"] = round(
        100.0 * accounted / max(1, stats["ambiguous_pairs_in"]), 2
    )
    stats["llm_calls_per_mention"] = round(stats["llm_calls"] / max(1, total_mentions), 4)
    stats["llm_mention_fraction"] = stats["llm_calls_per_mention"]
    stats["avg_sec_per_llm_call"] = round(
        llm_time_sec / stats["llm_calls"], 3
    ) if stats["llm_calls"] else None
    if stats["unaccounted_pairs"] != 0:
        print(
            f"WARNING: {stats['unaccounted_pairs']} ambiguous pairs unaccounted "
            f"(in={stats['ambiguous_pairs_in']}, accounted={accounted})"
        )
    return stats


# Ollama health check runs once per process for each (endpoint, llm, embed) triple.
_PREFLIGHT_DONE: set[tuple[str, str, str]] = set()


def run_cascade(mentions: list[dict], cfg: dict, enable_tier3: bool = True) -> dict:
    t0 = time.time()

    # Mandatory Tier3 preflight — refuse to start if Ollama/models are down.
    if enable_tier3 and (cfg.get("tier3") or {}).get("enabled", True):
        t3 = cfg.get("tier3") or {}
        model = t3.get("model", "qwen2.5:7b")
        endpoint = ollama_endpoint(cfg)
        embed_model = ((cfg.get("tier2") or {}).get("ollama") or {}).get("model") or "nomic-embed-text"
        pf_key = (endpoint, model, embed_model)
        if pf_key not in _PREFLIGHT_DONE:
            print(f"PREFLIGHT: endpoint={endpoint} llm={model} embed={embed_model}", flush=True)
            try:
                pf = preflight_ollama(endpoint=endpoint, llm_model=model, embed_model=embed_model)
                print(f"PREFLIGHT_OK: {json.dumps({k: pf[k] for k in ('generate_ping','embed_dim','ok')})}", flush=True)
            except PreflightError as e:
                print(f"PREFLIGHT_FAILED: {e}", flush=True)
                raise
            _PREFLIGHT_DONE.add(pf_key)

    journal_path = resolve_path(cfg, cfg["io"]["decisions_out"])
    journal = DecisionJournal(journal_path)
    uf = UnionFind()

    common_path = resolve_path(
        cfg,
        ((cfg.get("information_content_barrier") or {}).get("triggers") or [{}])[2].get(
            "list_path", "data/external/common_surnames.txt"
        )
        if len((cfg.get("information_content_barrier") or {}).get("triggers") or []) > 2
        else "data/external/common_surnames.txt",
    )
    # Robust path from config
    for trig in (cfg.get("information_content_barrier") or {}).get("triggers") or []:
        if trig.get("id") == "very_common_surname" and trig.get("list_path"):
            common_path = resolve_path(cfg, trig["list_path"])
            break
    common_surnames = load_common_surnames(common_path)

    hygiene_counts: dict[str, int] = {}
    quarantined: list[dict] = []
    if (cfg.get("mention_hygiene") or {}).get("enabled", True):
        mentions, quarantined, hygiene_counts = apply_hygiene(
            mentions, common_surnames, cfg=cfg
        )
        print(f"Mention hygiene: {hygiene_counts} (quarantined={len(quarantined)})", flush=True)

    by_id = {m["mention_id"]: m for m in mentions}

    # Teach the name_gate which tokens in this pool are surnames and which are
    # docket prose, so it compares surnames rather than trailing noise.
    cfg["_token_profile"] = build_token_profile_for_run(mentions, cfg)
    print(
        f"Name token profile: protected={len(cfg['_token_profile']['protected'])} "
        f"noise={len(cfg['_token_profile']['noise'])}",
        flush=True,
    )

    # FJC linking (soft dates) before Tier0 NID joins
    fjc_link_stats = None
    fjc_rule = next(
        (r for r in ((cfg.get("tier0") or {}).get("rules") or []) if r.get("id") == "fjc_nid_join"),
        None,
    )
    if fjc_rule:
        ext = fjc_rule.get("external") or {}
        fjc_path = resolve_path(cfg, ext.get("path", "data/judges_fjc.csv"))
        crosswalk = resolve_path(cfg, ext.get("crosswalk_path", "data/external/fjc_court_crosswalk.json"))
        if fjc_path.exists():
            honorifics = (cfg.get("normalization") or {}).get("strip_honorifics") or []
            strip_chars = (cfg.get("normalization") or {}).get("strip_chars") or ""
            fjc_index = load_fjc_index(fjc_path, crosswalk, honorifics, strip_chars)
            fjc_link_stats = link_mentions_to_fjc(mentions, fjc_index)
            fjc_link_stats["n_fjc_judges"] = fjc_index["n_judges"]
            print(f"FJC linked: {fjc_link_stats}")
        else:
            fjc_link_stats = {"fjc_file_missing": True, "path": str(fjc_path)}

    # Tier 0
    t0_stats = {}
    if (cfg.get("tier0") or {}).get("enabled", True):
        t0_stats = tier0_merge_groups(mentions, cfg, journal, uf)
    if fjc_link_stats is not None:
        t0_stats["fjc_link"] = fjc_link_stats
    if hygiene_counts:
        t0_stats["mention_hygiene"] = hygiene_counts
        t0_stats["quarantined_mentions"] = len(quarantined)

    # Within-UCID anchoring for one-token mentions. Runs after Tier0 (so exact
    # keys already fired) and before any similarity pairing, because a short
    # token must be resolved against its own case's full names rather than
    # compared to another short token.
    anchor_stats: dict[str, Any] = {}
    if (cfg.get("ucid_anchor") or {}).get("enabled", True):
        res = resolve_short_mentions(mentions, cfg)
        for mg in res["merges"]:
            uf.union(mg["anchor_id"], mg["short_id"])
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": mg["anchor_id"],
                    "mention_id_b": mg["short_id"],
                    "entity_type": cfg.get("entity_type"),
                    "decision": "MERGE_UCID_ANCHOR",
                    "confidence": 100,
                    "method": "ucid_anchor",
                    "rationale": (
                        f"Short mention {mg['short_name']!r} uniquely matches the "
                        f"{mg['matched_on']} of {mg['anchor_name']!r} in the same UCID."
                    ),
                    "signals": ["ucid_anchor", mg["matched_on"]],
                    "evidence": {
                        "short_name": mg["short_name"],
                        "anchor_name": mg["anchor_name"],
                        "matched_on": mg["matched_on"],
                        "ucid": mg["ucid"],
                    },
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
        for mid, out in res["outcomes"].items():
            if out.get("status") != "abstain":
                continue
            journal.log(
                {
                    "decision_id": f"dec_{journal.n:08d}",
                    "mention_id_a": mid,
                    "mention_id_b": None,
                    "entity_type": cfg.get("entity_type"),
                    "decision": "ABSTAIN",
                    "confidence": 100,
                    "method": "ucid_anchor.abstain",
                    "rationale": (
                        "Short mention has no unique full-name anchor in its UCID; "
                        "left unattached rather than guessed."
                    ),
                    "signals": ["ucid_anchor", out.get("reason") or "abstain"],
                    "evidence": {
                        k: v for k, v in out.items() if k != "status"
                    } | {"normalized_name": (by_id.get(mid) or {}).get("normalized_name")},
                    "timestamp": _now(),
                    "config_version": cfg.get("version"),
                }
            )
        mark_anchor_outcomes(mentions, res["outcomes"])
        anchor_stats = res["stats"]
        print(f"UCID anchoring: {anchor_stats}", flush=True)

    # Tier 1
    mention_blocks = build_profile_blocks(mentions, cfg)
    blocks = invert_blocks(mention_blocks)

    # Tier 2
    ambiguous = []
    t2_stats = {}
    embed_pack = None
    if (cfg.get("tier2") or {}).get("enabled", True):
        backend = ((cfg.get("tier2") or {}).get("backend") or "ollama_faiss").lower()
        # Tier2 only pairs two different UF groups that share a block. If no
        # block holds two groups, it cannot produce a pair: skip embedding.
        tier2_possible = any(
            len({uf.find(mid) for mid in mids}) >= 2 for mids in blocks.values()
        )
        if tier2_possible and backend in {"ollama_faiss", "ollama", "faiss"}:
            embed_pack = build_ollama_faiss_pack(mentions, cfg)
        pairs = (
            tier2_candidates(mentions, mention_blocks, blocks, uf, cfg, embed_pack=embed_pack)
            if tier2_possible
            else []
        )
        t2_out = apply_tier2_auto_merges(pairs, by_id, uf, cfg, journal, common_surnames)
        t2_stats = t2_out["stats"]
        t2_stats["candidate_pairs"] = len(pairs)
        t2_stats["embedding_backend"] = (embed_pack or {}).get(
            "backend", "rapidfuzz_fallback" if tier2_possible else "skipped_no_multi_group_block"
        )
        ambiguous = t2_out["ambiguous"]
        # Marginal recall: pairs found by Tier2 that Tier0 had not merged
        reports = resolve_path(cfg, cfg["io"]["reports_dir"])
        reports.mkdir(parents=True, exist_ok=True)
        marginal = {
            "note": "Pairs found by Tier2 similarity that Tier0 had not merged (same-block, sim>=min).",
            "tier2_new_candidate_pairs": len(pairs),
            "tier2_auto_merges": t2_stats.get("auto_merges", 0),
            "ambiguous_pairs": len(ambiguous),
            "embedding_backend": t2_stats["embedding_backend"],
            "marginal_recall_numerator": len(pairs),
            "definition": "numerator = |{(a,b) : same block, sim>=min, not already Tier0-merged}|",
        }
        with open(reports / "tier2_marginal_recall.json", "w", encoding="utf-8") as f:
            json.dump(marginal, f, indent=2)
        print(f"Tier2 marginal recall candidates (Tier0-missed pairs): {len(pairs)}")


    # Tier 3
    t3_stats = {}
    if enable_tier3 and (cfg.get("tier3") or {}).get("enabled", True):
        try:
            t3_stats = tier3_adjudicate(ambiguous, by_id, uf, cfg, journal, mention_blocks, len(mentions))
        except CascadeFailedError as e:
            elapsed = time.time() - t0
            fail_stats = dict(e.stats)
            fail_stats["status"] = "FAILED"
            fail_stats["fail_reason"] = str(e)
            summary = {
                "status": "FAILED",
                "fail_reason": str(e),
                "n_mentions": len(mentions),
                "n_quarantined": len(quarantined),
                "tier0": t0_stats,
                "tier1_blocks": len(blocks),
                "tier2": t2_stats,
                "tier3": fail_stats,
                "decisions_logged": journal.n,
                "elapsed_sec": round(elapsed, 2),
            }
            print("CASCADE status=FAILED — do not report entity counts as baseline.", flush=True)
            return {
                "uf": uf,
                "components": uf.components(),
                "journal": journal,
                "summary": summary,
                "by_id": by_id,
                "quarantined": quarantined,
                "mentions": mentions,
                "status": "FAILED",
            }
        cand = (t2_stats or {}).get("candidate_pairs") or 0
        t3_stats["tier2_candidate_pairs"] = cand
        t3_stats["pct_candidate_pairs_adjudicated"] = round(
            100.0 * t3_stats.get("adjudicated_pairs", 0) / max(1, cand), 2
        )
    else:
        t3_stats = {
            "llm_calls": 0,
            "skipped": len(ambiguous),
            "reason": "tier3_disabled",
            "ambiguous_pairs_in": len(ambiguous),
            "adjudicated_pairs": 0,
            "pct_ambiguous_pairs_adjudicated": 0.0,
            "pct_candidate_pairs_adjudicated": 0.0,
            "status": "OK",
        }

    components = uf.components()
    elapsed = time.time() - t0
    summary = {
        "n_mentions": len(mentions),
        "n_quarantined": len(quarantined),
        "n_entities_cascade": len(components),
        "n_entities": len(components) + len(quarantined),
        "tier0": t0_stats,
        "ucid_anchor": anchor_stats,
        "tier1_blocks": len(blocks),
        "tier2": t2_stats,
        "tier3": t3_stats,
        "decisions_logged": journal.n,
        "elapsed_sec": round(elapsed, 2),
        "llm_mention_fraction": t3_stats.get("llm_mention_fraction"),
        "mention_hygiene": hygiene_counts,
    }
    return {
        "uf": uf,
        "components": components,
        "journal": journal,
        "summary": summary,
        "by_id": by_id,
        "quarantined": quarantined,
        "mentions": mentions,
    }
