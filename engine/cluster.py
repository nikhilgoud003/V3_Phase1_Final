"""Union-find clustering + merge verification → entity registry.

A5 safety rail
--------------
1. **Cascade-time (primary):** ``engine.tiers.large_cluster_merge_blocked``
   prevents Tier2 auto-merge when both UF components are ≥
   ``clustering.large_cluster_verify_min_size``. Tier3 MATCH that would fuse
   two large clusters must meet ``barrier_llm_min_confidence`` (LLM
   re-verification). See ``tier3_adjudicate`` / ``apply_tier2_auto_merges``.

2. **Cluster-time (this module):** after UF components are materialised,
   split any component that contains incompatible surnames (name_gate
   rules) or distinct FJC NIDs / transfer conflicts. ``verify_merges``
   config flag enables the surname split rail.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config_loader import resolve_path
from .name_compat import names_compatible, punct_fold_compatible
from .tiers import (
    UnionFind,
    domain_is_non_identifying,
    fund_plan_type_tokens,
    generational_suffix_conflict,
    mention_office_geo,
    transfer_conflict,
)


def cluster_mentions(
    components: dict[str, list[str]],
    by_id: dict[str, dict],
    cfg: dict,
    uf: UnionFind | None = None,
) -> list[dict]:
    """Build entity records from UF components with safety checks."""
    reject = set((cfg.get("clustering") or {}).get("reject_if") or [])
    verify = bool((cfg.get("clustering") or {}).get("verify_merges", True))
    prefix = (cfg.get("clustering") or {}).get("id_prefix", "SJ")
    entities = []
    serial = 0

    # Sort components by size desc for stable IDs
    ordered = sorted(components.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    for root, members in ordered:
        member_mentions = [by_id[m] for m in members if m in by_id]
        if not member_mentions:
            continue

        # Safety: distinct FJC NIDs
        nids = {m.get("fjc_nid") for m in member_mentions if m.get("fjc_nid")}
        if "both_have_distinct_fjc_nids" in reject and len(nids) > 1:
            for m in member_mentions:
                entities.append(_entity_record(prefix, serial, [m], cfg, split_reason="distinct_fjc_nids"))
                serial += 1
            continue

        # Safety: transfer conflicts inside component → mark Ambiguous / split pairs
        if "transfer_clue_conflict" in reject:
            conflict = False
            for i in range(len(member_mentions)):
                for j in range(i + 1, len(member_mentions)):
                    if transfer_conflict(member_mentions[i], member_mentions[j]):
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                name_counts = Counter(m["normalized_name"] for m in member_mentions)
                top_name, _ = name_counts.most_common(1)[0]
                keep = [m for m in member_mentions if m["normalized_name"] == top_name]
                rest = [m for m in member_mentions if m["normalized_name"] != top_name]
                entities.append(_entity_record(prefix, serial, keep, cfg, split_reason=None))
                serial += 1
                for m in rest:
                    entities.append(
                        _entity_record(
                            prefix,
                            serial,
                            [m],
                            cfg,
                            split_reason="transfer_clue_conflict",
                            sentinel="Ambiguous",
                        )
                    )
                    serial += 1
                continue

        # Cluster-time identity rails (name_compat + optional parties splitters)
        cl = cfg.get("clustering") or {}
        groups: list[list[dict]] = [member_mentions]
        split_reason: str | None = None

        if verify and len(member_mentions) >= 2:
            new_groups = _split_by_name_compat(member_mentions, cfg)
            if len(new_groups) > 1:
                split_reason = "name_gate_cluster_split"
            groups = new_groups

        if cl.get("split_generational_suffixes") and any(len(g) >= 2 for g in groups):
            expanded: list[list[dict]] = []
            for grp in groups:
                sub = _split_by_generational_suffix(grp, cfg)
                if len(sub) > 1:
                    split_reason = "generational_cluster_split"
                expanded.extend(sub)
            groups = expanded

        if cl.get("split_fund_plan_types") and any(len(g) >= 2 for g in groups):
            expanded = []
            for grp in groups:
                sub = _split_by_fund_plan_type(grp, cfg)
                if len(sub) > 1:
                    split_reason = "fund_plan_cluster_split"
                expanded.extend(sub)
            groups = expanded

        if len(groups) > 1:
            for grp in groups:
                entities.append(
                    _entity_record(
                        prefix,
                        serial,
                        grp,
                        cfg,
                        split_reason=split_reason,
                    )
                )
                serial += 1
            continue

        entities.append(_entity_record(prefix, serial, groups[0], cfg))
        serial += 1

    if (cfg.get("clustering") or {}).get("union_punct_fold_variants"):
        entities = _union_punct_fold_variants(entities, by_id, cfg, prefix)

    out = resolve_path(cfg, cfg["io"]["clusters_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in entities:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Always write the run-local registry next to entities.jsonl. Do not write
    # incremental.registry_path here — that path is the *target* for REUSE
    # matching (recall-fix), and overwriting it from a later cascade would
    # clobber the NID-corrected file.
    if out.name.endswith("_entities.jsonl"):
        reg = out.with_name(out.name.replace("_entities.jsonl", "_entity_registry.jsonl"))
    else:
        reg = out.parent / "judges_entity_registry.jsonl"
    with open(reg, "w", encoding="utf-8") as f:
        for e in entities:
            f.write(json.dumps(registry_record(e), ensure_ascii=False) + "\n")

    return entities


def registry_record(e: dict) -> dict:
    """Slim registry row. Must include fjc_nids — NID-first REUSE/CREATE depends on it.

    Firms also persist identifying domains + office geo for signature REUSE.
    """
    return {
        "entity_id": e["entity_id"],
        "entity_type": e.get("entity_type"),
        "canonical_name": e["canonical_name"],
        "normalized_name": e["normalized_name"],
        "courts": e.get("courts") or [],
        "sjid": e["sjid"],
        "mention_ids": e.get("mention_ids") or [],
        "fjc_nids": e.get("fjc_nids") or [],
        "domains": e.get("domains") or [],
        "office_state": e.get("office_state"),
        "office_city": e.get("office_city"),
    }


def _punct_fold_keys(e: dict, cfg: dict) -> set[str]:
    from .name_compat import _punct_fold_collapsed_and, _punct_fold_tokens

    suffixes = list((cfg.get("normalization") or {}).get("strip_corp_suffixes") or [])
    keys: set[str] = set()
    for name in e.get("name_variants") or [e.get("normalized_name") or ""]:
        toks = _punct_fold_tokens(name, suffixes)
        if toks:
            keys.add("tok:" + " ".join(toks))
            joined = "".join(toks)
            if len(joined) >= 6:
                keys.add("join:" + joined)
        collapsed = _punct_fold_collapsed_and(name, suffixes)
        if len(collapsed) >= 3:
            keys.add("and:" + collapsed)
    return keys


def _punct_fold_link(ea: dict, eb: dict, by_id: dict[str, dict], cfg: dict) -> bool:
    """True when two entities are punctuation variants of one legal name.

    Exact identical name-sets are not linked. Those stay the cross-court
    no-anchor limitation. A punct-different partner can still pull exact
    copies into the same component transitively.
    """
    names_a = set(ea.get("name_variants") or [ea.get("normalized_name") or ""])
    names_b = set(eb.get("name_variants") or [eb.get("normalized_name") or ""])
    names_a.discard("")
    names_b.discard("")
    if not names_a or not names_b or names_a == names_b:
        return False

    def reps(e: dict) -> list[dict]:
        out = []
        seen: set[str] = set()
        for mid in e.get("mention_ids") or []:
            m = by_id.get(mid)
            if not m:
                continue
            nn = m.get("normalized_name") or ""
            if nn in seen:
                continue
            seen.add(nn)
            out.append(m)
        return out

    ma_list, mb_list = reps(ea), reps(eb)
    if not ma_list or not mb_list:
        return False
    linked = False
    for ma in ma_list:
        for mb in mb_list:
            if (ma.get("normalized_name") or "") == (mb.get("normalized_name") or ""):
                continue
            ok, _ = punct_fold_compatible(ma, mb, cfg=cfg)
            if not ok:
                continue
            # Slash in "Anchor/Darling" trips division_subsidiary; equal
            # punct-fold cores are not a parent/subunit split. Group / Services
            # remain blocked because those cores are not equal.
            if generational_suffix_conflict(ma, mb, cfg):
                continue
            linked = True
            break
        if linked:
            break
    return linked


def _union_punct_fold_variants(
    entities: list[dict],
    by_id: dict[str, dict],
    cfg: dict,
    prefix: str,
) -> list[dict]:
    """Join entities split only by hyphen/space/punctuation. Parties config."""
    n = len(entities)
    if n < 2:
        return entities
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Only compare entities that share a punct-folded core. Exact same-name
    # groups never get an edge unless a punctuation-different partner exists.
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entities):
        for key in _punct_fold_keys(e, cfg):
            by_key[key].append(i)
    for idxs in by_key.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if find(ia) == find(ib):
                    continue
                if _punct_fold_link(entities[ia], entities[ib], by_id, cfg):
                    union(ia, ib)

    buckets: dict[int, list[dict]] = defaultdict(list)
    for i, e in enumerate(entities):
        buckets[find(i)].append(e)

    merged: list[dict] = []
    ordered = sorted(buckets.values(), key=lambda g: min(_entity_serial(e) for e in g))
    serial = 0
    for group in ordered:
        if len(group) == 1:
            rec = dict(group[0])
            rec["entity_id"] = f"ent_{serial:06d}"
            rec["sjid"] = f"{prefix}{serial:06d}"
            merged.append(rec)
            serial += 1
            continue
        mids: list[str] = []
        seen: set[str] = set()
        for e in group:
            for mid in e.get("mention_ids") or []:
                if mid not in seen and mid in by_id:
                    seen.add(mid)
                    mids.append(mid)
        mentions = [by_id[mid] for mid in mids]
        reasons = [e.get("split_reason") for e in group if e.get("split_reason")]
        rec = _entity_record(prefix, serial, mentions, cfg, split_reason=reasons[0] if reasons else None)
        merged.append(rec)
        serial += 1
    return merged


def _entity_serial(e: dict) -> int:
    sjid = str(e.get("sjid") or "")
    digits = "".join(ch for ch in sjid if ch.isdigit())
    return int(digits) if digits else 10**9


def _split_by_name_compat(mentions: list[dict], cfg: dict) -> list[list[dict]]:
    """Union-find over mentions using names_compatible as the edge predicate."""
    n = len(mentions)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            ok, _ = names_compatible(mentions[i], mentions[j], cfg=cfg)
            if ok:
                union(i, j)

    buckets: dict[int, list[dict]] = defaultdict(list)
    for i, m in enumerate(mentions):
        buckets[find(i)].append(m)
    return list(buckets.values())


def _split_by_generational_suffix(mentions: list[dict], cfg: dict) -> list[list[dict]]:
    """Split components where Sr/Jr (etc.) conflict — config-gated for parties."""
    if len(mentions) < 2:
        return [mentions]
    n = len(mentions)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if not generational_suffix_conflict(mentions[i], mentions[j], cfg):
                union(i, j)

    buckets: dict[int, list[dict]] = defaultdict(list)
    for i, m in enumerate(mentions):
        buckets[find(i)].append(m)
    return list(buckets.values())


def _split_by_fund_plan_type(mentions: list[dict], cfg: dict) -> list[list[dict]]:
    """Split typed fund/plan bridges without relying on pairwise-only guards.

    Mentions with nonempty plan-type tokens are partitioned so disjoint types
    never share a component. Untyped mentions (e.g. "…Institute of Chicago")
    may join at most one typed group — if they match multiple typed groups by
    name similarity, they become their own entity (breaks Cement Masons bridge).
    """
    if len(mentions) < 2:
        return [mentions]

    typed: list[tuple[int, dict, frozenset[str]]] = []
    untyped: list[tuple[int, dict]] = []
    for i, m in enumerate(mentions):
        types = frozenset(fund_plan_type_tokens(m, cfg))
        if types:
            typed.append((i, m, types))
        else:
            untyped.append((i, m))

    if not typed:
        return [mentions]

    # UF over typed mentions: edge iff type sets overlap
    parent = {i: i for i, _, _ in typed}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for a in range(len(typed)):
        for b in range(a + 1, len(typed)):
            ia, _, ta = typed[a]
            ib, _, tb = typed[b]
            if ta & tb:
                union(ia, ib)

    typed_groups: dict[int, list[dict]] = defaultdict(list)
    typed_roots: dict[int, frozenset[str]] = {}
    for i, m, types in typed:
        r = find(i)
        typed_groups[r].append(m)
        typed_roots[r] = typed_roots.get(r, frozenset()) | types

    if len(typed_groups) <= 1 and not untyped:
        return [mentions]
    if len(typed_groups) <= 1 and untyped:
        # Single typed color — keep untyped with it (no multi-type bridge risk)
        only = next(iter(typed_groups.values()))
        return [only + [m for _, m in untyped]]

    # Multiple typed colors: assign each untyped to ≤1 group by name overlap
    from rapidfuzz import fuzz

    def _nn(m: dict) -> str:
        return (m.get("normalized_name") or "").lower()

    assigned: dict[int, list[dict]] = {r: list(ms) for r, ms in typed_groups.items()}
    leftovers: list[dict] = []
    for _, um in untyped:
        scores = []
        for r, ms in typed_groups.items():
            best = max(fuzz.token_set_ratio(_nn(um), _nn(tm)) for tm in ms)
            scores.append((best, r))
        scores.sort(reverse=True)
        if scores and scores[0][0] >= 85 and (
            len(scores) == 1 or scores[0][0] - scores[1][0] >= 8
        ):
            assigned[scores[0][1]].append(um)
        else:
            leftovers.append(um)

    out = list(assigned.values())
    for m in leftovers:
        out.append([m])
    return out


def _entity_record(
    prefix: str,
    serial: int,
    mentions: list[dict],
    cfg: dict,
    split_reason: str | None = None,
    sentinel: str | None = None,
) -> dict[str, Any]:
    sjid = sentinel or f"{prefix}{serial:06d}"
    names = Counter(m["normalized_name"] for m in mentions)
    canonical_norm, _ = names.most_common(1)[0]
    presentable = next(
        (m["presentable_name"] for m in mentions if m["normalized_name"] == canonical_norm),
        mentions[0]["presentable_name"],
    )
    courts = sorted({m.get("court") for m in mentions if m.get("court")})
    ucids = sorted({m.get("ucid") for m in mentions if m.get("ucid")})
    # Identifying domains only (shared/gov/free-mail excluded) — firms registry key.
    domains = sorted(
        {
            (m.get("domain") or "").strip().lower()
            for m in mentions
            if (m.get("domain") or "").strip()
            and not domain_is_non_identifying(m.get("domain"), cfg)
        }
    )
    # Majority office geo from address-derived mention fields (institutional + private).
    state_counts: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    for m in mentions:
        st, city = mention_office_geo(m, cfg)
        if st:
            state_counts[st] += 1
        if city:
            city_counts[city] += 1
    office_state = state_counts.most_common(1)[0][0] if state_counts else None
    office_city = city_counts.most_common(1)[0][0] if city_counts else None
    return {
        "entity_id": f"ent_{serial:06d}",
        "sjid": sjid,
        "entity_type": cfg.get("entity_type"),
        "canonical_name": presentable,
        "normalized_name": canonical_norm,
        "name_variants": sorted(names.keys()),
        "courts": courts,
        "ucids": ucids,
        "n_mentions": len(mentions),
        "mention_ids": [m["mention_id"] for m in mentions],
        "fjc_nids": sorted({m.get("fjc_nid") for m in mentions if m.get("fjc_nid")}),
        "fjc_match_methods": sorted(
            {
                m.get("fjc_match_method")
                for m in mentions
                if m.get("fjc_nid") and m.get("fjc_match_method")
            }
        ),
        "domains": domains,
        "office_state": office_state,
        "office_city": office_city,
        "split_reason": split_reason,
        "sentinel": sentinel,
    }
