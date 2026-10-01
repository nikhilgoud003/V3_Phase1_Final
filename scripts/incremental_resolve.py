"""Incremental resolve: match one file against the saved registry.

Does not edit Tier0–3 rules. It calls the existing functions in engine.tiers
(tier0_merge_groups, apply_tier2_auto_merges, tier3_adjudicate, run_cascade)
on only:
  - mentions from the current file (within-file cascade)
  - pairs whose one side is a new mention and the other is a saved entity prototype

Old mentions are never passed back into run_cascade.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from engine.cluster import cluster_mentions
from engine.embeddings import cached_vectors, compact_embed_text, embed_texts_ollama, seed_vector_cache
from engine.poc_party_evidence import adjudicate_poc_evidence_via_tier3, rebuild_components
from engine.config_loader import load_config_cached, ollama_endpoint, resolve_path
from engine.provenance import DecisionJournal
from engine.tiers import (
    UnionFind,
    apply_tier2_auto_merges,
    build_profile_blocks,
    coparty_conflict,
    load_common_surnames,
    party_slot,
    run_cascade,
    tier0_merge_groups,
    tier3_adjudicate,
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_ALIAS_CACHE: dict[str, dict[str, str]] = {}
_SURNAME_CACHE: dict[str, set[str]] = {}

# Set by the driver: run-wide cache folder that survives the per-file work_dir wipe.
RUN_CACHE_DIR: Path | None = None


def _alias_index(cfg: dict) -> dict[str, str]:
    key = str(cfg.get("_config_path"))
    if key not in _ALIAS_CACHE:
        _ALIAS_CACHE[key] = _alias_index_uncached(cfg)
    return _ALIAS_CACHE[key]


def _alias_index_uncached(cfg: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for grp in (cfg.get("tier0") or {}).get("alias_groups") or []:
        gid = str(grp.get("id") or "alias")
        for name in grp.get("names") or []:
            nn = " ".join(str(name).lower().split())
            if nn:
                out[nn] = gid
    return out


def _common_surnames(cfg: dict) -> set[str]:
    key = str(cfg.get("_config_path"))
    if key not in _SURNAME_CACHE:
        _SURNAME_CACHE[key] = _common_surnames_uncached(cfg)
    return _SURNAME_CACHE[key]


def _common_surnames_uncached(cfg: dict) -> set[str]:
    path = None
    for trig in (cfg.get("information_content_barrier") or {}).get("triggers") or []:
        if trig.get("id") == "very_common_surname" and trig.get("list_path"):
            path = resolve_path(cfg, trig["list_path"])
            break
    if path is None:
        path = resolve_path(cfg, "data/external/common_surnames.txt")
    return load_common_surnames(path) if path.is_file() else set()


class EmbedCache:
    """Disk copy of the process-wide vector cache (engine.embeddings).

    Loaded once at start and saved once at the end, so a later run that adds
    files does not re-embed names it has already seen.
    """

    def __init__(self, path: Path, model: str = "nomic-embed-text") -> None:
        self.path = path.with_suffix(".npz")
        self.model = model
        legacy = path.with_suffix(".json")
        if self.path.is_file():
            with np.load(self.path, allow_pickle=False) as z:
                seed_vector_cache(model, dict(zip(z["texts"].tolist(), z["vectors"])))
        elif legacy.is_file() and legacy.stat().st_size:
            payload = json.loads(legacy.read_text(encoding="utf-8"))
            seed_vector_cache(model, {k: np.asarray(v, dtype=np.float32) for k, v in payload.items()})

    def save(self) -> None:
        vecs = cached_vectors(self.model)
        if not vecs:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        texts = list(vecs)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, texts=np.asarray(texts), vectors=np.vstack([vecs[t] for t in texts]))
        os.replace(tmp, self.path)

    def vectors_for(self, mentions: list[dict], cfg: dict) -> dict[str, np.ndarray]:
        texts = {m["mention_id"]: compact_embed_text(m) for m in mentions}
        model = (cfg.get("tier2") or {}).get("ollama_embed_model") or "nomic-embed-text"
        order = list(texts)
        mat = embed_texts_ollama([texts[mid] for mid in order], model=model, endpoint=ollama_endpoint(cfg))
        return {mid: mat[i] for i, mid in enumerate(order)}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _rows_text(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def load_checkpoint(out_root: Path) -> dict[str, Any] | None:
    state_path = out_root / "checkpoint" / "state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    ck = out_root / "checkpoint"

    def _load(name: str) -> list[dict]:
        p = ck / name
        if not p.is_file():
            return []
        return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

    state["entities"] = {
        "judge": _load("entities_judge.jsonl"),
        "firm": _load("entities_firm.jsonl"),
        "party": _load("entities_party.jsonl"),
    }
    state["mentions"] = {
        "judge": _load("mentions_judge.jsonl"),
        "firm": _load("mentions_firm.jsonl"),
        "party": _load("mentions_party.jsonl"),
    }
    state["decisions"] = _load("decisions.jsonl")
    return state


def save_checkpoint(out_root: Path, state: dict[str, Any]) -> None:
    ck = out_root / "checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    meta = {
        "processed": state["processed"],
        "next_serial": state["next_serial"],
        "timings": state["timings"],
        "poc_evidence": state.get("poc_evidence") or [],
        "cascade_last": state.get("cascade_last") or {},
    }
    _atomic_write(ck / "state.json", json.dumps(meta, indent=2, default=str))
    for et in ("judge", "firm", "party"):
        _atomic_write(ck / f"entities_{et}.jsonl", _rows_text(state["entities"][et]))
        _atomic_write(ck / f"mentions_{et}.jsonl", _rows_text(state["mentions"][et]))
    _atomic_write(ck / "decisions.jsonl", _rows_text(state["decisions"]))
    if state.get("embed_cache") is not None:
        state["embed_cache"].save()
    public_entities = []
    public_mentions = []
    for et in ("judge", "firm", "party"):
        for e in state["entities"][et]:
            row = {k: v for k, v in e.items() if k != "_proto"}
            row["type"] = et
            row["entity_type"] = et
            public_entities.append(row)
        for m in state["mentions"][et]:
            row = dict(m)
            row["type"] = et
            row.setdefault("entity_type", et)
            public_mentions.append(row)
    _atomic_write(out_root / "entities.jsonl", _rows_text(public_entities))
    _atomic_write(out_root / "mentions.jsonl", _rows_text(public_mentions))
    _atomic_write(out_root / "decisions.jsonl", _rows_text(state["decisions"]))
    summary = state.get("summary_partial") or {"checkpoint": True, "processed": state["processed"]}
    _atomic_write(out_root / "summary.json", json.dumps(summary, indent=2, default=str))


def _proto(entity: dict, by_id: dict[str, dict]) -> dict:
    if entity.get("_proto"):
        return entity["_proto"]
    for mid in entity.get("mention_ids") or []:
        if mid in by_id:
            return by_id[mid]
    return {}


def _absorb(entity: dict, new_entity: dict, new_mentions: list[dict]) -> None:
    ids = list(entity.get("mention_ids") or [])
    have = set(ids)
    for mid in new_entity.get("mention_ids") or []:
        if mid not in have:
            ids.append(mid)
            have.add(mid)
    entity["mention_ids"] = ids
    entity["n_mentions"] = len(ids)
    courts = set(entity.get("courts") or [])
    ucids = set(entity.get("ucids") or [])
    variants = set(entity.get("name_variants") or [])
    for m in new_mentions:
        if m.get("court"):
            courts.add(m["court"])
        if m.get("ucid"):
            ucids.add(m["ucid"])
        if m.get("normalized_name"):
            variants.add(m["normalized_name"])
    entity["courts"] = sorted(courts)
    entity["ucids"] = sorted(ucids)
    entity["name_variants"] = sorted(variants)
    nids = set(entity.get("fjc_nids") or [])
    nids.update(new_entity.get("fjc_nids") or [])
    entity["fjc_nids"] = sorted(nids)


class SavedIndex:
    """Lookup table from block key / alias group to saved entities.

    Replaces re-blocking every saved prototype for every new entity. The
    saved list only grows by appending, so entity positions are stable and
    candidates come back in the same (saved-list) order as before.
    """

    def __init__(self) -> None:
        self.n = 0
        self.by_block: dict[str, list[int]] = defaultdict(list)
        self.by_alias: dict[str, list[int]] = defaultdict(list)
        self.by_nid: dict[str, set[int]] = defaultdict(set)
        self.pos: dict[int, int] = {}
        # Saved mentions by id (filled by the driver), for the co-party barrier.
        self.mentions: dict[str, dict] = {}

    def sync(self, saved: list[dict], cfg: dict) -> None:
        if self.n >= len(saved):
            return
        alias = _alias_index(cfg)
        new = saved[self.n :]
        protos = [e.get("_proto") or {} for e in new]
        blocks = build_profile_blocks([p for p in protos if p.get("mention_id")], cfg)
        for i, (e, proto) in enumerate(zip(new, protos), start=self.n):
            self.pos[id(e)] = i
            for nid in e.get("fjc_nids") or []:
                self.by_nid[str(nid)].add(i)
            gid = alias.get(" ".join((proto.get("normalized_name") or e.get("normalized_name") or "").lower().split()))
            if gid:
                self.by_alias[gid].append(i)
            if not e.get("_proto"):
                continue
            for k in blocks.get(proto.get("mention_id") or "") or []:
                self.by_block[k].append(i)
        self.n = len(saved)

    def note_absorb(self, entity: dict) -> None:
        """Index FJC ids a saved entity gained by absorbing a new one."""
        i = self.pos.get(id(entity))
        if i is None:
            return
        for nid in entity.get("fjc_nids") or []:
            self.by_nid[str(nid)].add(i)

    def fjc_hits(self, saved: list[dict], nids: set[str], cfg: dict) -> list[dict]:
        self.sync(saved, cfg)
        hit: set[int] = set()
        for nid in nids:
            hit |= self.by_nid.get(str(nid)) or set()
        return [saved[i] for i in sorted(hit)]

    def candidates(self, saved: list[dict], new_mentions: list[dict], cfg: dict) -> list[dict]:
        if not saved:
            return []
        self.sync(saved, cfg)
        alias = _alias_index(cfg)
        hit: set[int] = set()
        for m in new_mentions:
            gid = alias.get(" ".join((m.get("normalized_name") or "").lower().split()))
            if gid:
                hit.update(self.by_alias.get(gid) or ())
        blocks = build_profile_blocks(new_mentions, cfg)
        for m in new_mentions:
            for k in blocks.get(m["mention_id"]) or []:
                hit.update(self.by_block.get(k) or ())
        return [saved[i] for i in sorted(hit)]


def _coparty_ok(entity: dict, members: list[dict], index: SavedIndex, by_id: dict, cfg: dict) -> bool:
    """False when linking would put two separately listed co-parties of one case together."""
    if not (cfg.get("coparty_barrier") or {}).get("enabled"):
        return True
    slots_e = set()
    for mid in entity.get("mention_ids") or []:
        m = index.mentions.get(mid) or by_id.get(mid)
        slot = party_slot(m, cfg) if m else None
        if slot:
            slots_e.add(slot)
    slots_n = {s for s in (party_slot(m, cfg) for m in members) if s}
    return coparty_conflict(slots_e, slots_n) is None


def link_against_saved(
    new_entities: list[dict],
    by_id: dict[str, dict],
    saved: list[dict],
    cfg: dict,
    embed_cache: EmbedCache,
    journal: DecisionJournal,
    index: SavedIndex | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Attach each new entity to one saved entity, or return it as new.

    Saved entities are never merged with each other.
    """
    stats = {"fjc_links": 0, "tier0_links": 0, "tier2_links": 0, "tier3_links": 0, "new": 0, "tier3_calls": 0}
    still_new: list[dict] = []
    alias = _alias_index(cfg)
    surnames = _common_surnames(cfg)
    min_sim = float(((cfg.get("tier2") or {}).get("search") or {}).get("min_similarity", 0.72))
    if index is None:
        index = SavedIndex()

    for ent in new_entities:
        members = [by_id[mid] for mid in (ent.get("mention_ids") or []) if mid in by_id]
        if not members:
            still_new.append(ent)
            stats["new"] += 1
            continue
        # Same FJC id = same judge, in any court: link directly. Different FJC
        # ids = different judges: never link those, whatever the name says.
        nids = {str(x) for x in ent.get("fjc_nids") or []}
        if nids:
            nid_hits = index.fjc_hits(saved, nids, cfg)
            if nid_hits:
                chosen = sorted(nid_hits, key=lambda h: str(h.get("sjid") or ""))[0]
                _absorb(chosen, ent, members)
                index.note_absorb(chosen)
                stats["fjc_links"] += 1
                journal.log(
                    {
                        "decision": "MERGE_TIER0",
                        "method": "incremental.link_fjc_nid",
                        "mention_id_a": members[0]["mention_id"],
                        "mention_id_b": (chosen.get("_proto") or {}).get("mention_id"),
                        "entity_type": cfg.get("entity_type"),
                        "confidence": 100,
                        "rationale": f"Same FJC id {sorted(nids & set(map(str, chosen.get('fjc_nids') or [])))} as saved {chosen.get('sjid')}",
                        "signals": ["incremental", "fjc_nid_match"],
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
                continue
        cands = index.candidates(saved, members, cfg)
        if nids:
            cands = [c for c in cands if not (c.get("fjc_nids") and not nids & {str(x) for x in c["fjc_nids"]})]
        if not cands:
            still_new.append(ent)
            stats["new"] += 1
            continue

        pool = members + [c["_proto"] for c in cands if c.get("_proto")]
        uf = UnionFind()
        # Keep the within-file entity as one blob. Tier0 may then union it with
        # a saved prototype. Saved prototypes are not unioned with each other
        # in the registry even if this temporary UF joins them.
        for m in members:
            uf.add(m["mention_id"])
        for m in members[1:]:
            uf.union(members[0]["mention_id"], m["mention_id"])
        tier0_merge_groups(pool, cfg, journal, uf)
        root_new = uf.find(members[0]["mention_id"])
        hits = []
        for c in cands:
            proto = c.get("_proto") or {}
            if proto.get("mention_id") and uf.find(proto["mention_id"]) == root_new:
                hits.append(c)
        hits = [h for h in hits if _coparty_ok(h, members, index, by_id, cfg)]
        if hits:
            chosen = _pick_hit(hits, ent)
            _absorb(chosen, ent, members)
            index.note_absorb(chosen)
            stats["tier0_links"] += 1
            journal.log(
                {
                    "decision": "MERGE_TIER0",
                    "method": "incremental.link_tier0",
                    "mention_id_a": members[0]["mention_id"],
                    "mention_id_b": (chosen.get("_proto") or {}).get("mention_id"),
                    "entity_type": cfg.get("entity_type"),
                    "confidence": 100,
                    "rationale": f"New file linked to saved {chosen.get('sjid')} by existing Tier0 rules",
                    "signals": ["incremental", "tier0"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            continue

        # Tier2/3: new prototype vs each saved prototype. Not saved vs saved.
        proto_new = members[0]
        protos = [c["_proto"] for c in cands if c.get("_proto")]
        vecs = embed_cache.vectors_for([proto_new] + protos, cfg)
        by = {m["mention_id"]: m for m in [proto_new] + protos}
        pairs = []
        vnew = vecs.get(proto_new["mention_id"])
        if vnew is not None:
            for c in cands:
                proto = c.get("_proto") or {}
                pid = proto.get("mention_id")
                vold = vecs.get(pid) if pid else None
                if vold is None:
                    continue
                sim = float(np.dot(vnew, vold))
                if sim >= min_sim:
                    pairs.append((proto_new["mention_id"], pid, sim))
        uf2 = UnionFind()
        for m in by.values():
            uf2.add(m["mention_id"])
        t2 = apply_tier2_auto_merges(pairs, by, uf2, cfg, journal, surnames)
        linked = None
        for c in cands:
            pid = (c.get("_proto") or {}).get("mention_id")
            if pid and uf2.find(proto_new["mention_id"]) == uf2.find(pid):
                linked = c
                break
        if linked is not None and not _coparty_ok(linked, members, index, by_id, cfg):
            linked = None
            stats["coparty_blocked"] = stats.get("coparty_blocked", 0) + 1
        if linked is not None:
            _absorb(linked, ent, members)
            index.note_absorb(linked)
            stats["tier2_links"] += 1
            continue
        ambiguous = [p for p in (t2.get("ambiguous") or []) if proto_new["mention_id"] in (p[0], p[1])]
        if ambiguous:
            blocks = build_profile_blocks(list(by.values()), cfg)
            t3 = tier3_adjudicate(
                ambiguous, by, uf2, cfg, journal, blocks, total_mentions=len(by)
            )
            stats["tier3_calls"] += int(t3.get("llm_calls") or 0)
            for c in cands:
                pid = (c.get("_proto") or {}).get("mention_id")
                if pid and uf2.find(proto_new["mention_id"]) == uf2.find(pid):
                    linked = c
                    break
            if linked is not None and not _coparty_ok(linked, members, index, by_id, cfg):
                linked = None
                stats["coparty_blocked"] = stats.get("coparty_blocked", 0) + 1
            if linked is not None:
                _absorb(linked, ent, members)
                index.note_absorb(linked)
                stats["tier3_links"] += 1
                continue
        still_new.append(ent)
        stats["new"] += 1
    return saved, still_new, stats


def _pick_hit(hits: list[dict], new_ent: dict) -> dict:
    nn = " ".join((new_ent.get("normalized_name") or "").lower().split())
    same = [h for h in hits if " ".join((h.get("normalized_name") or "").lower().split()) == nn]
    pool = same or hits
    nids = set(new_ent.get("fjc_nids") or [])
    if nids:
        nid_hits = [h for h in pool if nids & set(h.get("fjc_nids") or [])]
        if len(nid_hits) == 1:
            return nid_hits[0]
        if nid_hits:
            pool = nid_hits
    return sorted(pool, key=lambda h: str(h.get("sjid") or ""))[0]


def resolve_within_file(mentions: list[dict], transfers: list[dict], cfg: dict, work_dir: Path) -> dict:
    """Cascade only this file's mentions. Caller sets TIER_V3_OUTPUT_DIR to work_dir."""
    from engine.extract import _finalize_mentions

    os.environ["TIER_V3_OUTPUT_DIR"] = str(work_dir)
    cfg = load_config_cached(cfg["_config_path"])
    if RUN_CACHE_DIR is not None:
        llm_nv = (cfg.get("name_validity") or {}).get("llm_validation")
        if isinstance(llm_nv, dict):
            llm_nv["cache_path"] = str(RUN_CACHE_DIR / "llm_name_validity_cache.jsonl")
    finalized = _finalize_mentions(
        cfg,
        list(mentions),
        list(transfers),
        write=True,
        transfer_out_rel="data/mentions/transfer_clues.jsonl",
    )
    if not finalized:
        return {"entities": [], "by_id": {}, "mentions": [], "summary": {}, "decisions": [], "poc": {}}
    result = run_cascade(finalized, cfg, enable_tier3=True)
    uf = result["uf"]
    by_id = result["by_id"]
    poc: dict[str, Any] = {}
    if cfg.get("entity_type") == "party":
        journal = DecisionJournal(work_dir / "decisions" / "poc_party_evidence_decisions.jsonl", fresh=True)
        poc_stats = adjudicate_poc_evidence_via_tier3(
            finalized, uf, cfg, journal=journal, max_pairs=25
        )
        if poc_stats["merges"]:
            comps = rebuild_components(uf, [m["mention_id"] for m in finalized])
            entities = cluster_mentions(comps, by_id, cfg, uf)
        else:
            entities = cluster_mentions(result["components"], by_id, cfg, uf)
        poc = {
            "candidates": poc_stats["candidates"],
            "tier3_calls": poc_stats["tier3_calls"],
            "merges_after_citation_match": poc_stats["merges"],
        }
    else:
        entities = cluster_mentions(result["components"], by_id, cfg, uf)
    decisions = []
    for p in (work_dir / "decisions").glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                d.setdefault("entity_type", cfg.get("entity_type"))
                decisions.append(d)
    return {
        "entities": entities,
        "by_id": by_id,
        "mentions": list(by_id.values()),
        "summary": result.get("summary") or {},
        "decisions": decisions,
        "poc": poc,
    }


def stamp_new_entities(entities: list[dict], by_id: dict[str, dict], prefix: str, serial: int) -> tuple[list[dict], int]:
    out = []
    for e in entities:
        serial += 1
        row = dict(e)
        row["sjid"] = f"{prefix}{serial:06d}"
        row["entity_id"] = f"ent_{prefix}_{serial:06d}"
        row["entity_type"] = row.get("entity_type")
        members = [by_id[m] for m in (row.get("mention_ids") or []) if m in by_id]
        proto = None
        nn = row.get("normalized_name")
        for m in members:
            if m.get("normalized_name") == nn:
                proto = m
                break
        row["_proto"] = proto or (members[0] if members else {})
        out.append(row)
    return out, serial
