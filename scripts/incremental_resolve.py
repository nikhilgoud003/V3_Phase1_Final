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
from pathlib import Path
from typing import Any

import numpy as np

from engine.cluster import cluster_mentions
from engine.embeddings import compact_embed_text, embed_texts_ollama
from engine.poc_party_evidence import adjudicate_poc_evidence_via_tier3, rebuild_components
from engine.config_loader import load_config, ollama_endpoint, resolve_path
from engine.provenance import DecisionJournal
from engine.tiers import (
    UnionFind,
    apply_tier2_auto_merges,
    build_profile_blocks,
    load_common_surnames,
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


def _alias_index(cfg: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for grp in (cfg.get("tier0") or {}).get("alias_groups") or []:
        gid = str(grp.get("id") or "alias")
        for name in grp.get("names") or []:
            nn = " ".join(str(name).lower().split())
            if nn:
                out[nn] = gid
    return out


def _common_surnames(cfg: dict) -> set[str]:
    path = None
    for trig in (cfg.get("information_content_barrier") or {}).get("triggers") or []:
        if trig.get("id") == "very_common_surname" and trig.get("list_path"):
            path = resolve_path(cfg, trig["list_path"])
            break
    if path is None:
        path = resolve_path(cfg, "data/external/common_surnames.txt")
    return load_common_surnames(path) if path.is_file() else set()


class EmbedCache:
    """Same profile string is embedded once and reused from disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.mem: dict[str, np.ndarray] = {}
        if path.is_file():
            with path.open("rb") as f:
                blob = f.read()
            if blob:
                payload = json.loads(blob.decode("utf-8"))
                for key, vec in payload.items():
                    self.mem[key] = np.asarray(vec, dtype=np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.tolist() for k, v in self.mem.items()}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)

    def vectors_for(self, mentions: list[dict], cfg: dict) -> dict[str, np.ndarray]:
        texts = {m["mention_id"]: compact_embed_text(m) for m in mentions}
        missing = []
        seen = set()
        for t in texts.values():
            if t not in self.mem and t not in seen:
                seen.add(t)
                missing.append(t)
        if missing:
            model = (cfg.get("tier2") or {}).get("ollama_embed_model") or "nomic-embed-text"
            mat = embed_texts_ollama(missing, model=model, endpoint=ollama_endpoint(cfg))
            for t, row in zip(missing, mat):
                self.mem[t] = np.asarray(row, dtype=np.float32)
        return {mid: self.mem[t] for mid, t in texts.items()}


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


def _candidates(saved: list[dict], new_mentions: list[dict], cfg: dict) -> list[dict]:
    if not saved:
        return []
    alias = _alias_index(cfg)
    blocks = build_profile_blocks(new_mentions + [_proto(e, {}) for e in saved if e.get("_proto")], cfg)
    want: set[str] = set()
    for m in new_mentions:
        for k in blocks.get(m["mention_id"]) or []:
            want.add(k)
    new_alias = {alias.get(" ".join((m.get("normalized_name") or "").lower().split())) for m in new_mentions}
    new_alias.discard(None)
    out = []
    for e in saved:
        proto = e.get("_proto") or {}
        gid = alias.get(" ".join((proto.get("normalized_name") or e.get("normalized_name") or "").lower().split()))
        if gid and gid in new_alias:
            out.append(e)
            continue
        keys = set(blocks.get(proto.get("mention_id") or "") or [])
        if keys & want:
            out.append(e)
    return out


def link_against_saved(
    new_entities: list[dict],
    by_id: dict[str, dict],
    saved: list[dict],
    cfg: dict,
    embed_cache: EmbedCache,
    journal: DecisionJournal,
) -> tuple[list[dict], list[dict], dict]:
    """Attach each new entity to one saved entity, or return it as new.

    Saved entities are never merged with each other.
    """
    stats = {"tier0_links": 0, "tier2_links": 0, "tier3_links": 0, "new": 0, "tier3_calls": 0}
    still_new: list[dict] = []
    alias = _alias_index(cfg)
    surnames = _common_surnames(cfg)
    min_sim = float(((cfg.get("tier2") or {}).get("search") or {}).get("min_similarity", 0.72))

    for ent in new_entities:
        members = [by_id[mid] for mid in (ent.get("mention_ids") or []) if mid in by_id]
        if not members:
            still_new.append(ent)
            stats["new"] += 1
            continue
        cands = _candidates(saved, members, cfg)
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
        if hits:
            chosen = _pick_hit(hits, ent)
            _absorb(chosen, ent, members)
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
        if linked is not None:
            _absorb(linked, ent, members)
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
            if linked is not None:
                _absorb(linked, ent, members)
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
    cfg = load_config(cfg["_config_path"])
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
