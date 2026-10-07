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
from engine.embeddings import compact_embed_text, embed_texts_ollama, seed_vector_cache
from engine.poc_party_evidence import adjudicate_poc_evidence_via_tier3, rebuild_components
from engine.config_loader import load_config_cached, ollama_endpoint, resolve_path
from engine.provenance import DecisionJournal
from engine.tiers import (
    STAGE_SEC,
    UnionFind,
    apply_tier2_auto_merges,
    build_profile_blocks,
    coparty_conflict,
    legal_entity_form_set,
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

    Loaded once at start. Each checkpoint writes only the vectors added since
    the last one, as a new shard (embed_cache.NNNNN.npz); checkpoint/state.json
    records how many shards are committed. A legacy embed_cache.npz is loaded too.
    """

    def __init__(self, path: Path, model: str = "nomic-embed-text") -> None:
        self.path = path.with_suffix(".npz")
        self.model = model
        self.saved: set[str] = set()
        legacy = path.with_suffix(".json")
        if self.path.is_file():
            with np.load(self.path, allow_pickle=False) as z:
                texts = z["texts"].tolist()
                seed_vector_cache(model, dict(zip(texts, z["vectors"])))
                self.saved.update(texts)
        elif legacy.is_file() and legacy.stat().st_size:
            payload = json.loads(legacy.read_text(encoding="utf-8"))
            seed_vector_cache(model, {k: np.asarray(v, dtype=np.float32) for k, v in payload.items()})
        self.shards = 0
        st = self.path.parent / "state.json"
        if st.is_file():
            self.shards = int(json.loads(st.read_text(encoding="utf-8")).get("embed_shards") or 0)
        for k in range(self.shards):
            with np.load(self._shard(k), allow_pickle=False) as z:
                texts = z["texts"].tolist()
                seed_vector_cache(model, dict(zip(texts, z["vectors"])))
                self.saved.update(texts)
        for extra in self.path.parent.glob(self.path.stem + ".[0-9]*.npz"):
            if int(extra.name.split(".")[1]) >= self.shards:
                extra.unlink()  # written after the last committed checkpoint

    def _shard(self, k: int) -> Path:
        return self.path.with_name(f"{self.path.stem}.{k:05d}.npz")

    def save(self) -> int:
        """Write vectors added since the last save; return the shard count to commit."""
        new = [(t, v) for (m, t), v in list(_VEC_CACHE_ITEMS()) if m == self.model and t not in self.saved]
        if not new:
            return self.shards
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._shard(self.shards).with_name(f"tmp.{self._shard(self.shards).name}")
        with tmp.open("wb") as f:
            np.savez(f, texts=np.asarray([t for t, _ in new]), vectors=np.vstack([v for _, v in new]))
        os.replace(tmp, self._shard(self.shards))
        self.saved.update(t for t, _ in new)
        self.shards += 1
        return self.shards

    def vectors_for(self, mentions: list[dict], cfg: dict) -> dict[str, np.ndarray]:
        texts = {m["mention_id"]: compact_embed_text(m) for m in mentions}
        model = (cfg.get("tier2") or {}).get("ollama_embed_model") or "nomic-embed-text"
        order = list(texts)
        mat = embed_texts_ollama([texts[mid] for mid in order], model=model, endpoint=ollama_endpoint(cfg))
        return {mid: mat[i] for i, mid in enumerate(order)}


def _VEC_CACHE_ITEMS():
    from engine import embeddings

    return embeddings._VEC_CACHE.items()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_rows(path: Path, rows) -> None:
    """Write JSONL row by row (no whole-file string in memory), atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Checkpoint format 2: append-only streams + one entity generation per save.
#
#   state.json            small; written LAST (atomic) = the commit point. Holds
#                         the committed byte size of every append stream, the
#                         entity generation and the embed shard count.
#   processed / timings / case_names / mentions_{type} / decisions /
#   llm_prompt_cache .jsonl   append-only; bytes past the committed size are
#                         dropped on load (a save that was cut off).
#   entities_{type}.gNNNNNN.jsonl, judge_confirm.gNNNNNN.json   rewritten per save.
#
# Storage only; what is loaded equals what was in memory:
#   - co_mentions lists of >= CO_REF_MIN names are stored once per case in
#     case_names.jsonl ({"_ref": id} in the mention row) when the list is
#     exactly that case's names minus the mention's own name; else inline.
#   - entity _proto is stored as {"_ref": mention_id} when it is the saved
#     mention itself; loaded back as that same mention object.
#   - decisions are kept on disk only (not in memory) once saved.
# ---------------------------------------------------------------------------
CK_FORMAT = 2
CO_REF_MIN = 50
_STREAMS = ("processed", "timings", "case_names", "mentions_judge", "mentions_firm", "mentions_party",
            "decisions", "llm_prompt_cache")


def _read_stream(path: Path, nbytes: int | None) -> list[dict]:
    if not path.is_file():
        return []
    if nbytes is not None and path.stat().st_size > nbytes:
        with path.open("r+b") as f:
            f.truncate(nbytes)
    with path.open("rb") as f:
        return [json.loads(l) for l in f if l.strip()]


def _append_stream(path: Path, offset: int, rows) -> int:
    """Append rows at the committed offset (dropping any cut-off tail); return new size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r+b" if path.exists() else "wb") as f:
        f.seek(offset)
        f.truncate()
        for r in rows:
            f.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
        return f.tell()


def checkpoint_rows(out_root: Path, name: str):
    """Iterate the committed rows of one append stream (used for final decisions.jsonl)."""
    ck = out_root / "checkpoint"
    meta = json.loads((ck / "state.json").read_text(encoding="utf-8"))
    nbytes = (meta.get("bytes") or {}).get(name)
    path = ck / f"{name}.jsonl"
    if not path.is_file():
        return
    with path.open("rb") as f:
        read = 0
        for line in f:
            read += len(line)
            if nbytes is not None and read > nbytes:
                break
            if line.strip():
                yield json.loads(line)


def _is_case_list(co: list, name: str, universe: set) -> bool:
    if len(co) != len(universe) - (1 if name in universe else 0) or name in co:
        return False
    return all(a < b for a, b in zip(co, co[1:])) and all(x in universe for x in co)


def _mention_rows(ms: list[dict], et: str, ck: dict, case_rows: list[dict]):
    universe: dict[str, set] = {}
    for m in ms:
        co = m.get("co_mentions")
        if isinstance(co, list) and len(co) >= CO_REF_MIN:
            u = universe.setdefault(m.get("ucid") or "", set())
            u.update(co)
            u.add(m.get("normalized_name"))
    refs: dict[str, int] = {}
    for m in ms:
        co = m.get("co_mentions")
        ucid = m.get("ucid") or ""
        if isinstance(co, list) and len(co) >= CO_REF_MIN and _is_case_list(co, m.get("normalized_name"), universe[ucid]):
            if ucid not in refs:
                ck["case_seq"] = ck.get("case_seq", 0) + 1
                refs[ucid] = ck["case_seq"]
                case_rows.append({"id": refs[ucid], "type": et, "ucid": ucid, "names": sorted(universe[ucid])})
            row = dict(m)
            row["co_mentions"] = {"_ref": refs[ucid]}
            yield row
        else:
            yield m


def _entity_rows(entities: list[dict], by_id: dict[str, dict]):
    for e in entities:
        proto = e.get("_proto")
        mid = proto.get("mention_id") if isinstance(proto, dict) else None
        if mid and (by_id.get(mid) is proto or by_id.get(mid) == proto):
            row = dict(e)
            row["_proto"] = {"_ref": mid}
            yield row
        else:
            yield e


def load_checkpoint(out_root: Path) -> dict[str, Any] | None:
    state_path = out_root / "checkpoint" / "state.json"
    if not state_path.is_file():
        return None
    meta = json.loads(state_path.read_text(encoding="utf-8"))
    if meta.get("format") != CK_FORMAT:
        return _load_checkpoint_v1(out_root, meta)
    ck = out_root / "checkpoint"
    nbytes = meta.get("bytes") or {}
    rd = {name: _read_stream(ck / f"{name}.jsonl", nbytes.get(name, 0)) for name in _STREAMS if name != "llm_prompt_cache"}
    # The LLM cache stream is read by load_llm_memo; drop a cut-off tail here.
    _read_stream(ck / "llm_prompt_cache.jsonl", nbytes.get("llm_prompt_cache", 0))
    cases = {r["id"]: r["names"] for r in rd["case_names"]}
    state: dict[str, Any] = {
        "processed": rd["processed"],
        "timings": rd["timings"],
        "next_serial": meta["next_serial"],
        "poc_evidence": meta.get("poc_evidence") or [],
        "cascade_last": meta.get("cascade_last") or {},
        "decisions": [],
        "mentions": {},
        "entities": {},
    }
    gen = int(meta.get("gen") or 0)
    for et in ("judge", "firm", "party"):
        ms = rd[f"mentions_{et}"]
        for m in ms:
            co = m.get("co_mentions")
            if isinstance(co, dict) and "_ref" in co:
                own = m.get("normalized_name")
                m["co_mentions"] = [x for x in cases[co["_ref"]] if x != own]
        state["mentions"][et] = ms
        by_id = {m["mention_id"]: m for m in ms}
        ents = _read_stream(ck / f"entities_{et}.g{gen:06d}.jsonl", None)
        for e in ents:
            p = e.get("_proto")
            if isinstance(p, dict) and "_ref" in p:
                e["_proto"] = by_id[p["_ref"]]
        state["entities"][et] = ents
    jc = ck / f"judge_confirm.g{gen:06d}.json"
    if jc.is_file():
        state["judge_confirm"] = json.loads(jc.read_text(encoding="utf-8"))
    for old in list(ck.glob("entities_*.g*.jsonl")) + list(ck.glob("judge_confirm.g*.json")):
        if f".g{gen:06d}." not in old.name:
            old.unlink()
    state["_ck"] = {
        "bytes": dict(nbytes),
        "rows": {
            "processed": len(state["processed"]),
            "timings": len(state["timings"]),
            **{f"mentions_{et}": len(state["mentions"][et]) for et in ("judge", "firm", "party")},
            "llm_prompt_cache": int((meta.get("rows") or {}).get("llm_prompt_cache") or 0),
        },
        "gen": gen,
        "case_seq": int(meta.get("case_seq") or 0),
    }
    return state


def _load_checkpoint_v1(out_root: Path, state: dict) -> dict[str, Any]:
    """Format-1 checkpoint (every file rewritten per save). The next save writes format 2."""
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
    jc = ck / "judge_confirm.json"
    if jc.is_file():
        state["judge_confirm"] = json.loads(jc.read_text(encoding="utf-8"))
    return state


def save_checkpoint(out_root: Path, state: dict[str, Any]) -> None:
    """Append what is new since the last save, rewrite entities, then commit state.json."""
    from engine.tiers import _LLM_MEMO

    ck = out_root / "checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    c = state.setdefault("_ck", {"bytes": {}, "rows": {}, "gen": 0, "case_seq": 0})
    first = not c["bytes"]  # fresh run or a format-1 checkpoint: write every stream from 0
    nb, nr = c["bytes"], c["rows"]

    def put(name: str, rows) -> None:
        nb[name] = _append_stream(ck / f"{name}.jsonl", 0 if first else nb.get(name, 0), rows)

    put("processed", state["processed"][nr.get("processed", 0):])
    put("timings", state["timings"][nr.get("timings", 0):])
    nr["processed"], nr["timings"] = len(state["processed"]), len(state["timings"])
    case_rows: list[dict] = []
    for et in ("judge", "firm", "party"):
        key = f"mentions_{et}"
        put(key, _mention_rows(state["mentions"][et][nr.get(key, 0):], et, c, case_rows))
        nr[key] = len(state["mentions"][et])
    put("case_names", case_rows)
    put("decisions", state["decisions"])
    state["decisions"].clear()  # on disk now; the final decisions.jsonl streams from there
    memo = list(_LLM_MEMO.items())
    put("llm_prompt_cache", ({"key": k, "response": v} for k, v in memo[nr.get("llm_prompt_cache", 0):]))
    nr["llm_prompt_cache"] = len(memo)

    gen = int(c.get("gen") or 0) + 1
    for et in ("judge", "firm", "party"):
        by_id = {m["mention_id"]: m for m in state["mentions"][et]}
        _write_rows(ck / f"entities_{et}.g{gen:06d}.jsonl", _entity_rows(state["entities"][et], by_id))
    if state.get("judge_confirm"):
        _atomic_write(ck / f"judge_confirm.g{gen:06d}.json", json.dumps(state["judge_confirm"], ensure_ascii=False))
    shards = state["embed_cache"].save() if state.get("embed_cache") is not None else 0

    meta = {
        "format": CK_FORMAT,
        "processed_count": len(state["processed"]),
        "next_serial": state["next_serial"],
        "poc_evidence": state.get("poc_evidence") or [],
        "cascade_last": state.get("cascade_last") or {},
        "gen": gen,
        "embed_shards": shards,
        "case_seq": c.get("case_seq", 0),
        "bytes": nb,
        "rows": {"llm_prompt_cache": nr["llm_prompt_cache"]},
    }
    _atomic_write(ck / "state.json", json.dumps(meta, indent=2, default=str))  # commit point
    c["gen"] = gen
    for old in list(ck.glob("entities_*.jsonl")) + list(ck.glob("judge_confirm*.json")):
        if f".g{gen:06d}." not in old.name and old.name.startswith(("entities_", "judge_confirm")):
            old.unlink()


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
        # Per-entity co-party slots, built once and updated on absorb.
        self.slot_cache: dict[int, set] = {}
        # Per-entity legal-form families (parties), built once and updated on absorb.
        self.form_cache: dict[int, set] = {}

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

    def note_absorb(self, entity: dict, members: list[dict] | None = None, cfg: dict | None = None) -> None:
        """Index FJC ids and co-party slots a saved entity gained by absorbing a new one."""
        i = self.pos.get(id(entity))
        if i is None:
            return
        for nid in entity.get("fjc_nids") or []:
            self.by_nid[str(nid)].add(i)
        if i in self.slot_cache and members and cfg is not None:
            for m in members:
                slot = party_slot(m, cfg)
                if slot:
                    self.slot_cache[i].add(slot)
        if i in self.form_cache and members and cfg is not None:
            for m in members:
                self.form_cache[i] |= legal_entity_form_set(m, cfg)

    def entity_forms(self, entity: dict, by_id: dict, cfg: dict) -> set:
        i = self.pos.get(id(entity))
        if i is not None and i in self.form_cache:
            return self.form_cache[i]
        forms: set = set()
        for mid in entity.get("mention_ids") or []:
            m = self.mentions.get(mid) or by_id.get(mid)
            if m:
                forms |= legal_entity_form_set(m, cfg)
        if i is not None:
            self.form_cache[i] = forms
        return forms

    def entity_slots(self, entity: dict, by_id: dict, cfg: dict) -> set:
        i = self.pos.get(id(entity))
        if i is not None and i in self.slot_cache:
            return self.slot_cache[i]
        slots = set()
        for mid in entity.get("mention_ids") or []:
            m = self.mentions.get(mid) or by_id.get(mid)
            slot = party_slot(m, cfg) if m else None
            if slot:
                slots.add(slot)
        if i is not None:
            self.slot_cache[i] = slots
        return slots

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
    slots_e = index.entity_slots(entity, by_id, cfg)
    slots_n = {s for s in (party_slot(m, cfg) for m in members) if s}
    return coparty_conflict(slots_e, slots_n) is None


class _ScratchJournal:
    """Collects the comparison rows made while linking one new entity.

    Only the outcome (and any LLM decision) is written to the link log, so the
    log grows by one row per decision instead of one row per comparison.
    """

    def __init__(self) -> None:
        self.n = 0
        self.rows: list[dict] = []

    def log(self, record: dict) -> None:
        self.n += 1
        record.setdefault("decision_id", f"cmp_{self.n:08d}")
        self.rows.append(record)


def _log_link(
    journal: DecisionJournal,
    *,
    decision: str,
    method: str,
    rationale: str,
    members: list[dict],
    chosen: dict | None,
    n_candidates: int,
    scratch: _ScratchJournal | None,
    cfg: dict,
    signals: list[str],
) -> None:
    compared: dict[str, int] = defaultdict(int)
    if scratch is not None:
        for r in scratch.rows:
            if str(r.get("method") or "").startswith("tier3.ollama"):
                journal.log(dict(r))  # LLM decisions are kept as they are
            compared[f"{r.get('method')}:{r.get('decision')}"] += 1
    journal.log(
        {
            "decision": decision,
            "method": method,
            "mention_id_a": members[0]["mention_id"],
            "mention_id_b": (chosen.get("_proto") or {}).get("mention_id") if chosen else None,
            "entity_type": cfg.get("entity_type"),
            "confidence": 100,
            "rationale": rationale,
            "signals": signals,
            "evidence": {
                "linked_sjid": chosen.get("sjid") if chosen else None,
                "n_candidates": n_candidates,
                "comparisons": dict(compared),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )


def _forms_ok(entity: dict, members: list[dict], index: SavedIndex, by_id: dict, cfg: dict) -> bool:
    """False when the saved entity and the new one carry legal-form families that
    do not overlap at all (AG vs Corp, LLC vs PLC). Only where a Tier0 rule opts in."""
    if not any((r.get("match") or {}).get("forbid_legal_form_conflict") for r in (cfg.get("tier0") or {}).get("rules") or []):
        return True
    fe = index.entity_forms(entity, by_id, cfg)
    fn: set = set()
    for m in members:
        fn |= legal_entity_form_set(m, cfg)
    return not (fe and fn and not (fe & fn))


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
                index.note_absorb(chosen, members, cfg)
                stats["fjc_links"] += 1
                _log_link(
                    journal, decision="MERGE_TIER0", method="incremental.link_fjc_nid",
                    rationale=f"Same FJC id {sorted(nids & set(map(str, chosen.get('fjc_nids') or [])))} as saved {chosen.get('sjid')}",
                    members=members, chosen=chosen, n_candidates=len(nid_hits), scratch=None, cfg=cfg,
                    signals=["incremental", "fjc_nid_match"],
                )
                continue
        cands = index.candidates(saved, members, cfg)
        if nids:
            cands = [c for c in cands if not (c.get("fjc_nids") and not nids & {str(x) for x in c["fjc_nids"]})]
        if not cands:
            still_new.append(ent)
            stats["new"] += 1
            _log_link(
                journal, decision="NEW_ENTITY", method="incremental.new_entity",
                rationale="No saved entity shares a block, alias group or FJC id",
                members=members, chosen=None, n_candidates=0, scratch=None, cfg=cfg, signals=["incremental", "no_candidates"],
            )
            continue

        scratch = _ScratchJournal()
        pool = members + [c["_proto"] for c in cands if c.get("_proto")]
        uf = UnionFind()
        # Keep the within-file entity as one blob. Tier0 may then union it with
        # a saved prototype. Saved prototypes are not unioned with each other
        # in the registry even if this temporary UF joins them.
        for m in members:
            uf.add(m["mention_id"])
        for m in members[1:]:
            uf.union(members[0]["mention_id"], m["mention_id"])
        tier0_merge_groups(pool, cfg, scratch, uf)
        root_new = uf.find(members[0]["mention_id"])
        hits = []
        for c in cands:
            proto = c.get("_proto") or {}
            if proto.get("mention_id") and uf.find(proto["mention_id"]) == root_new:
                hits.append(c)
        hits = [h for h in hits if _coparty_ok(h, members, index, by_id, cfg) and _forms_ok(h, members, index, by_id, cfg)]
        if hits:
            chosen = _pick_hit(hits, ent)
            _absorb(chosen, ent, members)
            index.note_absorb(chosen, members, cfg)
            stats["tier0_links"] += 1
            _log_link(
                journal, decision="MERGE_TIER0", method="incremental.link_tier0",
                rationale=f"New file linked to saved {chosen.get('sjid')} by existing Tier0 rules",
                members=members, chosen=chosen, n_candidates=len(cands), scratch=scratch, cfg=cfg,
                signals=["incremental", "tier0"],
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
        t2 = apply_tier2_auto_merges(pairs, by, uf2, cfg, scratch, surnames)
        linked = None
        for c in cands:
            pid = (c.get("_proto") or {}).get("mention_id")
            if pid and uf2.find(proto_new["mention_id"]) == uf2.find(pid):
                linked = c
                break
        if linked is not None and not (_coparty_ok(linked, members, index, by_id, cfg) and _forms_ok(linked, members, index, by_id, cfg)):
            linked = None
            stats["coparty_blocked"] = stats.get("coparty_blocked", 0) + 1
        if linked is not None:
            _absorb(linked, ent, members)
            index.note_absorb(linked, members, cfg)
            stats["tier2_links"] += 1
            _log_link(
                journal, decision="MERGE_TIER2", method="incremental.link_tier2",
                rationale=f"New file linked to saved {linked.get('sjid')} by Tier2 embedding similarity",
                members=members, chosen=linked, n_candidates=len(cands), scratch=scratch, cfg=cfg,
                signals=["incremental", "tier2"],
            )
            continue
        ambiguous = [p for p in (t2.get("ambiguous") or []) if proto_new["mention_id"] in (p[0], p[1])]
        if ambiguous:
            blocks = build_profile_blocks(list(by.values()), cfg)
            t3 = tier3_adjudicate(
                ambiguous, by, uf2, cfg, scratch, blocks, total_mentions=len(by)
            )
            stats["tier3_calls"] += int(t3.get("llm_calls") or 0)
            for c in cands:
                pid = (c.get("_proto") or {}).get("mention_id")
                if pid and uf2.find(proto_new["mention_id"]) == uf2.find(pid):
                    linked = c
                    break
            if linked is not None and not (_coparty_ok(linked, members, index, by_id, cfg) and _forms_ok(linked, members, index, by_id, cfg)):
                linked = None
                stats["coparty_blocked"] = stats.get("coparty_blocked", 0) + 1
            if linked is not None:
                _absorb(linked, ent, members)
                index.note_absorb(linked, members, cfg)
                stats["tier3_links"] += 1
                _log_link(
                    journal, decision="MERGE_TIER3", method="incremental.link_tier3",
                    rationale=f"New file linked to saved {linked.get('sjid')} by Tier3 (LLM)",
                    members=members, chosen=linked, n_candidates=len(cands), scratch=scratch, cfg=cfg,
                    signals=["incremental", "tier3"],
                )
                continue
        still_new.append(ent)
        stats["new"] += 1
        _log_link(
            journal, decision="NEW_ENTITY", method="incremental.new_entity",
            rationale="No saved candidate matched (Tier0/Tier2/Tier3)",
            members=members, chosen=None, n_candidates=len(cands), scratch=scratch, cfg=cfg,
            signals=["incremental", "no_match"],
        )
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
    _t = time.perf_counter()
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
    STAGE_SEC["extract_finalize_and_name_validity"] += time.perf_counter() - _t
    if not finalized:
        return {"entities": [], "by_id": {}, "mentions": [], "summary": {}, "decisions": [], "poc": {}}
    _t = time.perf_counter()
    result = run_cascade(finalized, cfg, enable_tier3=True)
    STAGE_SEC["cascade_total"] += time.perf_counter() - _t
    _t = time.perf_counter()
    uf = result["uf"]
    by_id = result["by_id"]
    poc: dict[str, Any] = {}
    if cfg.get("entity_type") == "party":
        journal = DecisionJournal(
            work_dir / "decisions" / "poc_party_evidence_decisions.jsonl", fresh=True, buffered=True
        )
        poc_stats = adjudicate_poc_evidence_via_tier3(
            finalized, uf, cfg, journal=journal, max_pairs=25
        )
        journal.flush()
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
    STAGE_SEC["party_evidence_and_cluster"] += time.perf_counter() - _t
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
