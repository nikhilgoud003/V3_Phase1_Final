#!/usr/bin/env python3
"""Invariant checks against current V3 outputs. Report violations — do not auto-fix."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config, resolve_path
from engine.extract import _collect_party_counsel_names, _norm_set
from engine.normalize import normalize_name, tokens
from engine.tiers import information_barrier, load_common_surnames, transfer_conflict


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Invariant checks against a Phase C run.")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Run folder (sets TIER_V3_OUTPUT_DIR). Default: env or config data/ paths.",
    )
    args, _ = ap.parse_known_args()
    if args.run_dir:
        os.environ["TIER_V3_OUTPUT_DIR"] = str(Path(args.run_dir).resolve())

    cfg = load_config(ROOT / "configs/judges.yaml")
    entities = load_jsonl(resolve_path(cfg, cfg["io"]["clusters_out"]))
    decisions = load_jsonl(resolve_path(cfg, cfg["io"]["decisions_out"]))
    mentions = load_jsonl(resolve_path(cfg, cfg["io"]["mentions_out"]))
    by_id = {m["mention_id"]: m for m in mentions}
    honorifics = (cfg.get("normalization") or {}).get("strip_honorifics") or []
    strip_chars = (cfg.get("normalization") or {}).get("strip_chars") or ""

    if not entities or not mentions:
        print(
            json.dumps(
                {
                    "pass": False,
                    "error": "empty entities or mentions — wrong --run-dir?",
                    "run_dir": os.environ.get("TIER_V3_OUTPUT_DIR"),
                },
                indent=2,
            )
        )
        return 1

    violations: dict[str, list] = defaultdict(list)

    # --- 1. No multi-NID entities ---
    for e in entities:
        nids = sorted({n for n in (e.get("fjc_nids") or []) if n})
        if len(nids) > 1:
            violations["multi_nid_entity"].append(
                {
                    "entity_id": e.get("entity_id"),
                    "sjid": e.get("sjid"),
                    "normalized_name": e.get("normalized_name"),
                    "fjc_nids": nids,
                    "n_mentions": e.get("n_mentions"),
                }
            )

    # --- 2. Provenance completeness: merge graph connects multi-mention entities ---
    merge_decisions = {
        "MERGE_TIER0",
        "MERGE_TIER2",
        "MERGE_TIER3",
        "MERGE_UCID_ANCHOR",
    }
    merge_edges: dict[str, set[str]] = defaultdict(set)
    merge_records = 0
    for d in decisions:
        if d.get("decision") not in merge_decisions:
            continue
        a, b = d.get("mention_id_a"), d.get("mention_id_b")
        if not a or not b:
            violations["merge_missing_mention_ids"].append(d.get("decision_id"))
            continue
        merge_edges[a].add(b)
        merge_edges[b].add(a)
        merge_records += 1
        if a not in by_id or b not in by_id:
            violations["merge_unknown_mention_id"].append(
                {"decision_id": d.get("decision_id"), "a": a, "b": b}
            )

    def connected(members: list[str]) -> bool:
        if len(members) <= 1:
            return True
        start = members[0]
        seen = {start}
        stack = [start]
        member_set = set(members)
        while stack:
            cur = stack.pop()
            for nb in merge_edges.get(cur, ()):
                if nb in member_set and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen == member_set

    for e in entities:
        mids = e.get("mention_ids") or []
        if len(mids) <= 1:
            continue
        if not connected(mids):
            violations["provenance_incomplete_component"].append(
                {
                    "entity_id": e.get("entity_id"),
                    "sjid": e.get("sjid"),
                    "normalized_name": e.get("normalized_name"),
                    "n_mentions": len(mids),
                    "example_mentions": mids[:5],
                }
            )

    # --- 3. Deterministic cache replay (Tier3 rows vs cache + current rails) ---
    cache_path = resolve_path(cfg, (cfg.get("tier3") or {}).get("decision_cache", {}).get("path", "data/decisions/tier3_cache.jsonl"))
    cache = {}
    if cache_path.exists():
        for row in load_jsonl(cache_path):
            cache[row["cache_key"]] = row
    abstain_below = int((cfg.get("tier3") or {}).get("abstain_confidence_below", 60))
    barrier_min = int((cfg.get("tier3") or {}).get("barrier_llm_min_confidence", 80))
    common = load_common_surnames(
        resolve_path(cfg, "data/external/common_surnames.txt")
    )

    def expected_from_cache(d: dict) -> str | None:
        ck = d.get("cache_key")
        if not ck or ck not in cache:
            return None
        rec = cache[ck]
        decision = (rec.get("decision") or "UNCERTAIN").upper()
        conf = int(rec.get("confidence") or 0)
        ma, mb = by_id.get(d["mention_id_a"], {}), by_id.get(d["mention_id_b"], {})
        ba, _ = information_barrier(ma, cfg, common)
        bb, _ = information_barrier(mb, cfg, common)
        barrier = ba or bb
        shared_nid = bool(ma.get("fjc_nid") and ma.get("fjc_nid") == mb.get("fjc_nid"))
        if decision in {"MATCH", "NO_MATCH"} and conf < abstain_below:
            if not (decision == "MATCH" and barrier and not shared_nid):
                decision = "UNCERTAIN"
        if decision == "MATCH" and barrier and not shared_nid and conf < barrier_min:
            decision = "UNCERTAIN"
        if decision == "MATCH":
            return "MERGE_TIER3"
        if decision == "NO_MATCH":
            return "NO_MATCH"
        return "UNCERTAIN"

    replay_checked = 0
    for d in decisions:
        if d.get("decision") not in {"MERGE_TIER3", "NO_MATCH", "UNCERTAIN"}:
            continue
        if not str(d.get("method") or "").startswith("tier3"):
            continue
        exp = expected_from_cache(d)
        if exp is None:
            continue
        replay_checked += 1
        if exp != d.get("decision"):
            violations["cache_replay_mismatch"].append(
                {
                    "decision_id": d.get("decision_id"),
                    "journal": d.get("decision"),
                    "expected_from_cache_rails": exp,
                    "cache_key": d.get("cache_key"),
                    "confidence": d.get("confidence"),
                }
            )

    # --- 4. No negative-list (party/counsel) mentions inside entities ---
    # Build party/counsel negatives per UCID from pilot JSONs (sample all files once).
    json_dir = resolve_path(cfg, "data/json/pilot_1000")
    neg_by_ucid: dict[str, set[str]] = {}
    for fp in sorted(json_dir.glob("*.json")):
        try:
            case = json.loads(fp.read_text())
        except Exception:
            continue
        ucid = case.get("ucid") or ""
        raw_p, raw_c = _collect_party_counsel_names(case)
        neg_by_ucid[ucid] = _norm_set(list(raw_p) + list(raw_c), honorifics, strip_chars)

    for e in entities:
        for mid in e.get("mention_ids") or []:
            m = by_id.get(mid)
            if not m:
                continue
            # Only docket NER / line_entry can be party FP; header judges are intentional
            if m.get("docket_source") not in {"line_entry", None} and m.get("extraction_method") not in {
                "SPACY_EN_CORE_WEB_SM",
                "spacy",
            }:
                # still check line_entry and spacy
                pass
            if m.get("docket_source") != "line_entry":
                continue
            ucid = m.get("ucid") or ""
            neg = neg_by_ucid.get(ucid) or set()
            nn = m.get("normalized_name") or ""
            if nn and nn in neg:
                violations["negative_list_mention_in_entity"].append(
                    {
                        "entity_id": e.get("entity_id"),
                        "mention_id": mid,
                        "normalized_name": nn,
                        "ucid": ucid,
                        "docket_source": m.get("docket_source"),
                    }
                )

    # --- 5. Transfer-clue consistency ---
    for e in entities:
        ms = [by_id[m] for m in (e.get("mention_ids") or []) if m in by_id]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if transfer_conflict(ms[i], ms[j]):
                    violations["transfer_clue_conflict_in_entity"].append(
                        {
                            "entity_id": e.get("entity_id"),
                            "a": ms[i].get("normalized_name"),
                            "b": ms[j].get("normalized_name"),
                            "ucid_a": ms[i].get("ucid"),
                            "ucid_b": ms[j].get("ucid"),
                        }
                    )

    # --- 6. Same-UCID same-name cohesion ---
    # Mentions with identical (ucid, normalized_name) should share an entity.
    mid_to_ent = {}
    for e in entities:
        for mid in e.get("mention_ids") or []:
            mid_to_ent[mid] = e.get("entity_id")
    groups: dict[tuple, list[str]] = defaultdict(list)
    for m in mentions:
        key = (m.get("ucid"), m.get("normalized_name"))
        if key[0] and key[1]:
            groups[key].append(m["mention_id"])
    for key, mids in groups.items():
        if len(mids) < 2:
            continue
        ents = {mid_to_ent.get(mid) for mid in mids}
        if None in ents:
            ents.discard(None)
        if len(ents) > 1:
            violations["same_ucid_same_name_split"].append(
                {
                    "ucid": key[0],
                    "normalized_name": key[1],
                    "entity_ids": sorted(ents),
                    "mention_ids": mids[:8],
                }
            )

    # Cap examples per category
    report = {
        "n_entities": len(entities),
        "n_mentions": len(mentions),
        "n_decisions": len(decisions),
        "n_merge_records": merge_records,
        "cache_replay_checked": replay_checked,
        "violation_counts": {k: len(v) for k, v in sorted(violations.items())},
        "examples": {k: v[:10] for k, v in sorted(violations.items())},
        "pass": all(len(v) == 0 for v in violations.values()),
    }
    out = resolve_path(cfg, cfg["io"]["reports_dir"]) / "invariant_violations.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"violation_counts": report["violation_counts"], "pass": report["pass"],
                      "n_entities": report["n_entities"], "cache_replay_checked": replay_checked}, indent=2))
    print(f"Wrote {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
