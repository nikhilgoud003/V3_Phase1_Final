#!/usr/bin/env python3
"""Evaluate V3 entity clusters against Path B gold SEL/JEL (pairwise + B³)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gold_clusters_from_sel(sel_rows: list[dict]) -> dict[str, set[str]]:
    """Map gold SJID → set of mention keys (ucid||normalized Extracted_Entity)."""
    clusters: dict[str, set[str]] = defaultdict(set)
    for r in sel_rows:
        sjid = r.get("SJID") or r.get("sjid")
        if not sjid:
            continue
        ucid = r.get("ucid") or ""
        name = (r.get("Parent_Entity") or r.get("Extracted_Entity") or "").strip().lower()
        name = " ".join(name.split())
        if not name:
            continue
        clusters[str(sjid)].add(f"{ucid}||{name}")
    return dict(clusters)


def pred_clusters_from_entities(ents: list[dict], mentions_by_id: dict[str, dict] | None = None) -> dict[str, set[str]]:
    """Map pred entity_id → set of mention keys (ucid||normalized_name)."""
    clusters: dict[str, set[str]] = {}
    for e in ents:
        keys: set[str] = set()
        mids = e.get("mention_ids") or []
        if mentions_by_id and mids:
            for mid in mids:
                m = mentions_by_id.get(mid)
                if not m:
                    continue
                nn = (m.get("normalized_name") or "").strip().lower()
                ucid = m.get("ucid") or ""
                if nn:
                    keys.add(f"{ucid}||{nn}")
        else:
            # fallback: entity-level name × ucids
            nn = (e.get("normalized_name") or "").strip().lower()
            for ucid in e.get("ucids") or []:
                if nn:
                    keys.add(f"{ucid}||{nn}")
            for v in e.get("name_variants") or []:
                vv = (v or "").strip().lower()
                for ucid in e.get("ucids") or []:
                    if vv:
                        keys.add(f"{ucid}||{vv}")
        if keys:
            clusters[e.get("entity_id") or e.get("sjid") or str(len(clusters))] = keys
    return clusters


def pairwise_sets(clusters: dict[str, set[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for members in clusters.values():
        ms = sorted(members)
        for a, b in combinations(ms, 2):
            pairs.add((a, b) if a < b else (b, a))
    return pairs


def b3_scores(gold: dict[str, set[str]], pred: dict[str, set[str]]) -> dict[str, float]:
    """B-cubed P/R/F1 over mention keys present in both."""
    # invert
    g_of: dict[str, str] = {}
    for cid, mems in gold.items():
        for m in mems:
            g_of[m] = cid
    p_of: dict[str, str] = {}
    for cid, mems in pred.items():
        for m in mems:
            p_of[m] = cid
    keys = sorted(set(g_of) & set(p_of))
    if not keys:
        return {"b3_precision": 0.0, "b3_recall": 0.0, "b3_f1": 0.0, "n_keys": 0}

    # precompute cluster member sets
    g_mem = gold
    p_mem = pred
    precs = []
    recs = []
    for m in keys:
        gset = g_mem[g_of[m]]
        pset = p_mem[p_of[m]]
        inter = len(gset & pset)
        precs.append(inter / max(1, len(pset)))
        recs.append(inter / max(1, len(gset)))
    p = sum(precs) / len(precs)
    r = sum(recs) / len(recs)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "b3_precision": round(p, 4),
        "b3_recall": round(r, 4),
        "b3_f1": round(f1, 4),
        "n_keys": len(keys),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier_V3 evaluation vs Path B gold")
    parser.add_argument("--jel", default="data/gold/JEL_pilot1000.jsonl")
    parser.add_argument("--sel", default="data/gold/SEL_pilot1000.jsonl")
    parser.add_argument("--pred", default="data/clusters/judges_entities.jsonl")
    parser.add_argument("--mentions", default="data/mentions/judges_mentions.jsonl")
    parser.add_argument("--label", default="current")
    parser.add_argument("--out", default=None, help="Write JSON report path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sel = load_jsonl(root / args.sel)
    jel = load_jsonl(root / args.jel)
    pred_ents = load_jsonl(root / args.pred)
    ments = load_jsonl(root / args.mentions)
    by_id = {m["mention_id"]: m for m in ments}

    if not sel:
        print("BLOCKER: gold SEL missing")
        return 2

    gold = gold_clusters_from_sel(sel)
    pred = pred_clusters_from_entities(pred_ents, by_id)

    gp = pairwise_sets(gold)
    pp = pairwise_sets(pred)
    tp = gp & pp
    fp = pp - gp
    fn = gp - pp
    precision = len(tp) / max(1, len(tp) + len(fp))
    recall = len(tp) / max(1, len(tp) + len(fn))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    b3 = b3_scores(gold, pred)

    report = {
        "label": args.label,
        "n_sel_rows": len(sel),
        "n_jel_rows": len(jel),
        "n_gold_entities": len(gold),
        "n_pred_entities": len(pred_ents),
        "n_pred_clusters_nonempty": len(pred),
        "pairwise": {
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "b3": b3,
        "note": "Pair keys = ucid||normalized_name; V2/V3 name normalization may differ → lower bound.",
    }
    print(json.dumps(report, indent=2))
    out = Path(args.out) if args.out else root / "data/reports" / f"eval_{args.label}.json"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
