#!/usr/bin/env python3
"""Phase E — align V3 entities to approved research_dev gold, then score.

Gold is header/parties-assigned judges (research_dev), not spaCy line NER.
Alignment is therefore by UCID + normalized name (with light variants).

FJC policy (locked): when a gold pair is split by V3 but both sides share the
same FJC NID under V3 Tier0 linking, do **not** count that as a V3 error —
treat V3 as more likely correct.

Usage:
  python3 scripts/phase_e_eval.py \\
    --run-dir data/runs/final_v3_14b_colab \\
    --sel data/gold/SEL_pilot1000.jsonl \\
    --jel data/gold/JEL_pilot1000.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config, resolve_path
from engine.fjc import link_mentions_to_fjc, load_fjc_index


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def norm_name(s: str | None) -> str:
    n = (s or "").strip().lower()
    n = re.sub(
        r"\b(honorable|hon\.?|judge|magistrate|chief|senior|district)\b",
        "",
        n,
        flags=re.I,
    )
    n = re.sub(r"[^\w\s\-\']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    # drop trailing generational for soft keying (kept in display)
    return n


_GEN = re.compile(r"\b(jr|sr|ii|iii|iv|2nd|3rd|4th)\b", re.I)
_US_PREFIX = re.compile(r"^(u\.?\s*s\.?\s*|us\s+)", re.I)


def align_norm(s: str | None) -> str:
    """Alignment-only normalize: drop generationals + leading US tags for matching."""
    n = norm_name(s)
    n = _US_PREFIX.sub("", n).strip()
    n = _GEN.sub("", n)
    n = re.sub(r"\s+", " ", n).strip()
    # apostrophe / hyphen soft forms
    n = n.replace("'", "").replace("'", "")
    return n


def align_exact_key(ucid: str, name: str) -> str:
    return f"{ucid}||{align_norm(name)}"


def soft_key(ucid: str, name: str) -> str:
    """ucid + surname + first-initial for near matches (J. Smith ~ John Smith)."""
    toks = align_norm(name).split()
    if not toks:
        return f"{ucid}||"
    sur = toks[-1]
    first = toks[0]
    init = first[0] if first else ""
    return f"{ucid}||{sur}||{init}"


def pairwise(clusters: dict[str, set[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for mems in clusters.values():
        ms = sorted(mems)
        for a, b in combinations(ms, 2):
            pairs.add((a, b) if a < b else (b, a))
    return pairs


def b3_scores(gold: dict[str, set[str]], pred: dict[str, set[str]]) -> dict[str, float]:
    # Mention → cluster id
    g_of: dict[str, str] = {}
    for cid, mems in gold.items():
        for m in mems:
            g_of[m] = cid
    p_of: dict[str, str] = {}
    for cid, mems in pred.items():
        for m in mems:
            p_of[m] = cid

    mentions = sorted(set(g_of) & set(p_of))
    if not mentions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_mentions": 0}

    prec_num = prec_den = 0.0
    rec_num = rec_den = 0.0
    # build full clusters restricted to shared mentions
    g_cl: dict[str, set[str]] = defaultdict(set)
    p_cl: dict[str, set[str]] = defaultdict(set)
    for m in mentions:
        g_cl[g_of[m]].add(m)
        p_cl[p_of[m]].add(m)

    for m in mentions:
        gset = g_cl[g_of[m]]
        pset = p_cl[p_of[m]]
        inter = len(gset & pset)
        prec_num += inter / len(pset)
        prec_den += 1
        rec_num += inter / len(gset)
        rec_den += 1

    p = prec_num / prec_den if prec_den else 0.0
    r = rec_num / rec_den if rec_den else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "n_mentions": len(mentions),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="data/runs/final_v3_14b_colab")
    ap.add_argument("--sel", default="data/gold/SEL_pilot1000.jsonl")
    ap.add_argument("--jel", default="data/gold/JEL_pilot1000.jsonl")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    sel_path = Path(args.sel)
    if not sel_path.is_absolute():
        sel_path = ROOT / sel_path
    jel_path = Path(args.jel)
    if not jel_path.is_absolute():
        jel_path = ROOT / jel_path

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "reports" / "phase_e"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = load_jsonl(sel_path)
    jel = load_jsonl(jel_path)
    ents = load_jsonl(run_dir / "clusters" / "judges_entities.jsonl")
    ments = load_jsonl(run_dir / "mentions" / "judges_mentions.jsonl")
    by_id = {m["mention_id"]: m for m in ments}

    # Re-link FJC onto V3 mentions for the disagreement policy
    cfg = load_config(ROOT / "configs/judges.yaml")
    nrm = cfg.get("normalization") or {}
    fjc_path = resolve_path(cfg, "data/judges_fjc.csv")
    crosswalk = resolve_path(cfg, "data/external/fjc_court_crosswalk.json")
    fjc_index = load_fjc_index(
        fjc_path, crosswalk, nrm.get("strip_honorifics") or [], nrm.get("strip_chars") or ""
    )
    link_mentions_to_fjc(ments, fjc_index)
    by_id = {m["mention_id"]: m for m in ments}

    # --- Gold keys ---
    gold_rows = []
    for r in sel:
        ucid = r.get("ucid") or ""
        raw = r.get("Extracted_Entity") or ""
        parent = r.get("Parent_Entity") or ""
        nn = norm_name(parent or raw)
        if not ucid or not nn:
            continue
        key = f"{ucid}||{nn}"
        gold_rows.append(
            {
                "key": key,
                "align": align_exact_key(ucid, nn),
                "soft": soft_key(ucid, nn),
                "ucid": ucid,
                "name": nn,
                "raw": raw,
                "SJID": r.get("SJID"),
                "source": r.get("_extracted_from"),
            }
        )

    # --- V3 keys from mentions ---
    mid_to_ent = {}
    for e in ents:
        for mid in e.get("mention_ids") or []:
            mid_to_ent[mid] = e.get("entity_id")

    v3_rows = []
    for m in ments:
        ucid = m.get("ucid") or ""
        nn = norm_name(m.get("normalized_name"))
        if not ucid or not nn:
            continue
        key = f"{ucid}||{nn}"
        v3_rows.append(
            {
                "key": key,
                "align": align_exact_key(ucid, nn),
                "soft": soft_key(ucid, nn),
                "ucid": ucid,
                "name": nn,
                "mention_id": m.get("mention_id"),
                "entity_id": mid_to_ent.get(m.get("mention_id")),
                "fjc_nid": m.get("fjc_nid"),
                "docket_source": m.get("docket_source"),
            }
        )

    gold_by_exact = defaultdict(list)
    gold_by_align = defaultdict(list)
    gold_by_soft = defaultdict(list)
    for g in gold_rows:
        gold_by_exact[g["key"]].append(g)
        gold_by_align[g["align"]].append(g)
        gold_by_soft[g["soft"]].append(g)

    v3_by_exact = defaultdict(list)
    v3_by_align = defaultdict(list)
    v3_by_soft = defaultdict(list)
    for v in v3_rows:
        v3_by_exact[v["key"]].append(v)
        v3_by_align[v["align"]].append(v)
        v3_by_soft[v["soft"]].append(v)

    # Align each gold row to a V3 key
    aligned = []  # gold_key, v3_key, how
    gold_unmatched = []
    used_v3_keys = set()
    for g in gold_rows:
        if g["key"] in v3_by_exact:
            vk = g["key"]
            aligned.append({"gold_key": g["key"], "v3_key": vk, "how": "exact", "SJID": g["SJID"]})
            used_v3_keys.add(vk)
        elif g["align"] in v3_by_align:
            cand = next(
                (x["key"] for x in v3_by_align[g["align"]] if x["key"] not in used_v3_keys),
                v3_by_align[g["align"]][0]["key"],
            )
            aligned.append({"gold_key": g["key"], "v3_key": cand, "how": "align_norm", "SJID": g["SJID"]})
            used_v3_keys.add(cand)
        elif g["soft"] in v3_by_soft:
            cand = next(
                (x["key"] for x in v3_by_soft[g["soft"]] if x["key"] not in used_v3_keys),
                v3_by_soft[g["soft"]][0]["key"],
            )
            aligned.append({"gold_key": g["key"], "v3_key": cand, "how": "soft", "SJID": g["SJID"]})
            used_v3_keys.add(cand)
        else:
            gold_unmatched.append(g)

    # Map gold_key → aligned canonical key (use v3_key as shared universe key)
    g2u = {a["gold_key"]: a["v3_key"] for a in aligned}
    # Also identity for exact
    shared_keys = sorted(set(g2u.values()))

    alignment = {
        "n_gold_rows": len(gold_rows),
        "n_v3_mention_keys": len({v["key"] for v in v3_rows}),
        "n_aligned": len(aligned),
        "n_aligned_exact": sum(1 for a in aligned if a["how"] == "exact"),
        "n_aligned_align_norm": sum(1 for a in aligned if a["how"] == "align_norm"),
        "n_aligned_soft": sum(1 for a in aligned if a["how"] == "soft"),
        "n_gold_unmatched": len(gold_unmatched),
        "gold_coverage_pct": round(100.0 * len(aligned) / len(gold_rows), 2) if gold_rows else 0.0,
        "align_how": dict(Counter(a["how"] for a in aligned)),
        "unmatched_by_source": dict(
            Counter(g.get("source") or "?" for g in gold_unmatched)
        ),
        "unmatched_examples": [
            {"ucid": g["ucid"], "name": g["name"], "raw": g["raw"], "SJID": g["SJID"], "source": g["source"]}
            for g in gold_unmatched[:40]
        ],
    }

    # Build gold / pred clusters on shared universe keys only
    # gold: SJID → set of aligned v3_keys
    gold_cl: dict[str, set[str]] = defaultdict(set)
    for a in aligned:
        gold_cl[str(a["SJID"])].add(a["v3_key"])

    # pred: entity_id → set of keys that appear in aligned universe
    key_to_ent: dict[str, str] = {}
    for v in v3_rows:
        if v["key"] in used_v3_keys and v.get("entity_id"):
            # if multiple mentions share key, keep first
            key_to_ent.setdefault(v["key"], v["entity_id"])

    pred_cl: dict[str, set[str]] = defaultdict(set)
    for k in shared_keys:
        eid = key_to_ent.get(k)
        if eid:
            pred_cl[eid].add(k)

    gp = pairwise(gold_cl)
    pp = pairwise(pred_cl)
    tp = gp & pp
    fp = pp - gp
    fn = gp - pp

    # FJC policy: excuse FP pairs where both keys share an FJC NID in V3
    # (V3 FJC-anchored merge vs gold split → trust V3, not a V3 error).
    key_to_nids: dict[str, set[str]] = defaultdict(set)
    for v in v3_rows:
        if v.get("fjc_nid"):
            key_to_nids[v["key"]].add(str(v["fjc_nid"]))
    # Also use entity-level fjc_nids when mention-level link is empty
    for e in ents:
        nids = [str(x) for x in (e.get("fjc_nids") or []) if x]
        if not nids:
            continue
        for mid in e.get("mention_ids") or []:
            m = by_id.get(mid) or {}
            nn = norm_name(m.get("normalized_name"))
            ucid = m.get("ucid") or ""
            if ucid and nn:
                key_to_nids[f"{ucid}||{nn}"].update(nids)

    fp_fjc_excused = []
    fp_counted = []
    for a, b in fp:
        na, nb = key_to_nids.get(a, set()), key_to_nids.get(b, set())
        shared_nid = na & nb
        if shared_nid:
            fp_fjc_excused.append({"a": a, "b": b, "shared_fjc_nid": sorted(shared_nid)})
        else:
            fp_counted.append((a, b))

    fp_adj = set(fp_counted)
    prec_raw = len(tp) / len(pp) if pp else 0.0
    prec_adj = len(tp) / (len(tp) + len(fp_adj)) if (len(tp) + len(fp_adj)) else 0.0
    rec = len(tp) / len(gp) if gp else 0.0
    f1_raw = (2 * prec_raw * rec / (prec_raw + rec)) if (prec_raw + rec) else 0.0
    f1_adj = (2 * prec_adj * rec / (prec_adj + rec)) if (prec_adj + rec) else 0.0

    b3 = b3_scores(gold_cl, pred_cl)

    pairwise_report = {
        "n_gold_pairs": len(gp),
        "n_pred_pairs": len(pp),
        "tp": len(tp),
        "fp_raw": len(fp),
        "fp_fjc_excused": len(fp_fjc_excused),
        "fp_counted_after_fjc_policy": len(fp_adj),
        "fn": len(fn),
        "precision_raw": round(prec_raw, 4),
        "precision_fjc_adjusted": round(prec_adj, 4),
        "recall": round(rec, 4),
        "f1_raw": round(f1_raw, 4),
        "f1_fjc_adjusted": round(f1_adj, 4),
        "fjc_excused_examples": fp_fjc_excused[:20],
        "fn_examples": [{"a": a, "b": b} for a, b in list(fn)[:20]],
        "fp_examples": [{"a": a, "b": b} for a, b in list(fp_adj)[:20]],
    }

    summary = {
        "run_dir": str(run_dir),
        "gold_sel": str(sel_path),
        "gold_jel_rows": len(jel),
        "gold_approved": True,
        "gold_source": "research_dev header/parties (Bug-1 fix)",
        "alignment": alignment,
        "pairwise": pairwise_report,
        "b3_shared_universe": b3,
        "policy": {
            "fjc_vs_gold": (
                "FP pairs where both mentions share a V3 FJC NID are excused "
                "(V3 FJC-anchored merge treated as more likely correct than gold split)."
            )
        },
    }

    (out_dir / "phase_e_alignment.json").write_text(json.dumps(alignment, indent=2))
    (out_dir / "phase_e_eval.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# Phase E — alignment + eval (approved research_dev gold)",
        "",
        f"**Run:** `{run_dir}`  ",
        f"**Gold:** SEL {len(sel)} / JEL {len(jel)} (research_dev header/parties, Bug-1 fix)  ",
        "",
        "## Alignment",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Gold rows | {alignment['n_gold_rows']} |",
        f"| Aligned to V3 | {alignment['n_aligned']} ({alignment['gold_coverage_pct']}%) |",
        f"| Exact / soft | {alignment['n_aligned_exact']} / {alignment['n_aligned_soft']} |",
        f"| Gold unmatched | {alignment['n_gold_unmatched']} |",
        "",
        "## Pairwise (aligned universe only)",
        "",
        f"| Metric | Raw | FJC-adjusted |",
        f"|--------|----:|-------------:|",
        f"| Precision | {pairwise_report['precision_raw']} | **{pairwise_report['precision_fjc_adjusted']}** |",
        f"| Recall | {pairwise_report['recall']} | {pairwise_report['recall']} |",
        f"| F1 | {pairwise_report['f1_raw']} | **{pairwise_report['f1_fjc_adjusted']}** |",
        f"| TP / FP / FN | {pairwise_report['tp']} / {pairwise_report['fp_raw']} / {pairwise_report['fn']} | FP excused by FJC: {pairwise_report['fp_fjc_excused']} |",
        "",
        f"## B³ (shared mentions)",
        "",
        f"- P={b3['precision']} R={b3['recall']} F1={b3['f1']} (n={b3['n_mentions']})",
        "",
        "## Policy",
        "",
        summary["policy"]["fjc_vs_gold"],
        "",
        "## Notes",
        "",
        "- Gold is header/parties case-assigned judges only — not full docket-line coverage.",
        "- FN pairs are mostly same-name cross-UCID under-merges in V3 (gold SJID merges across cases).",
        "- Unmatched gold often has residual name noise (`US …`, `(Settlement`, Jr/II suffixes) or V3 extraction gaps.",
        "",
    ]
    (out_dir / "phase_e_eval.md").write_text("\n".join(md))

    print(json.dumps({
        "alignment_coverage_pct": alignment["gold_coverage_pct"],
        "aligned": alignment["n_aligned"],
        "unmatched_gold": alignment["n_gold_unmatched"],
        "pairwise_precision_raw": pairwise_report["precision_raw"],
        "pairwise_precision_fjc_adjusted": pairwise_report["precision_fjc_adjusted"],
        "pairwise_recall": pairwise_report["recall"],
        "pairwise_f1_fjc_adjusted": pairwise_report["f1_fjc_adjusted"],
        "b3_f1": b3["f1"],
        "fp_fjc_excused": pairwise_report["fp_fjc_excused"],
        "fn": pairwise_report["fn"],
        "out_dir": str(out_dir),
    }, indent=2))
    print("Wrote", out_dir / "phase_e_eval.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
