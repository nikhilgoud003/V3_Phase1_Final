#!/usr/bin/env python3
"""Compare fresh research_dev JudgePipeline output vs Tier_V3 judges_clean_v2.

Produces an apples-to-apples scoped report (header+parties only) plus a clearly
labeled full-V3 coverage section (includes docket NER).

Artifacts default to:
  data/validation/research_dev_pilot1000/
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HONORIFICS = re.compile(
    r"(?i)^\s*(?:u\.?\s*s\.?\s+|us\s+|united\s+states\s+)?"
    r"(?:hon(?:orable)?\.?\s+|judge\s+|magistrate(?:\s+judge)?\s+|"
    r"district\s+judge\s+|chief\s+judge\s+|senior\s+judge\s+|sr\.?\s+|jr\.?\s+)*"
)
GEN_SUFFIX = re.compile(r"(?i)\b(?:jr|sr|ii|iii|iv|esq)\.?$")
NON_ALNUM = re.compile(r"[^a-z0-9\s]+")


def norm_name(s: str) -> str:
    t = (s or "").strip().lower()
    t = HONORIFICS.sub("", t)
    t = NON_ALNUM.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = GEN_SUFFIX.sub("", t).strip()
    return t


def soft_key(s: str) -> str:
    """Drop single-letter tokens (middle initials) for soft match."""
    toks = [t for t in norm_name(s).split() if len(t) > 1]
    return " ".join(toks)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def research_dev_unique_names(entities: list[dict]) -> dict[str, dict]:
    """One entry per research_dev judge (by SJID), keyed later by normalized name."""
    out: dict[str, dict] = {}
    for e in entities:
        sid = str(e.get("SJID") or e.get("sjid") or e.get("entity_id") or "")
        name = (
            e.get("Parent_Entity")
            or e.get("Extracted_Entity")
            or e.get("canonical_name")
            or e.get("normalized_name")
            or ""
        ).strip()
        nn = norm_name(e.get("normalized_name") or name)
        if not sid:
            sid = nn or f"anon_{len(out)}"
        out[sid] = {
            "sjid": sid,
            "raw_name": name or nn,
            "norm": nn,
            "soft": soft_key(name or nn),
            "n_mentions": int(e.get("n_mentions") or 0),
            "courts": list(e.get("courts") or ([e.get("court")] if e.get("court") else [])),
        }
    return out


def v3_entities_scoped(
    ents: list[dict],
    mentions: list[dict],
    *,
    header_sources: set[str],
) -> tuple[list[dict], list[dict], dict]:
    """Split V3 entities into header+parties-scoped vs line_entry-only."""
    by_id = {m["mention_id"]: m for m in mentions if m.get("mention_id")}
    scoped, line_only = [], []
    stats = {"header_party_mentions": 0, "line_entry_mentions": 0, "other_mentions": 0}
    for e in ents:
        mids = e.get("mention_ids") or []
        sources = set()
        for mid in mids:
            m = by_id.get(mid)
            if not m:
                continue
            src = (m.get("docket_source") or "").strip()
            sources.add(src)
            if src in header_sources:
                stats["header_party_mentions"] += 1
            elif src == "line_entry":
                stats["line_entry_mentions"] += 1
            else:
                stats["other_mentions"] += 1
        if sources & header_sources:
            scoped.append(e)
        elif sources == {"line_entry"} or (sources and not (sources & header_sources)):
            line_only.append(e)
        else:
            # no resolvable mentions — treat as out of scoped set
            line_only.append(e)
    return scoped, line_only, stats


def unique_name_index(rows: list[dict], *, name_field: str) -> dict[str, dict]:
    """Deduplicate by normalized name (keep first)."""
    out: dict[str, dict] = {}
    for r in rows:
        raw = (r.get(name_field) or r.get("normalized_name") or r.get("canonical_name") or "").strip()
        nn = norm_name(raw if name_field != "normalized_name" else (r.get("normalized_name") or raw))
        if not nn:
            continue
        if nn in out:
            continue
        out[nn] = {
            "id": r.get("entity_id") or r.get("sjid") or r.get("SJID"),
            "raw_name": r.get("canonical_name") or r.get("Parent_Entity") or raw,
            "norm": nn,
            "soft": soft_key(raw),
            "n_mentions": r.get("n_mentions"),
            "courts": r.get("courts") or [],
        }
    return out


def match_sets(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Name-level match A vs B using exact norm then soft key."""
    soft_b: dict[str, list[str]] = defaultdict(list)
    for nn, row in b.items():
        soft_b[row["soft"]].append(nn)

    both_exact, both_soft = [], []
    only_a, used_b = [], set()

    for nn, row in a.items():
        if nn in b:
            both_exact.append({"a": row, "b": b[nn], "how": "exact"})
            used_b.add(nn)
            continue
        soft = row["soft"]
        cands = [c for c in soft_b.get(soft, []) if c not in used_b]
        if soft and cands:
            c = cands[0]
            both_soft.append({"a": row, "b": b[c], "how": "soft"})
            used_b.add(c)
            continue
        only_a.append(row)

    only_b = [b[nn] for nn in b if nn not in used_b]
    return {
        "both_exact": both_exact,
        "both_soft": both_soft,
        "both": both_exact + both_soft,
        "only_a": only_a,
        "only_b": only_b,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--validation-dir",
        default="data/validation/research_dev_pilot1000",
        help="Folder with research_dev SEL/entities artifacts",
    )
    ap.add_argument(
        "--v3-run",
        default="data/runs/judges_clean_v2",
        help="Tier_V3 run dir",
    )
    args = ap.parse_args()

    val_dir = ROOT / args.validation_dir
    v3_run = ROOT / args.v3_run
    rd_ent_path = val_dir / "research_dev_entities_pilot1000.jsonl"
    rd_sel_path = val_dir / "research_dev_SEL_pilot1000.jsonl"
    v3_ent_path = v3_run / "clusters" / "judges_entities.jsonl"
    v3_men_path = v3_run / "mentions" / "judges_mentions.jsonl"

    rd_ents = load_jsonl(rd_ent_path)
    rd_sel = load_jsonl(rd_sel_path) if rd_sel_path.exists() else []
    v3_ents = load_jsonl(v3_ent_path)
    v3_mens = load_jsonl(v3_men_path)

    header_sources = {"case_header", "case_parties"}
    v3_scoped, v3_line_only, src_stats = v3_entities_scoped(
        v3_ents, v3_mens, header_sources=header_sources
    )

    # Unique judge names
    rd_by_sjid = research_dev_unique_names(rd_ents)
    # Dedup research_dev by normalized Parent_Entity (name-level)
    rd_by_name: dict[str, dict] = {}
    for row in rd_by_sjid.values():
        nn = row["norm"]
        if not nn:
            continue
        if nn not in rd_by_name or (row.get("n_mentions") or 0) > (rd_by_name[nn].get("n_mentions") or 0):
            rd_by_name[nn] = row

    v3_scoped_by_name = unique_name_index(v3_scoped, name_field="normalized_name")
    v3_full_by_name = unique_name_index(v3_ents, name_field="normalized_name")

    scoped_match = match_sets(rd_by_name, v3_scoped_by_name)
    # Also RD vs full V3 for reference
    full_match = match_sets(rd_by_name, v3_full_by_name)

    n_rd = len(rd_by_name)
    n_v3_scoped = len(v3_scoped_by_name)
    n_both = len(scoped_match["both"])
    n_rd_only = len(scoped_match["only_a"])
    n_v3_only = len(scoped_match["only_b"])
    n_v3_full = len(v3_full_by_name)
    n_line_only_ents = len(v3_line_only)
    n_v3_entity_rows = len(v3_ents)

    report = {
        "scope_note": (
            "Apples-to-apples: research_dev JudgePipeline extracts header + parties "
            "assigned judges only (no docket NER). V3 scoped set = entities with ≥1 "
            "mention from case_header or case_parties (line_entry-only entities excluded)."
        ),
        "inputs": {
            "research_dev_entities": str(rd_ent_path),
            "research_dev_sel": str(rd_sel_path),
            "v3_entities": str(v3_ent_path),
            "v3_mentions": str(v3_men_path),
        },
        "scoped_comparison_apples_to_apples": {
            "research_dev_unique_judges": n_rd,
            "research_dev_entity_rows": len(rd_ents),
            "v3_header_parties_unique_judges": n_v3_scoped,
            "v3_header_parties_entity_rows": len(v3_scoped),
            "overlap_both": n_both,
            "overlap_exact": len(scoped_match["both_exact"]),
            "overlap_soft": len(scoped_match["both_soft"]),
            "research_dev_only": n_rd_only,
            "v3_scoped_only": n_v3_only,
            "research_dev_coverage_in_v3_scoped_pct": round(100.0 * n_both / n_rd, 2) if n_rd else 0.0,
            "v3_scoped_coverage_in_research_dev_pct": round(100.0 * n_both / n_v3_scoped, 2) if n_v3_scoped else 0.0,
            "research_dev_only_examples": [
                {"name": r["raw_name"], "norm": r["norm"], "sjid": r.get("sjid"), "n_mentions": r.get("n_mentions")}
                for r in scoped_match["only_a"][:40]
            ],
            "v3_scoped_only_examples": [
                {"name": r["raw_name"], "norm": r["norm"], "entity_id": r.get("id"), "n_mentions": r.get("n_mentions")}
                for r in scoped_match["only_b"][:40]
            ],
        },
        "full_v3_coverage_not_apples_to_apples": {
            "label": "NOT apples-to-apples",
            "v3_full_entity_rows": n_v3_entity_rows,
            "v3_full_unique_normalized_names": n_v3_full,
            "v3_line_entry_only_entities": n_line_only_ents,
            "gap_entity_rows_vs_scoped_rows": n_v3_entity_rows - len(v3_scoped),
            "gap_unique_names_vs_scoped": n_v3_full - n_v3_scoped,
            "explanation": (
                f"Full judges_clean_v2 has {n_v3_entity_rows} entity rows "
                f"({n_v3_full} unique normalized names). "
                f"{n_line_only_ents} entities are line_entry-only (docket-text NER) — "
                "a source research_dev never looks at. That gap is coverage breadth, "
                "not necessarily higher accuracy on the shared header/parties scope."
            ),
            "research_dev_coverage_in_full_v3_pct": round(
                100.0 * len(full_match["both"]) / n_rd, 2
            )
            if n_rd
            else 0.0,
        },
        "mention_source_stats_v3": src_stats,
        "research_dev_sel_rows": len(rd_sel),
        "research_dev_entity_rows": len(rd_ents),
    }

    # Side-by-side markdown
    md = [
        "# Validation: research_dev vs judges_clean_v2",
        "",
        "## Scope difference (read this first)",
        "",
        "| Pipeline | Sources used |",
        "|----------|--------------|",
        "| **research_dev** `JudgePipeline` | Case header + parties assigned judges **only** |",
        "| **Tier_V3** `judges_clean_v2` (full) | Header + parties **plus** docket-line SpaCy NER |",
        "| **Tier_V3 scoped** (this comparison) | Entities with ≥1 header/parties mention — **excludes line_entry-only** |",
        "",
        "V3 “finding more” judges in the **full** file is expected: it looks at more fields.",
        "The **scoped** table below is the fair comparison.",
        "",
        "## 1. Scoped comparison (apples-to-apples)",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| research_dev unique judges | **{n_rd}** |",
        f"| V3 header+parties unique judges | **{n_v3_scoped}** (from {len(v3_scoped)} entity rows) |",
        f"| Found by **both** | **{n_both}** (exact {len(scoped_match['both_exact'])}, soft {len(scoped_match['both_soft'])}) |",
        f"| Found only by research_dev (V2) | **{n_rd_only}** |",
        f"| Found only by V3 (scoped) | **{n_v3_only}** |",
        f"| research_dev recovered in V3 scoped | **{report['scoped_comparison_apples_to_apples']['research_dev_coverage_in_v3_scoped_pct']}%** |",
        "",
        "### Side-by-side totals",
        "",
        "```",
        f"research_dev (header+parties):     {n_rd:>5} unique judges",
        f"V3 scoped (header+parties ents):   {n_v3_scoped:>5} unique judges",
        f"  ∩ both:                          {n_both:>5}",
        f"  ∪ research_dev only:             {n_rd_only:>5}",
        f"  ∪ V3 scoped only:                {n_v3_only:>5}",
        "```",
        "",
        "### research_dev-only examples (first 25)",
        "",
    ]
    for r in scoped_match["only_a"][:25]:
        md.append(f"- `{r['norm']}` ({r.get('sjid')}, n={r.get('n_mentions')})")
    if not scoped_match["only_a"]:
        md.append("- *(none)*")
    md += ["", "### V3-scoped-only examples (first 25)", ""]
    for r in scoped_match["only_b"][:25]:
        md.append(f"- `{r['norm']}` ({r.get('id')}, n={r.get('n_mentions')})")
    if not scoped_match["only_b"]:
        md.append("- *(none)*")

    md += [
        "",
        "## 2. Full V3 coverage (**not** apples-to-apples)",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| Full `judges_clean_v2` entity rows | **{n_v3_entity_rows}** |",
        f"| Full unique normalized names | **{n_v3_full}** |",
        f"| Line-entry-only entities (docket NER) | **{n_line_only_ents}** |",
        f"| Gap vs scoped (entity rows) | **{n_v3_entity_rows - len(v3_scoped)}** |",
        "",
        report["full_v3_coverage_not_apples_to_apples"]["explanation"],
        "",
        "```",
        f"research_dev:     {n_rd:>5}  ← header+parties only",
        f"V3 scoped:        {n_v3_scoped:>5}  ← same source scope (unique names)",
        f"V3 full rows:     {n_v3_entity_rows:>5}  ← includes docket-text judges V2 never looks for",
        f"V3 full unique:   {n_v3_full:>5}  ← unique normalized names in full file",
        "```",
        "",
        "## Artifacts",
        "",
        f"- research_dev SEL: `{rd_sel_path.relative_to(ROOT)}`",
        f"- research_dev entities: `{rd_ent_path.relative_to(ROOT)}`",
        f"- V3 entities: `{v3_ent_path.relative_to(ROOT)}`",
        f"- This report: `{val_dir.relative_to(ROOT)}/comparison_report.md`",
        "",
    ]

    out_json = val_dir / "comparison_report.json"
    out_md = val_dir / "comparison_report.md"
    # Also write name-level diff lists for future access
    diffs = {
        "both": [
            {"research_dev": m["a"]["norm"], "v3": m["b"]["norm"], "how": m["how"]}
            for m in scoped_match["both"]
        ],
        "research_dev_only": [r["norm"] for r in scoped_match["only_a"]],
        "v3_scoped_only": [r["norm"] for r in scoped_match["only_b"]],
    }
    (val_dir / "name_level_diff_scoped.json").write_text(
        json.dumps(diffs, indent=2), encoding="utf-8"
    )
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(out_md.read_text(encoding="utf-8"))
    print(f"\nWrote {out_md}")
    print(f"Wrote {out_json}")
    print(f"Wrote {val_dir / 'name_level_diff_scoped.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
