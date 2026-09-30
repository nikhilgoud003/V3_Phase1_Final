#!/usr/bin/env python3
"""Run Tier_V3 judges pipeline on the pilot JSON set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.cluster import cluster_mentions
from engine.config_loader import default_json_dir, load_config, resolve_path
from engine.extract import extract_mentions
from engine.rdf_emit import emit_ttl
from engine.tiers import run_cascade


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier_V3 pilot runner")
    parser.add_argument("--config", default="configs/judges.yaml")
    parser.add_argument(
        "--json-dir",
        default=None,
        help="PACER JSON dir (default: $TIER_V3_JSON_DIR or data/json/pilot_1000)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-tier3", action="store_true", help="Skip LLM adjudication")
    parser.add_argument("--tier3-only-if", type=int, default=None, help="Only run tier3 if mention count <= N")
    parser.add_argument(
        "--from-mentions",
        action="store_true",
        help="Skip extraction; load mentions from config io.mentions_out (resume)",
    )
    parser.add_argument(
        "--target-registry",
        default=None,
        help="Target registry JSONL for REUSE/CREATE on RDF emit. "
        "Default for the live demo: data/runs/judges_pilot_recall_fix/clusters/judges_entity_registry.jsonl",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    if args.json_dir:
        json_dir = Path(args.json_dir)
        if not json_dir.is_absolute():
            json_dir = ROOT / json_dir
    else:
        json_dir = default_json_dir(cfg)

    print("=" * 60)
    print("Stage 1: Extraction")
    print("=" * 60)
    if cfg.get("_output_dir"):
        print(f"TIER_V3_OUTPUT_DIR={cfg['_output_dir']}")
    if args.from_mentions:
        mentions_path = resolve_path(cfg, cfg["io"]["mentions_out"])
        print(f"Loading mentions from {mentions_path}")
        if not mentions_path.exists():
            print(f"ERROR: mentions file missing: {mentions_path}")
            return 1
        mentions = [json.loads(l) for l in mentions_path.open(encoding="utf-8") if l.strip()]
        if args.limit:
            mentions = mentions[: args.limit]
    else:
        print(f"json_dir={json_dir}")
        mentions = extract_mentions(cfg["_config_path"], json_dir=str(json_dir), limit=args.limit, write=True)
    if not mentions:
        print("ERROR: no mentions extracted")
        return 1

    cr = [m for m in mentions if m.get("case_type") == "cr"]
    cv = [m for m in mentions if m.get("case_type") == "cv"]
    print(f"Mentions: {len(mentions)} (cr={len(cr)}, cv={len(cv)})")
    print(f"Unique courts: {len({m.get('court') for m in mentions})}")
    print(f"Unique normalized names: {len({m.get('normalized_name') for m in mentions})}")

    enable_tier3 = not args.skip_tier3
    if args.tier3_only_if is not None and len(mentions) > args.tier3_only_if:
        print(f"Tier3 skipped: mentions {len(mentions)} > {args.tier3_only_if}")
        enable_tier3 = False

    print("=" * 60)
    print("Stage 2: Cascade (Tier0 → Tier1 → Tier2 → Tier3)")
    print("=" * 60)
    if enable_tier3:
        print(f"Tier3 model={(cfg.get('tier3') or {}).get('model')} endpoint={(cfg.get('tier3') or {}).get('endpoint')}")
    result = run_cascade(mentions, cfg, enable_tier3=enable_tier3)
    summary = result["summary"]
    print(json.dumps(summary, indent=2))

    print("=" * 60)
    print("Stage 3: Clustering")
    print("=" * 60)
    entities = cluster_mentions(result["components"], result["by_id"], cfg, result["uf"])
    print(f"Entities: {len(entities)}")

    print("=" * 60)
    print("Stage 4: RDF emit")
    print("=" * 60)
    ttl = emit_ttl(
        entities,
        mentions,
        resolve_path(cfg, cfg["io"]["decisions_out"]),
        cfg,
        target_registry=args.target_registry,
    )
    print(f"Wrote {ttl}")

    reports = resolve_path(cfg, cfg["io"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    gold_jel = resolve_path(cfg, "data/gold/JEL.jsonl")
    with open(reports / "pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "n_entities": len(entities),
                "cr_mentions": len(cr),
                "cv_mentions": len(cv),
                "env": {
                    "TIER_V3_OUTPUT_DIR": os.environ.get("TIER_V3_OUTPUT_DIR"),
                    "TIER_V3_DATA_DIR": os.environ.get("TIER_V3_DATA_DIR"),
                    "TIER_V3_JSON_DIR": os.environ.get("TIER_V3_JSON_DIR"),
                    "OLLAMA_HOST": os.environ.get("OLLAMA_HOST"),
                    "TIER_V3_LLM_MODEL": os.environ.get("TIER_V3_LLM_MODEL"),
                },
                "blockers": {
                    "fjc_csv_missing": bool((summary.get("tier0") or {}).get("fjc_skipped")),
                    "gold_jel_sel_missing": not gold_jel.exists(),
                    "tier2_backend": (summary.get("tier2") or {}).get("backend")
                    or "see config",
                },
            },
            f,
            indent=2,
        )
    print(f"Summary → {reports / 'pilot_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
