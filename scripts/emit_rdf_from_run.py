#!/usr/bin/env python3
"""Emit Turtle RDF from a unified PoC run folder (entities/mentions/decisions.jsonl)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config  # noqa: E402
from engine.rdf_emit import emit_ttl  # noqa: E402

TYPE_CFG = {
    "judge": ROOT / "configs/judges.yaml",
    "firm": ROOT / "configs/firms.yaml",
    "party": ROOT / "configs/parties.yaml",
}
OUT_NAME = {"judge": "judges.ttl", "firm": "firms.ttl", "party": "parties.ttl"}


def _etype(row: dict) -> str:
    return (row.get("type") or row.get("entity_type") or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit RDF TTL from unified run outputs")
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Run folder with entities.jsonl, mentions.jsonl, decisions.jsonl",
    )
    ap.add_argument(
        "--combined-name",
        default="entities.ttl",
        help="Combined TTL filename under <run-dir>/rdf/ (default: entities.ttl)",
    )
    args = ap.parse_args()

    run = Path(args.run_dir)
    if not run.is_absolute():
        run = ROOT / run
    for name in ("entities.jsonl", "mentions.jsonl", "decisions.jsonl"):
        if not (run / name).is_file():
            print(f"ERROR: missing {run / name}", file=sys.stderr)
            return 1

    rdf_dir = run / "rdf"
    rdf_dir.mkdir(parents=True, exist_ok=True)

    entities = [json.loads(l) for l in (run / "entities.jsonl").open(encoding="utf-8") if l.strip()]
    mentions = [json.loads(l) for l in (run / "mentions.jsonl").open(encoding="utf-8") if l.strip()]
    decisions = [json.loads(l) for l in (run / "decisions.jsonl").open(encoding="utf-8") if l.strip()]

    paths: list[Path] = []
    for t, cfg_path in TYPE_CFG.items():
        cfg = load_config(cfg_path)
        ents = [e for e in entities if _etype(e) == t]
        mens = [m for m in mentions if _etype(m) == t]
        mid_set = {m["mention_id"] for m in mens}
        decs = [
            d
            for d in decisions
            if (_etype(d) == t)
            or (
                not _etype(d)
                and (d.get("mention_id_a") in mid_set or d.get("mention_id_b") in mid_set)
            )
        ]
        dec_path = rdf_dir / f"_{t}_decisions_for_emit.jsonl"
        with dec_path.open("w", encoding="utf-8") as f:
            for d in decs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        out = rdf_dir / OUT_NAME[t]
        ttl = emit_ttl(ents, mens, dec_path, cfg, out_path=str(out))
        print(f"{t}: entities={len(ents)} mentions={len(mens)} decisions={len(decs)} -> {ttl}")
        paths.append(ttl)

    combined = rdf_dir / args.combined_name
    parts: list[str] = []
    for i, p in enumerate(paths):
        text = p.read_text(encoding="utf-8")
        if i > 0:
            kept = []
            for line in text.splitlines():
                if line.startswith("@prefix ") or line.startswith("# Graph:") or line.startswith(
                    "# Entity class:"
                ) or line.startswith("# SKOS"):
                    continue
                kept.append(line)
            text = "\n".join(kept).lstrip("\n")
        parts.append(text.rstrip() + "\n")
    combined.write_text("\n".join(parts), encoding="utf-8")
    print(f"COMBINED -> {combined} ({combined.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
