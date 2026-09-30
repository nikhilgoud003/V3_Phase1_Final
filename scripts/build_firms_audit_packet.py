#!/usr/bin/env python3
"""Stratified 100-decision human audit packet for firms (no self-grading)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/runs/firms_pilot_v1"
    rng = random.Random(42)
    decs = load_jsonl(run / "decisions" / "firms_decisions.jsonl")
    if not decs:
        decs = load_jsonl(run / "decisions" / "decisions.jsonl")
    ments = {m["mention_id"]: m for m in load_jsonl(run / "mentions" / "firms_mentions.jsonl")}

    by = {
        "MERGE_TIER0": [d for d in decs if d.get("decision") == "MERGE_TIER0"],
        "MERGE_TIER2": [d for d in decs if d.get("decision") == "MERGE_TIER2"],
        "MERGE_TIER3": [d for d in decs if d.get("decision") == "MERGE_TIER3"],
        "NO_MATCH": [d for d in decs if d.get("decision") == "NO_MATCH"],
        "UNCERTAIN": [d for d in decs if d.get("decision") == "UNCERTAIN"],
    }

    # Target 100: 40 Tier0, 20 Tier2, 20 Tier3 (or UNCERTAIN), 10 NO_MATCH, 10 random
    plan = [
        ("MERGE_TIER0", 40),
        ("MERGE_TIER2", 20),
        ("MERGE_TIER3", 15),
        ("UNCERTAIN", 10),
        ("NO_MATCH", 10),
    ]
    picked = []
    used = set()

    def take(pool: list[dict], n: int) -> list[dict]:
        cand = [d for d in pool if d.get("decision_id") not in used]
        rng.shuffle(cand)
        out = cand[:n]
        for d in out:
            used.add(d.get("decision_id"))
        return out

    for key, n in plan:
        got = take(by.get(key) or [], n)
        picked.extend(got)
        # backfill shortage from Tier0
        if len(got) < n:
            picked.extend(take(by["MERGE_TIER0"], n - len(got)))

    # random filler to 100
    rest = [d for d in decs if d.get("decision_id") not in used]
    rng.shuffle(rest)
    while len(picked) < 100 and rest:
        picked.append(rest.pop())
        used.add(picked[-1].get("decision_id"))

    lines = [
        "# Firms human audit packet (100 decisions)",
        "",
        f"Run: `{run}`",
        "",
        "Label each as **SAME** / **DIFFERENT** / **UNSURE**. Do not trust model confidence.",
        "",
    ]
    index = []
    for i, d in enumerate(picked, 1):
        a = ments.get(d.get("mention_id_a") or "", {})
        b = ments.get(d.get("mention_id_b") or "", {})
        lines += [
            f"## {i}. `{d.get('decision_id')}` — {d.get('decision')} ({d.get('method')})",
            "",
            f"- confidence: {d.get('confidence')}",
            f"- A: **{a.get('presentable_name') or a.get('normalized_name')}** | class={a.get('office_class')} | domain={a.get('domain')} | phone={a.get('phone')} | court={a.get('court')} | ucid={a.get('ucid')}",
            f"- B: **{b.get('presentable_name') or b.get('normalized_name')}** | class={b.get('office_class')} | domain={b.get('domain')} | phone={b.get('phone')} | court={b.get('court')} | ucid={b.get('ucid')}",
            f"- rationale: {d.get('rationale')}",
            "",
            "Your label: SAME / DIFFERENT / UNSURE",
            "",
            "---",
            "",
        ]
        index.append(
            {
                "i": i,
                "decision_id": d.get("decision_id"),
                "decision": d.get("decision"),
                "method": d.get("method"),
                "a": a.get("normalized_name"),
                "b": b.get("normalized_name"),
            }
        )

    out_dir = run / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "firms_human_audit_packet.md").write_text("\n".join(lines))
    (out_dir / "firms_human_audit_packet.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {out_dir / 'firms_human_audit_packet.md'} n={len(picked)}")
    print("strata", {k: sum(1 for d in picked if d.get("decision") == k) for k in by})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
