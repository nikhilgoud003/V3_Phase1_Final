#!/usr/bin/env python3
"""Stratified human audit packet — 150 decisions, no self-grading.

Strata:
  50 Tier3 MATCH (25 barrier-overruled + 25 other)
  30 NO_MATCH
  30 UNCERTAIN
  20 Tier2 auto-merges
  20 random (any decision type)

Output: data/reports/human_audit_packet.md (+ .json index)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config, resolve_path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def docket_snippet(json_dir: Path, mention: dict, window: int = 200) -> str:
    src = mention.get("source_file") or ""
    fp = json_dir / src
    if not fp.exists():
        return "(docket JSON not found)"
    try:
        case = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        return f"(JSON load error: {e})"
    docket = case.get("docket") or []
    idx = mention.get("docket_index")
    raw = (mention.get("raw_name") or mention.get("normalized_name") or "").strip()

    def clip(text: str) -> str:
        text = text.replace("\n", " ").strip()
        if raw and raw.lower() in text.lower():
            i = text.lower().find(raw.lower())
            lo = max(0, i - window // 3)
            hi = min(len(text), i + len(raw) + 2 * window // 3)
            return text[lo:hi].strip()
        return (text[:window] + "…") if len(text) > window else text

    if isinstance(idx, int) and 0 <= idx < len(docket):
        return clip(docket[idx].get("docket_text") or "")
    for entry in docket:
        text = entry.get("docket_text") or ""
        if raw and raw.lower() in text.lower():
            return clip(text)
    # header mentions: no docket line — use case name / profile
    if mention.get("docket_source") != "line_entry":
        return f"(header/party source={mention.get('docket_source')}; case={case.get('case_name') or case.get('case_id')})"
    return "(no matching docket text)"


def profile_block(m: dict | None, label: str) -> list[str]:
    if not m:
        return [f"**{label}:** _(missing mention)_", ""]
    return [
        f"**{label}** (`{m.get('mention_id')}`)",
        f"- name: **{m.get('normalized_name')}** / raw: {m.get('raw_name')}",
        f"- court `{m.get('court')}` | ucid `{m.get('ucid')}` | year `{m.get('year')}` | role `{m.get('role')}`",
        f"- source `{m.get('docket_source')}` / `{m.get('extraction_method')}` | fjc_nid `{m.get('fjc_nid')}`",
        f"- profile: {(m.get('profile') or '')[:320]}",
        "",
    ]


def sample_n(pool: list, n: int, rng: random.Random) -> list:
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def main() -> int:
    import os

    cfg = load_config(ROOT / "configs/judges.yaml")
    run_dir = Path(os.environ.get("TIER_V3_OUTPUT_DIR", "")).resolve() if os.environ.get("TIER_V3_OUTPUT_DIR") else None
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1]).resolve()
    if run_dir and run_dir.exists():
        reports = run_dir / "reports"
        decisions_path = run_dir / "decisions" / "decisions.jsonl"
        mentions_path = run_dir / "mentions" / "judges_mentions.jsonl"
    else:
        reports = resolve_path(cfg, cfg["io"]["reports_dir"])
        decisions_path = ROOT / "data/decisions/decisions.jsonl"
        mentions_path = ROOT / "data/mentions/judges_mentions.jsonl"
    reports.mkdir(parents=True, exist_ok=True)
    json_dir = ROOT / "data/json/pilot_1000"

    decisions = load_jsonl(decisions_path)
    mentions = load_jsonl(mentions_path)
    by_id = {m["mention_id"]: m for m in mentions}

    rng = random.Random(20260803)

    t3_match = [d for d in decisions if d.get("decision") == "MERGE_TIER3"]
    barrier = [
        d
        for d in t3_match
        if "barrier_overruled_by_llm" in set(d.get("signals") or [])
    ]
    t3_other = [d for d in t3_match if d not in barrier]
    no_match = [d for d in decisions if d.get("decision") == "NO_MATCH"]
    uncertain = [d for d in decisions if d.get("decision") == "UNCERTAIN"]
    t2 = [d for d in decisions if d.get("decision") == "MERGE_TIER2"]
    # random from all logged decisions (including Tier0)
    all_pool = list(decisions)

    strata = [
        ("T3_MATCH_barrier_overruled", sample_n(barrier, 25, rng)),
        ("T3_MATCH_other", sample_n(t3_other, 25, rng)),
        ("NO_MATCH", sample_n(no_match, 30, rng)),
        ("UNCERTAIN", sample_n(uncertain, 30, rng)),
        ("T2_AUTO_MERGE", sample_n(t2, 20, rng)),
        ("RANDOM", sample_n(all_pool, 20, rng)),
    ]

    # Deduplicate if random overlaps prior strata (keep first assignment)
    seen = set()
    packed: list[tuple[str, dict]] = []
    for label, rows in strata:
        for d in rows:
            did = d.get("decision_id")
            if did in seen:
                continue
            seen.add(did)
            packed.append((label, d))

    # Top up RANDOM if dedupe shrunk below 150
    target = 150
    if len(packed) < target:
        extras = [d for d in all_pool if d.get("decision_id") not in seen]
        for d in sample_n(extras, target - len(packed), rng):
            packed.append(("RANDOM_TOPUP", d))
            seen.add(d.get("decision_id"))

    packed = packed[:target]

    index = []
    md: list[str] = [
        "# Human audit packet — 150 stratified decisions",
        "",
        "Fill in one checkbox per decision. **Do not** rely on any auto-grade — none provided.",
        "",
        "## Stratum counts (requested → packed)",
        "",
        "| Stratum | Requested | In packet | Pool size |",
        "|---------|----------:|----------:|----------:|",
    ]
    pool_sizes = {
        "T3_MATCH_barrier_overruled": len(barrier),
        "T3_MATCH_other": len(t3_other),
        "NO_MATCH": len(no_match),
        "UNCERTAIN": len(uncertain),
        "T2_AUTO_MERGE": len(t2),
        "RANDOM": len(all_pool),
    }
    from collections import Counter

    got = Counter(label for label, _ in packed)
    for label, req in [
        ("T3_MATCH_barrier_overruled", 25),
        ("T3_MATCH_other", 25),
        ("NO_MATCH", 30),
        ("UNCERTAIN", 30),
        ("T2_AUTO_MERGE", 20),
        ("RANDOM", 20),
        ("RANDOM_TOPUP", 0),
    ]:
        if got.get(label, 0) == 0 and req == 0:
            continue
        md.append(
            f"| {label} | {req if label != 'RANDOM_TOPUP' else '—'} | {got.get(label, 0)} | "
            f"{pool_sizes.get(label, '—')} |"
        )
    md += ["", f"Total blocks: **{len(packed)}** (seed=20260803)", "", "---", ""]

    for i, (stratum, d) in enumerate(packed, 1):
        ma = by_id.get(d.get("mention_id_a") or "")
        mb = by_id.get(d.get("mention_id_b") or "")
        snip_a = docket_snippet(json_dir, ma) if ma else "(n/a)"
        snip_b = docket_snippet(json_dir, mb) if mb else "(n/a)"
        md.append(f"## Decision {i} — `{d.get('decision_id')}` [{stratum}]")
        md.append("")
        md.extend(profile_block(ma, "Mention A"))
        md.append(f"- docket snippet A: _{snip_a}_")
        md.append("")
        md.extend(profile_block(mb, "Mention B"))
        md.append(f"- docket snippet B: _{snip_b}_")
        md.append("")
        md.append("**System**")
        md.append(
            f"- decision: **{d.get('decision')}** | confidence: **{d.get('confidence')}** | "
            f"method: `{d.get('method')}`"
        )
        md.append(f"- signals: `{d.get('signals')}`")
        md.append(f"- rationale: {(d.get('rationale') or '')[:500]}")
        md.append("")
        md.append("[ ] correct  [ ] wrong  [ ] can't tell")
        md.append("")
        md.append("---")
        md.append("")
        index.append(
            {
                "packet_index": i,
                "stratum": stratum,
                "decision_id": d.get("decision_id"),
                "decision": d.get("decision"),
                "confidence": d.get("confidence"),
                "method": d.get("method"),
                "mention_id_a": d.get("mention_id_a"),
                "mention_id_b": d.get("mention_id_b"),
                "signals": d.get("signals"),
            }
        )

    out_md = reports / "human_audit_packet.md"
    out_json = reports / "human_audit_packet.json"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "n": len(packed),
                "seed": 20260803,
                "stratum_counts": dict(got),
                "pool_sizes": pool_sizes,
                "items": index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"n": len(packed), "stratum_counts": dict(got), "pool_sizes": pool_sizes}, indent=2))
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
