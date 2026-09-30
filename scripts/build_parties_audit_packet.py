#!/usr/bin/env python3
"""Phase 3 human audit packet for parties."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.config_loader import load_config


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def mention_line(m: dict | None, label: str) -> list[str]:
    if not m:
        return [f"**{label}:** _(missing)_", ""]
    return [
        f"**{label}** (`{m.get('mention_id')}`)",
        f"- name: **{m.get('normalized_name')}** / raw: {m.get('raw_name')}",
        f"- court `{m.get('court')}` | ucid `{m.get('ucid')}` | role `{m.get('party_role')}` | type `{m.get('party_type')}`",
        f"- office_class `{m.get('office_class')}` | tokens `{m.get('token_count')}` | surname `{m.get('surname')}`",
        "",
    ]


def is_org_shaped(m: dict) -> bool:
    oc = m.get("office_class") or ""
    if oc in {"corporate", "government"}:
        return True
    nn = m.get("normalized_name") or ""
    return any(t in nn.split()[-1:] for t in ("inc", "llc", "corp", "co", "ag", "company")) or "industries" in nn


def is_crane_pair(a: dict, b: dict) -> bool:
    names = f"{a.get('normalized_name','')} {b.get('normalized_name','')}"
    return "crane" in names


def is_ex_rel_pair(a: dict, b: dict) -> bool:
    names = f"{a.get('normalized_name','')} {b.get('normalized_name','')}"
    return " ex rel" in names or "state of" in names


def pick_unique(pool: list[dict], n: int, rng: random.Random, used: set[str]) -> list[dict]:
    rng.shuffle(pool)
    out = []
    for d in pool:
        did = d.get("decision_id") or f"{d.get('mention_id_a')}|{d.get('mention_id_b')}"
        if did in used:
            continue
        out.append(d)
        used.add(did)
        if len(out) >= n:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", default=str(ROOT / "data/runs/parties_pilot"))
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    round_n = args.round
    cfg = load_config(ROOT / "configs/parties.yaml")
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    decisions = load_jsonl(run / "decisions" / "parties_decisions.jsonl")
    if not decisions:
        decisions = load_jsonl(run / "decisions" / "decisions.jsonl")
    mentions = {m["mention_id"]: m for m in load_jsonl(run / "mentions" / "parties_mentions.jsonl")}
    ents = load_jsonl(run / "clusters" / "parties_entities.jsonl")
    ent_by_mid: dict[str, dict] = {}
    for e in ents:
        for mid in e.get("mention_ids") or []:
            ent_by_mid[mid] = e

    rng = random.Random({1: 20260902, 2: 20260903, 3: 20260904}.get(round_n, 20260900 + round_n))
    used: set[str] = set()

    t3 = [d for d in decisions if d.get("decision") == "MERGE_TIER3"]
    t3_barrier = [d for d in t3 if "barrier_overruled_by_llm" in set(d.get("signals") or [])]
    t2 = [d for d in decisions if d.get("decision") == "MERGE_TIER2"]
    t0 = [d for d in decisions if d.get("decision") == "MERGE_TIER0"]
    no_match = [d for d in decisions if d.get("decision") == "NO_MATCH"]
    uncertain = [d for d in decisions if d.get("decision") == "UNCERTAIN"]

    org_pool = [
        d
        for d in decisions
        if d.get("mention_id_a") in mentions
        and d.get("mention_id_b") in mentions
        and (is_org_shaped(mentions[d["mention_id_a"]]) or is_org_shaped(mentions[d["mention_id_b"]]))
    ]
    crane_pool = [
        d
        for d in decisions
        if d.get("mention_id_a") in mentions
        and d.get("mention_id_b") in mentions
        and is_crane_pair(mentions[d["mention_id_a"]], mentions[d["mention_id_b"]])
    ]
    exrel_pool = [
        d
        for d in decisions
        if d.get("mention_id_a") in mentions
        and d.get("mention_id_b") in mentions
        and is_ex_rel_pair(mentions[d["mention_id_a"]], mentions[d["mention_id_b"]])
    ]

    t3_pool = [
        d
        for d in decisions
        if str(d.get("method") or "").startswith("tier3.")
        and d.get("decision") in {"MERGE_TIER3", "NO_MATCH", "UNCERTAIN"}
    ] or t3

    if round_n >= 4:
        plan = [
            ("tier2_auto", t2, min(50, len(t2))),
            ("tier3_all", t3_pool, min(15, len(t3_pool))),
            ("ex_rel_risk", exrel_pool, min(10, len(exrel_pool))),
        ]
        title = f"Parties Phase 3 — Round {round_n} human audit packet"
        subtitle = (
            f"**Config:** `configs/parties.yaml` v{cfg.get('version', '?')} "
            "(v0.6 structural rails: corroboration + Sr/Jr + legal-form + fund-plan splitter)"
        )
        out_md = reports / f"phase3_audit_packet_round{round_n}.md"
        out_json = reports / f"phase3_audit_packet_round{round_n}.json"
        fill_target = 0
    elif round_n >= 3:
        plan = [
            ("tier2_auto", t2, min(45, len(t2))),
            ("tier3_all", t3_pool, min(15, len(t3_pool))),
            ("ex_rel_risk", exrel_pool, min(15, len(exrel_pool))),
        ]
        title = "Parties Phase 3 — Round 3 human audit packet"
        subtitle = (
            f"**Config:** `configs/parties.yaml` v{cfg.get('version', '?')} "
            "(Round-2 T2 same-shape guards; oversample tier2_auto)"
        )
        out_md = reports / "phase3_audit_packet_round3.md"
        out_json = reports / "phase3_audit_packet_round3.json"
        fill_target = 0
    elif round_n >= 2:
        plan = [
            ("tier3_all", t3_pool, min(30, len(t3_pool))),
            ("tier2_auto", t2, min(25, len(t2))),
            ("ex_rel_risk", exrel_pool, min(20, len(exrel_pool))),
        ]
        title = "Parties Phase 3 — Round 2 human audit packet"
        subtitle = (
            f"**Config:** `configs/parties.yaml` v{cfg.get('version', '?')} "
            "(Round-1 fix baseline; oversample tier3 / tier2_auto / ex_rel)"
        )
        out_md = reports / "phase3_audit_packet_round2.md"
        out_json = reports / "phase3_audit_packet_round2.json"
        fill_target = 0
    else:
        large_org_pairs: list[dict] = []
        for e in sorted(ents, key=lambda x: len(x.get("mention_ids") or []), reverse=True):
            if len(large_org_pairs) >= 30:
                break
            mids = e.get("mention_ids") or []
            if len(mids) < 15:
                continue
            sample_m = mentions.get(mids[0])
            if not sample_m or not is_org_shaped(sample_m):
                continue
            a, b = mids[0], mids[min(5, len(mids) - 1)]
            large_org_pairs.append(
                {
                    "decision_id": f"cluster_{e.get('entity_id')}",
                    "decision": "ENTITY_CLUSTER_SAMPLE",
                    "method": "cluster.large_org",
                    "confidence": None,
                    "mention_id_a": a,
                    "mention_id_b": b,
                    "rationale": f"Large org cluster size={len(mids)} canonical={e.get('canonical_name')}",
                    "entity_id": e.get("entity_id"),
                }
            )
        plan = [
            ("tier3_all", t3, min(25, len(t3))),
            ("tier3_barrier", t3_barrier, min(15, len(t3_barrier))),
            ("tier2_auto", t2, 20),
            ("tier0_court", [d for d in t0 if "court" in (d.get("method") or "")], 15),
            ("no_match", no_match, 15),
            ("uncertain", uncertain, 15),
            ("org_shaped", org_pool, 35),
            ("crane_risk", crane_pool, 10),
            ("ex_rel_risk", exrel_pool, 10),
            ("large_org_cluster", large_org_pairs, 10),
        ]
        title = "Parties Phase 3 — Round 1 human audit packet"
        subtitle = f"**Config:** `configs/parties.yaml` v{cfg.get('version', '?')} (corp-core baseline)"
        out_md = reports / "phase3_audit_packet_round1.md"
        out_json = reports / "phase3_audit_packet_round1.json"
        fill_target = 150

    picked: list[tuple[str, dict]] = []
    for label, pool, n in plan:
        for d in pick_unique(pool, n, rng, used):
            picked.append((label, d))

    while fill_target and len(picked) < fill_target and org_pool:
        extra = pick_unique(org_pool, 1, rng, used)
        if not extra:
            break
        picked.append(("org_shaped_fill", extra[0]))

    md = [
        f"# {title}",
        "",
        f"**Run:** `{run}`",
        subtitle,
        f"**Samples:** {len(picked)}",
        "",
        "Label each pair **SAME** / **DIFFERENT** / **UNSURE**. Do not trust model confidence.",
        "",
        "## Stratum counts",
        "",
    ]
    strata_counts: dict[str, int] = {}
    for label, _ in picked:
        strata_counts[label] = strata_counts.get(label, 0) + 1
    for k, v in sorted(strata_counts.items()):
        md.append(f"- `{k}`: {v}")
    md.append("")

    index = []
    for i, (label, d) in enumerate(picked, 1):
        a = mentions.get(d.get("mention_id_a") or "", {})
        b = mentions.get(d.get("mention_id_b") or "", {})
        md.append(f"## {i}. [{label}] {d.get('decision')} — {d.get('method')}")
        md.append("")
        md.append(f"- decision_id: `{d.get('decision_id')}`")
        if d.get("confidence") is not None:
            md.append(f"- confidence: {d.get('confidence')}")
        md.append(f"- rationale: {d.get('rationale', '')[:400]}")
        ea = ent_by_mid.get(d.get("mention_id_a") or "")
        eb = ent_by_mid.get(d.get("mention_id_b") or "")
        if ea and eb and ea.get("entity_id") == eb.get("entity_id"):
            md.append(f"- **same entity cluster:** `{ea.get('entity_id')}` ({ea.get('canonical_name')})")
        md.append("")
        md.extend(mention_line(a, "Mention A"))
        md.extend(mention_line(b, "Mention B"))
        md.append("**Reviewer:** SAME / DIFFERENT / UNSURE")
        md.append("")
        md.append("---")
        md.append("")
        index.append(
            {
                "n": i,
                "stratum": label,
                "decision_id": d.get("decision_id"),
                "decision": d.get("decision"),
                "method": d.get("method"),
                "mention_id_a": d.get("mention_id_a"),
                "mention_id_b": d.get("mention_id_b"),
            }
        )

    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "n_samples": len(picked),
                "strata_counts": strata_counts,
                "index": index,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_md} ({len(picked)} samples)")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
