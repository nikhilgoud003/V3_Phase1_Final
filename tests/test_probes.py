#!/usr/bin/env python3
"""Split probe + merge trap via the incremental holdout path.

Split probe: 5 perturbed FJC-linked judges should attach to the existing registry
entity (same NID / strong key), not create duplicates.

Merge trap: 2 synthetic same-surname judges in *different* courts must stay separate.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.cluster import cluster_mentions
from engine.config_loader import load_config, resolve_path
from engine.tiers import run_cascade


def _base_mention(**kwargs) -> dict:
    m = {
        "mention_id": kwargs.get("mention_id", "mnt_probe"),
        "entity_type": "judge",
        "raw_name": kwargs.get("raw_name", "Judge Probe"),
        "normalized_name": kwargs.get("normalized_name", "probe"),
        "presentable_name": kwargs.get("presentable_name", "Probe"),
        "surname": kwargs.get("surname", "probe"),
        "token_count": kwargs.get("token_count", 2),
        "role": kwargs.get("role", "assigned"),
        "docket_source": kwargs.get("docket_source", "case_header"),
        "extraction_method": "probe_synthetic",
        "prefix_category": None,
        "court": kwargs.get("court", "ilnd"),
        "ucid": kwargs.get("ucid", "ilnd;;9:99-cv-00001"),
        "case_id": "9:99-cv-00001",
        "case_type": "cv",
        "case_name": None,
        "year": kwargs.get("year", 2017),
        "filing_date": "01/01/2017",
        "terminating_date": None,
        "judge_enum": 0,
        "party_enum": None,
        "party_name": None,
        "party_role": None,
        "pacer_id": None,
        "office_name": None,
        "email": None,
        "source_file": "probe_synthetic.json",
        "fjc_nid": kwargs.get("fjc_nid"),
        "co_mentions": [],
        "transfer_partners": [],
        "profile": kwargs.get("profile", ""),
    }
    if not m["profile"]:
        m["profile"] = (
            f"Judge: {m['normalized_name']} | Presentable: {m['presentable_name']} "
            f"| Role: {m['role']} | Court: {m['court']} | CaseType: cv | UCID: {m['ucid']} "
            f"| Year: {m['year']} | Source: probe_synthetic"
        )
    return m


def entity_signature(e: dict) -> str:
    nids = e.get("fjc_nids") or []
    if nids:
        return f"nid:{sorted(nids)[0]}"
    courts = e.get("courts") or []
    court = courts[0] if courts else ""
    return f"name:{e.get('normalized_name')}|{court}"


def main() -> int:
    cfg = load_config(ROOT / "configs/judges.yaml")
    # Avoid wiping the live journal/cache
    import tempfile
    import os

    out = Path(tempfile.mkdtemp(prefix="tier_v3_probes_"))
    os.environ["TIER_V3_OUTPUT_DIR"] = str(out)
    cfg = load_config(ROOT / "configs/judges.yaml")  # refresh with env
    # Force rapidfuzz — probes should not need Ollama
    cfg.setdefault("tier2", {})["backend"] = "rapidfuzz"
    # Optional: keep hygiene on for production path; probes set scopes via synthetic fields
    cfg.setdefault("mention_hygiene", {})["enabled"] = True

    # OUTPUT_DIR remaps mentions_out — always read the live mention pool from ROOT
    ment_path = ROOT / "data/mentions/judges_mentions.jsonl"
    all_ments = [json.loads(l) for l in ment_path.open(encoding="utf-8")]
    base = [m for m in all_ments if m.get("docket_source") != "line_entry"][:2500]

    from engine.fjc import link_mentions_to_fjc, load_fjc_index
    from engine.config_loader import resolve_path as _rp

    # Mentions JSONL does not persist fjc_nid; link before picking split-probe seeds.
    honorifics = (cfg.get("normalization") or {}).get("strip_honorifics") or []
    strip_chars = (cfg.get("normalization") or {}).get("strip_chars") or ""
    fjc_path = ROOT / "data/judges_fjc.csv"
    crosswalk = ROOT / "data/external/fjc_court_crosswalk.json"
    if fjc_path.exists():
        idx = load_fjc_index(fjc_path, crosswalk, honorifics, strip_chars)
        link_mentions_to_fjc(base, idx)

    # Pick 5 FJC-linked judges from base for split probe
    fjc_ments = [m for m in base if m.get("fjc_nid")]
    by_nid: dict[str, dict] = {}
    for m in fjc_ments:
        by_nid.setdefault(str(m["fjc_nid"]), m)
    seeds = list(by_nid.values())[:5]
    if len(seeds) < 5:
        raise SystemExit(f"Need 5 FJC-linked seeds, found {len(seeds)}")

    print("=== BASE registry cascade ===")
    base_res = run_cascade(copy.deepcopy(base), cfg, enable_tier3=False)
    base_ents = cluster_mentions(base_res["components"], base_res["by_id"], cfg, base_res["uf"])
    base_sigs = {entity_signature(e) for e in base_ents}
    print(f"base mentions={len(base)} entities={len(base_ents)}")

    # --- Split probe: perturbed names, same NID/court/year ---
    split_ments = []
    for i, s in enumerate(seeds):
        # Perturb presentable/raw but keep normalized surname+nid+court
        toks = (s.get("normalized_name") or "").split()
        pert = " ".join(toks[:-1] + [toks[-1] + "x"]) if len(toks) >= 2 else (s.get("normalized_name") + "x")
        # Better perturbation: middle initial drop / Jr add while keeping fjc_nid
        nn = s["normalized_name"]
        if " jr" not in nn:
            raw = (s.get("presentable_name") or s.get("raw_name") or nn) + ", Jr."
            # Keep same normalized_name so Tier0 name keys still fire OR rely on NID
            # User asked "perturbed" — change surface form but keep NID for attach
            norm = nn  # NID join should attach even if we also tweak year slightly
        else:
            raw = (s.get("presentable_name") or nn).replace(" Jr", "")
            norm = nn.replace(" jr", "")
        split_ments.append(
            _base_mention(
                mention_id=f"mnt_split_probe_{i}",
                raw_name=raw,
                normalized_name=norm,
                presentable_name=raw,
                surname=s.get("surname"),
                token_count=len(norm.split()),
                court=s.get("court"),
                ucid=f"{s.get('court')};;9:99-cv-{1000+i:05d}",
                year=s.get("year") or 2017,
                fjc_nid=s.get("fjc_nid"),
                role="assigned",
            )
        )

    print("=== SPLIT probe cascade ===")
    split_res = run_cascade(copy.deepcopy(split_ments), cfg, enable_tier3=False)
    split_ents = cluster_mentions(split_res["components"], split_res["by_id"], cfg, split_res["uf"])
    split_results = []
    for e in split_ents:
        sig = entity_signature(e)
        split_results.append(
            {
                "probe_names": e.get("name_variants"),
                "fjc_nids": e.get("fjc_nids"),
                "signature": sig,
                "attached_to_registry": sig in base_sigs,
            }
        )
    split_ok = all(r["attached_to_registry"] or (r.get("fjc_nids") and f"nid:{r['fjc_nids'][0]}" in base_sigs) for r in split_results)
    # Attachment definition: after merging probe into base+probe together
    combo = copy.deepcopy(base) + copy.deepcopy(split_ments)
    combo_res = run_cascade(combo, cfg, enable_tier3=False)
    combo_ents = cluster_mentions(combo_res["components"], combo_res["by_id"], cfg, combo_res["uf"])
    # For each probe mention, entity should also contain an original seed with same NID
    mid_to_ent = {}
    for e in combo_ents:
        for mid in e.get("mention_ids") or []:
            mid_to_ent[mid] = e
    split_attach = []
    for i, s in enumerate(seeds):
        probe_mid = f"mnt_split_probe_{i}"
        e = mid_to_ent.get(probe_mid)
        members = set(e.get("mention_ids") or []) if e else set()
        # any base mention with same NID in entity?
        same_nid_base = [
            m["mention_id"]
            for m in base
            if m.get("fjc_nid") == s.get("fjc_nid") and m["mention_id"] in members
        ]
        split_attach.append(
            {
                "probe_mention_id": probe_mid,
                "seed_nid": s.get("fjc_nid"),
                "seed_name": s.get("normalized_name"),
                "attached": bool(same_nid_base),
                "entity_id": e.get("entity_id") if e else None,
                "n_mentions_in_entity": len(members),
            }
        )
    split_pass = all(r["attached"] for r in split_attach)

    # --- Merge trap: same surname, different courts ---
    trap_a = _base_mention(
        mention_id="mnt_merge_trap_a",
        raw_name="Hon. Jane Smith",
        normalized_name="jane smith",
        presentable_name="Jane Smith",
        surname="smith",
        token_count=2,
        court="cand",
        ucid="cand;;9:99-cv-00010",
        year=2018,
        fjc_nid=None,
    )
    trap_b = _base_mention(
        mention_id="mnt_merge_trap_b",
        raw_name="Hon. Jane Smith",
        normalized_name="jane smith",
        presentable_name="Jane Smith",
        surname="smith",
        token_count=2,
        court="nyed",
        ucid="nyed;;9:99-cv-00011",
        year=2018,
        fjc_nid=None,
    )
    print("=== MERGE trap cascade ===")
    trap_res = run_cascade([trap_a, trap_b], cfg, enable_tier3=False)
    trap_ents = cluster_mentions(trap_res["components"], trap_res["by_id"], cfg, trap_res["uf"])
    trap_pass = len(trap_ents) == 2
    trap_report = {
        "n_entities": len(trap_ents),
        "stay_separate": trap_pass,
        "entities": [
            {"entity_id": e.get("entity_id"), "courts": e.get("courts"), "normalized_name": e.get("normalized_name")}
            for e in trap_ents
        ],
    }

    report = {
        "output_dir": str(out),
        "split_probe": {"pass": split_pass, "cases": split_attach},
        "merge_trap": {"pass": trap_pass, **trap_report},
        "pass": split_pass and trap_pass,
    }
    # Write under repo reports (not only temp)
    rep = ROOT / "data/reports/probe_results.json"
    rep.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {rep}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
