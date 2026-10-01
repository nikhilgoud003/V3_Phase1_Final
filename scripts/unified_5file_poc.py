#!/usr/bin/env python3
"""Unified one-pass ER — PACER JSON cold/warm resolve (judges + firms + parties).

Orchestration only. Does NOT modify:
  - engine/tiers.py Tier0–3 logic
  - engine/tier3_citation.py
  - live Tentris

Process (per file, in order):
  1. Read the PACER JSON once.
  2. Extract via existing YAML sources + judges spaCy NER on docket.
  3. Walk unknown string leaves (cached qwen field typing).
  4. Resolve cumulatively with run_cascade + cluster.
  5. Write entities.jsonl, mentions.jsonl, decisions.jsonl, summary.json.

By default, if --output-dir already has entities.jsonl, SJIDs are REUSED
(--resume-from / auto-resume). Pass --cold-start to remint from zero.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.cluster import cluster_mentions  # noqa: E402
from engine.config_loader import load_config, load_config_cached, resolve_path  # noqa: E402
from engine.embeddings import EMBED_STATS  # noqa: E402
from engine.discovery_validity import (  # noqa: E402
    force_type_other,
    path_is_non_entity,
    value_is_discovery_junk,
)
from engine.extract import (  # noqa: E402
    DEFAULT_TYPE_CONFIGS,
    _emit_mention,
    _finalize_mentions,
)
from engine.normalize import normalize_name  # noqa: E402
from engine.parallel_extract import iter_extracted  # noqa: E402
from engine.poc_party_evidence import (  # noqa: E402
    adjudicate_poc_evidence_via_tier3,
    rebuild_components,
)
from engine.provenance import DecisionJournal  # noqa: E402
from engine.tiers import LLM_MEMO_STATS, load_llm_memo, run_cascade  # noqa: E402

# Part B selection (fixed order)
POC_FILES = [
    "azd-2-16-cv-01302.json",
    "azd-2-17-cv-01194.json",
    "azd-2-17-cv-01514.json",
    "azd-2-17-cv-02566.json",
    "azd-2-17-cv-03643.json",
]

KNOWN_PATH_PREFIXES = {
    # Covered by YAML sources / negative-evidence / transfer mining
    "judge",
    "referred_judges",
    "parties[].name",
    "parties[].judge",
    "parties[].referred_judges",
    "parties[].counsel[].name",
    "parties[].counsel[].entity_info.office_name",
    "parties[].counsel[].entity_info.email",
    "parties[].counsel[].entity_info.phone",
    "parties[].counsel[].entity_info.fax",
    "parties[].counsel[].entity_info.address",
    "docket[].docket_text",  # spaCy via judges.yaml ner_span
}

# Paths that are never entity-name fields
SKIP_PATH_SUFFIXES = (
    "ucid",
    "case_id",
    "case_name",
    "case_type",
    "court",
    "filing_date",
    "terminating_date",
    "nature_of_suit",
    "cause",
    "jury_demand",
    "jurisdiction",
    "mdl_code",
    "pacer_case_id",
    "download_url",
    "pdf_url",
    "docket_number",
    "date_filed",
    "date_entered",
    "document_number",
    "description",
    "party_type",
    "role",
    "type",
)

NAMEISH = re.compile(r"[A-Za-z]{2,}")
TOO_LONG = 120
TOO_SHORT = 3


def _ollama_endpoint() -> str:
    host = (os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_ENDPOINT") or "http://127.0.0.1:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    return host


def _post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def path_pattern(path: str) -> str:
    """Collapse numeric list indices to [] for cache keys."""
    parts = []
    for p in path.split("."):
        if p.isdigit():
            parts.append("[]")
        else:
            parts.append(p)
    # Fix "parties.[].name" style from walk — we emit parties[].name directly
    out = []
    for p in parts:
        if p == "[]" and out:
            out[-1] = out[-1] + "[]"
        else:
            out.append(p)
    return ".".join(out)


def walk_string_leaves(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(walk_string_leaves(v, p))
    elif isinstance(obj, list):
        for item in obj:
            p = f"{prefix}[]" if prefix else "[]"
            out.extend(walk_string_leaves(item, p))
    elif isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append((prefix, s))
    return out


def is_known_path(pattern: str) -> bool:
    if pattern in KNOWN_PATH_PREFIXES:
        return True
    for k in KNOWN_PATH_PREFIXES:
        if pattern == k or pattern.startswith(k + ".") or pattern.startswith(k + "[]"):
            return True
    # docket text is known
    if "docket_text" in pattern:
        return True
    return False


def looks_like_name_value(s: str) -> bool:
    if len(s) < TOO_SHORT or len(s) > TOO_LONG:
        return False
    if not NAMEISH.search(s):
        return False
    # skip pure dates / ids / urls
    if re.fullmatch(r"[\d\-/:.\s]+", s):
        return False
    if s.lower().startswith("http"):
        return False
    leaf = s.split()[-1] if s.split() else s
    # skip booleans-as-strings etc.
    if s.lower() in {"true", "false", "null", "none", "yes", "no"}:
        return False
    return True


def leaf_name(pattern: str) -> str:
    return pattern.split(".")[-1].replace("[]", "")


class FieldTypeCache:
    def __init__(self, path: Path):
        self.path = path
        self.cache: dict[str, str] = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["path_pattern"]] = rec["entity_type"]

    def get(self, pattern: str) -> str | None:
        return self.cache.get(pattern)

    def put(self, pattern: str, etype: str, raw_response: str | None = None) -> None:
        self.cache[pattern] = etype
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "path_pattern": pattern,
                        "entity_type": etype,
                        "raw_response": raw_response,
                        "ts": time.time(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def classify_field_type(pattern: str, sample_values: list[str], cache: FieldTypeCache) -> str:
    """Type a never-seen field. Deterministic junk/path gates beat cache + qwen."""
    samples = sample_values[:5]
    forced, why = force_type_other(pattern, samples)
    if forced:
        # Do not trust a prior bad cache entry for this path.
        cache.put(pattern, "other", raw_response=f"deterministic:{why}")
        return "other"

    hit = cache.get(pattern)
    if hit and hit != "other":
        # Re-check: cached entity type still invalid if path/values are junk
        forced2, why2 = force_type_other(pattern, samples)
        if forced2:
            cache.put(pattern, "other", raw_response=f"deterministic_override:{why2}")
            return "other"
        return hit
    if hit == "other":
        return "other"

    prompt = (
        "You classify a PACER JSON field by what kind of named entity its VALUES hold.\n"
        "Reply with EXACTLY one word: judge, firm, party, or other.\n"
        "Use other for: HTML/status boilerplate, case/UCID keys, metadata, flags, URLs.\n"
        "Only use judge/firm/party when values are actual person or organization names.\n"
        f"Field path pattern: {pattern}\n"
        f"Leaf key: {leaf_name(pattern)}\n"
        f"Sample values: {json.dumps(samples, ensure_ascii=False)}\n"
    )
    model = os.environ.get("TIER_V3_LLM_MODEL") or "qwen2.5:7b"
    try:
        resp = _post_json(
            f"{_ollama_endpoint()}/api/generate",
            {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=90.0,
        )
        text = (resp.get("response") or "").strip().lower()
    except Exception as e:
        text = f"error:{e}"
        etype = "other"
        cache.put(pattern, etype, raw_response=text)
        return etype
    etype = "other"
    for cand in ("judge", "firm", "party", "other"):
        if re.search(rf"\b{cand}\b", text):
            etype = cand
            break
    # Final validity gate — never emit entity type for junk samples
    if etype in {"judge", "firm", "party"}:
        forced3, why3 = force_type_other(pattern, samples)
        if forced3:
            etype = "other"
            text = f"{text}|forced_other:{why3}"
    cache.put(pattern, etype, raw_response=text[:500])
    return etype


def discover_unknown_mentions(
    case: dict,
    source_file: str,
    cfgs: dict[str, dict],
    cache: FieldTypeCache,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Return {etype: mentions}, and skip/quarantine log rows."""
    buckets: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
    log: list[dict] = []

    # Group values by path pattern
    by_pat: dict[str, list[str]] = defaultdict(list)
    for path, val in walk_string_leaves(case):
        pat = path_pattern(path)
        if is_known_path(pat):
            continue
        leaf = leaf_name(pat)
        if leaf in SKIP_PATH_SUFFIXES or any(pat.endswith("." + s) or pat.endswith(s) for s in SKIP_PATH_SUFFIXES):
            continue
        # Path-level gate (same class as name-validity: never ask / never emit)
        bad_path, path_why = path_is_non_entity(pat)
        if bad_path:
            log.append(
                {
                    "reason": "discovery_path_rejected",
                    "path_pattern": pat,
                    "reject_reason": path_why,
                    "sample": val[:160],
                    "source_file": source_file,
                }
            )
            continue
        junk_val, junk_why = value_is_discovery_junk(val)
        if junk_val:
            log.append(
                {
                    "reason": "discovery_value_rejected",
                    "path_pattern": pat,
                    "reject_reason": junk_why,
                    "raw": val[:200],
                    "source_file": source_file,
                }
            )
            continue
        if not looks_like_name_value(val):
            continue
        by_pat[pat].append(val)

    party_neg: set[str] = set()
    counsel_neg: set[str] = set()
    for pat, vals in by_pat.items():
        etype_label = classify_field_type(pat, vals, cache)
        if etype_label not in {"judge", "firm", "party"}:
            log.append(
                {
                    "reason": "field_typed_other_or_unknown",
                    "path_pattern": pat,
                    "entity_type": etype_label,
                    "n_values": len(vals),
                    "samples": vals[:3],
                    "source_file": source_file,
                }
            )
            continue
        cfg = cfgs[etype_label]
        source = {
            "id": f"discovered_{etype_label}",
            "role": "discovered",
            "docket_source": "schema_free_walk",
            "extraction_method": f"unknown_field_qwen:{pat}",
        }
        for val in vals:
            # Per-value gate again (mixed fields: some junk, some real)
            junk_val, junk_why = value_is_discovery_junk(val)
            if junk_val:
                log.append(
                    {
                        "reason": "discovery_value_rejected",
                        "path_pattern": pat,
                        "reject_reason": junk_why,
                        "raw": val[:200],
                        "source_file": source_file,
                    }
                )
                continue
            m = _emit_mention(
                raw=val,
                cfg=cfg,
                case=case,
                source=source,
                extra={"source_file": source_file, "discovered_path": pat},
                party_neg=party_neg,
                counsel_neg=counsel_neg,
            )
            if m is None:
                log.append(
                    {
                        "reason": "emit_dropped",
                        "path_pattern": pat,
                        "raw": val,
                        "source_file": source_file,
                    }
                )
                continue
            m["poc_discovered"] = True
            m["poc_path_pattern"] = pat
            buckets[etype_label].append(m)
            log.append(
                {
                    "reason": "discovered_kept",
                    "path_pattern": pat,
                    "entity_type": etype_label,
                    "raw": val,
                    "normalized_name": m.get("normalized_name"),
                    "mention_id": m.get("mention_id"),
                    "source_file": source_file,
                }
            )
    return buckets, log


def attach_party_case_context(
    party_mentions: list[dict],
    judge_mentions: list[dict],
    firm_mentions: list[dict],
    case: dict,
) -> None:
    """Attach PoC sidecar evidence (does not change Tier0–3 code).

    Existing engine already uses co_mentions (co-defendants) in Tier3 evidence.
    Same-case judge/firm + MDL are recorded here for audit; MATCH still follows
    existing YAML/engine rules until a future (approved) wiring step.
    """
    judges = sorted({m.get("normalized_name") for m in judge_mentions if m.get("normalized_name")})
    firms = sorted({m.get("normalized_name") for m in firm_mentions if m.get("normalized_name")})
    mdl = case.get("mdl_code") or case.get("mdl") or None
    is_mdl = bool(case.get("is_mdl") or mdl)
    for m in party_mentions:
        m["poc_case_judges"] = judges
        m["poc_case_firms"] = firms
        m["poc_mdl_code"] = mdl
        m["poc_is_mdl"] = is_mdl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _alias_name_to_group(cfgs: dict[str, dict]) -> dict[str, str]:
    """Map normalized alias strings → group id from YAML alias_groups (config data)."""
    out: dict[str, str] = {}
    for cfg in cfgs.values():
        for grp in (cfg.get("tier0") or {}).get("alias_groups") or []:
            gid = str(grp.get("id") or "alias")
            for name in grp.get("names") or []:
                nn = " ".join(str(name).lower().split())
                if nn:
                    out[nn] = gid
    return out


def poc_stable_key(e: dict, alias_to_gid: dict[str, str]) -> str:
    """Stable identity key across runs (no remint when the same entity reappears)."""
    from engine.rdf_emit import entity_signature

    et = (e.get("entity_type") or e.get("type") or "").strip()
    nn = " ".join((e.get("normalized_name") or "").lower().split())
    if et == "judge":
        nids = [str(x) for x in (e.get("fjc_nids") or []) if x]
        if nids:
            return f"judge:nid:{sorted(nids)[0]}"
        courts = "|".join(sorted(e.get("courts") or []))
        return f"judge:name:{nn}|{courts}"
    if et == "firm":
        return f"firm:{entity_signature({**e, 'entity_type': 'firm'})}"
    # party — alias-group members share one key (config-driven); others keyed by name+court
    gid = alias_to_gid.get(nn)
    if gid:
        return f"party:alias:{gid}"
    courts = "|".join(sorted(e.get("courts") or []))
    return f"party:name:{nn}|{courts}"


def load_prior_entities(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def remap_entity_ids(
    entities_by_type: dict[str, list[dict]],
    prior: list[dict],
    alias_to_gid: dict[str, str],
    prefixes: dict[str, str],
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """REUSE prior sjids by stable key; CREATE only for new keys (serial after max)."""
    by_key: dict[str, dict] = {}
    max_serial: dict[str, int] = {t: -1 for t in prefixes}
    for e in prior:
        et = (e.get("entity_type") or e.get("type") or "").strip()
        if et not in prefixes:
            continue
        key = poc_stable_key(e, alias_to_gid)
        by_key[key] = e
        sid = str(e.get("sjid") or "")
        pref = prefixes[et]
        if sid.startswith(pref) and sid[len(pref) :].isdigit():
            max_serial[et] = max(max_serial[et], int(sid[len(pref) :]))

    report = {"n_reuse": 0, "n_create": 0, "by_type": {}}
    out: dict[str, list[dict]] = {}
    for etype, ents in entities_by_type.items():
        pref = prefixes[etype]
        remapped: list[dict] = []
        t_reuse = t_create = 0
        for e in ents:
            row = dict(e)
            row["type"] = etype
            row["entity_type"] = etype
            key = poc_stable_key(row, alias_to_gid)
            hit = by_key.get(key)
            if hit and hit.get("sjid"):
                row["sjid"] = hit["sjid"]
                # Keep prior entity_id when reusing so downstream refs stay stable
                if hit.get("entity_id"):
                    row["entity_id"] = hit["entity_id"]
                t_reuse += 1
            else:
                max_serial[etype] += 1
                row["sjid"] = f"{pref}{max_serial[etype]:06d}"
                row["entity_id"] = f"ent_{etype[0]}_{max_serial[etype]:06d}"
                t_create += 1
                by_key[key] = row
            remapped.append(row)
        out[etype] = remapped
        report["by_type"][etype] = {"reuse": t_reuse, "create": t_create}
        report["n_reuse"] += t_reuse
        report["n_create"] += t_create
    return out, report


def entity_cross_file_report(entities: list[dict], by_id: dict[str, dict]) -> list[dict]:
    rows = []
    for e in entities:
        files: set[str] = set()
        norms: set[str] = set()
        for mid in e.get("mention_ids") or []:
            m = by_id.get(mid) or {}
            if m.get("source_file"):
                files.add(m["source_file"])
            if m.get("normalized_name"):
                norms.add(m["normalized_name"])
        if len(files) >= 2:
            rows.append(
                {
                    "entity_id": e.get("entity_id"),
                    "sjid": e.get("sjid"),
                    "canonical_name": e.get("canonical_name"),
                    "normalized_name": e.get("normalized_name"),
                    "n_mentions": len(e.get("mention_ids") or []),
                    "source_files": sorted(files),
                    "name_variants": sorted(norms),
                    "cross_file_match": True,
                }
            )
    return rows


def expected_repeats() -> dict[str, list[str]]:
    """Ground-truth strings that SHOULD match across files (from Part B)."""
    return {
        "judge": ["david g campbell"],  # after honorific strip / normalize
        "firm": [
            "nelson mullins riley & scarborough llc",  # location may strip
            "nelson mullins riley & scarborough llc - atlanta, ga",
            "farris riley & pitt llp",
            "matthews & associates",
            "freese & goss pllc",  # location may strip
            "freese & goss pllc - maple ave., dallas, tx",
        ],
        "party": [
            "c r bard incorporated",
            "bard peripheral vascular incorporated",
        ],
    }


def norm_for_expect(s: str, cfg: dict) -> str:
    n = cfg.get("normalization") or {}
    return normalize_name(
        s,
        honorifics=n.get("strip_honorifics") or [],
        strip_chars=n.get("strip_chars") or "",
        lowercase=bool(n.get("lowercase", True)),
        collapse_whitespace=bool(n.get("collapse_whitespace", True)),
        name_prefixes=n.get("strip_name_prefixes") or [],
        corp_suffixes=n.get("strip_corp_suffixes") or [],
        replace_tokens=n.get("replace_tokens") or {},
        replace_regex=n.get("replace_regex") or [],
        delete_chars=n.get("delete_chars") or "",
        strip_procedural_prefixes=bool(n.get("strip_procedural_prefixes")),
        procedural_prefix_patterns=n.get("procedural_prefix_patterns"),
    )


def _collect_decision_rows(work_dir: Path, by_id: dict[str, dict]) -> list[dict]:
    """Harvest cascade decision journals from work_dir; annotate with names/files."""
    mapping = [
        ("judge", work_dir / "decisions" / "decisions.jsonl"),
        ("firm", work_dir / "decisions" / "firms_decisions.jsonl"),
        ("party", work_dir / "decisions" / "parties_decisions.jsonl"),
        ("party", work_dir / "decisions" / "poc_party_evidence_decisions.jsonl"),
    ]
    rows: list[dict] = []
    for default_etype, path in mapping:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            a = by_id.get(d.get("mention_id_a") or "") or {}
            b = by_id.get(d.get("mention_id_b") or "") or {}
            etype = d.get("entity_type") or default_etype
            rows.append(
                {
                    **d,
                    "entity_type": etype,
                    "normalized_name_a": a.get("normalized_name"),
                    "normalized_name_b": b.get("normalized_name"),
                    "source_file_a": a.get("source_file"),
                    "source_file_b": b.get("source_file"),
                    "cross_file": bool(
                        a.get("source_file")
                        and b.get("source_file")
                        and a.get("source_file") != b.get("source_file")
                    ),
                }
            )
    return rows


def write_final_bundle(
    out_root: Path,
    *,
    entities_by_type: dict[str, list[dict]],
    mentions_by_type: dict[str, list[dict]],
    decisions: list[dict],
    summary: dict,
) -> None:
    """Write the single final output set (no per-file step folders)."""
    out_root.mkdir(parents=True, exist_ok=True)

    ent_rows: list[dict] = []
    for etype in ("judge", "firm", "party"):
        for e in entities_by_type.get(etype) or []:
            row = dict(e)
            row["type"] = etype
            row.setdefault("entity_type", etype)
            ent_rows.append(row)
    write_jsonl(out_root / "entities.jsonl", ent_rows)

    men_rows: list[dict] = []
    for etype in ("judge", "firm", "party"):
        for m in mentions_by_type.get(etype) or []:
            row = dict(m)
            row["type"] = etype
            row.setdefault("entity_type", etype)
            men_rows.append(row)
    write_jsonl(out_root / "mentions.jsonl", men_rows)

    write_jsonl(out_root / "decisions.jsonl", decisions)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def main() -> int:
    import argparse
    import shutil
    import tempfile

    ap = argparse.ArgumentParser(
        description="Unified one-pass ER PoC: one JSON at a time, cold-start cumulative resolve"
    )
    ap.add_argument(
        "--json-dir",
        default=str(ROOT / "data/json/pilot_1000"),
        help="Directory of PACER *.json files",
    )
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional ordered list of filenames under --json-dir "
        "(default: all *.json in the directory, sorted)",
    )
    ap.add_argument(
        "--poc-bard5",
        action="store_true",
        help="Use the original Part-B Bard 5-file subset only (ignores other JSONs)",
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "data/runs/unified_5file_poc"),
        help="Final run output directory (entities/mentions/decisions/summary)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Process at most N files")
    ap.add_argument(
        "--debug-steps",
        action="store_true",
        help="Also write per-file step_XX_* snapshot folders (off by default)",
    )
    ap.add_argument(
        "--resume-from",
        default=None,
        help="Prior run dir (or entities.jsonl) whose SJIDs to REUSE. "
        "Default: auto-resume from --output-dir/entities.jsonl when present.",
    )
    ap.add_argument(
        "--cold-start",
        action="store_true",
        help="Does not wipe a checkpoint. If checkpoint/state.json exists, the run resumes. "
        "Use --fresh to start over.",
    )
    ap.add_argument(
        "--schema-walk",
        choices=["on", "off"],
        default=None,
        help="Override configs/unified.yaml schema_free_walk.enabled for this run.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Processes for JSON reading + extraction (default: configs/unified.yaml "
        "extract_workers; 0 = CPU count - 1; 1 = in-process).",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Also save the checkpoint every N files (default 0: save once at the end).",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Start from file 1. Refuses if --output-dir already contains any file.",
    )
    args = ap.parse_args()

    json_root = Path(args.json_dir)
    if not json_root.is_absolute():
        json_root = ROOT / json_root
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    if args.fresh:
        existing = []
        if out_root.exists():
            existing = [p for p in out_root.rglob("*") if p.is_file()]
        if existing:
            print(
                f"ERROR: --fresh refused; output dir is not empty ({len(existing)} files in {out_root}). "
                "Pick an empty directory.",
                file=sys.stderr,
            )
            return 1
    out_root.mkdir(parents=True, exist_ok=True)

    if args.files is not None:
        names = list(args.files)
        files = [json_root / name for name in names]
        missing = [str(fp) for fp in files if not fp.is_file()]
        if missing:
            print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
            return 1
    elif args.poc_bard5:
        files = [json_root / name for name in POC_FILES]
        missing = [str(fp) for fp in files if not fp.is_file()]
        if missing:
            print("ERROR: missing Bard-5 files:\n  " + "\n  ".join(missing), file=sys.stderr)
            return 1
    else:
        # Default: every *.json in the folder (sorted). This is what you want for 10 or 1000.
        files = sorted(json_root.glob("*.json"))
        if not files:
            print(f"ERROR: no *.json in {json_root}", file=sys.stderr)
            return 1

    if args.limit is not None:
        files = files[: args.limit]
    print(f"Will process {len(files)} JSON file(s) from {json_root}", flush=True)
    file_names = [fp.name for fp in files]

    cfgs: dict[str, dict] = {}
    for etype, cpath in DEFAULT_TYPE_CONFIGS.items():
        cfg = load_config(cpath)
        cfgs[cfg.get("entity_type") or etype] = cfg

    unified_cfg = load_config(ROOT / "configs" / "unified.yaml")
    schema_walk = bool((unified_cfg.get("schema_free_walk") or {}).get("enabled", False))
    if args.schema_walk is not None:
        schema_walk = args.schema_walk == "on"
    print(f"Schema-free walk: {'on' if schema_walk else 'off'}", flush=True)
    workers = args.workers if args.workers is not None else int(unified_cfg.get("extract_workers") or 0)
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if schema_walk:
        workers = 1  # the walk asks Qwen per field and shares a cache: keep it in-process
    print(f"Extraction workers: {workers}", flush=True)

    sys.path.insert(0, str(ROOT / "scripts"))
    import incremental_resolve as inc  # noqa: E402

    discovery_log: list[dict] = []
    step_summaries: list[dict] = []
    mdl_by_ucid: dict[str, Any] = {}
    poc_evidence_all: list[dict] = []

    # Cascade scratch (overwritten each step). Final harvest from last step.
    # Field-type cache lives beside scratch so it survives per-step rmtree of work_dir.
    work_dir = Path(tempfile.mkdtemp(prefix="unified_poc_work_", dir=str(out_root)))
    cache_path = work_dir.parent / f".{work_dir.name}_field_type_cache.jsonl"
    field_cache = FieldTypeCache(cache_path)
    t_start = time.time()
    n_files = len(files)

    # A checkpoint wins over --cold-start. Only --fresh (and an empty dir) starts over.
    state = inc.load_checkpoint(out_root)
    if state is None:
        state = {
            "processed": [],
            "next_serial": {"judge": -1, "firm": -1, "party": -1},
            "timings": [],
            "entities": {"judge": [], "firm": [], "party": []},
            "mentions": {"judge": [], "firm": [], "party": []},
            "decisions": [],
            "poc_evidence": [],
            "cascade_last": {},
        }
    inc.RUN_CACHE_DIR = out_root / "checkpoint"
    n_memo = load_llm_memo(out_root / "checkpoint" / "llm_prompt_cache.jsonl")
    if n_memo:
        print(f"LLM prompt cache: {n_memo} answers loaded", flush=True)
    embed_cache = inc.EmbedCache(out_root / "checkpoint" / "embed_cache.json")
    state["embed_cache"] = embed_cache
    saved_index = {et: inc.SavedIndex() for et in ("judge", "firm", "party")}
    for et in ("judge", "firm", "party"):
        saved_index[et].mentions = {m["mention_id"]: m for m in state["mentions"][et]}
    processed_keys = {(p["file"], p["sha256"]) for p in state["processed"]}
    prefixes = {
        et: (cfgs[et].get("clustering") or {}).get("id_prefix", "SJ")
        for et in ("judge", "firm", "party")
    }
    link_journal = DecisionJournal(
        out_root / "checkpoint" / "link_decisions.jsonl", fresh=not processed_keys, buffered=True
    )

    final_entities: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
    final_mentions: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
    final_by_id: dict[str, dict] = {}
    final_cascade: dict[str, Any] = {}


    resume_at = None
    if state["processed"]:
        for step_i, fp in enumerate(files, start=1):
            if (fp.name, inc.file_sha256(fp)) not in processed_keys:
                resume_at = step_i
                break
        if resume_at is not None:
            note = " (--cold-start ignored because checkpoint/state.json exists)" if args.cold_start else ""
            print(f"RESUMING from file {resume_at}{note}", flush=True)
        else:
            print(f"RESUMING from file {len(files)+1} (all {len(files)} files already in checkpoint)", flush=True)

    digests = {fp: inc.file_sha256(fp) for fp in files}
    todo = [fp for fp in files if (fp.name, digests[fp]) not in processed_keys]
    extracted = iter_extracted(todo, {k: str(v) for k, v in DEFAULT_TYPE_CONFIGS.items()}, workers)

    try:
        for step_i, fp in enumerate(files, start=1):
            digest = digests[fp]
            if (fp.name, digest) in processed_keys:
                print(f"SKIP already processed step-file {fp.name} sha256={digest[:12]}", flush=True)
                continue
            t_file = time.perf_counter()
            got = next(extracted)
            assert got["file"] == fp.name, (got["file"], fp.name)
            case = got["case"]
            sec_read = got["sec_read"]

            print("\n" + "=" * 72)
            print(f"STEP {step_i}/{n_files}  file={fp.name}")
            print("=" * 72, flush=True)

            file_mentions: dict[str, list[dict]] = got["mentions"]
            file_xfers: dict[str, list[dict]] = got["xfers"]
            t_extract = time.perf_counter() - got["sec_extract"]
            if schema_walk:
                discovered, dlog = discover_unknown_mentions(case, fp.name, cfgs, field_cache)
                discovery_log.extend(dlog)
                for etype, ms in discovered.items():
                    file_mentions[etype].extend(ms)
            attach_party_case_context(
                file_mentions["party"],
                file_mentions["judge"],
                file_mentions["firm"],
                case,
            )
            sec_extract = time.perf_counter() - t_extract

            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

            step_cascade: dict[str, dict] = {}
            t_resolve = time.perf_counter()
            # Within-file cascade only. Old mentions stay in the registry.
            for etype, cfg_path in DEFAULT_TYPE_CONFIGS.items():
                print(
                    f"\n--- incremental entity_type={etype} n_new={len(file_mentions[etype])} "
                    f"n_saved_entities={len(state['entities'][etype])} ---",
                    flush=True,
                )
                resolved = inc.resolve_within_file(
                    file_mentions[etype], file_xfers[etype], cfgs[etype], work_dir
                )
                cfg = load_config_cached(cfg_path)
                journal = link_journal
                saved, fresh, link_stats = inc.link_against_saved(
                    resolved["entities"],
                    resolved["by_id"],
                    state["entities"][etype],
                    cfg,
                    embed_cache,
                    journal,
                    saved_index[etype],
                )
                fresh, state["next_serial"][etype] = inc.stamp_new_entities(
                    fresh, resolved["by_id"], prefixes[etype], state["next_serial"][etype]
                )
                state["entities"][etype] = saved + fresh
                state["mentions"][etype].extend(resolved["mentions"])
                saved_index[etype].mentions.update(resolved["by_id"])
                state["decisions"].extend(resolved["decisions"])
                step_cascade[etype] = {
                    "summary": resolved.get("summary"),
                    "n_mentions_new": len(resolved["mentions"]),
                    "n_entities_total": len(state["entities"][etype]),
                    "link": link_stats,
                    "poc_party_evidence": resolved.get("poc") or None,
                }
                if resolved.get("poc"):
                    poc_evidence_all.append({"step": step_i, "file": fp.name, **resolved["poc"]})
            link_journal.flush()  # one write per file
            sec_resolve = time.perf_counter() - t_resolve

            t_write = time.perf_counter()
            elapsed_file = time.perf_counter() - t_file
            timing = {
                "step": step_i,
                "file": fp.name,
                "sha256": digest,
                "sec_total": round(elapsed_file, 3),
                "sec_read": round(sec_read, 3),
                "sec_extract": round(sec_extract, 3),
                "sec_resolve": round(sec_resolve, 3),
                "sec_write": 0.0,
            }
            state["processed"].append({"file": fp.name, "sha256": digest, "step": step_i, "sec": timing["sec_total"]})
            processed_keys.add((fp.name, digest))
            state["timings"].append(timing)
            state["cascade_last"] = step_cascade
            final_entities = state["entities"]
            final_mentions = state["mentions"]
            final_cascade = step_cascade
            step_summaries.append({
                "step": step_i,
                "file": fp.name,
                "ucid": case.get("ucid"),
                "case_name": case.get("case_name"),
                "counts": {
                    etype: {
                        "file_mentions": len(file_mentions[etype]),
                        "cumulative_entities": len(state["entities"][etype]),
                        "cumulative_mentions": len(state["mentions"][etype]),
                    }
                    for etype in ("judge", "firm", "party")
                },
                "link": {etype: step_cascade[etype]["link"] for etype in ("judge", "firm", "party")},
                "timing": timing,
            })
            state["summary_partial"] = {
                "output_dir": str(out_root),
                "incremental": True,
                "processed_files": len(state["processed"]),
                "timings": state["timings"],
            }
            if args.checkpoint_every and len(state["processed"]) % args.checkpoint_every == 0:
                inc.save_checkpoint(out_root, state)
            timing["sec_write"] = round(time.perf_counter() - t_write, 3)
            print(
                f"FILE_SEC step={step_i} file={fp.name} sec={timing['sec_total']:.3f} "
                f"read={timing['sec_read']:.3f} extract={timing['sec_extract']:.3f} "
                f"resolve={timing['sec_resolve']:.3f}",
                flush=True,
            )
            print(f"CHECKPOINT file_done={step_i} name={fp.name}", flush=True)

        link_journal.flush()
        # One checkpoint save for the whole run (IDs and serials for later files).
        state["summary_partial"] = {
            "output_dir": str(out_root),
            "incremental": True,
            "processed_files": len(state["processed"]),
            "timings": state["timings"],
        }
        t_ck = time.perf_counter()
        inc.save_checkpoint(out_root, state)
        print(f"CHECKPOINT saved once: {len(state['processed'])} files in {time.perf_counter() - t_ck:.2f}s", flush=True)
        final_entities = state["entities"]
        final_mentions = state["mentions"]
        final_by_id = {}
        for etype in ("judge", "firm", "party"):
            for m in final_mentions[etype]:
                final_by_id[m["mention_id"]] = m
        id_report = {
            "cold_start": not bool(processed_keys),
            "incremental": True,
            "mode": "match_new_file_against_saved_registry",
        }

        decisions = list(state['decisions'])
        cross_final = {
            etype: entity_cross_file_report(final_entities[etype], {m["mention_id"]: m for m in final_mentions[etype]})
            for etype in ("judge", "firm", "party")
        }
        summary = {
            "output_dir": str(out_root),
            "cold_start": bool(id_report.get("cold_start", True)),
            "id_remap": id_report,
            "debug_steps": bool(args.debug_steps),
            "elapsed_sec": round(time.time() - t_start, 2),
            "files": [
                {"step": s["step"], "file": s["file"], "ucid": s.get("ucid"), "case_name": s.get("case_name")}
                for s in step_summaries
            ],
            "counts": {
                etype: {
                    "mentions": len(final_mentions[etype]),
                    "entities": len(final_entities[etype]),
                    "cross_file_entities": len(cross_final[etype]),
                }
                for etype in ("judge", "firm", "party")
            },
            "cross_file_entities": cross_final,
            "embedding_calls": dict(EMBED_STATS),
            "llm_prompt_cache": dict(LLM_MEMO_STATS),
            "per_file_cumulative": step_summaries,
            "cascade_final": final_cascade,
            "poc_party_evidence": poc_evidence_all,
            "discovery": {
                "schema_free_walk": schema_walk,
                "n_events": len(discovery_log),
                "kept": sum(1 for x in discovery_log if x.get("reason") == "discovered_kept"),
                "rejected": sum(
                    1
                    for x in discovery_log
                    if str(x.get("reason") or "").startswith("discovery_")
                ),
            },
            "artifacts": {
                "entities": "entities.jsonl",
                "mentions": "mentions.jsonl",
                "decisions": "decisions.jsonl",
                "summary": "summary.json",
            },
        }
        final_entities = {
            et: [{k: v for k, v in e.items() if k != "_proto"} for e in state["entities"][et]]
            for et in ("judge", "firm", "party")
        }
        final_mentions = state["mentions"]
        decisions = list(state["decisions"])
        write_final_bundle(
            out_root,
            entities_by_type=final_entities,
            mentions_by_type=final_mentions,
            decisions=decisions,
            summary=summary,
        )
        if args.debug_steps:
            if discovery_log:
                write_jsonl(out_root / "discovery_log.jsonl", discovery_log)
            if cache_path.is_file():
                shutil.copy2(cache_path, out_root / "field_type_cache.jsonl")

        print(f"\nRESULTS_WRITTEN {out_root}/entities.jsonl", flush=True)
        print(f"RESULTS_WRITTEN {out_root}/mentions.jsonl", flush=True)
        print(f"RESULTS_WRITTEN {out_root}/decisions.jsonl", flush=True)
        print(f"RESULTS_WRITTEN {out_root}/summary.json", flush=True)
        return 0
    finally:
        # Remove cascade scratch unless debug-steps requested a copy already
        if cache_path.exists() and not args.debug_steps:
            cache_path.unlink(missing_ok=True)
        if work_dir.exists() and not args.debug_steps:
            shutil.rmtree(work_dir, ignore_errors=True)
        elif work_dir.exists() and args.debug_steps:
            # Keep last scratch as _work for debug convenience
            final_work = out_root / "_work_last_step"
            if final_work.exists():
                shutil.rmtree(final_work, ignore_errors=True)
            try:
                work_dir.rename(final_work)
            except OSError:
                shutil.copytree(work_dir, final_work, dirs_exist_ok=True)
                shutil.rmtree(work_dir, ignore_errors=True)
            if cache_path.exists():
                cache_path.unlink(missing_ok=True)


def build_results_md(
    *,
    out_root: Path,
    step_summaries: list[dict],
    final_dir: Path,
    cfgs: dict[str, dict],
    discovery_log: list[dict],
    elapsed_sec: float,
) -> str:
    """Legacy markdown report (unused by default; kept for --debug-steps callers)."""
    lines: list[str] = []
    lines.append("# Unified PoC — results")
    lines.append("")
    lines.append(f"- Run dir: `{out_root}`")
    lines.append(f"- Elapsed: {elapsed_sec:.1f}s")
    lines.append("")
    for i, s in enumerate(step_summaries, 1):
        lines.append(f"{i}. `{s['file']}` — {s.get('ucid')}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
