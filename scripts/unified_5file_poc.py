#!/usr/bin/env python3
"""Unified one-pass ER — 5-file cold-start proof of concept.

Orchestration only. Does NOT modify:
  - configs/{judges,firms,parties}.yaml matching rules
  - engine/tiers.py Tier0–3 logic
  - engine/tier3_citation.py
  - live Tentris or pilot registries

Process (per file, in order):
  1. Read the PACER JSON once.
  2. Extract via existing YAML sources (known fields) + judges spaCy NER on docket.
  3. Walk remaining string leaves; for never-seen path patterns, one cached qwen
     call classifies field type (judge/firm/party/other); typed names become
     candidate mentions and still pass the type's finalize gates.
  4. Resolve cumulatively (mentions from files 1..k) with run_cascade + cluster.
  5. Write one final folder: entities.jsonl, mentions.jsonl, decisions.jsonl,
     summary.json. Per-file step_XX_* snapshots only with --debug-steps.

Cold start: no live pilot registries loaded.
"""

from __future__ import annotations

import copy
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
from engine.config_loader import load_config, resolve_path  # noqa: E402
from engine.discovery_validity import (  # noqa: E402
    force_type_other,
    path_is_non_entity,
    value_is_discovery_junk,
)
from engine.extract import (  # noqa: E402
    DEFAULT_TYPE_CONFIGS,
    _emit_mention,
    _finalize_mentions,
    extract_from_case,
)
from engine.normalize import normalize_name  # noqa: E402
from engine.poc_party_evidence import (  # noqa: E402
    adjudicate_poc_evidence_via_tier3,
    rebuild_components,
)
from engine.provenance import DecisionJournal  # noqa: E402
from engine.tiers import run_cascade  # noqa: E402

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
        help="Optional ordered list of filenames (default: the Part-B Bard 5-file set)",
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
    args = ap.parse_args()

    json_root = Path(args.json_dir)
    if not json_root.is_absolute():
        json_root = ROOT / json_root
    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.files:
        names = list(args.files)
    else:
        names = list(POC_FILES)
    files = [json_root / name for name in names]
    missing = [str(fp) for fp in files if not fp.is_file()]
    if missing:
        if args.files is None and not any((json_root / n).is_file() for n in POC_FILES):
            files = sorted(json_root.glob("*.json"))
            if args.limit:
                files = files[: args.limit]
            if not files:
                print(f"ERROR: no *.json in {json_root}", file=sys.stderr)
                return 1
        else:
            print("ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
            return 1
    if args.limit is not None:
        files = files[: args.limit]
    file_names = [fp.name for fp in files]

    cfgs: dict[str, dict] = {}
    for etype, cpath in DEFAULT_TYPE_CONFIGS.items():
        cfg = load_config(cpath)
        cfgs[cfg.get("entity_type") or etype] = cfg

    raw: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
    transfers: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
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

    final_entities: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
    final_mentions: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
    final_by_id: dict[str, dict] = {}
    final_cascade: dict[str, Any] = {}

    try:
        for step_i, fp in enumerate(files, start=1):
            step_dir: Path | None = None
            if args.debug_steps:
                step_dir = out_root / f"step_{step_i:02d}_{fp.stem}"
                step_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 72)
            print(f"STEP {step_i}/{n_files}  file={fp.name}")
            print("=" * 72, flush=True)

            with open(fp, encoding="utf-8") as f:
                case = json.load(f)

            ucid = case.get("ucid") or ""
            mdl_val = case.get("mdl_code")
            if mdl_val in (None, False, ""):
                mdl_val = None
            mdl_by_ucid[ucid] = mdl_val

            file_mentions: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
            for etype, cfg in cfgs.items():
                case_i = copy.deepcopy(case)
                mentions, xfers = extract_from_case(case_i, cfg, source_file=fp.name)
                for t in xfers:
                    t["ucid"] = case.get("ucid")
                    t["source_file"] = fp.name
                file_mentions[etype] = mentions
                raw[etype].extend(mentions)
                transfers[etype].extend(xfers)

            discovered, dlog = discover_unknown_mentions(case, fp.name, cfgs, field_cache)
            discovery_log.extend(dlog)
            for etype, ms in discovered.items():
                file_mentions[etype].extend(ms)
                raw[etype].extend(ms)

            attach_party_case_context(
                file_mentions["party"],
                file_mentions["judge"],
                file_mentions["firm"],
                case,
            )

            if step_dir is not None:
                write_jsonl(step_dir / "file_mentions_judge.jsonl", file_mentions["judge"])
                write_jsonl(step_dir / "file_mentions_firm.jsonl", file_mentions["firm"])
                write_jsonl(step_dir / "file_mentions_party.jsonl", file_mentions["party"])

            # Cascade writes under work_dir (single scratch; last step = final)
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TIER_V3_OUTPUT_DIR"] = str(work_dir)

            step_entities: dict[str, list[dict]] = {}
            step_by_id: dict[str, dict[str, dict]] = {}
            step_cascade: dict[str, dict] = {}

            for etype, cfg_path in DEFAULT_TYPE_CONFIGS.items():
                cfg = load_config(cfg_path)
                print(
                    f"\n--- finalize+cascade entity_type={etype} n_raw={len(raw[etype])} ---",
                    flush=True,
                )
                finalized = _finalize_mentions(
                    cfg,
                    list(raw[etype]),
                    list(transfers[etype]),
                    write=True,
                    transfer_out_rel=f"data/mentions/transfer_clues_{etype}.jsonl",
                )
                if etype == "party":
                    by_ucid_j: dict[str, list[dict]] = defaultdict(list)
                    by_ucid_f: dict[str, list[dict]] = defaultdict(list)
                    for m in raw["judge"]:
                        by_ucid_j[m.get("ucid") or ""].append(m)
                    for m in raw["firm"]:
                        by_ucid_f[m.get("ucid") or ""].append(m)
                    for m in finalized:
                        u = m.get("ucid") or ""
                        m["poc_case_judges"] = sorted(
                            {
                                x.get("normalized_name")
                                for x in by_ucid_j.get(u, [])
                                if x.get("normalized_name")
                            }
                        )
                        m["poc_case_firms"] = sorted(
                            {
                                x.get("normalized_name")
                                for x in by_ucid_f.get(u, [])
                                if x.get("normalized_name")
                            }
                        )
                        m["poc_mdl_code"] = mdl_by_ucid.get(u)
                        m["poc_is_mdl"] = bool(mdl_by_ucid.get(u))

                result = run_cascade(finalized, cfg, enable_tier3=True)
                uf = result["uf"]
                by_id = result["by_id"]
                poc_info: dict[str, Any] = {}

                if etype == "party":
                    journal = DecisionJournal(
                        resolve_path(cfg, "data/decisions/poc_party_evidence_decisions.jsonl"),
                        fresh=True,
                    )
                    poc_stats = adjudicate_poc_evidence_via_tier3(
                        finalized, uf, cfg, journal=journal, max_pairs=25
                    )
                    print(
                        f"PoC party evidence: candidates={poc_stats['candidates']} "
                        f"tier3_calls={poc_stats['tier3_calls']} "
                        f"merges_after_citation_MATCH={poc_stats['merges']}",
                        flush=True,
                    )
                    if poc_stats["merges"]:
                        comps = rebuild_components(uf, [m["mention_id"] for m in finalized])
                        entities = cluster_mentions(comps, by_id, cfg, uf)
                    else:
                        entities = cluster_mentions(result["components"], by_id, cfg, uf)
                    poc_info = {
                        "candidates": poc_stats["candidates"],
                        "tier3_calls": poc_stats["tier3_calls"],
                        "merges_after_citation_match": poc_stats["merges"],
                        "rows": poc_stats["rows"],
                    }
                    if step_dir is not None:
                        write_jsonl(
                            step_dir / "poc_party_tier3_adjudications.jsonl",
                            poc_stats["rows"],
                        )
                    poc_evidence_all.append({"step": step_i, "file": fp.name, **poc_info})
                else:
                    entities = cluster_mentions(result["components"], by_id, cfg, uf)

                step_entities[etype] = entities
                step_by_id[etype] = by_id
                step_cascade[etype] = {
                    "summary": result.get("summary"),
                    "n_mentions": len(finalized),
                    "n_entities": len(entities),
                    "poc_party_evidence": poc_info or None,
                }

            # Remember latest cumulative state for final bundle
            final_entities = step_entities
            final_mentions = {
                etype: list(step_by_id[etype].values()) for etype in ("judge", "firm", "party")
            }
            final_by_id = {}
            for etype in ("judge", "firm", "party"):
                final_by_id.update(step_by_id[etype])
            final_cascade = step_cascade

            cross = {
                etype: entity_cross_file_report(step_entities[etype], step_by_id[etype])
                for etype in ("judge", "firm", "party")
            }
            summary_step = {
                "step": step_i,
                "file": fp.name,
                "ucid": case.get("ucid"),
                "case_name": case.get("case_name"),
                "files_in_pool": file_names[:step_i],
                "counts": {
                    etype: {
                        "file_mentions": len(file_mentions[etype]),
                        "cumulative_raw": len(raw[etype]),
                        "cumulative_entities": len(step_entities[etype]),
                        "cross_file_entities": len(cross[etype]),
                    }
                    for etype in ("judge", "firm", "party")
                },
                "cascade": {
                    etype: {
                        "n_mentions": step_cascade[etype]["n_mentions"],
                        "n_entities": step_cascade[etype]["n_entities"],
                        "summary": step_cascade[etype].get("summary"),
                    }
                    for etype in ("judge", "firm", "party")
                },
                "poc_party_evidence": (poc_evidence_all[-1] if poc_evidence_all else None),
            }
            step_summaries.append(summary_step)
            print(json.dumps(summary_step["counts"], indent=2), flush=True)

            if step_dir is not None:
                write_jsonl(step_dir / "cross_file_entities_judge.jsonl", cross["judge"])
                write_jsonl(step_dir / "cross_file_entities_firm.jsonl", cross["firm"])
                write_jsonl(step_dir / "cross_file_entities_party.jsonl", cross["party"])
                (step_dir / "step_summary.json").write_text(
                    json.dumps(summary_step, indent=2, default=str), encoding="utf-8"
                )
                # Copy cascade scratch into step folder for inspection
                for sub in ("mentions", "clusters", "decisions"):
                    src = work_dir / sub
                    if src.is_dir():
                        dst = step_dir / sub
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)

        # --- Final single-folder outputs ---
        decisions = _collect_decision_rows(work_dir, final_by_id)
        cross_final = {
            etype: entity_cross_file_report(final_entities[etype], {m["mention_id"]: m for m in final_mentions[etype]})
            for etype in ("judge", "firm", "party")
        }
        summary = {
            "output_dir": str(out_root),
            "cold_start": True,
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
            "per_file_cumulative": step_summaries,
            "cascade_final": final_cascade,
            "poc_party_evidence": poc_evidence_all,
            "discovery": {
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
