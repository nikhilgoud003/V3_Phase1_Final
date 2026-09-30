#!/usr/bin/env python3
"""Part 1 regression: unified one-pass extract vs three separate extracts.

Reads the same JSON dir twice:
  A) three ``extract_mentions(config)`` calls (today's behavior)
  B) one ``extract_mentions_unified`` call (one filesystem pass)

Compares per-type mention lists for identity (count + content).

By default LLM name-validity is disabled (deterministic extract proof).
Pass ``--with-llm-name-validity`` to match production judges.yaml (LLM gate
on). Separate runs first so the shared ``TIER_V3_OUTPUT_DIR`` cache is warm
before unified — same production cache file both paths use.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.extract import extract_mentions, extract_mentions_unified


def _disable_llm_name_validity(cfg: dict) -> None:
    nv = cfg.setdefault("name_validity", {})
    llm = nv.setdefault("llm_validation", {})
    llm["enabled"] = False


def _canon_mention(m: dict) -> str:
    """Stable serialization for equality (sort keys; drop ephemeral if any)."""
    return json.dumps(m, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _fingerprint(mentions: list[dict]) -> str:
    lines = sorted(_canon_mention(m) for m in mentions)
    h = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return h


def _index_by_id(mentions: list[dict]) -> dict[str, dict]:
    out = {}
    for m in mentions:
        mid = m.get("mention_id")
        if mid in out:
            raise ValueError(f"duplicate mention_id {mid}")
        out[mid] = m
    return out


def compare_type(etype: str, a: list[dict], b: list[dict]) -> dict:
    rep: dict = {"entity_type": etype, "count_a": len(a), "count_b": len(b), "ok": True, "diffs": []}
    if len(a) != len(b):
        rep["ok"] = False
        rep["diffs"].append(f"count mismatch {len(a)} vs {len(b)}")
    ia, ib = _index_by_id(a), _index_by_id(b)
    only_a = sorted(set(ia) - set(ib))
    only_b = sorted(set(ib) - set(ia))
    if only_a or only_b:
        rep["ok"] = False
        rep["diffs"].append(f"id-only-a={len(only_a)} id-only-b={len(only_b)}")
        rep["sample_only_a"] = only_a[:5]
        rep["sample_only_b"] = only_b[:5]
    content_mismatches = 0
    samples = []
    for mid in sorted(set(ia) & set(ib)):
        ca, cb = _canon_mention(ia[mid]), _canon_mention(ib[mid])
        if ca != cb:
            content_mismatches += 1
            if len(samples) < 3:
                ka, kb = set(ia[mid]), set(ib[mid])
                field_diffs = []
                for k in sorted(ka | kb):
                    if ia[mid].get(k) != ib[mid].get(k):
                        field_diffs.append(k)
                samples.append({"mention_id": mid, "fields": field_diffs[:20]})
    if content_mismatches:
        rep["ok"] = False
        rep["diffs"].append(f"content mismatches={content_mismatches}")
        rep["sample_field_diffs"] = samples
    rep["sha256_a"] = _fingerprint(a)
    rep["sha256_b"] = _fingerprint(b)
    rep["sha_match"] = rep["sha256_a"] == rep["sha256_b"]
    if not rep["sha_match"]:
        rep["ok"] = False
    return rep


def compare_quarantine(a: list[dict], b: list[dict]) -> dict:
    """Compare quarantine rows by mention_id (+ content SHA of sorted rows)."""
    rep: dict = {
        "count_a": len(a),
        "count_b": len(b),
        "ok": True,
        "diffs": [],
    }
    if len(a) != len(b):
        rep["ok"] = False
        rep["diffs"].append(f"quarantine count mismatch {len(a)} vs {len(b)}")
    ia = {m.get("mention_id"): m for m in a}
    ib = {m.get("mention_id"): m for m in b}
    only_a = sorted(set(ia) - set(ib))
    only_b = sorted(set(ib) - set(ia))
    if only_a or only_b:
        rep["ok"] = False
        rep["diffs"].append(f"quarantine id-only-a={len(only_a)} id-only-b={len(only_b)}")
        rep["sample_only_a"] = only_a[:5]
        rep["sample_only_b"] = only_b[:5]
    mismatches = 0
    for mid in sorted(set(ia) & set(ib)):
        if _canon_mention(ia[mid]) != _canon_mention(ib[mid]):
            mismatches += 1
    if mismatches:
        rep["ok"] = False
        rep["diffs"].append(f"quarantine content mismatches={mismatches}")
    rep["sha256_a"] = _fingerprint(a)
    rep["sha256_b"] = _fingerprint(b)
    rep["sha_match"] = rep["sha256_a"] == rep["sha256_b"]
    if not rep["sha_match"]:
        rep["ok"] = False
    return rep


def main() -> int:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Part 1 unified vs separate extract regression")
    ap.add_argument(
        "--json-dir",
        default=str(ROOT / "data/json/nyed_connectivity_test"),
        help="PACER JSON directory",
    )
    ap.add_argument(
        "--report-out",
        default=None,
        help="Where to write JSON report (default under data/runs/)",
    )
    ap.add_argument(
        "--with-llm-name-validity",
        action="store_true",
        help="Keep production judges LLM name-validity enabled",
    )
    ap.add_argument(
        "--cold-isolated-caches",
        action="store_true",
        help=(
            "With --with-llm-name-validity: use SEPARATE empty OUTPUT_DIRs for "
            "separate vs unified arms so both make live Ollama calls (no shared warm cache)"
        ),
    )
    args = ap.parse_args()

    if args.cold_isolated_caches and not args.with_llm_name_validity:
        print("ERROR: --cold-isolated-caches requires --with-llm-name-validity", file=sys.stderr)
        return 1

    json_dir = Path(args.json_dir)
    if not json_dir.is_absolute():
        json_dir = ROOT / json_dir
    if not json_dir.is_dir():
        print(f"ERROR: missing {json_dir}", file=sys.stderr)
        return 1
    n_json = len(list(json_dir.glob("*.json")))
    if args.with_llm_name_validity and args.cold_isolated_caches:
        llm_mode = "enabled_production_cold_isolated_caches"
    elif args.with_llm_name_validity:
        llm_mode = "enabled_production_shared_warm_cache"
    else:
        llm_mode = "disabled_both_sides"
    print(f"JSON dir={json_dir} n_files={n_json} llm_name_validity={llm_mode}")

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    def _fresh_out_dir(name: str) -> Path:
        """Create an empty output dir (remove prior LLM cache/journal if present)."""
        import shutil

        p = ROOT / "data/runs" / name
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
        (p / "decisions").mkdir(parents=True, exist_ok=True)
        (p / "mentions").mkdir(parents=True, exist_ok=True)
        return p

    if args.cold_isolated_caches:
        out_separate = _fresh_out_dir(f"part1_extract_llm_cold_separate_{json_dir.name}")
        out_unified = _fresh_out_dir(f"part1_extract_llm_cold_unified_{json_dir.name}")
        print(f"SEPARATE TIER_V3_OUTPUT_DIR={out_separate} (cold)", flush=True)
        print(f"UNIFIED  TIER_V3_OUTPUT_DIR={out_unified} (cold)", flush=True)
        out_root = None
    else:
        out_root = ROOT / "data/runs" / f"part1_extract_llm_cmp_{json_dir.name}"
        out_root.mkdir(parents=True, exist_ok=True)
        os.environ["TIER_V3_OUTPUT_DIR"] = str(out_root)
        print(f"TIER_V3_OUTPUT_DIR={out_root} (shared cache/quarantine)", flush=True)
        out_separate = out_unified = out_root

    t0 = time.perf_counter()
    phase_times: dict[str, float] = {}
    quarantine_captures: dict[str, list[dict]] = {}
    llm_stats_captured: dict[str, dict] = {}

    from engine import extract as extract_mod
    from engine import config_loader as cl
    from engine import name_validity as nv_mod
    from engine import llm_name_validity as llm_nv_mod

    real_load = cl.load_config
    real_wq = nv_mod.write_quarantine
    real_apply_llm = llm_nv_mod.apply_llm_name_validity

    def _make_load_patched(disable_llm: bool):
        def load_patched(path, _real=real_load, _disable=disable_llm):
            cfg = _real(path)
            if _disable:
                _disable_llm_name_validity(cfg)
            return cfg

        return load_patched

    def _run_with_quarantine_capture(label: str, fn):
        captured: list[dict] = []
        llm_stats_box: dict = {}

        def wq_capture(path, rows, _real=real_wq):
            captured.clear()
            captured.extend(list(rows or []))
            return _real(path, rows)

        def apply_llm_wrap(*a, **kw):
            kept, q, stats = real_apply_llm(*a, **kw)
            llm_stats_box.clear()
            llm_stats_box.update(stats or {})
            return kept, q, stats

        nv_mod.write_quarantine = wq_capture  # type: ignore
        extract_mod.write_quarantine = wq_capture  # type: ignore
        llm_nv_mod.apply_llm_name_validity = apply_llm_wrap  # type: ignore
        extract_mod.apply_llm_name_validity = apply_llm_wrap  # type: ignore
        # _finalize_mentions imports apply_llm inside the function — patch module attr used after import
        try:
            result = fn()
        finally:
            nv_mod.write_quarantine = real_wq  # type: ignore
            extract_mod.write_quarantine = real_wq  # type: ignore
            llm_nv_mod.apply_llm_name_validity = real_apply_llm  # type: ignore
        quarantine_captures[label] = list(captured)
        llm_stats_captured[label] = dict(llm_stats_box)
        print(
            f"{label}: quarantined_captured={len(captured)} "
            f"llm_stats={json.dumps(llm_stats_box)}",
            flush=True,
        )
        return result

    disable_llm = not args.with_llm_name_validity
    load_patched = _make_load_patched(disable_llm)

    # Patch apply_llm inside extract._finalize_mentions's local import path:
    # extract.py does `from engine.llm_name_validity import apply_llm_name_validity`
    # INSIDE _finalize_mentions — so we must patch llm_name_validity module before each call.
    # (done in _run_with_quarantine_capture)

    # --- A: three separate extracts ---
    os.environ["TIER_V3_OUTPUT_DIR"] = str(out_separate)
    separate: dict[str, list[dict]] = {}
    for etype, cfg_path in [
        ("judge", "configs/judges.yaml"),
        ("firm", "configs/firms.yaml"),
        ("party", "configs/parties.yaml"),
    ]:
        print(f"\n=== SEPARATE extract: {etype} ===", flush=True)
        print(f"  OUTPUT_DIR={os.environ['TIER_V3_OUTPUT_DIR']}", flush=True)
        cl.load_config = load_patched  # type: ignore
        extract_mod.load_config = load_patched  # type: ignore
        t_phase = time.perf_counter()
        try:
            if etype == "judge":
                separate[etype] = _run_with_quarantine_capture(
                    "separate_judge",
                    lambda: extract_mentions(cfg_path, json_dir=str(json_dir), write=False),
                )
            else:
                separate[etype] = extract_mentions(
                    cfg_path, json_dir=str(json_dir), write=False
                )
        finally:
            cl.load_config = real_load  # type: ignore
            extract_mod.load_config = real_load  # type: ignore
        phase_times[f"separate_{etype}"] = round(time.perf_counter() - t_phase, 2)
        print(
            f"separate {etype}: {len(separate[etype])} mentions "
            f"({phase_times[f'separate_{etype}']}s)",
            flush=True,
        )

    # --- B: unified one-pass ---
    os.environ["TIER_V3_OUTPUT_DIR"] = str(out_unified)
    print("\n=== UNIFIED extract (one JSON pass) ===", flush=True)
    print(f"  OUTPUT_DIR={os.environ['TIER_V3_OUTPUT_DIR']}", flush=True)
    if args.cold_isolated_caches:
        print("  (cold isolated cache — expect live Ollama calls, not cache hits)", flush=True)
    cl.load_config = load_patched  # type: ignore
    extract_mod.load_config = load_patched  # type: ignore
    t_phase = time.perf_counter()
    try:
        unified = _run_with_quarantine_capture(
            "unified_judge",
            lambda: extract_mentions_unified(str(json_dir), write=False),
        )
    finally:
        cl.load_config = real_load  # type: ignore
        extract_mod.load_config = real_load  # type: ignore
    phase_times["unified_all"] = round(time.perf_counter() - t_phase, 2)
    print(f"unified done ({phase_times['unified_all']}s)", flush=True)

    if "unified_judge" in quarantine_captures:
        quarantine_captures["unified"] = quarantine_captures.pop("unified_judge")
    if "unified_judge" in llm_stats_captured:
        llm_stats_captured["unified"] = llm_stats_captured.pop("unified_judge")

    elapsed = time.perf_counter() - t0

    report: dict = {
        "json_dir": str(json_dir),
        "n_json": n_json,
        "llm_name_validity": llm_mode,
        "shared_output_dir": str(out_root) if out_root else None,
        "separate_output_dir": str(out_separate),
        "unified_output_dir": str(out_unified),
        "cold_isolated_caches": bool(args.cold_isolated_caches),
        "llm_stats": llm_stats_captured,
        "wall_clock_sec": round(elapsed, 2),
        "phase_times_sec": phase_times,
        "types": [],
        "quarantine": {},
    }
    all_ok = True
    for etype in ("judge", "firm", "party"):
        rep = compare_type(etype, separate[etype], unified.get(etype) or [])
        report["types"].append(rep)
        status = "PASS" if rep["ok"] else "FAIL"
        print(f"\n{status} {etype}: counts {rep['count_a']} vs {rep['count_b']} sha_match={rep['sha_match']}")
        if rep["diffs"]:
            print("  diffs:", rep["diffs"])
            if rep.get("sample_field_diffs"):
                print("  sample:", rep["sample_field_diffs"])
        all_ok = all_ok and rep["ok"]

    q_sep = quarantine_captures.get("separate_judge") or []
    q_uni = quarantine_captures.get("unified") or []
    qrep = compare_quarantine(q_sep, q_uni)
    report["quarantine"] = qrep
    qstatus = "PASS" if qrep["ok"] else "FAIL"
    print(
        f"\n{qstatus} quarantine: counts {qrep['count_a']} vs {qrep['count_b']} "
        f"sha_match={qrep['sha_match']}"
    )
    if qrep["diffs"]:
        print("  diffs:", qrep["diffs"])
    all_ok = all_ok and qrep["ok"]

    # Cold-isolated: both arms must have made live Ollama calls (not zero via shared cache)
    if args.cold_isolated_caches:
        sep_calls = int((llm_stats_captured.get("separate_judge") or {}).get("llm_calls") or 0)
        uni_calls = int((llm_stats_captured.get("unified") or {}).get("llm_calls") or 0)
        sep_cache = int((llm_stats_captured.get("separate_judge") or {}).get("cache_hits") or 0)
        uni_cache = int((llm_stats_captured.get("unified") or {}).get("cache_hits") or 0)
        live_ok = sep_calls > 0 and uni_calls > 0
        report["cold_live_calls_ok"] = live_ok
        report["cold_live_calls"] = {
            "separate_llm_calls": sep_calls,
            "separate_cache_hits": sep_cache,
            "unified_llm_calls": uni_calls,
            "unified_cache_hits": uni_cache,
        }
        print(
            f"\nCold-cache live calls: separate llm_calls={sep_calls} cache_hits={sep_cache}; "
            f"unified llm_calls={uni_calls} cache_hits={uni_cache} → "
            f"{'OK' if live_ok else 'FAIL (expected both > 0 live calls)'}"
        )
        if not live_ok:
            all_ok = False

    report["ok"] = all_ok
    if args.report_out:
        out = Path(args.report_out)
        if not out.is_absolute():
            out = ROOT / out
    else:
        suffix = (
            "llm_cold"
            if args.cold_isolated_caches
            else ("llm_on" if args.with_llm_name_validity else "llm_off")
        )
        out = ROOT / "data/runs" / f"part1_extract_unified_regression_{json_dir.name}_{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"WALL_CLOCK_SEC={elapsed:.2f}")
    print("PART1_REGRESSION", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
