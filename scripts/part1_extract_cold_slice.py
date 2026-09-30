#!/usr/bin/env python3
"""Cold-isolated Part 1 extract comparison: separate arm FIRST, then unified.

Two fresh TIER_V3_OUTPUT_DIR trees — both make live Ollama LLM name-validity calls.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse compare helpers without treating scripts/ as a package
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "part1_extract_regression",
    ROOT / "scripts/part1_extract_regression.py",
)
_reg = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_reg)
compare_type = _reg.compare_type
compare_quarantine = _reg.compare_quarantine

from engine.extract import extract_mentions, extract_mentions_unified  # noqa: E402


def _fresh(name: str) -> Path:
    p = ROOT / "data/runs" / name
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    (p / "decisions").mkdir()
    (p / "mentions").mkdir()
    return p


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--tag", default="cold40")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.is_absolute():
        json_dir = ROOT / json_dir
    n_json = len(list(json_dir.glob("*.json")))
    print(f"json_dir={json_dir} n_files={n_json}", flush=True)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from engine import extract as extract_mod
    from engine import config_loader as cl
    from engine import name_validity as nv_mod
    from engine import llm_name_validity as llm_nv_mod

    real_load = cl.load_config
    real_wq = nv_mod.write_quarantine
    real_apply = llm_nv_mod.apply_llm_name_validity

    def load_prod(path):
        return real_load(path)  # LLM validity left as in YAML

    quarantine: dict[str, list] = {}
    llm_stats: dict[str, dict] = {}

    def capture(label: str, fn):
        rows: list = []
        box: dict = {}

        def wq(path, r, _real=real_wq):
            rows.clear()
            rows.extend(list(r or []))
            return _real(path, r)

        def apply_wrap(*a, **kw):
            kept, q, st = real_apply(*a, **kw)
            box.clear()
            box.update(st or {})
            return kept, q, st

        nv_mod.write_quarantine = wq  # type: ignore
        extract_mod.write_quarantine = wq  # type: ignore
        llm_nv_mod.apply_llm_name_validity = apply_wrap  # type: ignore
        cl.load_config = load_prod  # type: ignore
        extract_mod.load_config = load_prod  # type: ignore
        try:
            out = fn()
        finally:
            nv_mod.write_quarantine = real_wq  # type: ignore
            extract_mod.write_quarantine = real_wq  # type: ignore
            llm_nv_mod.apply_llm_name_validity = real_apply  # type: ignore
            cl.load_config = real_load  # type: ignore
            extract_mod.load_config = real_load  # type: ignore
        quarantine[label] = list(rows)
        llm_stats[label] = dict(box)
        print(f"{label}: quarantine={len(rows)} llm={json.dumps(box)}", flush=True)
        return out

    # ---------- STEP 1: SEPARATE (must finish before unified) ----------
    out_sep = _fresh(f"part1_extract_llm_cold_separate_{args.tag}")
    os.environ["TIER_V3_OUTPUT_DIR"] = str(out_sep)
    print(f"\n===== STEP 1 SEPARATE (cold) OUTPUT_DIR={out_sep} =====", flush=True)
    t0 = time.perf_counter()
    separate: dict[str, list] = {}
    for etype, cfg in [
        ("judge", "configs/judges.yaml"),
        ("firm", "configs/firms.yaml"),
        ("party", "configs/parties.yaml"),
    ]:
        print(f"\n--- separate {etype} ---", flush=True)
        if etype == "judge":
            separate[etype] = capture(
                "separate_judge",
                lambda c=cfg: extract_mentions(c, json_dir=str(json_dir), write=False),
            )
        else:
            separate[etype] = extract_mentions(cfg, json_dir=str(json_dir), write=False)
        print(f"separate {etype}: n={len(separate[etype])}", flush=True)

    sep_sec = round(time.perf_counter() - t0, 2)
    sep_calls = int((llm_stats.get("separate_judge") or {}).get("llm_calls") or 0)
    sep_cache = int((llm_stats.get("separate_judge") or {}).get("cache_hits") or 0)
    print(
        f"\nSTEP1_COMPLETE separate_sec={sep_sec} "
        f"judge={len(separate['judge'])} firm={len(separate['firm'])} "
        f"party={len(separate['party'])} quarantine={len(quarantine.get('separate_judge') or [])} "
        f"llm_calls={sep_calls} cache_hits={sep_cache}",
        flush=True,
    )
    if sep_calls <= 0:
        print("ERROR: separate arm made 0 live LLM calls — not a valid cold test", file=sys.stderr)
        return 1

    # ---------- STEP 2: UNIFIED (only after step 1) ----------
    out_uni = _fresh(f"part1_extract_llm_cold_unified_{args.tag}")
    os.environ["TIER_V3_OUTPUT_DIR"] = str(out_uni)
    print(f"\n===== STEP 2 UNIFIED (cold) OUTPUT_DIR={out_uni} =====", flush=True)
    t1 = time.perf_counter()
    unified = capture(
        "unified",
        lambda: extract_mentions_unified(str(json_dir), write=False),
    )
    uni_sec = round(time.perf_counter() - t1, 2)
    uni_calls = int((llm_stats.get("unified") or {}).get("llm_calls") or 0)
    uni_cache = int((llm_stats.get("unified") or {}).get("cache_hits") or 0)
    print(
        f"\nSTEP2_COMPLETE unified_sec={uni_sec} "
        f"judge={len(unified.get('judge') or [])} firm={len(unified.get('firm') or [])} "
        f"party={len(unified.get('party') or [])} quarantine={len(quarantine.get('unified') or [])} "
        f"llm_calls={uni_calls} cache_hits={uni_cache}",
        flush=True,
    )
    if uni_calls <= 0:
        print("ERROR: unified arm made 0 live LLM calls — not a valid cold test", file=sys.stderr)
        return 1

    # ---------- COMPARE ----------
    report = {
        "json_dir": str(json_dir),
        "n_json": n_json,
        "llm_name_validity": "enabled_production_cold_isolated_sequential",
        "separate_output_dir": str(out_sep),
        "unified_output_dir": str(out_uni),
        "phase_times_sec": {"separate_all": sep_sec, "unified_all": uni_sec},
        "llm_stats": llm_stats,
        "cold_live_calls": {
            "separate_llm_calls": sep_calls,
            "separate_cache_hits": sep_cache,
            "unified_llm_calls": uni_calls,
            "unified_cache_hits": uni_cache,
        },
        "types": [],
        "quarantine": {},
    }
    ok = True
    for etype in ("judge", "firm", "party"):
        rep = compare_type(etype, separate[etype], unified.get(etype) or [])
        report["types"].append(rep)
        print(
            f"{'PASS' if rep['ok'] else 'FAIL'} {etype}: "
            f"{rep['count_a']} vs {rep['count_b']} sha_match={rep['sha_match']}"
        )
        ok = ok and rep["ok"]

    qrep = compare_quarantine(
        quarantine.get("separate_judge") or [],
        quarantine.get("unified") or [],
    )
    report["quarantine"] = qrep
    print(
        f"{'PASS' if qrep['ok'] else 'FAIL'} quarantine: "
        f"{qrep['count_a']} vs {qrep['count_b']} sha_match={qrep['sha_match']}"
    )
    ok = ok and qrep["ok"]
    report["ok"] = ok
    report["wall_clock_sec"] = round(sep_sec + uni_sec, 2)

    out = Path(args.report_out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("PART1_COLD_SLICE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
