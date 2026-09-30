#!/usr/bin/env python3
"""Phase D harness — one-line re-point to any Phase C run.

Runs, in order:
  1. ungameable cross-surname metric
  2. structural invariants
  3. synthetic split/merge probes
  4. FJC-anchored purity / cohesion eval
  5. incremental holdout (Tier0–2 only; no LLM)

Usage:
  python3 scripts/phase_d_harness.py data/runs/final_v3_14b_colab
  python3 scripts/phase_d_harness.py --run-dir data/runs/gatefix4_v3_7b_20260805 --skip-holdout
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, env: dict | None = None) -> dict:
    print("\n>>>", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    print(out[-4000:] if len(out) > 4000 else out, flush=True)
    return {"cmd": cmd, "returncode": proc.returncode, "tail": out[-2000:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--run-dir", dest="run_dir_opt", default=None)
    ap.add_argument("--skip-holdout", action="store_true")
    ap.add_argument("--skip-probes", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir_opt or args.run_dir or "data/runs/final_v3_14b_colab")
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    assert run_dir.exists(), f"missing run dir: {run_dir}"

    env = os.environ.copy()
    env["TIER_V3_OUTPUT_DIR"] = str(run_dir)
    env["TIER_V3_FAISS_THREADS"] = env.get("TIER_V3_FAISS_THREADS", "1")
    # Holdout / probes: skip LLM; use rapidfuzz Tier2 for speed + determinism.
    env.setdefault("TIER_V3_TIER2_BACKEND", "rapidfuzz")

    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {"run_dir": str(run_dir)}

    steps = [
        (
            "cross_surname",
            [sys.executable, "scripts/check_cross_surname.py", str(run_dir)],
        ),
        (
            "invariants",
            [sys.executable, "tests/test_invariants.py", "--run-dir", str(run_dir)],
        ),
    ]
    if not args.skip_probes:
        steps.append(
            ("probes", [sys.executable, "tests/test_probes.py"])
        )
    steps.append(
        (
            "fjc_anchored",
            [sys.executable, "scripts/eval_fjc_anchored.py", "--run-dir", str(run_dir)],
        )
    )
    if not args.skip_holdout:
        scratch = run_dir / "phase_d_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        hold_env = env.copy()
        hold_env["TIER_V3_OUTPUT_DIR"] = str(scratch)
        # Keep cascade journals out of the baseline run folder.
        steps.append(
            ("holdout", [sys.executable, "scripts/run_incremental_holdout.py"], hold_env)
        )

    overall = True
    for item in steps:
        if len(item) == 2:
            name, cmd = item
            step_env = env
        else:
            name, cmd, step_env = item
        r = run(cmd, env=step_env)
        results[name] = {"returncode": r["returncode"], "ok": r["returncode"] == 0}
        if r["returncode"] != 0:
            overall = False

    # Normalize step list for the markdown table
    step_names = [s[0] for s in steps]

    results["pass"] = overall
    out = reports / "phase_d_harness.json"
    out.write_text(json.dumps(results, indent=2))
    md = [
        f"# Phase D harness — `{run_dir.name}`",
        "",
        f"**Overall:** {'PASS' if overall else 'FAIL'}",
        "",
        "| Step | Result |",
        "|------|--------|",
    ]
    for name in step_names:
        ok = results[name]["ok"]
        md.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    md.append("")
    md.append(f"Machine-readable: `{out}`")
    (reports / "phase_d_harness.md").write_text("\n".join(md) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
