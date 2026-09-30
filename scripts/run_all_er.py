#!/usr/bin/env python3
"""One-shot entity resolution for judges + firms + parties on the same JSON batch.

Professor / operator expectation: point at PACER JSONs once → full ER for all
entity types (no per-config re-run).

Examples:
  python3 scripts/run_all_er.py --json-dir data/json/nyed_connectivity_test

  python3 scripts/run_all_er.py \\
    --json-dir data/json/nyed_connectivity_test \\
    --output-dir data/runs/nyed_connectivity_test_rerun \\
    --load-tentris-clone
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENTITY_TYPES = (
    ("judges", "configs/judges.yaml"),
    ("firms", "configs/firms.yaml"),
    ("parties", "configs/parties.yaml"),
)

DEFAULT_JUDGES_REGISTRY = (
    ROOT / "data/runs/judges_pilot_recall_fix/clusters/judges_entity_registry.jsonl"
)
LIVE_TENTRIS = ROOT / "data/tentris_judges_recall_fix_data"


def _run(cmd: list[str], env: dict[str, str]) -> int:
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72, flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def _tentris_bin() -> str:
    which = shutil.which("tentris")
    if which:
        return which
    home = Path.home() / ".local/bin/tentris"
    if home.is_file():
        return str(home)
    raise FileNotFoundError(
        "tentris not found on PATH or ~/.local/bin/tentris — install/fix PATH first"
    )


def load_into_clone(run_root: Path, clone_path: Path, port: int) -> None:
    """Clone live KG, load this batch's TTLs, serve on port (blocking print)."""
    tentris = _tentris_bin()
    if not LIVE_TENTRIS.is_dir():
        raise FileNotFoundError(f"live Tentris missing: {LIVE_TENTRIS}")

    if clone_path.exists():
        shutil.rmtree(clone_path)
    print(f"\nCloning {LIVE_TENTRIS} → {clone_path}")
    subprocess.check_call(["rsync", "-a", f"{LIVE_TENTRIS}/", f"{clone_path}/"])
    subprocess.check_call(["chmod", "-R", "700", str(clone_path)])

    for etype, _ in ENTITY_TYPES:
        ttl = run_root / etype / "rdf" / f"{etype}.ttl"
        if not ttl.is_file():
            print(f"WARNING: skip load, missing {ttl}")
            continue
        cmd = [
            tentris,
            "--datastore-path",
            str(clone_path),
            "load",
            "--format",
            "turtle",
            str(ttl),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.check_call(cmd)

    print(
        f"\nTTLs loaded into clone. Start Tentris with:\n"
        f"  {tentris} --datastore-path {clone_path} serve 127.0.0.1:{port}\n"
        f"Then SPARQL: http://127.0.0.1:{port}/sparql\n"
        f"(Live store {LIVE_TENTRIS} was not modified.)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full Tier_V3 ER (judges + firms + parties) on one JSON folder"
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Folder of PACER *.json dockets (upload destination)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Parent run dir (default: data/runs/<json-dir-name>_all_er)",
    )
    parser.add_argument(
        "--types",
        default="judges,firms,parties",
        help="Comma list subset, default all three",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max dockets (smoke)")
    parser.add_argument("--skip-tier3", action="store_true")
    parser.add_argument(
        "--target-registry",
        default=None,
        help="Judges registry for REUSE/CREATE (default: judges pilot registry if present)",
    )
    parser.add_argument(
        "--no-judges-registry",
        action="store_true",
        help="Mint fresh judge IDs; do not pass --target-registry",
    )
    parser.add_argument(
        "--load-tentris-clone",
        action="store_true",
        help="After ER: clone live Tentris and load this batch's TTLs (not live write)",
    )
    parser.add_argument(
        "--tentris-clone-path",
        default=None,
        help="Clone datastore path (default: data/tentris_<output-name>_clone)",
    )
    parser.add_argument("--tentris-port", type=int, default=9081)
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.is_absolute():
        json_dir = ROOT / json_dir
    if not json_dir.is_dir():
        print(f"ERROR: json-dir not found: {json_dir}", file=sys.stderr)
        return 1
    n_json = len(list(json_dir.glob("*.json")))
    if n_json == 0:
        print(f"ERROR: no *.json in {json_dir}", file=sys.stderr)
        return 1

    if args.output_dir:
        run_root = Path(args.output_dir)
        if not run_root.is_absolute():
            run_root = ROOT / run_root
    else:
        run_root = ROOT / "data" / "runs" / f"{json_dir.name}_all_er"

    want = {t.strip() for t in args.types.split(",") if t.strip()}
    unknown = want - {e for e, _ in ENTITY_TYPES}
    if unknown:
        print(f"ERROR: unknown --types: {unknown}", file=sys.stderr)
        return 1

    judges_registry = args.target_registry
    if judges_registry is None and not args.no_judges_registry and DEFAULT_JUDGES_REGISTRY.is_file():
        judges_registry = str(DEFAULT_JUDGES_REGISTRY)

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    path = env.get("PATH", "")
    local_bin = str(Path.home() / ".local/bin")
    if local_bin not in path.split(":"):
        env["PATH"] = f"{local_bin}:{path}"

    started = time.time()
    results: dict[str, object] = {
        "json_dir": str(json_dir),
        "n_json": n_json,
        "output_dir": str(run_root),
        "types": [],
        "ok": True,
    }

    print(
        f"One-shot ER: {n_json} JSON files → {run_root}\n"
        f"Entity types: {', '.join(t for t, _ in ENTITY_TYPES if t in want)}"
    )

    for etype, cfg in ENTITY_TYPES:
        if etype not in want:
            continue
        out = run_root / etype
        out.mkdir(parents=True, exist_ok=True)
        env["TIER_V3_OUTPUT_DIR"] = str(out)

        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_pilot.py"),
            "--config",
            cfg,
            "--json-dir",
            str(json_dir),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.skip_tier3:
            cmd.append("--skip-tier3")
        if etype == "judges" and judges_registry:
            cmd.extend(["--target-registry", judges_registry])

        rc = _run(cmd, env)
        entry = {"type": etype, "output": str(out), "exit_code": rc}
        results["types"].append(entry)  # type: ignore[attr-defined]
        if rc != 0:
            results["ok"] = False
            print(f"ERROR: {etype} failed with exit {rc}", file=sys.stderr)
            break

    if results["ok"] and args.load_tentris_clone:
        clone = (
            Path(args.tentris_clone_path)
            if args.tentris_clone_path
            else ROOT / "data" / f"tentris_{run_root.name}_clone"
        )
        if not clone.is_absolute():
            clone = ROOT / clone
        try:
            load_into_clone(run_root, clone, args.tentris_port)
            results["tentris_clone"] = str(clone)
        except Exception as e:
            results["ok"] = False
            results["tentris_error"] = f"{type(e).__name__}: {e}"
            print(f"ERROR Tentris clone/load: {e}", file=sys.stderr)

    results["elapsed_sec"] = round(time.time() - started, 1)
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "all_er_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    print(json.dumps(results, indent=2))
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
