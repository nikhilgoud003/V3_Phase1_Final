#!/usr/bin/env python3
"""Hard pre-insert URI check against a LIVE Tentris SPARQL endpoint.

Exit 1 on collision. No --force / skip flag. Does not read a TTL file on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.kg_insert_safety import InsertSafetyError, check_assignments_against_sparql


def main() -> int:
    p = argparse.ArgumentParser(
        description="Abort insert if CREATE/REUSE URIs are inconsistent with the live SPARQL graph"
    )
    p.add_argument("--assignments", required=True, help="*_sjid_assignments.json from registry-aware emit")
    p.add_argument(
        "--sparql-endpoint",
        required=True,
        help="Live Tentris SPARQL URL, e.g. http://127.0.0.1:9081/sparql",
    )
    p.add_argument("--report-out", default=None)
    args = p.parse_args()

    try:
        report = check_assignments_against_sparql(
            assignment_path=args.assignments,
            sparql_endpoint=args.sparql_endpoint,
        )
    except InsertSafetyError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    skip_keys = ("create_uri_already_in_target", "reuse_uri_missing_from_target")
    print(json.dumps({k: v for k, v in report.items() if k not in skip_keys}, indent=2))
    print("PASS: CREATE URIs are new; REUSE URIs exist in the live graph.")
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
