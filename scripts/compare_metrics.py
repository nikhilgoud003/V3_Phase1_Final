#!/usr/bin/env python3
"""Before/after table from two er_quality_metrics.py outputs.

Usage: compare_metrics.py BEFORE/metrics.json AFTER/metrics.json
"""

from __future__ import annotations

import json
import sys


def rows(m: dict) -> dict:
    calls = m.get("ollama_calls") or {}
    j, p = m["v3_judges"], m["v3_parties"]
    return {
        "Time (s)": m.get("elapsed_sec"),
        "Qwen calls (work)": calls.get("generate:work"),
        "All Ollama HTTP calls": calls.get("total_http_calls"),
        "Real judges found (distinct FJC judges)": j["fjc_judges"],
        "Judge entities": j["ids"],
        "FJC judge splits": j["fjc_splits"],
        "FJC wrong merges": j["fjc_wrong_merges"],
        "Party entities": p["ids"],
        "Same-name party splits (company/other)": f"{p['same_name_splits_org']}/{p['same_name_splits_person']}",
        "Co-defendant merges": p["coparty_wrong_merges"],
        "Placeholder ids across cases": p["placeholder_ids_across_cases"],
        "Junk judge-name entities": j["junk_name_entities"],
        "Junk walk mentions": m["v3_schema_walk"]["mentions"],
        "Party aliases": m["v3_party_aliases"]["mentions"],
    }


def main() -> int:
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    ra, rb = rows(a), rows(b)
    print(f"| Metric | Before | After |\n|---|---|---|")
    for k in ra:
        print(f"| {k} | {ra[k]} | {rb[k]} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
