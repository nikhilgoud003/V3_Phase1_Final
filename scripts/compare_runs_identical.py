#!/usr/bin/env python3
"""Check whether two unified runs produced the same results.

Compares entities (id -> members and fields), mentions (all fields) and
decisions (as a multiset, ignoring timestamps and journal counters).
Usage: compare_runs_identical.py OLD_RUN NEW_RUN
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

VOLATILE = {"timestamp", "decision_id", "ts", "latency_sec", "elapsed_sec", "llm_latency_sec"}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in VOLATILE}


def is_name_validity_cache_row(d: dict) -> bool:
    # Rows copied from llm_name_validity_cache.jsonl (the old work_dir glob picked them up).
    return d.get("method") == "llm_name_validation" and "cache_key" in d and "prompt_hash" in d


def main() -> int:
    old, new = Path(sys.argv[1]), Path(sys.argv[2])
    report: dict = {}

    eo = {e["entity_id"]: strip(e) for e in load(old / "entities.jsonl")}
    en = {e["entity_id"]: strip(e) for e in load(new / "entities.jsonl")}
    ent_diff = [k for k in sorted(set(eo) | set(en)) if eo.get(k) != en.get(k)]
    report["entities"] = {"old": len(eo), "new": len(en), "differ": len(ent_diff), "examples": ent_diff[:10]}

    mo = {m["mention_id"]: strip(m) for m in load(old / "mentions.jsonl")}
    mn = {m["mention_id"]: strip(m) for m in load(new / "mentions.jsonl")}
    men_diff = [k for k in sorted(set(mo) | set(mn)) if mo.get(k) != mn.get(k)]
    fields = Counter()
    for k in men_diff:
        a, b = mo.get(k) or {}, mn.get(k) or {}
        for f in set(a) | set(b):
            if a.get(f) != b.get(f):
                fields[f] += 1
    report["mentions"] = {"old": len(mo), "new": len(mn), "differ": len(men_diff), "fields": dict(fields)}

    do = load(old / "decisions.jsonl")
    dn = load(new / "decisions.jsonl")
    cache_rows_old = sum(is_name_validity_cache_row(d) for d in do)
    cache_rows_new = sum(is_name_validity_cache_row(d) for d in dn)
    key = lambda d: json.dumps(strip(d), sort_keys=True, default=str)  # noqa: E731
    co = Counter(key(d) for d in do if not is_name_validity_cache_row(d))
    cn = Counter(key(d) for d in dn if not is_name_validity_cache_row(d))
    only_old = co - cn
    only_new = cn - co
    report["decisions"] = {
        "old": len(do),
        "new": len(dn),
        "name_validity_cache_rows_old": cache_rows_old,
        "name_validity_cache_rows_new": cache_rows_new,
        "differ_only_old": sum(only_old.values()),
        "differ_only_new": sum(only_new.values()),
        "methods_only_old": dict(Counter(json.loads(k).get("method") for k in only_old.elements())),
        "methods_only_new": dict(Counter(json.loads(k).get("method") for k in only_new.elements())),
    }

    so = json.loads((old / "checkpoint" / "state.json").read_text())
    sn = json.loads((new / "checkpoint" / "state.json").read_text())
    report["next_serial_same"] = so.get("next_serial") == sn.get("next_serial")
    report["identical"] = (
        not ent_diff and not men_diff and not only_old and not only_new and report["next_serial_same"]
    )
    print(json.dumps(report, indent=2))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
