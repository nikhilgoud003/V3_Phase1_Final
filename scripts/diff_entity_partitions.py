#!/usr/bin/env python3
"""List entities that a rule change merged or split, between two runs.

A "new merge" is an AFTER entity whose mentions sat in 2+ BEFORE entities.
A "new split" is a BEFORE entity whose mentions sit in 2+ AFTER entities.
Usage: diff_entity_partitions.py BEFORE_RUN AFTER_RUN [--type party] [--json out.json]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def partition(run: Path, etype: str | None) -> tuple[dict[str, str], dict[str, dict]]:
    m2e: dict[str, str] = {}
    ents: dict[str, dict] = {}
    for e in load(run / "entities.jsonl"):
        if etype and e.get("entity_type") != etype:
            continue
        ents[e["entity_id"]] = e
        for mid in e.get("mention_ids") or []:
            m2e[mid] = e["entity_id"]
    return m2e, ents


def describe(eid: str, mids: list[str], ments: dict[str, dict]) -> dict:
    ms = [ments[m] for m in mids if m in ments]
    return {
        "entity_id": eid,
        "names": sorted({m.get("raw_name") for m in ms}),
        "courts": sorted({m.get("court") for m in ms if m.get("court")}),
        "cases": sorted({m.get("ucid") for m in ms if m.get("ucid")}),
        "n_mentions": len(ms),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--type", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    before, after = Path(args.before), Path(args.after)
    b_m2e, _ = partition(before, args.type)
    a_m2e, _ = partition(after, args.type)
    ments = {m["mention_id"]: m for m in load(after / "mentions.jsonl")}
    ments.update({m["mention_id"]: m for m in load(before / "mentions.jsonl") if m["mention_id"] not in ments})

    a_groups: dict[str, list[str]] = collections.defaultdict(list)
    for mid, eid in a_m2e.items():
        a_groups[eid].append(mid)
    b_groups: dict[str, list[str]] = collections.defaultdict(list)
    for mid, eid in b_m2e.items():
        b_groups[eid].append(mid)

    merges = []
    for eid, mids in a_groups.items():
        olds = collections.defaultdict(list)
        for mid in mids:
            olds[b_m2e.get(mid, "<new mention>")].append(mid)
        if len(olds) > 1:
            d = describe(eid, mids, ments)
            d["before_entities"] = [describe(o, ms, ments) for o, ms in sorted(olds.items())]
            merges.append(d)
    splits = []
    for eid, mids in b_groups.items():
        news = collections.defaultdict(list)
        for mid in mids:
            news[a_m2e.get(mid, "<dropped mention>")].append(mid)
        if len(news) > 1:
            d = describe(eid, mids, ments)
            d["after_entities"] = [describe(n, ms, ments) for n, ms in sorted(news.items())]
            splits.append(d)

    out = {
        "n_entities_before": len(b_groups),
        "n_entities_after": len(a_groups),
        "new_merges": merges,
        "new_splits": splits,
        "mentions_only_before": sorted(set(b_m2e) - set(a_m2e))[:50],
        "mentions_only_after": sorted(set(a_m2e) - set(b_m2e))[:50],
    }
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"entities {out['n_entities_before']} -> {out['n_entities_after']}; "
          f"new merges {len(merges)}; new splits {len(splits)}; "
          f"mentions only before {len(set(b_m2e) - set(a_m2e))}, only after {len(set(a_m2e) - set(b_m2e))}")
    for d in merges:
        print(f"\nMERGE {d['entity_id']}: {d['names']} | courts {d['courts']} | cases {d['cases']}")
        for b in d["before_entities"]:
            print(f"   was {b['entity_id']}: {b['names']} courts {b['courts']} cases {b['cases']}")
    for d in splits:
        print(f"\nSPLIT {d['entity_id']}: {d['names']}")
        for a in d["after_entities"]:
            print(f"   now {a['entity_id']}: {a['names']} courts {a['courts']} cases {a['cases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
