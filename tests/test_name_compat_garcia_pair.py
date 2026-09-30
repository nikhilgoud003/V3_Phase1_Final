#!/usr/bin/env python3
"""Regression: Garcia/Garcia same-UCID homonym must not Tier2 auto-merge."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.name_compat import names_compatible
from engine.tiers import DecisionJournal, UnionFind, apply_tier2_auto_merges

CFG = {
    "entity_type": "judge",
    "version": "0.1.0-test",
    "name_compat": {"enabled": True, "max_surname_edit_distance": 1},
    "tier2": {"search": {"auto_merge_min": 0.92, "ambiguous_low": 0.72, "ambiguous_high": 0.92}},
    "information_content_barrier": {"enabled": False},
}


def _m(mid: str, name: str, *, ucid: str = "txsd;;5:16-cr-00022", court: str = "txsd") -> dict:
    toks = name.split()
    return {
        "mention_id": mid,
        "normalized_name": name,
        "presentable_name": name.title(),
        "surname": toks[-1] if toks else "",
        "token_count": len(toks),
        "ucid": ucid,
        "court": court,
        "role": "mentioned",
        "profile": name,
        "co_mentions": [],
    }


class TestGarciaHomonymGate(unittest.TestCase):
    def test_names_compatible_blocks_guillermo_vs_marina(self):
        a = _m("a", "guillermo r garcia")
        b = _m("b", "marina garcia")
        ok, reason = names_compatible(a, b, cfg=CFG)
        self.assertFalse(ok, reason)
        self.assertTrue(reason.startswith("incompatible_given:"), reason)

    def test_tier2_auto_merge_blocked_for_garcia_pair(self):
        a = _m("a", "guillermo r garcia")
        b = _m("b", "marina garcia")
        by_id = {a["mention_id"]: a, b["mention_id"]: b}
        uf = UnionFind()
        for m in (a, b):
            uf.add(m["mention_id"])

        with tempfile.TemporaryDirectory() as td:
            journal = DecisionJournal(Path(td) / "dec.jsonl")
            out = apply_tier2_auto_merges(
                [(a["mention_id"], b["mention_id"], 0.97)],
                by_id,
                uf,
                CFG,
                journal,
                set(),
            )
            stats = out["stats"]
            self.assertEqual(stats["auto_merges"], 0)
            self.assertEqual(stats["name_gate"], 1)
            self.assertNotEqual(uf.find(a["mention_id"]), uf.find(b["mention_id"]))
            rows = [json.loads(l) for l in Path(td).joinpath("dec.jsonl").read_text().splitlines() if l.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["decision"], "NO_MATCH")
            self.assertEqual(rows[0]["method"], "name_gate")
            self.assertIn("incompatible_given:guillermo|marina", rows[0]["evidence"]["name_gate_reason"])

    def test_truncated_marmolejo_still_compatible(self):
        a = _m("a", "garcia marmolejo")
        b = _m("b", "marina garcia marmolejo")
        ok, _ = names_compatible(a, b, cfg=CFG)
        self.assertTrue(ok)

    def test_guillermo_blocked_from_garcia_marmolejo_truncated_span(self):
        a = _m("a", "guillermo r garcia")
        b = _m("b", "garcia marmolejo")
        ok, reason = names_compatible(a, b, cfg=CFG)
        self.assertFalse(ok, reason)
        self.assertTrue(
            reason.startswith("truncated_span_missing_given:")
            or reason.startswith("incompatible_given:"),
            reason,
        )

    def test_initial_expansion_still_allowed(self):
        a = _m("a", "g r garcia")
        b = _m("b", "guillermo r garcia")
        ok, _ = names_compatible(a, b, cfg=CFG)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
