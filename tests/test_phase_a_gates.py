#!/usr/bin/env python3
"""Phase A unit tests: name_gate, name_validity A2/A3, low-info A4, prompt hash."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from engine.name_compat import names_compatible, surnames_compatible
from engine.name_validity import classify_name_validity, build_corpus_name_rescue
from engine.mention_hygiene import apply_hygiene, pair_allowed, is_common_first_name_only
from engine.tiers import (
    DecisionJournal,
    UnionFind,
    apply_tier2_auto_merges,
    prompt_sha256,
    tier3_adjudicate,
)
from engine.cluster import cluster_mentions, _split_by_name_compat


CFG = {
    "entity_type": "judge",
    "version": "0.1.0-test",
    "name_compat": {"enabled": True, "max_surname_edit_distance": 1},
    "mention_hygiene": {"enabled": True, "low_info_within_ucid_only": True},
    "name_validity": {"enabled": True},
    "clustering": {
        "verify_merges": True,
        "large_cluster_verify_min_size": 5,
        "id_prefix": "SJ",
        "reject_if": ["transfer_clue_conflict", "both_have_distinct_fjc_nids"],
    },
    "tier2": {
        "search": {"auto_merge_min": 0.92, "ambiguous_low": 0.72, "ambiguous_high": 0.92}
    },
    "information_content_barrier": {"enabled": False},
    "tier3": {
        "enabled": True,
        "model": "test-model",
        "endpoint": "http://127.0.0.1:9",
        "prompt_path": "prompts/tier3_adjudicate.txt",
        "json_schema_path": "prompts/tier3_output.schema.json",
        "decision_cache": {"enabled": True, "path": "data/decisions/_test_cache.jsonl"},
        "routing": {"require_same_block": False},
        "budget": {"target_mention_fraction_max": 1.0},
        "abstain_confidence_below": 60,
        "barrier_llm_min_confidence": 80,
        "progress_log_every": 0,
    },
    "io": {
        "decisions_out": "data/decisions/_test_decisions.jsonl",
        "clusters_out": "data/clusters/_test_entities.jsonl",
    },
    "_config_path": str(ROOT / "configs" / "judges.yaml"),
    "_repo_root": str(ROOT),
}


def _m(mid, name, surname=None, ucid="u1", court="akd", **kw):
    toks = name.split()
    return {
        "mention_id": mid,
        "normalized_name": name,
        "presentable_name": name.title(),
        "surname": surname if surname is not None else (toks[-1] if toks else ""),
        "token_count": len(toks),
        "ucid": ucid,
        "court": court,
        "role": "assigned",
        "profile": name,
        "co_mentions": [],
        **kw,
    }


class TestNameCompatA1(unittest.TestCase):
    def test_surnames_incompatible_examples(self):
        for a, b in [
            ("mehalchick", "munley"),
            ("darrah", "der-yeghiayan"),
            ("anderson", "rose"),
        ]:
            self.assertFalse(surnames_compatible(a, b), f"{a}|{b}")

    def test_surnames_edit_distance_1_ok(self):
        self.assertTrue(surnames_compatible("smith", "smyth"))  # dist 1? s-m-i-t-h vs s-m-y-t-h = 1
        self.assertTrue(surnames_compatible("marten", "marten"))

    def test_cascade_zero_cross_surname_match(self):
        """Synthetic cascade: LLM would MATCH everything — name_gate must yield 0 MATCH."""
        pairs = [
            ("mehalchick", "munley"),
            ("darrah", "der-yeghiayan"),
            ("anderson", "rose"),
        ]
        mentions = []
        ambiguous = []
        for i, (sa, sb) in enumerate(pairs):
            a = _m(f"a{i}", f"karoline {sa}", surname=sa, ucid=f"u{i}")
            b = _m(f"b{i}", f"james {sb}", surname=sb, ucid=f"u{i}")
            mentions.extend([a, b])
            ambiguous.append((a["mention_id"], b["mention_id"], 0.85, False, []))

        by_id = {m["mention_id"]: m for m in mentions}
        uf = UnionFind()
        for m in mentions:
            uf.add(m["mention_id"])

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg = dict(CFG)
            cfg = json.loads(json.dumps(CFG))  # deep-ish copy
            cfg["tier3"] = dict(CFG["tier3"])
            cfg["tier3"]["decision_cache"] = {
                "enabled": True,
                "path": str(td / "cache.jsonl"),
            }
            cfg["io"] = {"decisions_out": str(td / "dec.jsonl"), "clusters_out": str(td / "ent.jsonl")}
            cfg["_repo_root"] = str(ROOT)
            # resolve_path uses cfg roots — set absolute prompt paths via monkeypatch load
            journal = DecisionJournal(td / "dec.jsonl")

            def fake_load_prompt(c):
                return (ROOT / "prompts" / "tier3_adjudicate.txt").read_text()

            def fake_schema(c):
                return json.loads((ROOT / "prompts" / "tier3_output.schema.json").read_text())

            def boom(*a, **k):
                raise AssertionError("LLM must not be called for cross-surname pairs")

            with mock.patch("engine.tiers.load_prompt", fake_load_prompt), mock.patch(
                "engine.tiers.load_output_schema", fake_schema
            ), mock.patch("engine.tiers.call_ollama_json", boom), mock.patch(
                "engine.tiers.resolve_path", lambda c, p: Path(p) if str(p).startswith("/") else ROOT / p
            ):
                # cache path absolute
                cfg["tier3"]["decision_cache"]["path"] = str(td / "cache.jsonl")
                stats = tier3_adjudicate(
                    ambiguous, by_id, uf, cfg, journal, {m["mention_id"]: ["b"] for m in mentions}, len(mentions)
                )

            rows = [json.loads(l) for l in (td / "dec.jsonl").read_text().splitlines() if l.strip()]
            matches = [r for r in rows if r["decision"] in {"MERGE_TIER3", "MATCH"}]
            self.assertEqual(matches, [], matches)
            self.assertEqual(stats["name_gate"], 3)
            self.assertEqual(stats["llm_calls"], 0)
            self.assertTrue(all(r["method"] == "name_gate" for r in rows))


class TestNameValidityA2A3(unittest.TestCase):
    def test_dictionary_rejects_english_only(self):
        for name in ["set", "respect", "because", "only", "rules", "miscellaneous", "granted"]:
            ok, reasons = classify_name_validity(name)
            self.assertFalse(ok, name)
            self.assertTrue(
                "all_tokens_english_dictionary" in reasons
                or "procedural_stopword_full_name" in reasons
                or "header_drop_phrase" in reasons,
                (name, reasons),
            )

    def test_leading_initial_kept(self):
        for name in ["j thomas marten", "g murray snow", "f keith ball"]:
            ok, reasons = classify_name_validity(name)
            self.assertTrue(ok, (name, reasons))
            self.assertNotIn("single_char_token_non_middle", reasons)

    def test_jr_mcguire_quarantined(self):
        ok, reasons = classify_name_validity("jr mcguire")
        self.assertFalse(ok, reasons)
        self.assertIn("generational_as_given_name", reasons)

    def test_corpus_rescue_gregory(self):
        corpus_surnames, corpus_tokens = build_corpus_name_rescue(
            ["roger gregory", "gregory"]
        )
        ok, reasons = classify_name_validity(
            "gregory",
            corpus_surnames=corpus_surnames,
            corpus_tokens=corpus_tokens,
        )
        # research_dev minimum: single-token names rejected even with corpus surname rescue
        self.assertFalse(ok, reasons)
        self.assertIn("too_few_name_tokens", reasons)

    def test_strip_trailing_non_name(self):
        from engine.name_validity import strip_trailing_procedural, DEFAULT_PROCEDURAL_STOPWORDS

        cleaned = strip_trailing_procedural("john smith added", DEFAULT_PROCEDURAL_STOPWORDS)
        self.assertEqual(cleaned, "john smith")


class TestLowInfoA4(unittest.TestCase):
    def test_first_name_only_within_ucid(self):
        kept, quar, counts = apply_hygiene(
            [_m("1", "mary", ucid="A"), _m("2", "william", ucid="B"), _m("3", "john smith", ucid="A")],
            cfg=CFG,
        )
        mary = next(m for m in kept if m["mention_id"] == "1")
        self.assertTrue(mary.get("low_info_first_name"))
        self.assertEqual(mary.get("hygiene_scope"), "same_ucid")
        self.assertTrue(is_common_first_name_only(mary))
        other = _m("x", "mary jones", surname="jones", ucid="Z")
        self.assertFalse(pair_allowed(mary, other))
        same = _m("y", "mary ellis", surname="ellis", ucid="A")
        # first-name-only vs multi-token same UCID allowed by hygiene
        self.assertTrue(pair_allowed(mary, same))


class TestClusterA5(unittest.TestCase):
    def test_split_incompatible_surnames(self):
        ms = [
            _m("1", "karoline mehalchick", surname="mehalchick"),
            _m("2", "james munley", surname="munley"),
            _m("3", "karoline mehalchick", surname="mehalchick", ucid="u2"),
        ]
        groups = _split_by_name_compat(ms, CFG)
        self.assertEqual(len(groups), 2)

    def test_cluster_path_doc(self):
        # Smoke: cluster_mentions splits cross-surname component
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg = json.loads(json.dumps(CFG))
            cfg["io"] = {
                "clusters_out": str(td / "ent.jsonl"),
            }
            cfg["incremental"] = {"registry_path": str(td / "reg.jsonl")}
            ms = [
                _m("1", "a mehalchick", surname="mehalchick"),
                _m("2", "b munley", surname="munley"),
            ]
            by_id = {m["mention_id"]: m for m in ms}
            with mock.patch("engine.cluster.resolve_path", lambda c, p: Path(p)):
                ents = cluster_mentions({"r": ["1", "2"]}, by_id, cfg)
            self.assertEqual(len(ents), 2)
            self.assertTrue(all(e.get("split_reason") == "name_gate_cluster_split" for e in ents))


class TestPromptA6(unittest.TestCase):
    def test_prompt_v2_contents_and_hash(self):
        text = (ROOT / "prompts" / "tier3_adjudicate.txt").read_text()
        self.assertIn("name-evidence-primary", text.lower().replace(" ", "-") if False else text)
        self.assertIn("Absence of contradicting evidence is NOT evidence of identity", text)
        self.assertIn("Mehalchick", text)
        self.assertIn("Coughenour", text)
        h = prompt_sha256(text)
        self.assertEqual(len(h), 64)
        # Stable for this file content
        self.assertEqual(h, prompt_sha256(text))


if __name__ == "__main__":
    unittest.main()
