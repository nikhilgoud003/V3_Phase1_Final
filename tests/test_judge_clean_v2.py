#!/usr/bin/env python3
"""Unit tests: deterministic + LLM name-validity contract cases."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.name_validity import (
    classify_name_validity,
    gate_mention,
    load_fjc_name_sets,
    preclean_header_junk,
    strip_trailing_lexemes,
)


def _fjc():
    honorifics = ["honorable", "hon", "judge", "magistrate", "chief", "senior", "district"]
    return load_fjc_name_sets(ROOT / "data/judges_fjc.csv", honorifics, ".,;:\"()[]{}")


CFG = {
    "name_validity": {
        "enabled": True,
        "min_name_tokens": 2,
        "reject_single_char_second_token": True,
        "whitelist": {"use_fjc_surnames": True, "use_fjc_full_names": True},
    },
    "normalization": {
        "strip_honorifics": [
            "honorable", "hon", "judge", "magistrate", "chief", "senior", "district",
        ],
        "strip_chars": ".,;:\"()[]{}",
    },
}


def test_denise_page_hood_survives():
    """Real MIED judge — 'page' must not be killed by lexicon mid-name cuts."""
    fjc_s, fjc_f = _fjc()
    ok, reasons = classify_name_validity(
        "denise page hood",
        fjc_surnames=fjc_s,
        fjc_full_names=fjc_f,
    )
    assert ok, reasons
    m = {
        "mention_id": "t1",
        "raw_name": "Denise Page Hood",
        "normalized_name": "denise page hood",
        "presentable_name": "Denise Page Hood",
    }
    kept, quar = gate_mention(m, cfg=CFG, fjc_surnames=fjc_s, fjc_full_names=fjc_f)
    assert quar is None, quar
    assert kept is not None
    assert kept["normalized_name"] == "denise page hood"
    # Docket NER must not truncate at mid-name 'Page'
    from engine.docket_ner import _clean_name, extract_judges_from_docket_text

    assert _clean_name("Denise Page Hood") == "Denise Page Hood"
    text = (
        "Signed by District Judge Denise Page Hood. (SSch) "
        "(Entered: 04/23/2020)"
    )
    spans = extract_judges_from_docket_text(text)
    names = {s["raw"] for s in spans}
    assert any(n == "Denise Page Hood" for n in names), names


def test_wexler_trial_rules_invalid():
    fjc_s, fjc_f = _fjc()
    assert preclean_header_junk("Wexler's Trial Rules") == ""
    m = {
        "mention_id": "t2",
        "raw_name": "Wexler'S Trial Rules",
        "normalized_name": "wexler s trial rules",
        "presentable_name": "Wexler'S Trial Rules",
    }
    kept, quar = gate_mention(m, cfg=CFG, fjc_surnames=fjc_s, fjc_full_names=fjc_f)
    assert kept is None
    assert quar is not None
    assert "possessive_construction" in (quar.get("name_validity_reasons") or [])


def test_garcia_cdomadi_normalizes():
    fjc_s, fjc_f = _fjc()
    cleaned = preclean_header_junk("Guillermo R. Garcia.(Cdomadi")
    assert "cdomadi" not in cleaned.lower()
    assert "garcia" in cleaned.lower()
    m = {
        "mention_id": "t3",
        "raw_name": "Guillermo R. Garcia.(Cdomadi",
        "normalized_name": "guillermo r garcia cdomadi",
        "presentable_name": "Guillermo R. Garcia.(Cdomadi",
    }
    kept, quar = gate_mention(m, cfg=CFG, fjc_surnames=fjc_s, fjc_full_names=fjc_f)
    assert quar is None, quar
    assert kept is not None
    assert kept["normalized_name"] == "guillermo r garcia"
    assert "cdomadi" not in kept["normalized_name"]


def test_trailing_lexeme_and_code_strip():
    assert strip_trailing_lexemes("t lane wilson modified") == "t lane wilson"
    assert strip_trailing_lexemes("anita b brody plea") == "anita b brody"
    assert strip_trailing_lexemes("landya b mccafferty de") == "landya b mccafferty"


def test_sentence_fragment_and_interpreter_rejected():
    fjc_s, fjc_f = _fjc()
    for raw in (
        "is continued until further",
        "Interpreter Needed](bw*)COPIES",
        "Guilty Plea",
    ):
        kept, quar = gate_mention(
            {
                "mention_id": raw,
                "raw_name": raw,
                "normalized_name": raw.lower(),
            },
            cfg=CFG,
            fjc_surnames=fjc_s,
            fjc_full_names=fjc_f,
        )
        assert kept is None, raw
        assert quar is not None


def test_llm_pass_uses_method_tag():
    from engine.llm_name_validity import apply_llm_name_validity

    cfg = {
        "name_validity": {
            "enabled": True,
            "llm_validation": {
                "enabled": True,
                "max_calls": 5,
                "cache_path": "/tmp/tier_v3_llm_nv_cache_test.jsonl",
                "journal_path": "/tmp/tier_v3_llm_nv_journal_test.jsonl",
                "prompt_path": "prompts/llm_name_validity.txt",
            },
        },
        "tier3": {"model": "qwen2.5:7b", "endpoint": "http://localhost:11434"},
        "normalization": CFG["normalization"],
        "_repo_root": str(ROOT),
        "_config_path": str(ROOT / "configs/judges.yaml"),
    }
    Path("/tmp/tier_v3_llm_nv_cache_test.jsonl").write_text("")
    Path("/tmp/tier_v3_llm_nv_journal_test.jsonl").write_text("")

    mentions = [
        {
            "mention_id": "m1",
            "raw_name": "Some Odd Span Rules",
            "normalized_name": "some odd span rules",
            "docket_source": "line_entry",
        }
    ]

    def fake_ollama(model, prompt, endpoint="http://localhost:11434", **kw):
        return {"decision": "INVALID", "clean_name": "", "confidence": 95}

    with mock.patch("engine.tiers.call_ollama_json", fake_ollama):
        kept, quar, stats = apply_llm_name_validity(
            mentions, cfg=cfg, fjc_surnames=set(), fjc_full_names=set()
        )
    assert stats["llm_calls"] + stats["cache_hits"] >= 1
    assert quar and quar[0]["name_validity_method"] == "llm_name_validation"
