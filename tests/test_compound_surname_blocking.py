#!/usr/bin/env python3
"""Compound-surname Tier1 keys: Hanovice/Palermo-style names share a block."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.normalize import surname_block_keys
from engine.tiers import build_profile_blocks, same_block

COMPOUNDS = ["garcia marmolejo", "gonzales ramos", "hanovice palermo", "dos santos"]

CFG = {
    "tier1": {
        "compound_surnames": COMPOUNDS,
        "strategies": [
            {
                "id": "court_surname",
                "key_template": "{court}|{surname}",
            },
            {
                "id": "court_initials",
                "key_template": "{court}|{initials}",
            },
        ],
        "residual_bucket": "_UNBLOCKED_",
    }
}


def _m(mid: str, name: str, court: str = "txsd") -> dict:
    toks = name.split()
    return {
        "mention_id": mid,
        "normalized_name": name,
        "surname": toks[-1] if toks else "",
        "court": court,
        "year": 2017,
    }


def test_surname_block_keys_hanovice_palermo():
    short = surname_block_keys("dena hanovice", compound_surnames=COMPOUNDS)
    long = surname_block_keys("dena hanovice palermo", compound_surnames=COMPOUNDS)
    assert "hanovice" in short
    assert "palermo" in long
    assert "hanovice" in long
    assert set(short) & set(long)


def test_surname_block_keys_garcia_marmolejo_and_hyphen():
    a = surname_block_keys("marina garcia marmolejo", compound_surnames=COMPOUNDS)
    b = surname_block_keys("marina garcia", compound_surnames=COMPOUNDS)
    c = surname_block_keys("marina garcia-marmolejo", compound_surnames=COMPOUNDS)
    assert "garcia" in a and "marmolejo" in a
    assert "garcia" in b
    assert "garcia" in c and "marmolejo" in c
    assert set(a) & set(b)
    assert set(b) & set(c)


def test_hanovice_palermo_share_court_surname_block():
    ments = [
        _m("a", "dena hanovice"),
        _m("b", "dena hanovice palermo"),
        _m("c", "dena palermo"),
        _m("d", "stephen smith"),
    ]
    blocks = build_profile_blocks(ments, CFG)
    assert same_block("a", "b", blocks), (blocks["a"], blocks["b"])
    assert same_block("b", "c", blocks), (blocks["b"], blocks["c"])
    assert not same_block("a", "d", blocks)


def test_plain_last_token_still_blocks_smith():
    ments = [_m("s1", "stephen smith"), _m("s2", "stephen wm smith")]
    blocks = build_profile_blocks(ments, CFG)
    assert same_block("s1", "s2", blocks)


def test_judges_initials_helper_still_blocks():
    """judges.yaml maps initials → first_last_initials; must not skip the strategy."""
    cfg = {
        "tier1": {
            "strategies": [
                {
                    "id": "court_initials",
                    "key_template": "{court}|{initials}",
                    "fields": {"court": "court", "initials": "first_last_initials"},
                }
            ],
            "residual_bucket": "_UNBLOCKED_",
        }
    }
    ments = [
        {
            "mention_id": "a",
            "normalized_name": "john smith",
            "surname": "smith",
            "court": "nyed",
            "year": 2017,
        },
        {
            "mention_id": "b",
            "normalized_name": "jane smith",
            "surname": "smith",
            "court": "nyed",
            "year": 2017,
        },
    ]
    blocks = build_profile_blocks(ments, cfg)
    assert any(k.startswith("court_initials::") for k in blocks["a"])
    assert same_block("a", "b", blocks)


def test_surname_strips_generational_suffixes():
    from engine.normalize import surname

    assert surname("joseph c wilkinson jr") == "wilkinson"
    assert surname("robert smith ii") == "smith"
    assert surname("maria garcia iii") == "garcia"
    assert surname("john doe") == "doe"
