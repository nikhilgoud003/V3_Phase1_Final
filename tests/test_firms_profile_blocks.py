#!/usr/bin/env python3
"""Firms Tier1 keys: domain/phone/name_prefix slots, not judges-only format args."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.tiers import build_profile_blocks, same_block

FIRMS_T1 = {
    "tier1": {
        "strategies": [
            {
                "id": "domain",
                "key_template": "domain|{court}|{domain}",
                "fields": {"court": "court", "domain": "domain"},
            },
            {
                "id": "phone",
                "key_template": "phone|{phone}",
                "fields": {"phone": "phone"},
            },
            {
                "id": "court_name_prefix",
                "key_template": "{court}|{name_prefix}",
                "fields": {"court": "court", "name_prefix": "first_two_tokens"},
            },
        ],
        "residual_bucket": "_UNBLOCKED_",
    }
}


def _firm(
    mid: str,
    *,
    name: str,
    court: str,
    domain: str | None = None,
    phone: str | None = None,
) -> dict:
    toks = name.split()
    return {
        "mention_id": mid,
        "normalized_name": name,
        "surname": toks[-1] if toks else "",
        "court": court,
        "year": 2016,
        "domain": domain,
        "phone": phone,
    }


def test_empty_domain_does_not_emit_domain_key():
    ments = [_firm("a", name="solo practice", court="azd", domain=None, phone=None)]
    keys = build_profile_blocks(ments, FIRMS_T1)["a"]
    assert not any(k.startswith("domain::") for k in keys)
    assert not any(k.startswith("phone::") for k in keys)


def test_fpd_same_court_domain_and_phone_share_blocks():
    """akd FPD: two addresses, one phone, one fd.org — each key sufficient."""
    a = _firm(
        "fpd1",
        name="federal public defender s agency",
        court="akd",
        domain="fd.org",
        phone="907-646-3400",
    )
    b = _firm(
        "fpd2",
        name="federal public defender s agency",
        court="akd",
        domain="fd.org",
        phone="907-646-3400",
    )
    blocks = build_profile_blocks([a, b], FIRMS_T1)
    assert same_block("fpd1", "fpd2", blocks)
    assert any(k.startswith("domain::") for k in blocks["fpd1"])
    assert any(k.startswith("phone::") for k in blocks["fpd1"])


def test_usdoj_gov_does_not_share_unscoped_domain_block_across_courts():
    """usdoj.gov must not put every USAO nationwide in one domain block."""
    ak = _firm(
        "ak",
        name="u s attorney s office anch",
        court="akd",
        domain="usdoj.gov",
        phone="907-271-5071",
    )
    ny = _firm(
        "ny",
        name="united states attorneys office eastern district of new york",
        court="nyed",
        domain="usdoj.gov",
        phone="718-254-7000",
    )
    blocks = build_profile_blocks([ak, ny], FIRMS_T1)
    ak_dom = [k for k in blocks["ak"] if k.startswith("domain::")]
    ny_dom = [k for k in blocks["ny"] if k.startswith("domain::")]
    assert ak_dom and ny_dom
    assert set(ak_dom).isdisjoint(set(ny_dom))
    assert not same_block("ak", "ny", blocks)


def test_name_prefix_helper_emits_first_two_tokens():
    m = _firm("n", name="nelson mullins riley", court="azd", domain="nelsonmullins.com")
    blocks = build_profile_blocks([m], FIRMS_T1)
    assert any("azd|nelson mullins" in k for k in blocks["n"])
