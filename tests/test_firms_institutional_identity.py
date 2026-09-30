"""Class 1/2 firms institutional: non-identifying domains + office_state."""

from __future__ import annotations

from engine.config_loader import load_config
from engine.normalize import parse_office_address_geo
from engine.provenance import DecisionJournal
from engine.tiers import (
    UnionFind,
    build_tier3_evidence,
    domain_is_non_identifying,
    institutional_office_state_conflict,
    tier0_merge_groups,
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_parse_office_address_geo_city_state_zip():
    g = parse_office_address_geo("615 CHESTNUT ST SUITE 1250\nPHILADELPHIA, PA 19106")
    assert g["office_state"] == "PA"
    assert (g["office_city"] or "").upper().startswith("PHILADELPHIA")


def test_parse_office_address_geo_puerto_rico():
    g = parse_office_address_geo("350 CARLOS CHARDON ST\nSAN JUAN, PR 00918")
    assert g["office_state"] == "PR"


def test_parse_office_address_geo_dc():
    g = parse_office_address_geo("P.O. Box 7611\nWashington, DC 20044-7611")
    assert g["office_state"] == "DC"


def test_parse_office_address_geo_trailing_na_junk():
    g = parse_office_address_geo("1400 New York Ave NW\nWashington, DC 20530\n**NA**")
    assert g["office_state"] == "DC"
    assert (g["office_city"] or "").upper().startswith("WASHINGTON")
    g2 = parse_office_address_geo("2100 Jamieson Avenue\nAlexandria, VA 22314\nNA")
    assert g2["office_state"] == "VA"


def test_usdoj_is_non_identifying_from_config():
    cfg = load_config(ROOT / "configs/firms.yaml")
    assert domain_is_non_identifying("usdoj.gov", cfg)
    assert domain_is_non_identifying("fd.org", cfg)
    assert domain_is_non_identifying("something.gov", cfg)
    assert not domain_is_non_identifying("akerman.com", cfg)


def test_tier3_evidence_suppresses_same_domain_for_usdoj():
    cfg = load_config(ROOT / "configs/firms.yaml")
    ma = {"domain": "usdoj.gov", "court": "paed", "year": 2016, "ucid": "x", "office_class": "doj",
          "office_state": "PR", "co_mentions": []}
    mb = {"domain": "usdoj.gov", "court": "paed", "year": 2016, "ucid": "x", "office_class": "doj",
          "office_state": "PA", "co_mentions": []}
    ev = build_tier3_evidence(ma, mb, barrier=False, reasons=[], cfg=cfg)
    assert ev["same_domain"] is False
    assert ev["shared_non_identifying_domain"] is True
    assert ev["office_state_conflict"] is True


def test_tier3_evidence_emits_same_domain_for_firm():
    cfg = load_config(ROOT / "configs/firms.yaml")
    ma = {"domain": "akerman.com", "court": "nhd", "year": 2016, "ucid": "a", "office_class": "private_other",
          "co_mentions": []}
    mb = {"domain": "akerman.com", "court": "nvd", "year": 2016, "ucid": "b", "office_class": "private_other",
          "co_mentions": []}
    ev = build_tier3_evidence(ma, mb, barrier=False, reasons=[], cfg=cfg)
    assert ev["same_domain"] is True
    assert "shared_non_identifying_domain" not in ev


def test_institutional_office_state_conflict_gate():
    cfg = load_config(ROOT / "configs/firms.yaml")
    a = {"office_class": "doj", "office_state": "PR"}
    b = {"office_class": "doj", "office_state": "PA"}
    assert institutional_office_state_conflict(a, b, cfg)
    assert not institutional_office_state_conflict(
        {"office_class": "doj", "office_state": "PA"},
        {"office_class": "doj", "office_state": "PA"},
        cfg,
    )
    # private firms: no conflict even if states differ
    assert not institutional_office_state_conflict(
        {"office_class": "private_other", "office_state": "IL"},
        {"office_class": "private_other", "office_state": "MO"},
        cfg,
    )


def test_tier0_blocks_institutional_cross_state_same_name_domain(tmp_path):
    cfg = load_config(ROOT / "configs/firms.yaml")
    # Keep only office_domain rule for a tight test
    cfg = {
        **cfg,
        "tier0": {
            "enabled": True,
            "rules": [
                r for r in cfg["tier0"]["rules"] if r["id"] == "office_domain"
            ],
        },
    }
    ments = [
        {
            "mention_id": "a",
            "normalized_name": "united states attorneys office",
            "domain": "usdoj.gov",
            "court": "paed",
            "office_class": "doj",
            "office_state": "PR",
        },
        {
            "mention_id": "b",
            "normalized_name": "united states attorneys office",
            "domain": "usdoj.gov",
            "court": "paed",
            "office_class": "doj",
            "office_state": "PA",
        },
    ]
    uf = UnionFind()
    tier0_merge_groups(ments, cfg, DecisionJournal(tmp_path / "d.jsonl"), uf)
    assert uf.find("a") != uf.find("b")
