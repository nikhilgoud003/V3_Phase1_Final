"""Parties v0.6 structural rails: corroboration, Sr/Jr, legal form, fund splitter."""

from __future__ import annotations

from engine.cluster import _split_by_fund_plan_type, _split_by_generational_suffix
from engine.config_loader import ROOT, load_config
from engine.tiers import (
    compatible_name_core,
    generational_suffix_conflict,
    identity_pair_conflict,
    legal_entity_form_conflict,
    parties_tier2_lacks_corroboration,
)


def _cfg():
    return load_config(ROOT / "configs/parties.yaml")


def _m(**kw):
    row = {
        "normalized_name": "",
        "raw_name": "",
        "office_class": "corporate",
        "ucid": "x;;1:10-cv-00001",
        "party_role": "Defendant",
        "party_type": "defendant",
        "pacer_id": None,
    }
    row.update(kw)
    if not row.get("raw_name"):
        row["raw_name"] = row["normalized_name"]
    return row


def test_wells_fargo_vs_boa_lacks_corroboration():
    cfg = _cfg()
    a = _m(normalized_name="wells fargo bank n a", office_class="private_other")
    b = _m(normalized_name="bank of america n a", office_class="private_other")
    assert parties_tier2_lacks_corroboration(a, b, cfg)
    assert not compatible_name_core(a, b, cfg)


def test_clean_variants_have_name_core_corroboration():
    cfg = _cfg()
    pairs = [
        ("patterson kelly division", "patterson-kelly division"),
        ("crown cork & seal", "crown cork and seal company incorporated"),
        ("mcmaster carr supply", "mcmaster-carr supply"),
        ("the okonite", "okonite incorporated"),
        ("felt products manufacturing", "felt products incorporated"),
        ("eagle-picher industries", "eagle-picher industries incorporated"),
        ("durabala manufacturing", "durabla manufacturing"),
    ]
    for na, nb in pairs:
        a, b = _m(normalized_name=na), _m(normalized_name=nb)
        assert compatible_name_core(a, b, cfg), (na, nb)
        assert not parties_tier2_lacks_corroboration(a, b, cfg), (na, nb)


def test_pinto_sr_vs_jr_hard_block_even_same_ucid_role():
    cfg = _cfg()
    sr = _m(
        normalized_name="wilfrido pinto sr",
        office_class="nominal_person",
        party_role="Plaintiff",
        party_type="plaintiff",
        ucid="ilnd;;1:04-cv-02352",
    )
    jr = _m(
        normalized_name="wilfrido pinto jr",
        office_class="nominal_person",
        party_role="Plaintiff",
        party_type="plaintiff",
        ucid="ilnd;;1:04-cv-02352",
    )
    assert generational_suffix_conflict(sr, jr, cfg)
    sig, _ = identity_pair_conflict(sr, jr, cfg)
    assert sig == "generational_suffix"
    # Cluster splitter also separates them
    groups = _split_by_generational_suffix([sr, jr], cfg)
    assert len(groups) == 2


def test_one_sided_jr_allowed_same_ucid_role():
    cfg = _cfg()
    bare = _m(
        normalized_name="wilfrido pinto",
        office_class="nominal_person",
        party_role="Plaintiff",
        party_type="plaintiff",
        ucid="ilnd;;1:04-cv-02352",
    )
    jr = _m(
        normalized_name="wilfrido pinto jr",
        office_class="nominal_person",
        party_role="Plaintiff",
        party_type="plaintiff",
        ucid="ilnd;;1:04-cv-02352",
    )
    assert not generational_suffix_conflict(bare, jr, cfg)


def test_legal_entity_form_inc_vs_trust():
    cfg = _cfg()
    a = _m(normalized_name="acme holdings inc")
    b = _m(normalized_name="acme holdings trust")
    assert legal_entity_form_conflict(a, b, cfg)
    sig, _ = identity_pair_conflict(a, b, cfg)
    assert sig == "legal_entity_form"
    # One-sided form is OK (Inc vs bare)
    assert not legal_entity_form_conflict(a, _m(normalized_name="acme holdings"), cfg)


def test_cement_masons_fund_plan_cluster_split():
    cfg = _cfg()
    pension = _m(
        normalized_name="trustees of the cement masons pension fund local 502",
        office_class="private_other",
    )
    savings = _m(
        normalized_name="trustees of the cement masons savings fund local 502",
        office_class="private_other",
    )
    apprentice = _m(
        normalized_name="trustees of the cement masons apprentice education and training fund local 502",
        office_class="private_other",
    )
    institute = _m(
        normalized_name="trustees of the cement masons institute of chicago illinois",
        office_class="private_other",
    )
    groups = _split_by_fund_plan_type([pension, savings, apprentice, institute], cfg)
    assert len(groups) >= 3
    # Typed funds must not share a group
    def group_of(m):
        for g in groups:
            if m in g:
                return id(g)
        return None

    assert group_of(pension) != group_of(savings)
    assert group_of(pension) != group_of(apprentice)
    assert group_of(savings) != group_of(apprentice)


def test_firms_yaml_corroboration_gate_stays_off():
    firms = load_config(ROOT / "configs/firms.yaml")
    a = _m(normalized_name="wells fargo bank n a")
    b = _m(normalized_name="bank of america n a")
    assert not parties_tier2_lacks_corroboration(a, b, firms)


def test_harbison_walker_group_division_marker():
    """F-003: trailing Group is a division/subsidiary marker (Round 5 #46)."""
    cfg = _cfg()
    a = _m(normalized_name="harbison-walker refractories")
    b = _m(normalized_name="harbison-walker refractories group")
    sig, _ = identity_pair_conflict(a, b, cfg)
    assert sig == "division_subsidiary"
    # Same-name Group variants must still be compatible with each other
    g1 = _m(normalized_name="harbison-walker refractories group")
    g2 = _m(normalized_name="harbison walker refractories group")
    assert identity_pair_conflict(g1, g2, cfg) is None
