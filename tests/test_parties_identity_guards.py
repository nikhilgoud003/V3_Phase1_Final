"""Parties Round-1 identity guards: corp-core stays; T2/T3 exclusions are config-driven."""

from __future__ import annotations

from engine.config_loader import ROOT, load_config
from engine.name_compat import (
    all_initials_incompatible,
    names_compatible,
    person_initials_expand_compatible,
)
from engine.path_extract import apply_office_classification
from engine.tiers import (
    division_subsidiary_conflict,
    fund_plan_type_conflict,
    generic_word_overlap_conflict,
    government_jurisdiction_conflict,
    identity_pair_conflict,
    information_barrier,
    opposing_roles_conflict,
    parent_subunit_conflict,
    person_title_prefix_conflict,
    person_vs_org_conflict,
    placeholder_identity_conflict,
    ucid_only_corroboration,
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


def test_office_classification_john_doe_anchored():
    cfg = _cfg()
    m = apply_office_classification(
        {"normalized_name": "john doe", "raw_name": "John Doe"}, cfg
    )
    assert m["office_class"] == "placeholder"
    m2 = apply_office_classification(
        {
            "normalized_name": "john doe subscriber assigned ip address 65 78 84 40",
            "raw_name": "John Doe Subscriber assigned IP Address 65.78.84.40",
        },
        cfg,
    )
    assert m2["office_class"] == "placeholder"


def test_government_jurisdiction_140_146():
    cfg = _cfg()
    ct = _m(
        normalized_name="state of connecticut ex rel",
        office_class="government",
        party_role="Plaintiff",
        party_type="plaintiff",
        ucid="gand;;1:10-cv-01614",
    )
    de = _m(
        normalized_name="state of delaware ex rel",
        office_class="government",
        party_role="Plaintiff",
        party_type="plaintiff",
        ucid="gand;;1:10-cv-01614",
    )
    ri = _m(
        normalized_name="state of rhode island ex rel",
        office_class="government",
        party_role="Plaintiff",
        party_type="plaintiff",
        ucid="gand;;1:10-cv-01614",
    )
    assert government_jurisdiction_conflict(ct, de, cfg)
    assert government_jurisdiction_conflict(ct, ri, cfg)
    sig, _ = identity_pair_conflict(ct, de, cfg)
    assert sig == "government_jurisdiction"
    same = _m(normalized_name="state of connecticut", office_class="government")
    assert not government_jurisdiction_conflict(ct, same, cfg)


def test_division_subsidiary_named_pairs():
    cfg = _cfg()
    gs = _m(normalized_name="general signal")
    aurora = _m(normalized_name="aurora pump division of general signal")
    assert division_subsidiary_conflict(gs, aurora, cfg)
    goodrich = _m(normalized_name="b f goodrich")
    fairbanks = _m(
        normalized_name="b f goodrich co /fairbanks morse engine division",
        raw_name="B.F. Goodrich Co. /Fairbanks Morse Engine Division",
        office_class="private_other",
    )
    assert division_subsidiary_conflict(goodrich, fairbanks, cfg)
    copes = _m(normalized_name="copes vulcan")
    dezurik = _m(
        normalized_name="dezurik/copes-vulcan",
        raw_name="Dezurik/Copes-Vulcan",
        office_class="private_other",
    )
    assert division_subsidiary_conflict(copes, dezurik, cfg)


def test_placeholder_never_cross_ucid():
    cfg = _cfg()
    a = _m(
        normalized_name="john doe",
        office_class="placeholder",
        ucid="paed;;2:05-cv-02485",
    )
    b = _m(
        normalized_name="john doe subscriber assigned ip address 65 78 84 40",
        office_class="placeholder",
        ucid="paed;;5:21-cv-05191",
    )
    assert placeholder_identity_conflict(a, b, cfg)
    same = _m(normalized_name="john doe", office_class="placeholder", ucid=a["ucid"])
    assert not placeholder_identity_conflict(a, same, cfg)


def test_bare_initials_person_name_gate():
    cfg = _cfg()
    im = _m(normalized_name="i m", office_class="private_other", party_type="plaintiff")
    cm = _m(normalized_name="c m", office_class="private_other", party_type="plaintiff")
    assert all_initials_incompatible(im, cm, cfg=cfg)[0]
    ok, reason = names_compatible(im, cm, cfg=cfg)
    assert not ok and reason.startswith("incompatible_initials")
    corey = _m(normalized_name="corey mitchell", office_class="private_other")
    assert person_initials_expand_compatible(cm, corey)
    assert not person_initials_expand_compatible(im, corey)
    ba, ra = information_barrier(im, cfg, set())
    assert ba and "bare_initials_person" in ra


def test_opposing_roles_and_po_prefix():
    cfg = _cfg()
    p = _m(
        normalized_name="state farm fire & casualty",
        party_role="Plaintiff",
        party_type="plaintiff",
        ucid="paed;;2:13-cv-02562",
    )
    d = _m(
        normalized_name="state farm insurance",
        party_role="Defendant",
        party_type="defendant",
        ucid="paed;;2:20-cv-03148",
    )
    assert opposing_roles_conflict(p, d, cfg)
    crane_d = _m(normalized_name="john crane", party_role="Defendant", party_type="defendant")
    crane_x = _m(
        normalized_name="john crane",
        party_role="Cross Defendant",
        party_type="other_party",
    )
    assert not opposing_roles_conflict(crane_d, crane_x, cfg)
    co = _m(normalized_name="collins equipment", office_class="corporate")
    po = _m(
        normalized_name="p/o michael collins",
        raw_name="P/O Michael Collins",
        office_class="private_other",
    )
    assert person_title_prefix_conflict(co, po, cfg)


def test_ucid_only_corroboration_weak():
    cfg = _cfg()
    a = _m(normalized_name="industrial rubber and specialty", ucid="paed;;2:08-cv-61631")
    b = _m(normalized_name="e i dupont de nemours and", ucid="paed;;2:08-cv-61631")
    assert ucid_only_corroboration(a, b, cfg)
    b2 = dict(b)
    b2["normalized_name"] = a["normalized_name"]
    assert not ucid_only_corroboration(a, b2, cfg)


def test_gaf_and_owens_not_regressed():
    cfg = _cfg()
    g1 = _m(normalized_name="gaf corporation")
    g2 = _m(normalized_name="gaf corp")
    assert identity_pair_conflict(g1, g2, cfg) is None
    oc = _m(normalized_name="owens corning fiberglas")
    oi = _m(normalized_name="owens illinois")
    sig, _ = identity_pair_conflict(oc, oi, cfg)
    assert sig == "corporate_shared_prefix"


def test_fund_plan_type_is_general_not_sampled_id():
    """Round 1 #43/#49 were NEVER given a plan-type rule — only name_gate when
    sponsor tokens already differed. Round 2 #50 is the same failure class.
    """
    cfg = _cfg()
    health = _m(
        normalized_name="locals 302 and 612 of the international union of operating engineers construction industry health and security fund",
        office_class="private_other",
    )
    retire = _m(
        normalized_name="locals 302 and 612 of the international union of operating engineers-employers construction industry retirement fund",
        office_class="private_other",
    )
    assert fund_plan_type_conflict(health, retire, cfg)
    sig, _ = identity_pair_conflict(health, retire, cfg)
    assert sig == "fund_plan_type"

    ibew_hw = _m(
        normalized_name="international brotherhood of electrical workers local union no 380 health & welfare fund"
    )
    ibew_pen = _m(
        normalized_name="i b e w local union no 380 money purchase pension trust fund",
        office_class="private_other",
    )
    assert fund_plan_type_conflict(ibew_hw, ibew_pen, cfg)

    pen = _m(normalized_name="trustees of the cement masons pension fund local 502", office_class="private_other")
    sav = _m(normalized_name="trustees of the cement masons savings fund local 502", office_class="private_other")
    assert fund_plan_type_conflict(pen, sav, cfg)

    same_health = _m(normalized_name="ibe w local 380 health and welfare fund")
    assert not fund_plan_type_conflict(ibew_hw, same_health, cfg)


def test_generic_word_overlap_asbestos_and_gasket():
    """Round 1 Flintkote/T&N was not a general trust rule; Round 2 #51 repeats it."""
    cfg = _cfg()
    arm = _m(normalized_name="armstrong world industries asbestos trust")
    tn = _m(normalized_name="t & n subfund federal mogul asbestos trust")
    assert generic_word_overlap_conflict(arm, tn, cfg)
    flint = _m(normalized_name="flintkote asbestos trust", office_class="private_other")
    assert generic_word_overlap_conflict(flint, tn, cfg)
    g1 = _m(normalized_name="gasket holdings")
    g2 = _m(normalized_name="metallo gasket")
    assert generic_word_overlap_conflict(g1, g2, cfg)
    sig, _ = identity_pair_conflict(g1, g2, cfg)
    assert sig in {"generic_word_overlap", "division_subsidiary"}
    # Shared two distinctive tokens: suffix/generic extra is allowed.
    gc = _m(normalized_name="general cable")
    gci = _m(normalized_name="general cable industries")
    assert not generic_word_overlap_conflict(gc, gci, cfg)
    assert identity_pair_conflict(gc, gci, cfg) is None


def test_parent_subunit_city_and_shareholder_and_bell():
    cfg = _cfg()
    city = _m(normalized_name="city of philadelphia", office_class="nominal_person")
    police = _m(
        normalized_name="city of philadelphia police department",
        office_class="private_other",
    )
    assert parent_subunit_conflict(city, police, cfg)
    sig, _ = identity_pair_conflict(city, police, cfg)
    assert sig in {"government_jurisdiction", "division_subsidiary"}

    bank = _m(normalized_name="capital one bank", office_class="nominal_person")
    dept = _m(normalized_name="capital one bank shareholder dept", office_class="private_other")
    assert parent_subunit_conflict(bank, dept, cfg)
    sig, _ = identity_pair_conflict(bank, dept, cfg)
    assert sig == "division_subsidiary"

    bell = _m(normalized_name="bell &")
    gossett = _m(normalized_name="bell & gossett itt industries")
    assert parent_subunit_conflict(bell, gossett, cfg)

    pump = _m(normalized_name="fairbanks morse", office_class="nominal_person")
    pump_co = _m(normalized_name="fairbanks morse pump")
    assert not parent_subunit_conflict(pump, pump_co, cfg)
    pc = _m(normalized_name="pittsburgh corning")
    pcf = _m(normalized_name="pittsburgh corning fiberglas")
    assert not parent_subunit_conflict(pc, pcf, cfg)


def test_person_vs_org_surname_only():
    cfg = _cfg()
    person = _m(normalized_name="robert mcweeney", office_class="nominal_person")
    co = _m(normalized_name="mcweeney enterprises")
    assert person_vs_org_conflict(person, co, cfg)
    sig, _ = identity_pair_conflict(person, co, cfg)
    assert sig == "person_vs_org"
    # Two shared distinctive tokens: person-shaped docket label of the same company.
    fm = _m(normalized_name="fairbanks morse", office_class="nominal_person")
    fmp = _m(normalized_name="fairbanks morse pump")
    assert not person_vs_org_conflict(fm, fmp, cfg)
    assert identity_pair_conflict(fm, fmp, cfg) is None


def test_true_same_suffix_and_typo_variants_still_merge():
    cfg = _cfg()
    pairs = [
        (_m(normalized_name="eagle-picher industries incorporated"), _m(normalized_name="eagle-picher industries")),
        (_m(normalized_name="seatrain lines"), _m(normalized_name="seatrain lines incorporated")),
        (_m(normalized_name="the okonite"), _m(normalized_name="okonite incorporated")),
        (_m(normalized_name="wr grace & company - conn"), _m(normalized_name="wr grace & co - conn")),
        (_m(normalized_name="merck sharp & dohme"), _m(normalized_name="merck sharp and dohme")),
        (_m(normalized_name="durabala manufacturing"), _m(normalized_name="durabla manufacturing")),
        (
            _m(normalized_name="a-c product liability trust", office_class="private_other"),
            _m(normalized_name="a-c product liability trust et al", office_class="private_other"),
        ),
        (_m(normalized_name="gaf corporation"), _m(normalized_name="gaf corp")),
        (_m(normalized_name="capital one"), _m(normalized_name="capital one bank usa na")),
        (_m(normalized_name="crown cork & seal"), _m(normalized_name="crown cork and seal company incorporated")),
    ]
    for a, b in pairs:
        hit = identity_pair_conflict(a, b, cfg)
        assert hit is None, (a["normalized_name"], b["normalized_name"], hit)


def test_punct_fold_name_gate_hyphen_space_variants():
    """Surname gate must not split hyphen/space/initialism org variants."""
    cfg = _cfg()
    cfg["_token_profile"] = {
        "protected": [],
        "noise": ["refractories", "supply", "company", "corp", "co"],
        "prose": ["refractories", "supply", "company", "corp", "co"],
    }
    pairs = [
        ("harbison-walker refractories", "harbison walker refractories"),
        ("mcmaster-carr supply", "mcmaster carr supply"),
        ("w r grace & company - conn", "wr grace & company-conn"),
        ("w r grace & company - conn", "w r grace & company-conn"),
        ("babcock & wilcox", "babcock-wilcox"),
        ("certainteed", "certain teed"),
        ("acands", "a c and s"),
        ("owens-corning", "owens corning"),
        ("foster wheeler", "foster-wheeler"),
        ("glaxosmithkline", "glaxosmith kline"),
        ("c s r", "csr"),
        ("does 1-10", "does 1 - 10"),
    ]
    for a, b in pairs:
        ok, reason = names_compatible(_m(normalized_name=a), _m(normalized_name=b), cfg=cfg)
        assert ok, (a, b, reason)
        assert reason.startswith("punct_fold") or reason == "surname_ok", (a, b, reason)

    # Division remainder and different second tokens stay incompatible.
    noise_pairs = [
        ("harbison-walker refractories", "harbison walker refractories group"),
        ("harbison walker refractories", "general refractories"),
        ("w r grace & company", "wr grace & company-conn"),
    ]
    for a, b in noise_pairs:
        ok, reason = names_compatible(_m(normalized_name=a), _m(normalized_name=b), cfg=cfg)
        assert not ok, (a, b, reason)

    # Distinct person names stay split; equal punct-cores may fold even if
    # the extractor tagged a company string as nominal_person.
    person_a = _m(normalized_name="guillermo garcia", office_class="nominal_person")
    person_b = _m(normalized_name="marina garcia", office_class="nominal_person")
    ok, _ = names_compatible(person_a, person_b, cfg=cfg)
    assert not ok
    placeholder = _m(normalized_name="john doe", office_class="placeholder")
    other = _m(normalized_name="john-doe", office_class="placeholder")
    ok, _ = names_compatible(placeholder, other, cfg=cfg)
    assert not ok
    judges = load_config(ROOT / "configs/judges.yaml")
    assert not (judges.get("name_compat") or {}).get("punct_fold")


def test_firms_yaml_does_not_enable_parties_guards():
    firms = load_config(ROOT / "configs/firms.yaml")
    a = _m(normalized_name="gasket holdings")
    b = _m(normalized_name="metallo gasket")
    assert identity_pair_conflict(a, b, firms) is None
    assert not fund_plan_type_conflict(
        _m(normalized_name="health and security fund", office_class="private_other"),
        _m(normalized_name="retirement fund", office_class="private_other"),
        firms,
    )
