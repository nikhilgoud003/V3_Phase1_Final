"""Class 4/5: institutional office_city conflict + private cross-geo corroboration."""

from __future__ import annotations

from engine.config_loader import load_config
from engine.tiers import (
    institutional_office_geo_conflict,
    institutional_office_state_conflict,
    mention_office_geo,
    private_cross_geo_lacks_corroboration,
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_office_city_conflict_sacramento_san_diego():
    cfg = load_config(ROOT / "configs/firms.yaml")
    a = {
        "office_class": "fpd",
        "office_state": "CA",
        "office_city": "Sacramento",
        "address": "801 I Street\n3rd Floor\nSacramento, CA 95814",
    }
    b = {
        "office_class": "fpd",
        "office_state": "CA",
        "office_city": "San Diego",
        "address": "225 Broadway\nSuite 900\nSan Diego, CA 92101",
    }
    conflict, sig = institutional_office_geo_conflict(a, b, cfg)
    assert conflict and sig == "office_city_conflict"
    assert institutional_office_state_conflict(a, b, cfg)


def test_office_city_conflict_dallas_del_rio():
    cfg = load_config(ROOT / "configs/firms.yaml")
    a = {
        "office_class": "doj",
        "office_state": "TX",
        "office_city": "Dallas",
        "address": "1100 Commerce Street\n3rd Floor\nDallas, TX 75242",
    }
    b = {
        "office_class": "doj",
        "office_state": "TX",
        "office_city": "Del Rio",
        "address": "111 E. Broadway, A300\nDel Rio, TX 78840",
    }
    conflict, sig = institutional_office_geo_conflict(a, b, cfg)
    assert conflict and sig == "office_city_conflict"


def test_dallas_el_paso_still_city_conflict():
    cfg = load_config(ROOT / "configs/firms.yaml")
    a = {"office_class": "doj", "office_state": "TX", "office_city": "Dallas"}
    b = {"office_class": "doj", "office_state": "TX", "office_city": "El Paso"}
    conflict, sig = institutional_office_geo_conflict(a, b, cfg)
    assert conflict and sig == "office_city_conflict"


def test_same_city_no_conflict():
    cfg = load_config(ROOT / "configs/firms.yaml")
    a = {"office_class": "doj", "office_state": "TX", "office_city": "Dallas"}
    b = {"office_class": "doj", "office_state": "TX", "office_city": "Dallas"}
    conflict, sig = institutional_office_geo_conflict(a, b, cfg)
    assert not conflict and sig == ""


def test_dc_address_not_court_fallback():
    """#95: DC address must stay DC even when case court is vaed."""
    cfg = load_config(ROOT / "configs/firms.yaml")
    m = {
        "office_class": "doj",
        "office_state": "DC",
        "office_city": "Washington",
        "address": "1400 New York Ave NW\nWashington, DC 20530\n**NA**",
        "court": "vaed",
    }
    st, city = mention_office_geo(m, cfg)
    assert st == "DC"
    assert city and "washington" in city
    # Never invent from court
    assert st != "VA"


def test_chernov_cross_state_lacks_corroboration():
    cfg = load_config(ROOT / "configs/firms.yaml")
    a = {
        "office_class": "private_other",
        "normalized_name": "chernov croen stern and mahoney",
        "domain": None,
        "phone": "312-222-8700",
        "address": "330 N. WABASH AVENUE\n#200\nCHICAGO, IL 60611",
    }
    b = {
        "office_class": "private_other",
        "normalized_name": "chernov stern and krings",
        "domain": None,
        "phone": "414-273-4000",
        "address": "330 E. KILBOURN AVENUE\nSUITE 1275\nMILWAUKEE, WI 53202",
    }
    assert private_cross_geo_lacks_corroboration(a, b, cfg)


def test_akerman_cross_state_has_domain_corroboration():
    cfg = load_config(ROOT / "configs/firms.yaml")
    a = {
        "office_class": "private_other",
        "domain": "akerman.com",
        "phone": "954-759-8930",
        "address": "350 E Las Olas Blvd\nFort Lauderdale, FL 33301",
    }
    b = {
        "office_class": "private_other",
        "domain": "akerman.com",
        "phone": "702-634-5000",
        "address": "1635 VillageCenter Circle\nLas Vegas, NV 89134",
    }
    assert not private_cross_geo_lacks_corroboration(a, b, cfg)
