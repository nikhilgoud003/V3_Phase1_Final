"""Class 3: procedural office-name strip + surname-only contact refuse."""

from __future__ import annotations

from engine.name_compat import (
    firm_contacts_all_differ,
    firm_surname_only_contacts_block,
    names_compatible,
)
from engine.normalize import normalize_name, strip_procedural_office_prefixes


def test_strip_counsel_not_admitted_prefix():
    raw = "COUNSEL NOT ADMITTED TO USDC-NJ BAR\nBEASLEY ALLEN CROW METHVIN PORTIS & MILES"
    stripped = strip_procedural_office_prefixes(raw)
    assert "beasley" in stripped.lower()
    assert "counsel not admitted" not in stripped.lower()
    assert "seyfarth" not in stripped.lower()


def test_strip_seyfarth_counsel_prefix():
    raw = "COUNSEL NOT ADMITTED TO USDC-NJ BAR\nSEYFARTH SHAW LLP"
    stripped = strip_procedural_office_prefixes(raw)
    assert stripped.lower().startswith("seyfarth")
    assert "not admitted" not in stripped.lower()


def test_normalize_name_strips_procedural_when_enabled():
    raw = "Counsel Not Admitted To Usdc-Nj Bar Seyfarth Shaw LLP"
    nn = normalize_name(
        raw,
        honorifics=[],
        corp_suffixes=["llp"],
        strip_procedural_prefixes=True,
    )
    assert nn == "seyfarth shaw"
    assert "counsel" not in nn
    assert "admitted" not in nn


def test_dowd_surname_only_blocked_when_contacts_differ():
    cfg = {
        "entity_type": "firm",
        "name_compat": {"enabled": True, "refuse_surname_only_when_contacts_differ": True},
    }
    a = {
        "normalized_name": "dowd bloch bennett cervone auerbach and yokich",
        "domain": "laboradvocates.com",
        "phone": "(312) 372-1361",
        "address": "8 South Michigan Avenue 19th Floor Chicago, IL 60603",
    }
    b = {
        "normalized_name": "dowd bennett",
        "domain": "dowdbennett.com",
        "phone": "314-889-7300",
        "address": "Suite 1900 7733 Forsyth Blvd. St. Louis, MO 63105",
    }
    assert firm_contacts_all_differ(a, b)
    blocked, reason = firm_surname_only_contacts_block(a, b, cfg=cfg)
    assert blocked
    assert reason.startswith("surname_only_contacts_differ")
    ok, why = names_compatible(a, b, cfg=cfg)
    assert not ok
    assert "surname_only_contacts_differ" in why


def test_reed_smith_allowed_when_domain_shared():
    cfg = {
        "entity_type": "firm",
        "name_compat": {"enabled": True, "refuse_surname_only_when_contacts_differ": True},
    }
    a = {
        "normalized_name": "reed smith",
        "domain": "reedsmith.com",
        "phone": "215-851-8100",
        "address": "Three Logan Square 1717 Arch Street Philadelphia, PA 19103",
    }
    b = {
        "normalized_name": "reed smith llp princeton forrestal village",
        "domain": "reedsmith.com",
        "phone": "609-987-0050",
        "address": "Princeton Forrestal Village 136 Main Street Princeton, NJ 08540",
    }
    assert not firm_contacts_all_differ(a, b)  # same domain
    ok, why = names_compatible(a, b, cfg=cfg)
    assert ok, why


def test_beasley_seyfarth_incompatible_after_strip():
    cfg = {
        "entity_type": "firm",
        "name_compat": {"enabled": True, "refuse_surname_only_when_contacts_differ": True},
    }
    a = {
        "normalized_name": "beasley allen crow methvin portis and miles",
        "domain": "beasleyallen.com",
        "phone": "334-269-2343",
        "address": "218 Commerce St Montgomery, AL 36104",
    }
    b = {
        "normalized_name": "seyfarth shaw",
        "domain": "seyfarth.com",
        "phone": "202-463-2400",
        "address": "975 F Street NW Washington, DC 20004",
    }
    ok, why = names_compatible(a, b, cfg=cfg)
    assert not ok
    assert "cross_surname" in why or "surname_only" in why


def test_dowd_bloch_and_bennett_short_form_missing_domain():
    """Missing domain on one side still counts as contact disagreement."""
    cfg = {
        "entity_type": "firm",
        "name_compat": {"enabled": True, "refuse_surname_only_when_contacts_differ": True},
    }
    a = {
        "normalized_name": "dowd bloch and bennett",
        "domain": None,
        "phone": "(312) 372-1361",
        "address": "8 South Michigan Avenue 19th Floor Chicago, IL 60603",
    }
    b = {
        "normalized_name": "dowd bennett",
        "domain": "dowdbennett.com",
        "phone": "314-889-7345",
        "address": "7733 Forsyth Blvd Suite 1900 St. Louis, MO 63105",
    }
    assert firm_contacts_all_differ(a, b)
    ok, why = names_compatible(a, b, cfg=cfg)
    assert not ok
    assert "surname_only_contacts_differ" in why
    cfg = {"entity_type": "judge", "name_compat": {"enabled": True}}
    a = {"normalized_name": "john smith", "domain": "a.com", "phone": "111", "address": "111 Main St City, ST 00000"}
    b = {"normalized_name": "jane smith", "domain": "b.com", "phone": "222", "address": "222 Oak Ave City, ST 00000"}
    # firm rail must not fire for judges
    blocked, _ = firm_surname_only_contacts_block(a, b, cfg=cfg)
    assert not blocked
