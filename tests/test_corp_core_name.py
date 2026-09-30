"""Tests for generic corp_core_name matching key."""

from __future__ import annotations

from engine.normalize import corp_core_name

SUFFIXES = ["inc", "inc.", "corp", "corporation", "llc", "co", "company", "ag"]


def test_gaf_variants_unify():
    assert corp_core_name("gaf corporation", corp_suffixes=SUFFIXES) == "gaf"
    assert corp_core_name("gaf corp", corp_suffixes=SUFFIXES) == "gaf"


def test_owens_corning_hyphen_space():
    a = corp_core_name("owens-corning fiberglas corp", corp_suffixes=SUFFIXES, hyphen_to_space=True)
    b = corp_core_name("owens corning fiberglas corporation", corp_suffixes=SUFFIXES, hyphen_to_space=True)
    assert a == b == "owens corning fiberglas"


def test_owens_illinois_distinct_from_corning():
    oc = corp_core_name("owens-corning fiberglas", corp_suffixes=SUFFIXES, hyphen_to_space=True)
    oi = corp_core_name("owens illinois inc", corp_suffixes=SUFFIXES, hyphen_to_space=True)
    assert oc != oi
    assert oc == "owens corning fiberglas"
    assert oi == "owens illinois"


def test_dana_quigley_single_token_core():
    assert corp_core_name("dana corporation", corp_suffixes=SUFFIXES) == "dana"
    assert corp_core_name("quigley company inc", corp_suffixes=SUFFIXES) == "quigley"
