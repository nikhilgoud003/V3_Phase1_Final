"""Discovery validity: reject HTML boilerplate and case-tracking codes."""

from __future__ import annotations

from engine.discovery_validity import (
    force_type_other,
    path_is_non_entity,
    value_is_discovery_junk,
)


def test_raw_info_path_rejected():
    bad, why = path_is_non_entity("parties[].counsel[].entity_info.raw_info")
    assert bad
    assert why and "raw_info" in why


def test_member_case_key_path_rejected():
    bad, why = path_is_non_entity("member_case_key")
    assert bad
    assert why and "member_case_key" in why


def test_html_counsel_boilerplate_value():
    raw = (
        "(See above for address)\n"
        "<br/><i>LEAD ATTORNEY</i>\n"
        "<br/><i>ATTORNEY TO BE NOTICED</i>"
    )
    junk, why = value_is_discovery_junk(raw)
    assert junk
    assert why in {"html_markup", "counsel_status_boilerplate", "see_above_address_boilerplate"}


def test_ucid_case_key_value():
    junk, why = value_is_discovery_junk("azd;;2:15-md-02641")
    assert junk
    assert why == "case_tracking_code"


def test_real_party_name_kept():
    junk, _ = value_is_discovery_junk("C R Bard Incorporated")
    assert not junk
    bad, _ = path_is_non_entity("parties[].alias_name")
    assert not bad


def test_force_type_other_on_junk_field():
    forced, why = force_type_other(
        "parties[].counsel[].entity_info.raw_info",
        ["(See above for address)<br/><i>LEAD ATTORNEY</i>"],
    )
    assert forced
    assert why
