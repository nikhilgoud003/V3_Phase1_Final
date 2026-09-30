"""Unit tests for generic nested path extraction (firms / future types)."""

from __future__ import annotations

from engine.path_extract import derive_field, resolve_carry_value, walk_path


def test_walk_counsel_office_name():
    case = {
        "parties": [
            {
                "name": "USA",
                "counsel": [
                    {
                        "name": "A Atty",
                        "entity_info": {
                            "office_name": "Kirkland & Ellis LLP",
                            "email": "a@kirkland.com",
                            "phone": "312-555-0100",
                            "address": "300 N LaSalle\nChicago, IL",
                        },
                    }
                ],
            }
        ]
    }
    hits = walk_path(case, "parties[].counsel[].entity_info.office_name")
    assert len(hits) == 1
    val, ctx = hits[0]
    assert val == "Kirkland & Ellis LLP"
    assert ctx["party_index"] == 0
    assert ctx["counsel_index"] == 0


def test_walk_header_judge_and_referred():
    case = {
        "judge": "Hon. Jane Doe",
        "referred_judges": ["John Smith", "Pat Lee"],
        "parties": [{"name": "USA", "role": "Plaintiff", "pacer_id": "p1", "judge": "Jane Doe"}],
        "docket": [{"docket_text": "Signed by Judge Jane Doe on 1/1/2017"}],
    }
    hits = walk_path(case, "judge")
    assert len(hits) == 1 and hits[0][0] == "Hon. Jane Doe"
    hits = walk_path(case, "referred_judges")
    assert hits[0][0] == ["John Smith", "Pat Lee"]
    hits = walk_path(case, "parties[].judge")
    assert hits[0][0] == "Jane Doe"
    assert hits[0][1]["party_index"] == 0
    hits = walk_path(case, "docket[].docket_text")
    assert hits[0][0].startswith("Signed by")
    assert hits[0][1]["docket_index"] == 0


def test_dollar_index_is_innermost_array():
    case = {
        "parties": [
            {"name": "A", "role": "D", "pacer_id": "1", "judge": "X"},
            {"name": "B", "role": "D", "pacer_id": "2", "judge": "Y"},
        ]
    }
    hits = walk_path(case, "parties[].judge")
    assert resolve_carry_value(case, "$index", hits[1][1]) == 1
    assert resolve_carry_value(case, "$party_index", hits[1][1]) == 1
    assert resolve_carry_value(case, "parties[].name", hits[1][1]) == "B"


def test_carry_and_derive_domain():
    case = {
        "parties": [
            {
                "name": "Def",
                "role": "Defendant",
                "counsel": [
                    {
                        "name": "B",
                        "entity_info": {
                            "office_name": "Office",
                            "email": "x@usdoj.gov",
                            "phone": "202-555-0100",
                            "address": "Main St",
                        },
                    }
                ],
            }
        ]
    }
    hits = walk_path(case, "parties[].counsel[].entity_info.office_name")
    _, ctx = hits[0]
    email = resolve_carry_value(case, "parties[].counsel[].entity_info.email", ctx)
    assert email == "x@usdoj.gov"
    assert derive_field("email_domain", {"email": email}) == "usdoj.gov"
    assert resolve_carry_value(case, "$party_index", ctx) == 0
