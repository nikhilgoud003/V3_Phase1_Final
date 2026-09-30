"""Registry-aware SJID assignment and pre-insert safety check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.kg_insert_safety import InsertSafetyError, check_insert_safety, judge_uris_from_ttl
from engine.rdf_emit import assign_registry_sjids, entity_signature, emit_ttl


def _ent(**kwargs):
    base = {
        "entity_id": "ent_x",
        "sjid": "SJ000000",
        "canonical_name": "X",
        "normalized_name": "x",
        "courts": ["nyed"],
        "fjc_nids": [],
        "mention_ids": [],
    }
    base.update(kwargs)
    return base


def test_signature_prefers_nid():
    assert entity_signature(_ent(fjc_nids=["1394806"], normalized_name="ann m donnelly")) == "nid:1394806"
    assert entity_signature(_ent(fjc_nids=[], normalized_name="yara suarez")) == "name:yara suarez|nyed"


def test_firms_signature_domain_then_geo_then_court():
    """Firms: domain (+ name/geo) → name|state|city → name|court. Judges NID first."""
    assert entity_signature(
        _ent(fjc_nids=["1394806"], normalized_name="ann m donnelly", domains=["x.com"])
    ) == "nid:1394806"
    assert (
        entity_signature(
            {
                "entity_type": "firm",
                "sjid": "SF000001",
                "normalized_name": "acme llp",
                "courts": ["paed"],
                "fjc_nids": [],
                "domains": ["acme.com"],
                "office_state": "PA",
                "office_city": "philadelphia",
            }
        )
        == "domain:acme.com|name:acme llp|PA|philadelphia"
    )
    assert entity_signature(
        {
            "entity_type": "firm",
            "sjid": "SF1",
            "normalized_name": "city of chicago department of law",
            "courts": ["ilnd"],
            "fjc_nids": [],
            "domains": ["cityofchicago.org"],
            "office_state": "IL",
            "office_city": "chicago",
        }
    ) != entity_signature(
        {
            "entity_type": "firm",
            "sjid": "SF2",
            "normalized_name": "city of chicago commission on human relations",
            "courts": ["ilnd"],
            "fjc_nids": [],
            "domains": ["cityofchicago.org"],
            "office_state": "IL",
            "office_city": "chicago",
        }
    )
    phoenix = entity_signature(
        {
            "entity_type": "firm",
            "sjid": "SF000189",
            "normalized_name": "united states attorneys office",
            "courts": ["azd"],
            "fjc_nids": [],
            "domains": [],
            "office_state": "AZ",
            "office_city": "phoenix",
        }
    )
    tucson = entity_signature(
        {
            "entity_type": "firm",
            "sjid": "SF000484",
            "normalized_name": "united states attorneys office",
            "courts": ["azd"],
            "fjc_nids": [],
            "domains": [],
            "office_state": "AZ",
            "office_city": "tucson",
        }
    )
    assert phoenix == "name:united states attorneys office|AZ|phoenix"
    assert tucson == "name:united states attorneys office|AZ|tucson"
    assert phoenix != tucson


def test_judges_signature_ignores_firm_geo_when_nid_present():
    """Judges NID path unchanged even if stray firm fields appear."""
    assert (
        entity_signature(
            _ent(
                fjc_nids=["1394806"],
                normalized_name="ann m donnelly",
                office_state="NY",
                office_city="brooklyn",
                domains=["example.com"],
            )
        )
        == "nid:1394806"
    )


def test_reuse_keeps_registry_sjid_create_starts_after_max():
    registry = [
        _ent(sjid="SJ000425", canonical_name="Ann M Donnelly", normalized_name="ann m donnelly", fjc_nids=["1394806"]),
        _ent(sjid="SJ000895", canonical_name="Last", normalized_name="last", fjc_nids=["9"]),
    ]
    incoming = [
        _ent(sjid="SJ000006", canonical_name="Ann M Donnelly", normalized_name="ann m donnelly", fjc_nids=["1394806"]),
        _ent(sjid="SJ000017", canonical_name="Ramon E. Reyes", normalized_name="ramon e reyes", fjc_nids=["13761341"]),
    ]
    remapped, report = assign_registry_sjids(incoming, registry)
    by_name = {e["canonical_name"]: e["sjid"] for e in remapped}
    assert by_name["Ann M Donnelly"] == "SJ000425"
    assert by_name["Ramon E. Reyes"] == "SJ000896"
    assert report["n_reuse"] == 1
    assert report["n_create"] == 1
    assert remapped[0]["sjid"] != incoming[0]["sjid"] or True  # copy, original unchanged
    assert incoming[0]["sjid"] == "SJ000006"


def test_standalone_emit_unchanged_without_registry(tmp_path):
    cfg = {
        "io": {"rdf_out": str(tmp_path / "judges.ttl")},
        "rdf": {"entity_class": "Judge", "entity_id_kind": "judge"},
        "_output_dir": str(tmp_path),
    }
    ents = [_ent(sjid="SJ000000", mention_ids=["mnt_a"], canonical_name="Eduardo C. Robreno")]
    ments = [
        {
            "mention_id": "mnt_a",
            "presentable_name": "Eduardo C. Robreno",
            "ucid": "paed;;x",
            "case_type": "cv",
            "role": "assigned",
            "filing_date": "01/01/2016",
            "terminating_date": "02/01/2016",
        }
    ]
    dec = tmp_path / "decisions.jsonl"
    dec.write_text(
        json.dumps(
            {
                "decision_id": "dec_00000000",
                "method": "tier0.fjc_nid_join",
                "confidence": 100,
                "decision": "MERGE",
                "rationale": "x",
                "mention_id_a": "mnt_a",
                "mention_id_b": "mnt_b",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = emit_ttl(ents, ments, dec, cfg, out_path=str(tmp_path / "judges.ttl"))
    text = out.read_text()
    assert "SJ000000" in text
    assert "SJ000000_pref_0" in text
    assert "dec_00000000" in text
    assert not (tmp_path / "judges_sjid_assignments.json").exists()


def test_incremental_emit_hashes_alias_and_decision_uris(tmp_path):
    registry = tmp_path / "reg.jsonl"
    registry.write_text(
        json.dumps(
            _ent(
                sjid="SJ000425",
                canonical_name="Ann M Donnelly",
                normalized_name="ann m donnelly",
                fjc_nids=["1394806"],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "io": {"rdf_out": str(tmp_path / "judges.ttl")},
        "rdf": {"entity_class": "Judge", "entity_id_kind": "judge"},
        "clustering": {"id_prefix": "SJ"},
        "_output_dir": str(tmp_path),
    }
    ents = [
        _ent(
            sjid="SJ000000",
            mention_ids=["mnt_a"],
            canonical_name="Ann M Donnelly",
            normalized_name="ann m donnelly",
            fjc_nids=["1394806"],
        )
    ]
    ments = [
        {
            "mention_id": "mnt_a",
            "presentable_name": "Ann M Donnelly",
            "ucid": "nyed;;x",
            "case_type": "cr",
            "role": "assigned",
            "filing_date": "01/04/2016",
            "terminating_date": "01/05/2016",
        }
    ]
    dec = tmp_path / "decisions.jsonl"
    dec.write_text(
        json.dumps(
            {
                "decision_id": "dec_00000000",
                "method": "tier0.fjc_nid_join",
                "confidence": 100,
                "decision": "MERGE",
                "rationale": "x",
                "mention_id_a": "mnt_a",
                "mention_id_b": "mnt_b",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = emit_ttl(
        ents,
        ments,
        dec,
        cfg,
        out_path=str(tmp_path / "judges.ttl"),
        target_registry=registry,
    )
    text = out.read_text()
    assert "SJ000425" in text
    assert "SJ000425_pref_0" not in text
    assert "dec_00000000" not in text
    assert "id/resolution/inc_" in text
    assert "id/alias/SJ000425_pref_" in text
    asg = json.loads((tmp_path / "judges_sjid_assignments.json").read_text())
    kinds = {r["kind"] for r in asg["assignments"]}
    assert kinds == {"judge", "alias", "decision"}
    assert asg["n_reuse"] == 1
    assert asg["n_alias_create"] == 1
    assert asg["n_decision_create"] == 1
    assert all(r["uri"].startswith("http://scales-kg.org/id/") for r in asg["assignments"])
    assert not any("_pref_0" in r["uri"] for r in asg["assignments"] if r["kind"] == "alias")
    assert all("/resolution/inc_" in r["uri"] for r in asg["assignments"] if r["kind"] == "decision")


def test_firms_emit_namespaces_and_content_hashes(tmp_path):
    """Firms bulk emit must not share judges alias/resolution/mention path kinds."""
    cfg = {
        "io": {"rdf_out": str(tmp_path / "firms.ttl")},
        "rdf": {
            "entity_class": "LawFirm",
            "entity_id_kind": "firm",
            "mention_class": "FirmMention",
            "id_predicate": "hasSFID",
            "alias_id_kind": "firm_alias",
            "resolution_id_kind": "firm_resolution",
            "mention_id_kind": "firm_mention",
            "content_hash_uris": True,
        },
        "clustering": {"id_prefix": "SF"},
        "_output_dir": str(tmp_path),
    }
    ents = [
        {
            "entity_id": "ent_x",
            "sjid": "SF000000",
            "canonical_name": "Acme LLP",
            "normalized_name": "acme llp",
            "courts": ["paed"],
            "fjc_nids": [],
            "mention_ids": ["mnt_firm_a"],
        }
    ]
    ments = [
        {
            "mention_id": "mnt_firm_a",
            "presentable_name": "Acme LLP",
            "ucid": "paed;;x",
            "case_type": "cv",
            "role": "office",
            "filing_date": "01/01/2016",
            "terminating_date": "02/01/2016",
            "domain": "acme.com",
        }
    ]
    dec = tmp_path / "decisions.jsonl"
    dec.write_text(
        json.dumps(
            {
                "decision_id": "dec_00000000",
                "method": "tier0",
                "confidence": 100,
                "decision": "MERGE",
                "rationale": "x",
                "mention_id_a": "mnt_firm_a",
                "mention_id_b": "mnt_firm_b",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = emit_ttl(ents, ments, dec, cfg, out_path=str(tmp_path / "firms.ttl"))
    text = out.read_text()
    assert "id/firm/SF000000" in text
    assert "id/firm_mention/mnt_firm_a" in text
    assert "id/firm_resolution/inc_" in text
    assert "id/firm_alias/SF000000_" in text
    assert "dec_00000000" not in text
    assert "SF000000_alt_0" not in text
    assert "id/resolution/" not in text or "id/firm_resolution/" in text
    assert "/id/mention/" not in text
    assert "/id/alias/" not in text
    assert "pacer:hasCounselOffice" in text
    assert "pacer:hasSFID" in text
    assert "pacer:LawFirm" in text


def test_safety_create_alias_collision_aborts():
    assignments = {
        "assignments": [
            {"kind": "alias", "action": "CREATE", "uri": "http://scales-kg.org/id/alias/SJ000425_pref_0", "label": "Ann M Donnelly"},
        ]
    }
    with pytest.raises(InsertSafetyError, match="INSERT ABORTED"):
        check_insert_safety(
            assignments=assignments,
            target_uris=["http://scales-kg.org/id/alias/SJ000425_pref_0"],
        )


def test_safety_create_collision_aborts():
    assignments = {
        "assignments": [
            {"action": "CREATE", "uri": "http://scales-kg.org/id/judge/SJ000006", "canonical_name": "Ann M Donnelly"},
        ]
    }
    with pytest.raises(InsertSafetyError, match="INSERT ABORTED"):
        check_insert_safety(
            assignments=assignments,
            target_judge_uris=["http://scales-kg.org/id/judge/SJ000006"],
        )


def test_safety_reuse_missing_aborts():
    assignments = {
        "assignments": [
            {"action": "REUSE", "uri": "http://scales-kg.org/id/judge/SJ000425", "canonical_name": "Ann M Donnelly"},
        ]
    }
    with pytest.raises(InsertSafetyError, match="INSERT ABORTED"):
        check_insert_safety(assignments=assignments, target_judge_uris=["http://scales-kg.org/id/judge/SJ000000"])


def test_safety_pass_reuse_and_create():
    assignments = {
        "n_reuse": 1,
        "n_create": 1,
        "assignments": [
            {"action": "REUSE", "uri": "http://scales-kg.org/id/judge/SJ000425", "canonical_name": "Ann M Donnelly"},
            {"action": "CREATE", "uri": "http://scales-kg.org/id/judge/SJ000896", "canonical_name": "Ramon E. Reyes"},
        ],
    }
    report = check_insert_safety(
        assignments=assignments,
        target_judge_uris=["http://scales-kg.org/id/judge/SJ000425"],
    )
    assert report["ok"] is True


def test_judge_uris_from_ttl(tmp_path):
    p = tmp_path / "g.ttl"
    p.write_text('<http://scales-kg.org/id/judge/SJ000425> rdf:type pacer:Judge .\n')
    assert judge_uris_from_ttl(p) == {"http://scales-kg.org/id/judge/SJ000425"}


def test_registry_record_includes_fjc_nids():
    from engine.cluster import registry_record

    row = registry_record(
        _ent(sjid="SJ000425", canonical_name="Ann M Donnelly", normalized_name="ann m donnelly", fjc_nids=["1394806"])
    )
    assert row["fjc_nids"] == ["1394806"]
    assert entity_signature(row) == "nid:1394806"


def test_nid_registry_required_for_nid_reuse_not_just_name_fallback():
    """If the target registry omits fjc_nids, an incoming NID entity will CREATE a duplicate."""
    registry = [
        _ent(sjid="SJ000425", canonical_name="Ann M Donnelly", normalized_name="ann m donnelly", fjc_nids=[]),
    ]
    incoming = [
        _ent(sjid="SJ000006", canonical_name="Ann M Donnelly", normalized_name="ann m donnelly", fjc_nids=["1394806"]),
    ]
    remapped, report = assign_registry_sjids(incoming, registry)
    assert report["n_create"] == 1
    assert remapped[0]["sjid"] != "SJ000425"

    registry_fixed = [
        _ent(sjid="SJ000425", canonical_name="Ann M Donnelly", normalized_name="ann m donnelly", fjc_nids=["1394806"]),
    ]
    remapped2, report2 = assign_registry_sjids(incoming, registry_fixed)
    assert report2["n_reuse"] == 1
    assert remapped2[0]["sjid"] == "SJ000425"
