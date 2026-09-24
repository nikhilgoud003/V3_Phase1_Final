"""Emit RDF triples using SCALES PACER vocabulary (+ V3 provenance + SKOS names)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .config_loader import resolve_path

PACER = "http://scales-kg.org/pacer#"
PACER_ID = "http://scales-kg.org/id/"
SKOS = "http://www.w3.org/2004/02/skos/core#"
DCT = "http://purl.org/dc/terms/"


def _lit(s: Any) -> str:
    if s is None:
        return '""'
    text = str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


def _uri(kind: str, key: str) -> str:
    return f"<{PACER_ID}{kind}/{quote(str(key), safe='')}>"


def _uri_abs(kind: str, key: str) -> str:
    return f"{PACER_ID}{kind}/{quote(str(key), safe='')}"


def _content_id(*parts: Any, n: int = 12) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(payload.encode()).hexdigest()[:n]


def alias_key(
    sjid: str,
    form: str,
    label: str,
    index: int,
    *,
    content_hash: bool,
    entity_id_kind: str = "judge",
) -> str:
    """Index scheme `{sjid}_{pref|alt}_{i}` vs content-hash (registry-aware / firms).

    Hashed keys never collide with live `{sjid}_pref_0` index URIs. `entity_id_kind`
    is mixed into the hash so firm/judge payloads differ even under a shared path.
    """
    if content_hash:
        return f"{sjid}_{form}_{_content_id(entity_id_kind, sjid, form, label)}"
    return f"{sjid}_{form}_{index}"


def decision_key(
    d: dict,
    *,
    content_hash: bool,
    entity_id_kind: str = "judge",
) -> str:
    """Sequential `dec_00000…` (legacy standalone judges) vs `inc_{hash}`."""
    sequential = d.get("decision_id") or "unknown"
    if not content_hash:
        return str(sequential)
    return "inc_" + _content_id(
        entity_id_kind,
        sequential,
        d.get("mention_id_a"),
        d.get("mention_id_b"),
        d.get("method"),
        d.get("decision"),
        n=16,
    )


_DATE = re.compile(r"(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}")


def _iso_date(raw: str | None) -> str | None:
    """Best-effort ISO date from PACER filing/terminating strings."""
    if not raw:
        return None
    s = str(raw).strip()
    # MM/DD/YYYY
    m = re.match(r"(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m = _DATE.search(s.replace("/", "-"))
    if m:
        return m.group(0).replace("/", "-")
    return None


def _party_mention_role(m: dict) -> str | None:
    """Map a party mention to plaintiff|defendant|other. Not a matching signal."""
    raw = (m.get("party_type") or "").strip().lower()
    if raw == "plaintiff":
        return "plaintiff"
    if raw == "defendant":
        return "defendant"
    if raw or (m.get("party_role") or "").strip():
        return "other"
    return None


def _name_forms_for_entity(
    e: dict, by_id: dict[str, dict]
) -> dict[str, dict[str, str | None]]:
    """Map distinct mention name-forms → {valid_from, valid_to} from case dates."""
    forms: dict[str, dict[str, str | None]] = {}
    for mid in e.get("mention_ids") or []:
        m = by_id.get(mid) or {}
        # Prefer presentable (display) then normalized
        label = (m.get("presentable_name") or m.get("normalized_name") or "").strip()
        if not label:
            continue
        key = label
        dates = [
            _iso_date(m.get("filing_date")),
            _iso_date(m.get("terminating_date")),
        ]
        dates = [d for d in dates if d]
        cur = forms.setdefault(key, {"valid_from": None, "valid_to": None})
        if dates:
            lo, hi = min(dates), max(dates)
            if cur["valid_from"] is None or lo < cur["valid_from"]:
                cur["valid_from"] = lo
            if cur["valid_to"] is None or hi > cur["valid_to"]:
                cur["valid_to"] = hi
    return forms


_SJID_SERIAL = re.compile(r"^([A-Za-z]+)(\d+)$")


def entity_signature(e: dict) -> str:
    """Match key used by incremental REUSE/CREATE (same as run_incremental_holdout).

    Judges (unchanged): ``nid:{fjc}`` else ``name:{normalized}|{court}``.

    Firms only (``entity_type == firm`` or SFID prefix), when fields present:
      1. identifying domain(s) + name (+ office geo when present)
      2. ``name:…|{office_state}|{office_city}``
      3. ``name:…|{court}`` fallback

    Parties (``entity_type == party`` or SPID prefix): ``name|court`` alone is not
    unique — USA / Does / intentional cluster splits legitimately share a name in
    the same court. Disambiguate with a stable hash of the entity's mention-id set
    so the registry stays collision-free for bulk emit and same-membership REUSE.
    """
    nids = e.get("fjc_nids") or []
    if nids:
        return f"nid:{sorted(nids)[0]}"
    nn = e.get("normalized_name") or ""
    sjid = str(e.get("sjid") or "")
    is_firm = (e.get("entity_type") == "firm") or sjid.startswith("SF")
    if is_firm:
        domains = sorted(
            {str(d).strip().lower() for d in (e.get("domains") or []) if str(d).strip()}
        )
        st = (e.get("office_state") or "").strip().upper()
        city = (e.get("office_city") or "").strip().lower()
        if domains:
            dom = "|".join(domains)
            if st and city:
                return f"domain:{dom}|name:{nn}|{st}|{city}"
            return f"domain:{dom}|name:{nn}"
        if st and city:
            return f"name:{nn}|{st}|{city}"
    courts = e.get("courts") or []
    court = courts[0] if courts else ""
    is_party = (e.get("entity_type") == "party") or sjid.startswith("SP")
    if is_party:
        mids = sorted(str(m) for m in (e.get("mention_ids") or []) if m)
        if mids:
            mh = hashlib.sha1("|".join(mids).encode()).hexdigest()[:12]
            return f"mentions:{mh}|name:{nn}|{court}"
        eid = e.get("entity_id") or ""
        if eid:
            return f"entity:{eid}|name:{nn}|{court}"
    return f"name:{nn}|{court}"


def sjid_serial(sjid: str | None) -> tuple[str, int] | None:
    """Return (prefix, integer) for SJ000425 → ('SJ', 425). None if not serial-form."""
    m = _SJID_SERIAL.match(str(sjid or "").strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def load_registry_records(path: str | Path) -> list[dict]:
    """Load a target registry JSONL. Join fjc_nids from sibling entities file if omitted."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"target registry not found: {p}")
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    if rows and not any(r.get("fjc_nids") for r in rows):
        sibling = p.with_name(p.name.replace("_entity_registry.jsonl", "_entities.jsonl"))
        if sibling == p:
            sibling = p.parent / "judges_entities.jsonl"
        if sibling.exists():
            by_sjid = {}
            for line in sibling.open(encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec.get("sjid")
                if sid:
                    by_sjid[sid] = rec
            for r in rows:
                extra = by_sjid.get(r.get("sjid")) or {}
                if extra.get("fjc_nids") and not r.get("fjc_nids"):
                    r["fjc_nids"] = extra["fjc_nids"]
                if extra.get("normalized_name") and not r.get("normalized_name"):
                    r["normalized_name"] = extra["normalized_name"]
                if extra.get("courts") and not r.get("courts"):
                    r["courts"] = extra["courts"]
                for fld in ("domains", "office_state", "office_city"):
                    if extra.get(fld) and not r.get(fld):
                        r[fld] = extra[fld]
    return rows


def assign_registry_sjids(
    entities: list[dict],
    registry: list[dict],
    *,
    prefix: str = "SJ",
    entity_id_kind: str = "judge",
) -> tuple[list[dict], dict[str, Any]]:
    """Remap a run's entities onto a target registry.

    REUSE → existing SJID. CREATE → prefix + (max_registry_serial + 1, …).
    Returns (copied entities with sjid rewritten, assignment report).
    """
    by_sig: dict[str, dict] = {}
    collisions: list[dict] = []
    max_serial = -1
    registry_sjids: set[str] = set()
    for rec in registry:
        sid = rec.get("sjid")
        if sid:
            registry_sjids.add(str(sid))
        parsed = sjid_serial(sid)
        if parsed:
            max_serial = max(max_serial, parsed[1])
        sig = entity_signature(rec)
        if sig in by_sig and by_sig[sig].get("sjid") != sid:
            collisions.append({"signature": sig, "a": by_sig[sig].get("sjid"), "b": sid})
        else:
            by_sig[sig] = rec
    if collisions:
        raise ValueError(f"target registry has duplicate signatures with distinct SJIDs: {collisions}")

    next_serial = max_serial + 1
    remapped: list[dict] = []
    assignments: list[dict] = []
    n_reuse = n_create = 0
    for e in entities:
        out = dict(e)
        sig = entity_signature(e)
        hit = by_sig.get(sig)
        run_sjid = e.get("sjid")
        if hit and hit.get("sjid"):
            out["sjid"] = hit["sjid"]
            action = "REUSE"
            n_reuse += 1
        else:
            new_id = f"{prefix}{next_serial:06d}"
            if new_id in registry_sjids:
                raise ValueError(f"CREATE SJID {new_id} already exists in target registry")
            out["sjid"] = new_id
            next_serial += 1
            action = "CREATE"
            n_create += 1
        remapped.append(out)
        assignments.append(
            {
                "kind": entity_id_kind,
                "action": action,
                "signature": sig,
                "run_sjid": run_sjid,
                "sjid": out["sjid"],
                "canonical_name": out.get("canonical_name"),
                "normalized_name": out.get("normalized_name"),
                "fjc_nids": out.get("fjc_nids") or [],
                "courts": out.get("courts") or [],
                "uri": _uri_abs(entity_id_kind, out["sjid"]),
            }
        )
    report = {
        "n_entities": len(remapped),
        "n_reuse": n_reuse,
        "n_create": n_create,
        "registry_size": len(registry),
        "registry_max_serial": max_serial,
        "next_create_serial_after": next_serial,
        "assignments": assignments,
    }
    return remapped, report


def emit_ttl(
    entities: list[dict],
    mentions: list[dict],
    decisions_path: str | Path,
    cfg: dict,
    out_path: str | None = None,
    target_registry: str | Path | None = None,
    assignment_out: str | Path | None = None,
) -> Path:
    rdf_cfg = cfg.get("rdf") or {}
    graph = rdf_cfg.get("graph_uri", "http://scales-kg.org/graph/tier_v3_judges")
    # Config-driven entity typing (judges default; firms override in YAML)
    entity_class = rdf_cfg.get("entity_class") or "Judge"
    entity_id_kind = rdf_cfg.get("entity_id_kind") or "judge"
    mention_class = rdf_cfg.get("mention_class") or "JudgeMention"
    id_predicate = rdf_cfg.get("id_predicate") or "hasSJID"
    # URI path kinds: firms namespace alias/resolution/mention away from judges.
    alias_id_kind = rdf_cfg.get("alias_id_kind") or "alias"
    resolution_id_kind = rdf_cfg.get("resolution_id_kind") or "resolution"
    mention_id_kind = rdf_cfg.get("mention_id_kind") or "mention"
    out = resolve_path(cfg, out_path or cfg["io"]["rdf_out"])
    out.parent.mkdir(parents=True, exist_ok=True)

    incremental = bool(target_registry)
    # Content-hash when registry-aware, or when YAML requests it (firms bulk emit).
    content_hash_uris = incremental or bool(rdf_cfg.get("content_hash_uris"))
    assignment_report: dict[str, Any] | None = None
    assignment_dest: Path | None = None
    extra_uri_rows: list[dict] = []
    if target_registry:
        prefix = (cfg.get("clustering") or {}).get("id_prefix", "SJ")
        registry_rows = load_registry_records(target_registry)
        entities, assignment_report = assign_registry_sjids(
            entities, registry_rows, prefix=prefix, entity_id_kind=entity_id_kind
        )
        assignment_dest = Path(assignment_out) if assignment_out else out.with_name(out.stem + "_sjid_assignments.json")
        assignment_dest.parent.mkdir(parents=True, exist_ok=True)

    by_id = {m["mention_id"]: m for m in mentions}
    mention_to_sjid = {}
    for e in entities:
        for mid in e.get("mention_ids") or []:
            mention_to_sjid[mid] = e["sjid"]

    lines = [
        f"@prefix pacer: <{PACER}> .",
        f"@prefix skos: <{SKOS}> .",
        f"@prefix dct: <{DCT}> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"# Graph: {graph}",
        f"# Entity class: pacer:{entity_class} (config-driven)",
        "# SKOS name assertions: prefLabel=canonical; altLabel=mention forms with validity intervals.",
        "",
    ]

    for e in entities:
        juri = _uri(entity_id_kind, e["sjid"])
        canon = e.get("canonical_name") or e.get("normalized_name") or ""
        lines.append(f"{juri} rdf:type pacer:{entity_class} ;")
        lines.append(f"  pacer:hasName {_lit(canon)} ;")
        lines.append(f"  skos:prefLabel {_lit(canon)} ;")
        lines.append(f"  pacer:{id_predicate} {_lit(e.get('sjid'))} .")
        for court in e.get("courts") or []:
            lines.append(f"{juri} pacer:hasCode {_lit(court)} .")
        for nid in e.get("fjc_nids") or []:
            lines.append(f"{juri} pacer:hasFjcNid {_lit(nid)} .")
        # Firm contact keys when present on member mentions
        domains = sorted({(by_id.get(mid) or {}).get("domain") for mid in (e.get("mention_ids") or []) if (by_id.get(mid) or {}).get("domain")})
        for dom in domains:
            lines.append(f"{juri} pacer:hasDomain {_lit(dom)} .")

        forms = _name_forms_for_entity(e, by_id)
        pref_l = (canon or "").strip().lower()
        for i, (label, span) in enumerate(sorted(forms.items())):
            if label.strip().lower() == pref_l:
                # Validity on the preferred form (rebrand hook even when only one name)
                if span.get("valid_from") or span.get("valid_to"):
                    akey = alias_key(
                        e["sjid"],
                        "pref",
                        label,
                        i,
                        content_hash=content_hash_uris,
                        entity_id_kind=entity_id_kind,
                    )
                    auri = _uri(alias_id_kind, akey)
                    lines.append(f"{auri} rdf:type pacer:NameAssertion ;")
                    lines.append(f"  skos:prefLabel {_lit(label)} ;")
                    if span.get("valid_from"):
                        lines.append(f"  dct:valid {_lit(span['valid_from'])}^^xsd:date ;")
                        lines.append(f"  pacer:aliasValidFrom {_lit(span['valid_from'])}^^xsd:date ;")
                    if span.get("valid_to"):
                        lines.append(f"  pacer:aliasValidTo {_lit(span['valid_to'])}^^xsd:date ;")
                    lines.append(f"  pacer:assertsNameFor {juri} .")
                    if assignment_report is not None:
                        extra_uri_rows.append(
                            {
                                "kind": "alias",
                                "action": "CREATE",
                                "uri": _uri_abs(alias_id_kind, akey),
                                "sjid": e["sjid"],
                                "label": label,
                                "form": "pref",
                            }
                        )
                continue
            # Distinct alt form
            lines.append(f"{juri} skos:altLabel {_lit(label)} .")
            akey = alias_key(
                e["sjid"],
                "alt",
                label,
                i,
                content_hash=content_hash_uris,
                entity_id_kind=entity_id_kind,
            )
            auri = _uri(alias_id_kind, akey)
            lines.append(f"{auri} rdf:type pacer:NameAssertion ;")
            lines.append(f"  skos:altLabel {_lit(label)} ;")
            if span.get("valid_from"):
                lines.append(f"  pacer:aliasValidFrom {_lit(span['valid_from'])}^^xsd:date ;")
            if span.get("valid_to"):
                lines.append(f"  pacer:aliasValidTo {_lit(span['valid_to'])}^^xsd:date ;")
            lines.append(f"  pacer:assertsNameFor {juri} .")
            if assignment_report is not None:
                extra_uri_rows.append(
                    {
                        "kind": "alias",
                        "action": "CREATE",
                        "uri": _uri_abs(alias_id_kind, akey),
                        "sjid": e["sjid"],
                        "label": label,
                        "form": "alt",
                    }
                )
        lines.append("")

    for m in mentions:
        muri = _uri(mention_id_kind, m["mention_id"])
        curi = _uri("case", m.get("ucid") or m.get("case_id") or "unknown")
        lines.append(f"{curi} rdf:type pacer:Case ;")
        lines.append(f"  pacer:hasUcid {_lit(m.get('ucid'))} ;")
        lines.append(f"  pacer:hasCaseType {_lit(m.get('case_type'))} .")
        lines.append(f"{muri} rdf:type pacer:{mention_class} ;")
        lines.append(f"  pacer:hasName {_lit(m.get('presentable_name'))} ;")
        # Role is mention-scoped only. A party can be plaintiff in one case
        # and defendant in another — never attach this to the Party node.
        party_role_lit = _party_mention_role(m) if entity_class == "Party" else None
        if party_role_lit:
            lines.append(f"  pacer:hasRole {_lit(party_role_lit)} ;")
        lines.append(f"  pacer:mentionsIn {curi} .")
        sjid = mention_to_sjid.get(m["mention_id"])
        if sjid:
            lines.append(f"{muri} pacer:resolvedTo {_uri(entity_id_kind, sjid)} .")
            role = m.get("role")
            if entity_class == "Judge":
                if role == "assigned":
                    lines.append(f"{curi} pacer:assignedJudge {_uri(entity_id_kind, sjid)} .")
                elif role == "referred":
                    lines.append(f"{curi} pacer:referredTo {_uri(entity_id_kind, sjid)} .")
            elif role == "office":
                lines.append(f"{curi} pacer:hasCounselOffice {_uri(entity_id_kind, sjid)} .")
        lines.append("")

    # Decisions
    dpath = Path(decisions_path)
    if dpath.exists():

        for line in dpath.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            dkey = decision_key(
                d, content_hash=content_hash_uris, entity_id_kind=entity_id_kind
            )
            duri = _uri(resolution_id_kind, dkey)
            lines.append(f"{duri} rdf:type pacer:ResolutionDecision ;")
            lines.append(f"  pacer:hasMethod {_lit(d.get('method'))} ;")
            lines.append(f"  pacer:hasConfidence {_lit(d.get('confidence'))} ;")
            lines.append(f"  pacer:hasDecision {_lit(d.get('decision'))} ;")
            lines.append(f"  pacer:hasRationale {_lit(d.get('rationale'))} .")
            if d.get("mention_id_a"):
                lines.append(
                    f"{duri} pacer:from {_uri(mention_id_kind, d['mention_id_a'])} ."
                )
            if d.get("mention_id_b"):
                lines.append(
                    f"{duri} pacer:to {_uri(mention_id_kind, d['mention_id_b'])} ."
                )
            if assignment_report is not None:
                extra_uri_rows.append(
                    {
                        "kind": "decision",
                        "action": "CREATE",
                        "uri": _uri_abs(resolution_id_kind, dkey),
                        "run_decision_id": d.get("decision_id"),
                        "method": d.get("method"),
                        "decision": d.get("decision"),
                    }
                )
            lines.append("")

    if assignment_report is not None and assignment_dest is not None:
        assignment_report["assignments"] = list(assignment_report.get("assignments") or []) + extra_uri_rows
        assignment_report["n_alias_create"] = sum(1 for r in extra_uri_rows if r.get("kind") == "alias")
        assignment_report["n_decision_create"] = sum(1 for r in extra_uri_rows if r.get("kind") == "decision")
        assignment_dest.write_text(json.dumps(assignment_report, indent=2), encoding="utf-8")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
