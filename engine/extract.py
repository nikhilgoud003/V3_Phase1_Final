"""Config-driven mention extraction from PACER case JSON."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .config_loader import load_config, resolve_path
from .docket_ner import extract_judges_from_docket_text
from .name_validity import gate_mention, load_fjc_name_sets, write_quarantine
from .normalize import (
    apply_replace_regex,
    normalize_name,
    parse_office_address_geo,
    presentable_name,
    split_trailing_location,
    surname,
    tokens,
    year_from_date,
)
from .path_extract import (
    apply_office_classification,
    apply_record_copy_fill,
    derive_field,
    resolve_carry_value,
    walk_path,
)


def _mention_id(parts: Iterable[Any]) -> str:
    h = hashlib.sha1("|".join("" if p is None else str(p) for p in parts).encode()).hexdigest()[:16]
    return f"mnt_{h}"


def _is_drop_literal(raw: str, drop_literals: list[str]) -> bool:
    low = (raw or "").strip().lower()
    if not low:
        return True
    return any(low == d.lower() or low.replace(".", "") == d.lower() for d in drop_literals)


def _norm_set(names: Iterable[str], honorifics, strip_chars: str) -> set[str]:
    out = set()
    for n in names:
        nn = normalize_name(n, honorifics=honorifics, strip_chars=strip_chars)
        if nn:
            out.add(nn)
    return out


def mine_court_personnel_negatives(
    case: dict,
    cfg: dict,
    honorifics,
    strip_chars: str,
) -> set[str]:
    """Normalized names of USPO / reporter / clerk-signature staff (config patterns)."""
    ne = (cfg.get("extraction") or {}).get("negative_evidence") or {}
    cp = ne.get("court_personnel") or {}
    if not cp.get("enabled"):
        return set()
    path = cp.get("path") or "docket[].docket_text"
    raws: list[str] = []
    compiled: list[tuple[re.Pattern[str], bool]] = []
    for pat in cp.get("patterns") or []:
        rx = pat.get("regex")
        if not rx:
            continue
        compiled.append((re.compile(rx), bool(pat.get("last_first"))))
    if not compiled:
        return set()
    for val, _ctx in walk_path(case, path):
        if not isinstance(val, str) or not val:
            continue
        for cre, last_first in compiled:
            for m in cre.finditer(val):
                if last_first and m.lastindex and m.lastindex >= 2:
                    raws.append(f"{m.group(2)} {m.group(1)}")
                elif m.lastindex:
                    raws.append(re.sub(r"'s$", "", m.group(1).strip()))
    names = _norm_set(raws, honorifics, strip_chars)
    if cp.get("match_token_suffixes"):
        for n in list(names):
            t = n.split()
            for k in range(1, len(t) - 1):
                names.add(" ".join(t[k:]))
    return names


def personnel_neg_for_source(
    source: dict,
    cfg: dict,
    personnel_neg: set[str] | None,
) -> set[str]:
    """Apply court-personnel drop only to docket-text spans.

    Header / party-assigned names are never eligible. A clerk signature
    ``(Ramos, Edgardo) (Entered: …)`` must not delete the assigned judge.
    Config: ``court_personnel.apply_to.docket_sources`` / ``kinds``; default
    is docket-line sources only.
    """
    if not personnel_neg:
        return set()
    cp = ((cfg.get("extraction") or {}).get("negative_evidence") or {}).get("court_personnel") or {}
    if not cp.get("enabled"):
        return set()
    apply_to = cp.get("apply_to") or {}
    allowed_ds = apply_to.get("docket_sources")
    allowed_kinds = apply_to.get("kinds")
    ds = source.get("docket_source")
    kind = source.get("kind")
    path = str(source.get("path") or "")
    if allowed_ds or allowed_kinds:
        if allowed_ds and ds not in allowed_ds:
            return set()
        if allowed_kinds and kind not in allowed_kinds:
            return set()
        return personnel_neg
    if ds in {"case_header", "case_parties"}:
        return set()
    if ds in {"line_entry", "case_docket"} or path.startswith("docket"):
        return personnel_neg
    return set()


def _collect_party_counsel_names(case: dict) -> tuple[set[str], set[str]]:
    parties, counsels = [], []
    for p in case.get("parties") or []:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("name"), str):
            parties.append(p["name"])
        for c in p.get("counsel") or []:
            if isinstance(c, dict) and isinstance(c.get("name"), str):
                counsels.append(c["name"])
    return set(parties), set(counsels)


def mine_transfer_pairs(case: dict, patterns: list[dict], honorifics, strip_chars: str) -> list[dict]:
    """Return list of {from_raw, to_norm, snippet, pattern} from docket text."""
    compiled = [(p["name"], re.compile(p["regex"])) for p in patterns]
    pairs = []
    for entry in case.get("docket") or []:
        if not isinstance(entry, dict):
            continue
        text = entry.get("docket_text") or ""
        if not text:
            continue
        for pname, cre in compiled:
            for m in cre.finditer(text):
                to_raw = m.group(1).strip()
                to_norm = normalize_name(to_raw, honorifics=honorifics, strip_chars=strip_chars)
                if not to_norm:
                    continue
                # crude "from" capture: look for Judge X before transfer verb in a window
                start = max(0, m.start() - 80)
                window = text[start : m.end() + 40]
                from_m = re.search(
                    r"(?i)(?:judge|hon\.?|honorable)\s+([A-Z][A-Za-z\.\\'\-]+(?:\s+[A-Z][A-Za-z\.\\'\-]+){0,3}).{0,40}transfer",
                    window,
                )
                from_raw = from_m.group(1).strip() if from_m else None
                from_norm = (
                    normalize_name(from_raw, honorifics=honorifics, strip_chars=strip_chars)
                    if from_raw
                    else None
                )
                pairs.append(
                    {
                        "pattern": pname,
                        "from_raw": from_raw,
                        "from_norm": from_norm,
                        "to_raw": to_raw,
                        "to_norm": to_norm,
                        "snippet": text[max(0, m.start() - 40) : m.end() + 40],
                    }
                )
    return pairs


def _context_should_skip(ctx: dict, rules: list | None) -> bool:
    """Generic skip: context object field equals a configured value (e.g. is_pro_se)."""
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        obj_key = rule.get("context_object")
        obj = ctx.get(obj_key) if obj_key else None
        if not isinstance(obj, dict):
            continue
        field = rule.get("field")
        if not field:
            continue
        if obj.get(field) == rule.get("equals", True):
            return True
    return False


def _emit_mention(
    *,
    raw: str,
    cfg: dict,
    case: dict,
    source: dict,
    extra: dict | None = None,
    party_neg: set[str],
    counsel_neg: set[str],
    personnel_neg: set[str] | None = None,
) -> dict | None:
    ext = cfg["extraction"]
    norm_cfg = cfg["normalization"]
    honorifics = norm_cfg.get("strip_honorifics", [])
    strip_chars = norm_cfg.get("strip_chars", "")

    if _is_drop_literal(raw, ext.get("drop_literals") or []):
        return None

    # General docket-prose cleanup (config normalization.name_cleaning).
    raw_uncleaned = None
    if norm_cfg.get("name_cleaning"):
        from engine.normalize import clean_name_span

        cleaned = clean_name_span(raw, norm_cfg["name_cleaning"])
        if cleaned != raw:
            raw_uncleaned, raw = raw, cleaned

    loc_cfg = (norm_cfg.get("trailing_location") or {})
    name_for_norm = raw
    office_location = extra.get("office_location") if extra else None
    if loc_cfg.get("enabled"):
        from engine.normalize import strip_trailing_address_contamination

        core, loc = split_trailing_location(
            raw,
            separators=loc_cfg.get("separators") or [],
            remainder_patterns=loc_cfg.get("remainder_patterns") or [],
        )
        # Second pass: suite/street/courthouse glued without "Name - City" dash.
        core2, loc2 = strip_trailing_address_contamination(
            core,
            patterns=loc_cfg.get("trailing_junk_patterns") or [],
            junk_chars=loc_cfg.get("trailing_junk_chars"),
            min_core_tokens=int(loc_cfg.get("min_core_tokens") or 2),
        )
        if loc2:
            core = core2
            loc = f"{loc} {loc2}".strip() if loc else loc2
        if loc:
            name_for_norm = core
            office_location = loc
            extra = dict(extra or {})
            extra["office_location"] = loc
            if str(ext.get("office_name_default") or "").lower() == "raw":
                extra["office_name"] = core

    normalized = normalize_name(
        name_for_norm,
        honorifics=honorifics,
        strip_chars=strip_chars,
        lowercase=bool(norm_cfg.get("lowercase", True)),
        collapse_whitespace=bool(norm_cfg.get("collapse_whitespace", True)),
        name_prefixes=norm_cfg.get("strip_name_prefixes") or [],
        corp_suffixes=norm_cfg.get("strip_corp_suffixes") or [],
        replace_tokens=norm_cfg.get("replace_tokens") or {},
        replace_regex=norm_cfg.get("replace_regex") or [],
        delete_chars=norm_cfg.get("delete_chars") or "",
        strip_procedural_prefixes=bool(norm_cfg.get("strip_procedural_prefixes")),
        procedural_prefix_patterns=norm_cfg.get("procedural_prefix_patterns"),
    )
    # Config-driven full-string expansions (e.g. parties.yaml expand_abbreviations).
    # Applied after normalize so both punctuated and stripped forms can match.
    normalized = apply_replace_regex(normalized, norm_cfg.get("expand_abbreviations") or [])
    if not normalized or len(normalized) <= 1:
        return None

    # Negative evidence: exact normalized equality with party/counsel/court staff
    if normalized in party_neg or normalized in counsel_neg or normalized in (personnel_neg or set()):
        return None

    filing = case.get("filing_date")
    year = year_from_date(filing)
    ucid = case.get("ucid")
    court = case.get("court")
    extra = extra or {}

    # Judge hashes stay the original 7-tuple (no counsel_enum). Firms include
    # counsel_enum so two counsel sharing an office_name on one party do not collide.
    id_parts: list[Any] = [
        ucid,
        source.get("id"),
        raw,
        extra.get("party_enum"),
        extra.get("judge_enum"),
        extra.get("docket_index"),
        source.get("role"),
    ]
    if extra.get("counsel_enum") is not None:
        id_parts.append(extra.get("counsel_enum"))
    mid = _mention_id(id_parts)

    # Presentable follows the same procedural-prefix strip as matching keys.
    presentable_src = name_for_norm
    if norm_cfg.get("strip_procedural_prefixes") or norm_cfg.get("procedural_prefix_patterns"):
        from engine.normalize import strip_procedural_office_prefixes

        presentable_src = strip_procedural_office_prefixes(
            name_for_norm, norm_cfg.get("procedural_prefix_patterns")
        )

    rec = {
        "mention_id": mid,
        "entity_type": cfg.get("entity_type"),
        "raw_name": raw,
        "normalized_name": normalized,
        "presentable_name": presentable_name(presentable_src, honorifics),
        "surname": surname(normalized),
        "token_count": len(tokens(normalized)),
        "role": source.get("role"),
        "docket_source": source.get("docket_source"),
        "extraction_method": source.get("extraction_method"),
        "ucid": ucid,
        "court": court,
        "case_id": case.get("case_id"),
        "case_type": case.get("case_type"),
        "case_name": case.get("case_name"),
        "filing_date": filing,
        "terminating_date": case.get("terminating_date"),
        "year": year,
        "email": extra.get("email"),
        "office_name": extra.get("office_name") or normalized,
        "address": extra.get("address"),
        "phone": extra.get("phone"),
        "fax": extra.get("fax"),
        "domain": extra.get("domain"),
        "office_class": extra.get("office_class"),
        "office_subclass": extra.get("office_subclass"),
        "fjc_nid": None,
        "source_file": extra.get("source_file"),
        "party_enum": extra.get("party_enum"),
        "party_name": extra.get("party_name"),
        "party_role": extra.get("party_role"),
        "pacer_id": extra.get("pacer_id"),
        "counsel_enum": extra.get("counsel_enum"),
        "counsel_name": extra.get("counsel_name"),
        "judge_enum": extra.get("judge_enum"),
        "prefix_category": None,
        "co_mentions": [],
        "transfer_partners": [],
    }
    if loc_cfg.get("enabled"):
        rec["office_location"] = extra.get("office_location") or office_location
    if extra.get("party_type") is not None:
        rec["party_type"] = extra["party_type"]
    if raw_uncleaned is not None:
        rec["raw_name_uncleaned"] = raw_uncleaned
    return rec


def _extract_via_generic_path(
    case: dict,
    cfg: dict,
    source: dict,
    source_file: str | None,
    party_neg: set[str],
    counsel_neg: set[str],
    personnel_neg: set[str] | None = None,
) -> list[dict]:
    """Emit mentions for an arbitrary nested path (string / list / NER span)."""
    path = source.get("path") or ""
    kind = source.get("kind")
    hits = walk_path(case, path)
    out: list[dict] = []
    carry_specs = source.get("carry_fields") or {}
    derive_specs = source.get("derive_fields") or {}
    ne_cfg = (cfg.get("extraction") or {}).get("negative_evidence") or {}
    skip_rules = (cfg.get("extraction") or {}).get("skip_record_if") or source.get("skip_record_if") or []
    # Config: apply_counsel_names (default true). Firms set false — office
    # strings may equal a counsel person name and must not be dropped.
    neg_counsel = counsel_neg if ne_cfg.get("apply_counsel_names", True) else set()
    # Config: apply_party_names (default true). Parties set false — the
    # extracted span IS the party name and must not be dropped by the
    # party-name negative list (judges/firms still default true).
    neg_party = party_neg if ne_cfg.get("apply_party_names", True) else set()

    ner_cfg = source.get("ner") or {}
    ner_model = ner_cfg.get("model") or "en_core_web_sm"
    ner_labels = tuple(ner_cfg.get("label_allowlist") or ["PERSON"])

    for val, ctx in hits:
        if _context_should_skip(ctx, skip_rules):
            continue
        values: list[tuple[str, dict[str, Any]]] = []
        if kind == "string_or_null":
            if isinstance(val, str) and val.strip():
                values = [(val.strip(), {})]
        elif kind == "list_of_strings":
            if isinstance(val, list):
                values = [(v.strip(), {}) for v in val if isinstance(v, str) and v.strip()]
        elif kind == "ner_span":
            text = val if isinstance(val, str) else ""
            if not text:
                continue
            d_idx = ctx.get("docket_index")
            if d_idx is None:
                for seg, i in reversed(ctx.get("indices") or []):
                    if seg == "docket":
                        d_idx = i
                        break
            spans = extract_judges_from_docket_text(text, model=ner_model, label_allowlist=ner_labels)
            for j_enum, span in enumerate(spans):
                values.append(
                    (
                        span["raw"],
                        {
                            "judge_enum": j_enum,
                            "docket_index": d_idx,
                            "ent_span_start": span["start"],
                            "ent_span_end": span["end"],
                            "ner_method": span["method"],
                            "prefix_category": span["method"],
                        },
                    )
                )
        else:
            continue

        base_extra: dict[str, Any] = {"source_file": source_file}
        for field, spec in carry_specs.items():
            base_extra[field] = resolve_carry_value(case, spec, ctx)
        if "party_enum" not in base_extra and ctx.get("party_index") is not None:
            base_extra["party_enum"] = ctx["party_index"]
        if "counsel_enum" not in base_extra and ctx.get("counsel_index") is not None:
            base_extra["counsel_enum"] = ctx["counsel_index"]

        for derived_field, helper in derive_specs.items():
            base_extra[derived_field] = derive_field(str(helper), base_extra)

        for i, (raw, span_extra) in enumerate(values):
            extra = dict(base_extra)
            extra.update(span_extra)
            extra.setdefault("judge_enum", i)
            # Config: office_name_default=raw (firms) vs unset → normalized_name
            # (judges; matches pre-walker header/party extract).
            if str((cfg.get("extraction") or {}).get("office_name_default") or "").lower() == "raw":
                extra.setdefault("office_name", raw)
            m = _emit_mention(
                raw=raw,
                cfg=cfg,
                case=case,
                source=source,
                extra=extra,
                party_neg=neg_party,
                counsel_neg=neg_counsel,
                personnel_neg=personnel_neg_for_source(source, cfg, personnel_neg),
            )
            if not m:
                continue
            if "docket_index" in extra and extra["docket_index"] is not None:
                m["docket_index"] = extra["docket_index"]
            if extra.get("prefix_category"):
                m["prefix_category"] = extra["prefix_category"]
            m = apply_office_classification(m, cfg)
            if m.pop("_drop_classified", False):
                continue
            # Office geo from address for institutional firms only — never for
            # judges (keeps frozen-mention sha256), never from case court.
            inst_classes = set(
                ((cfg.get("identity_exclusions") or {}).get("institutional_office_classes"))
                or []
            )
            if inst_classes and (m.get("office_class") or "") in inst_classes:
                geo = parse_office_address_geo(m.get("address"))
                m["office_state"] = geo.get("office_state")
                m["office_city"] = geo.get("office_city")
            out.append(m)
    return out



def extract_from_case(case: dict, cfg: dict, source_file: str | None = None) -> tuple[list[dict], list[dict]]:
    ext = cfg["extraction"]
    norm_cfg = cfg["normalization"]
    honorifics = norm_cfg.get("strip_honorifics", [])
    strip_chars = norm_cfg.get("strip_chars", "")

    fill_spec = ext.get("record_copy_fill") or {}
    if fill_spec.get("enabled"):
        case, fill_stats = apply_record_copy_fill(copy.deepcopy(case), fill_spec)
        bucket = cfg.setdefault("_extract_stats", {})
        acc = bucket.setdefault("record_copy_fill", {"flagged": 0, "empty_flagged": 0, "filled": 0, "no_donor": 0})
        for k, v in fill_stats.items():
            acc[k] = acc.get(k, 0) + v

    raw_parties, raw_counsels = _collect_party_counsel_names(case)
    party_neg = _norm_set(raw_parties, honorifics, strip_chars)
    counsel_neg = _norm_set(raw_counsels, honorifics, strip_chars)
    personnel_neg = mine_court_personnel_negatives(case, cfg, honorifics, strip_chars)

    transfer_pairs = []
    tc = ext.get("transfer_clues") or {}
    if tc.get("enabled"):
        transfer_pairs = mine_transfer_pairs(case, tc.get("patterns") or [], honorifics, strip_chars)

    mentions: list[dict] = []
    for source in ext.get("sources") or []:
        if source.get("enabled") is False:
            continue
        kind = source.get("kind")
        path = source.get("path") or ""
        if kind in {"string_or_null", "list_of_strings", "ner_span"} and path:
            mentions.extend(
                _extract_via_generic_path(
                    case, cfg, source, source_file, party_neg, counsel_neg, personnel_neg
                )
            )

    alias_spec = cfg.get("alias_extraction") or {}
    if alias_spec.get("enabled"):
        mentions.extend(
            _extract_party_aliases(
                case, cfg, mentions, alias_spec, source_file, party_neg, counsel_neg, personnel_neg
            )
        )

    # Attach co-mentions and transfer partners within case
    norms = [m["normalized_name"] for m in mentions]
    for m in mentions:
        m["co_mentions"] = sorted({n for n in norms if n != m["normalized_name"]})
        partners = set()
        for tp in transfer_pairs:
            if tp.get("from_norm") == m["normalized_name"] and tp.get("to_norm"):
                partners.add(tp["to_norm"])
            if tp.get("to_norm") == m["normalized_name"] and tp.get("from_norm"):
                partners.add(tp["from_norm"])
        m["transfer_partners"] = sorted(partners)

    return mentions, transfer_pairs


def _extract_party_aliases(
    case: dict,
    cfg: dict,
    mentions: list[dict],
    spec: dict,
    source_file: str | None,
    party_neg: set[str],
    counsel_neg: set[str],
    personnel_neg: set[str],
) -> list[dict]:
    """fka/aka/dba (same entity) and successor/alter ego/... (related entity)
    names from each listed party's raw_info (config alias_extraction).

    A name equal to any listed party or counsel name in the case is dropped.
    Same-entity aliases carry alias_of (the mention they name); the Tier0
    explicit_alias_link rule joins them.
    """
    from engine.party_alias import parse_raw_info

    main_by_enum = {
        m["party_enum"]: m
        for m in mentions
        if m.get("docket_source") == "case_parties" and m.get("party_enum") is not None
    }
    out: list[dict] = []
    seen: set[str] = set()
    field = spec.get("source_field") or "raw_info"
    for i, party in enumerate(case.get("parties") or []):
        main = main_by_enum.get(i)
        if main is None:
            continue
        raw_info = (party.get("entity_info") or {}).get(field) or ""
        related_ids: dict[str, str] = {}
        for item in parse_raw_info(raw_info, spec):
            rel, same = item["relationship"], item["same_entity"]
            source = {
                "id": "party_alias",
                "role": main.get("role"),
                "docket_source": "party_alias",
                "extraction_method": f"raw_info_{rel}",
            }
            extra = {
                "party_enum": i,
                "party_name": main.get("party_name"),
                "party_role": main.get("party_role"),
                "source_file": source_file,
            }
            m = _emit_mention(
                raw=item["name"],
                cfg=cfg,
                case=case,
                source=source,
                extra=extra,
                party_neg=party_neg,
                counsel_neg=counsel_neg,
                personnel_neg=personnel_neg,
            )
            if not m or m["mention_id"] in seen:
                continue
            m = apply_office_classification(m, cfg)
            if m.pop("_drop_classified", False):
                continue
            m["relationship_type"] = rel
            m["same_entity"] = same
            m["related_party"] = main.get("raw_name")
            if same:
                target = main["mention_id"] if item["subject"] is None else related_ids.get(item["subject"])
                if target is None:
                    continue
                m["alias_of"] = target
            else:
                m["related_to_mention"] = main["mention_id"]
                related_ids[item["name"]] = m["mention_id"]
            seen.add(m["mention_id"])
            out.append(m)
    return out


def _resolve_json_files(cfg: dict, json_dir: str | None, limit: int | None) -> list[Path]:
    if json_dir:
        files = sorted(Path(json_dir).glob("*.json"))
    else:
        glob_pat = cfg["io"]["json_glob"]
        files = sorted(resolve_path(cfg, glob_pat.replace("*.json", "")).glob("*.json"))
        if not files:
            parent = resolve_path(cfg, str(Path(glob_pat).parent))
            files = sorted(parent.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    return files


def _finalize_mentions(
    cfg: dict,
    all_mentions: list[dict],
    all_transfers: list[dict],
    *,
    write: bool = True,
    transfer_out_rel: str | None = None,
) -> list[dict]:
    """Name-validity, name-repair, profiles, optional write — same pipeline as single-type extract."""
    quarantined: list[dict] = []
    nv = cfg.get("name_validity") or {}
    if nv.get("enabled"):
        from engine.name_validity import build_corpus_name_rescue

        honorifics = (cfg.get("normalization") or {}).get("strip_honorifics") or []
        strip_chars = (cfg.get("normalization") or {}).get("strip_chars") or ""
        fjc_surnames: set[str] = set()
        fjc_full: set[str] = set()
        wl = nv.get("whitelist") or {}
        if wl.get("use_fjc_surnames", True) or wl.get("use_fjc_full_names", True):
            fjc_path = resolve_path(cfg, "data/judges_fjc.csv")
            fjc_surnames, fjc_full = load_fjc_name_sets(fjc_path, honorifics, strip_chars)
            if not wl.get("use_fjc_surnames", True):
                fjc_surnames = set()
            if not wl.get("use_fjc_full_names", True):
                fjc_full = set()
        corpus_surnames, corpus_tokens = build_corpus_name_rescue(
            [m.get("normalized_name") or "" for m in all_mentions]
        )
        kept: list[dict] = []
        for m in all_mentions:
            km, qm = gate_mention(
                m,
                cfg=cfg,
                fjc_surnames=fjc_surnames,
                fjc_full_names=fjc_full,
                corpus_surnames=corpus_surnames,
                corpus_tokens=corpus_tokens,
            )
            if km is not None:
                kept.append(km)
            if qm is not None:
                quarantined.append(qm)
        all_mentions = kept
        from engine.llm_name_validity import apply_llm_name_validity

        kept2, q_llm, llm_stats = apply_llm_name_validity(
            all_mentions,
            cfg=cfg,
            fjc_surnames=fjc_surnames,
            fjc_full_names=fjc_full,
        )
        if llm_stats.get("enabled"):
            all_mentions = kept2
            quarantined.extend(q_llm)
            print(
                f"LLM name-validity: candidates={llm_stats.get('unique_candidates')} "
                f"calls={llm_stats.get('llm_calls')} cache={llm_stats.get('cache_hits')} "
                f"valid={llm_stats.get('valid')} invalid={llm_stats.get('invalid')} "
                f"renamed={llm_stats.get('renamed')}",
                flush=True,
            )
        q_out = resolve_path(cfg, nv.get("quarantine_out") or "data/mentions/quarantine_name_validity.jsonl")
        write_quarantine(q_out, quarantined)
        print(f"Name-validity: kept={len(all_mentions)} quarantined={len(quarantined)} → {q_out}")

    if (cfg.get("name_repair") or {}).get("enabled", True):
        from engine.name_repair import repair_mention_names

        rep = repair_mention_names(all_mentions, cfg)
        print(
            f"Name repair: mentions_repaired={rep['mentions_repaired']} "
            f"distinct={rep['distinct_repairs']}",
            flush=True,
        )

    tmpl = (cfg.get("profile_template") or "").replace("\n", " ").strip()

    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return ""

    for m in all_mentions:
        m["profile"] = tmpl.format_map(
            _Safe(
                {
                    "normalized_name": m.get("normalized_name"),
                    "presentable_name": m.get("presentable_name"),
                    "role": m.get("role"),
                    "court": m.get("court"),
                    "case_type": m.get("case_type"),
                    "ucid": m.get("ucid"),
                    "year": m.get("year"),
                    "extraction_method": m.get("extraction_method"),
                    "prefix_category": m.get("prefix_category"),
                    "party_name": m.get("party_name"),
                    "co_mentions": ", ".join(m.get("co_mentions") or []),
                    "office_class": m.get("office_class"),
                    "office_subclass": m.get("office_subclass"),
                    "domain": m.get("domain"),
                    "phone": m.get("phone"),
                    "office_location": m.get("office_location") or "",
                    "office_state": m.get("office_state") or "",
                    "office_city": m.get("office_city") or "",
                    "address": (m.get("address") or "").replace("\n", " | ")[:160],
                }
            )
        )

    if write:
        out = resolve_path(cfg, cfg["io"]["mentions_out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for m in all_mentions:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        t_rel = transfer_out_rel or "data/mentions/transfer_clues.jsonl"
        t_out = resolve_path(cfg, t_rel)
        t_out.parent.mkdir(parents=True, exist_ok=True)
        with open(t_out, "w", encoding="utf-8") as f:
            for t in all_transfers:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_mentions)} mentions → {out}")
        print(f"Wrote {len(all_transfers)} transfer clues → {t_out}")
        st = cfg.get("_extract_stats")
        if st:
            print(f"Extract stats: {json.dumps(st)}", flush=True)

    return all_mentions


def extract_mentions(
    config_path: str,
    json_dir: str | None = None,
    limit: int | None = None,
    write: bool = True,
) -> list[dict]:
    cfg = load_config(config_path)
    files = _resolve_json_files(cfg, json_dir, limit)

    all_mentions: list[dict] = []
    all_transfers: list[dict] = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                case = json.load(f)
        except Exception as e:
            print(f"skip {fp.name}: {e}")
            continue
        mentions, transfers = extract_from_case(case, cfg, source_file=fp.name)
        for t in transfers:
            t["ucid"] = case.get("ucid")
            t["source_file"] = fp.name
        all_mentions.extend(mentions)
        all_transfers.extend(transfers)

    return _finalize_mentions(cfg, all_mentions, all_transfers, write=write)


DEFAULT_TYPE_CONFIGS: dict[str, str] = {
    "judge": "configs/judges.yaml",
    "firm": "configs/firms.yaml",
    "party": "configs/parties.yaml",
}


def extract_mentions_unified(
    json_dir: str,
    *,
    config_paths: dict[str, str] | None = None,
    limit: int | None = None,
    write: bool = True,
    combined_out_rel: str = "data/mentions/all_mentions.jsonl",
) -> dict[str, list[dict]]:
    """One filesystem pass over PACER JSONs; extract all entity types via their configs.

    Each JSON file is read once. Per-type extraction still uses that type's YAML
    (``extract_from_case`` + the same finalize gates as ``extract_mentions``).
    Mentions remain tagged with ``entity_type`` from each config.

    Returns ``{entity_type: [mentions...]}`` (keys: judge / firm / party).
    """
    paths = dict(DEFAULT_TYPE_CONFIGS)
    if config_paths:
        paths.update(config_paths)

    cfgs: dict[str, dict] = {}
    for etype, cpath in paths.items():
        cfg = load_config(cpath)
        # Ensure key matches config entity_type when present
        cfg_etype = cfg.get("entity_type") or etype
        cfgs[cfg_etype] = cfg

    # File list from the first config (same glob semantics); json_dir overrides
    first_cfg = next(iter(cfgs.values()))
    files = _resolve_json_files(first_cfg, json_dir, limit)

    buckets: dict[str, list[dict]] = {etype: [] for etype in cfgs}
    transfers: dict[str, list[dict]] = {etype: [] for etype in cfgs}

    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                case = json.load(f)
        except Exception as e:
            print(f"skip {fp.name}: {e}")
            continue
        # Isolate per-type mutation (record_copy_fill etc.) without re-reading disk
        for etype, cfg in cfgs.items():
            case_i = copy.deepcopy(case)
            mentions, xfers = extract_from_case(case_i, cfg, source_file=fp.name)
            for t in xfers:
                t["ucid"] = case.get("ucid")
                t["source_file"] = fp.name
            buckets[etype].extend(mentions)
            transfers[etype].extend(xfers)

    finalized: dict[str, list[dict]] = {}
    for etype, cfg in cfgs.items():
        print(f"--- finalize entity_type={etype} ---", flush=True)
        finalized[etype] = _finalize_mentions(
            cfg,
            buckets[etype],
            transfers[etype],
            write=write,
            # Avoid clobbering one transfer_clues.jsonl when all share OUTPUT_DIR
            transfer_out_rel=f"data/mentions/transfer_clues_{etype}.jsonl",
        )

    if write:
        # Combined type-tagged stream (one file); per-type files already written above
        combined = resolve_path(first_cfg, combined_out_rel)
        combined.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(combined, "w", encoding="utf-8") as f:
            for etype in ("judge", "firm", "party"):
                for m in finalized.get(etype) or []:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    n += 1
        print(f"Wrote {n} combined type-tagged mentions → {combined}")

    return finalized


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/judges.yaml")
    ap.add_argument("--json-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--unified",
        action="store_true",
        help="One-pass extract for judges+firms+parties (requires --json-dir)",
    )
    args = ap.parse_args()
    if args.unified:
        if not args.json_dir:
            raise SystemExit("--unified requires --json-dir")
        extract_mentions_unified(args.json_dir, limit=args.limit)
    else:
        extract_mentions(args.config, json_dir=args.json_dir, limit=args.limit)
