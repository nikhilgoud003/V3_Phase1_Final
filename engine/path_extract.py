"""Generic nested JSON path extraction (type-agnostic).

Supports paths like ``parties[].counsel[].entity_info.office_name`` with
``carry_fields`` and ``derive_fields`` from entity-type YAML — used by firms
and any future type without hardcoded Python branches per entity.
"""

from __future__ import annotations

import copy
import re
from typing import Any

_DEFAULT_EMPTY_IGNORE = frozenset({"raw_info", "terminating_date"})


def _payload_is_empty(obj: Any, ignore: set[str] | frozenset[str]) -> bool:
    if not isinstance(obj, dict):
        return True
    for k, v in obj.items():
        if k in ignore:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if v not in (None, "", [], {}):
            return False
    return True


def _payload_fill_score(obj: Any, ignore: set[str] | frozenset[str]) -> int:
    if not isinstance(obj, dict):
        return 0
    n = 0
    for k, v in obj.items():
        if k in ignore:
            continue
        if isinstance(v, str) and v.strip():
            n += 1
        elif v not in (None, "", [], {}):
            n += 1
    return n


def apply_record_copy_fill(case: dict, spec: dict) -> tuple[dict, dict]:
    """Copy empty payload fields from a same-case sibling sharing match_field.

    Generic (no entity-type branches). Typical firms use: walk counsel
    records, when ``has_see_above`` and empty ``entity_info``, fill from
    another counsel with the same ``name`` whose payload is populated.
    """
    stats = {
        "flagged": 0,
        "empty_flagged": 0,
        "filled": 0,
        "no_donor": 0,
    }
    if not spec or not spec.get("enabled"):
        return case, stats
    list_path = spec.get("list_path") or ""
    match_field = spec.get("match_field") or "name"
    flag_field = spec.get("flag_field") or "has_see_above"
    payload_field = spec.get("payload_field") or "entity_info"
    donor_key = spec.get("donor_requires_key") or "office_name"
    ignore = frozenset(spec.get("ignore_payload_keys") or _DEFAULT_EMPTY_IGNORE)

    hits = walk_path(case, list_path)
    records: list[dict] = []
    for val, _ctx in hits:
        if isinstance(val, dict):
            records.append(val)
    if not records:
        return case, stats

    groups: dict[str, list[dict]] = {}
    for rec in records:
        key = rec.get(match_field)
        if not isinstance(key, str) or not key.strip():
            continue
        groups.setdefault(key.strip(), []).append(rec)

    def _is_donor(rec: dict) -> bool:
        payload = rec.get(payload_field)
        if not isinstance(payload, dict):
            return False
        req = payload.get(donor_key)
        if not isinstance(req, str) or not req.strip():
            return False
        return not _payload_is_empty(payload, ignore)

    for rec in records:
        if rec.get(flag_field) is not True:
            continue
        stats["flagged"] += 1
        payload = rec.get(payload_field) if isinstance(rec.get(payload_field), dict) else {}
        if not _payload_is_empty(payload, ignore):
            continue
        stats["empty_flagged"] += 1
        name = rec.get(match_field)
        if not isinstance(name, str) or not name.strip():
            stats["no_donor"] += 1
            continue
        donors = [d for d in groups.get(name.strip(), []) if d is not rec and _is_donor(d)]
        if not donors:
            stats["no_donor"] += 1
            continue
        donors.sort(
            key=lambda d: _payload_fill_score(d.get(payload_field), ignore),
            reverse=True,
        )
        src = donors[0].get(payload_field) or {}
        dest = rec.setdefault(payload_field, {})
        if not isinstance(dest, dict):
            rec[payload_field] = copy.deepcopy(src)
            stats["filled"] += 1
            continue
        changed = False
        for k, v in src.items():
            if dest.get(k) in (None, "", [], {}):
                dest[k] = copy.deepcopy(v)
                changed = True
        if changed:
            stats["filled"] += 1
        else:
            stats["no_donor"] += 1
    return case, stats


def walk_path(root: Any, path: str) -> list[tuple[Any, dict[str, Any]]]:
    """Expand a dotted path with ``[]`` array wildcards.

    Returns list of ``(value, context)`` where context maps:
      - ``indices``: list of (segment_name, index) for each ``[]`` hop
      - ``objects``: list of parent dicts for each object hop
      - ``{segment}_index``: last index for that segment (e.g. parties_index)
    """
    if not path:
        return []
    parts = path.split(".")
    states: list[tuple[Any, dict[str, Any]]] = [(root, {"indices": [], "objects": []})]

    def _copy_aliases(src: dict[str, Any], dst: dict[str, Any]) -> None:
        for k, v in src.items():
            if k in {"indices", "objects"}:
                continue
            dst[k] = v

    for part in parts:
        is_array = part.endswith("[]")
        key = part[:-2] if is_array else part
        nxt: list[tuple[Any, dict[str, Any]]] = []
        for cur, ctx in states:
            if not isinstance(cur, dict):
                continue
            if key not in cur:
                continue
            val = cur[key]
            if is_array:
                if not isinstance(val, list):
                    continue
                for i, item in enumerate(val):
                    nctx: dict[str, Any] = {
                        "indices": list(ctx["indices"]) + [(key, i)],
                        "objects": list(ctx["objects"])
                        + ([item] if isinstance(item, dict) else []),
                    }
                    _copy_aliases(ctx, nctx)
                    nctx[f"{key}_index"] = i
                    if key == "parties":
                        nctx["party_index"] = i
                        nctx["party"] = item if isinstance(item, dict) else None
                    if key == "counsel":
                        nctx["counsel_index"] = i
                        nctx["counsel"] = item if isinstance(item, dict) else None
                    nxt.append((item, nctx))
            else:
                nctx = {
                    "indices": list(ctx["indices"]),
                    "objects": list(ctx["objects"]) + ([cur] if isinstance(cur, dict) else []),
                }
                _copy_aliases(ctx, nctx)
                nxt.append((val, nctx))
        states = nxt
    return states


def resolve_carry_value(case: dict, spec: str, ctx: dict[str, Any]) -> Any:
    """Resolve a carry_fields spec against the walk context.

    Spec forms:
      - ``$party_index`` / ``$counsel_index`` → context index
      - ``parties[].name`` → name of the party in context
      - ``parties[].counsel[].entity_info.address`` → field on current counsel entity_info
      - plain string leaf relative to deepest matching object
    """
    if spec is None:
        return None
    s = str(spec)
    if s.startswith("$"):
        key = s[1:]
        if key == "index":
            indices = ctx.get("indices") or []
            if indices:
                return indices[-1][1]
            return ctx.get("party_index") if ctx.get("party_index") is not None else ctx.get("docket_index")
        return ctx.get(key)

    # Prefer resolving with context indices when path has []
    if "[]" in s:
        parts = s.split(".")
        cur: Any = case
        for part in parts:
            is_array = part.endswith("[]")
            key = part[:-2] if is_array else part
            if not isinstance(cur, dict):
                return None
            if is_array:
                idx = ctx.get(f"{key}_index")
                if idx is None:
                    # fallback: last index of that name in indices list
                    for seg, i in reversed(ctx.get("indices") or []):
                        if seg == key:
                            idx = i
                            break
                arr = cur.get(key)
                if not isinstance(arr, list) or idx is None or idx >= len(arr):
                    return None
                cur = arr[idx]
            else:
                cur = cur.get(key)
        return cur

    # Simple top-level or relative key
    if isinstance(case, dict) and s in case:
        return case.get(s)
    counsel = ctx.get("counsel")
    if isinstance(counsel, dict):
        ei = counsel.get("entity_info") if isinstance(counsel.get("entity_info"), dict) else {}
        if s in ei:
            return ei.get(s)
        if s in counsel:
            return counsel.get(s)
    party = ctx.get("party")
    if isinstance(party, dict) and s in party:
        return party.get(s)
    return None


def derive_field(kind: str, values: dict[str, Any]) -> Any:
    """Config-driven derived fields (generic helpers)."""
    if kind == "email_domain":
        email = (values.get("email") or "").strip()
        if "@" in email:
            return email.split("@", 1)[1].strip().lower() or None
        return None
    return None


def apply_office_classification(mention: dict, cfg: dict) -> dict:
    """Attach office_class / office_subclass from config patterns+lists. Generic."""
    oc = cfg.get("office_classification") or {}
    if not oc.get("enabled"):
        return mention
    name = (mention.get("normalized_name") or "").strip()
    raw = (mention.get("raw_name") or name).strip()
    probe = f"{name} {raw}".lower()

    for cls in oc.get("classes") or []:
        cid = cls.get("id") or "unknown"
        # list_path membership (Big Law cores etc.)
        list_path = cls.get("list_path")
        if list_path:
            from pathlib import Path
            from engine.config_loader import resolve_path

            p = resolve_path(cfg, list_path)
            if p.exists():
                cores = {
                    ln.strip().lower()
                    for ln in p.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.strip().startswith("#")
                }
                # substring / equality match on normalized name
                hit = None
                for core in cores:
                    if name == core or core in name or name in core:
                        hit = core
                        break
                if hit:
                    mention["office_class"] = cid
                    mention["office_subclass"] = hit.replace(" ", "_")[:80]
                    if cls.get("drop_mention"):
                        mention["_drop_classified"] = True
                    return mention

        for pat in cls.get("patterns") or []:
            # Match name and raw independently so ^...$ anchors work. The
            # concatenated probe is last (substring patterns that span both).
            m = None
            for text in (name, raw, probe):
                if not text:
                    continue
                try:
                    m = re.search(pat, text)
                except re.error:
                    m = None
                    break
                if m:
                    break
            if not m:
                continue
            mention["office_class"] = cid
            subclass = None
            sfrom = cls.get("subclass_from")
            if sfrom == "capture_group_1" and m.lastindex:
                g1 = m.group(1)
                if g1:
                    subclass = re.sub(r"\s+", "_", g1.strip().lower())[:80]
            elif sfrom in {"state_match", "gpe_match", "court_or_name", "biglaw_id"}:
                g0 = m.group(0)
                subclass = (g0.strip().lower()[:80] if g0 else cid)
            mention["office_subclass"] = subclass
            if cls.get("drop_mention"):
                mention["_drop_classified"] = True
            return mention

    mention.setdefault("office_class", "private_other")
    mention.setdefault("office_subclass", None)
    return mention
