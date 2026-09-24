"""Hard pre-insert URI safety check. Failures abort; there is no skip flag.

The insert CLI queries a live Tentris SPARQL endpoint. TTL parsing remains
only as a unit-test helper and must not be used immediately before a real insert.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

JUDGE_URI_RE = re.compile(r"https?://scales-kg\.org/id/judge/(SJ[0-9A-Za-z]+)")

EXISTING_URI_QUERY = """
PREFIX pacer: <http://scales-kg.org/pacer#>
SELECT DISTINCT ?s WHERE {
  { ?s a pacer:Judge }
  UNION { ?s a pacer:LawFirm }
  UNION { ?s a pacer:NameAssertion }
  UNION { ?s a pacer:ResolutionDecision }
}
"""


class InsertSafetyError(RuntimeError):
    """Raised when a TTL must not be inserted into the target graph."""


def judge_uris_from_ttl(path: str | Path) -> set[str]:
    text = Path(path).read_text(encoding="utf-8")
    return {f"http://scales-kg.org/id/judge/{m.group(1)}" for m in JUDGE_URI_RE.finditer(text)}


def load_assignments(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise InsertSafetyError(f"assignment file missing (cannot skip safety check): {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not data.get("assignments"):
        raise InsertSafetyError(f"assignment file has no assignments: {p}")
    return data


def parse_sparql_uri_bindings(data: dict[str, Any]) -> set[str]:
    """Extract subject URIs from a SPARQL JSON results document."""
    bindings = (data.get("results") or {}).get("bindings") or []
    out: set[str] = set()
    for b in bindings:
        val = (b.get("s") or {}).get("value")
        if val:
            out.add(val)
    return out


def uris_from_sparql(endpoint: str, timeout: float = 120.0) -> set[str]:
    """Query a live SPARQL endpoint for existing entity / alias / decision URIs."""
    if not endpoint or not str(endpoint).strip():
        raise InsertSafetyError("SPARQL endpoint is required (cannot check against a TTL file)")
    body = urllib.parse.urlencode({"query": EXISTING_URI_QUERY}).encode()
    last_err: Exception | None = None
    for accept in ("application/json", "application/sparql-results+json"):
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Accept": accept,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
            data = json.loads(raw)
            return parse_sparql_uri_bindings(data)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as e:
            last_err = e
            continue
    raise InsertSafetyError(f"SPARQL query failed against {endpoint}: {last_err}")


def check_insert_safety(
    *,
    assignments: dict[str, Any],
    target_uris: Iterable[str] | None = None,
    target_judge_uris: Iterable[str] | None = None,
    allow_empty_target: bool = False,
) -> dict[str, Any]:
    """Abort-worthy check: CREATE URIs must be new; REUSE URIs must already exist."""
    target = set(target_uris if target_uris is not None else (target_judge_uris or []))
    if not target and not allow_empty_target:
        raise InsertSafetyError(
            "target URI set is empty — refusing to treat the graph as empty "
            "(live SPARQL must return existing entity/alias/decision URIs)"
        )
    create_collisions: list[dict] = []
    reuse_missing: list[dict] = []
    unlabeled: list[dict] = []

    for row in assignments.get("assignments") or []:
        action = (row.get("action") or "").upper()
        uri = row.get("uri") or ""
        if not uri:
            unlabeled.append(row)
            continue
        if action == "CREATE":
            if uri in target:
                create_collisions.append(row)
        elif action == "REUSE":
            if uri not in target:
                reuse_missing.append(row)
        else:
            unlabeled.append(row)

    ok = not create_collisions and not reuse_missing and not unlabeled
    report = {
        "ok": ok,
        "n_target_uris": len(target),
        "n_target_judge_uris": len(target),
        "n_assignments": len(assignments.get("assignments") or []),
        "n_reuse": assignments.get("n_reuse"),
        "n_create": assignments.get("n_create"),
        "n_alias_create": assignments.get("n_alias_create"),
        "n_decision_create": assignments.get("n_decision_create"),
        "create_uri_already_in_target": create_collisions,
        "reuse_uri_missing_from_target": reuse_missing,
        "unlabeled_or_missing_uri": unlabeled,
    }
    if not ok:
        parts = []
        if create_collisions:
            parts.append(
                f"CREATE URI already exists in target ({len(create_collisions)}): "
                + ", ".join(
                    (r.get("kind") or "uri") + "=" + r.get("uri", "") + "/" + str(r.get("canonical_name") or r.get("label") or r.get("run_decision_id") or "")
                    for r in create_collisions[:8]
                )
            )
        if reuse_missing:
            parts.append(
                f"REUSE URI missing from target ({len(reuse_missing)}): "
                + ", ".join(r.get("uri", "") + "=" + str(r.get("canonical_name")) for r in reuse_missing[:8])
            )
        if unlabeled:
            parts.append(f"{len(unlabeled)} assignment(s) missing action/uri")
        raise InsertSafetyError("INSERT ABORTED. " + " | ".join(parts))
    return report


def check_ttl_against_target(
    *,
    assignment_path: str | Path,
    target_ttl: str | Path,
) -> dict[str, Any]:
    """Unit-test helper only. Real inserts must use check_assignments_against_sparql."""
    assignments = load_assignments(assignment_path)
    return check_insert_safety(
        assignments=assignments,
        target_judge_uris=judge_uris_from_ttl(target_ttl),
        allow_empty_target=True,
    )


def check_assignments_against_sparql(
    *,
    assignment_path: str | Path,
    sparql_endpoint: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Required path immediately before a real insert: live SPARQL, not a TTL file."""
    assignments = load_assignments(assignment_path)
    uris = uris_from_sparql(sparql_endpoint, timeout=timeout)
    report = check_insert_safety(assignments=assignments, target_uris=uris)
    report["sparql_endpoint"] = sparql_endpoint
    return report
