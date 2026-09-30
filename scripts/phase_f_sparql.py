#!/usr/bin/env python3
"""Phase F — load judges.ttl into Oxigraph and run acceptance SPARQL queries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyoxigraph import Literal, RdfFormat, Store


QUERIES = {
    "a_llm_merges_conf_lt_80": """
PREFIX pacer: <http://scales-kg.org/pacer#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?dec ?method ?conf ?decision ?rationale ?from ?to
WHERE {
  ?dec a pacer:ResolutionDecision ;
       pacer:hasMethod ?method ;
       pacer:hasConfidence ?conf ;
       pacer:hasDecision ?decision ;
       pacer:hasRationale ?rationale .
  OPTIONAL { ?dec pacer:from ?from . }
  OPTIONAL { ?dec pacer:to ?to . }
  FILTER(STRSTARTS(STR(?method), "tier3"))
  FILTER(?decision = "MERGE_TIER3" || ?decision = "MATCH")
  FILTER(xsd:integer(STR(?conf)) < 80)
}
ORDER BY ?conf
LIMIT 50
""",
    "b_evidence_trail_named_judge": """
PREFIX pacer: <http://scales-kg.org/pacer#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?judge ?pref ?sjid ?mention ?mname ?case ?ucid ?dec ?method ?conf ?rationale
WHERE {
  ?judge a pacer:Judge ;
         skos:prefLabel ?pref ;
         pacer:hasSJID ?sjid .
  FILTER(CONTAINS(LCASE(STR(?pref)), "zagel"))
  OPTIONAL {
    ?mention pacer:resolvedTo ?judge ;
             pacer:hasName ?mname ;
             pacer:mentionsIn ?case .
    OPTIONAL { ?case pacer:hasUcid ?ucid . }
  }
  OPTIONAL {
    ?dec a pacer:ResolutionDecision ;
         pacer:hasMethod ?method ;
         pacer:hasConfidence ?conf ;
         pacer:hasRationale ?rationale .
    { ?dec pacer:from ?mention } UNION { ?dec pacer:to ?mention }
  }
}
LIMIT 100
""",
    "c_entity_count_per_court": """
PREFIX pacer: <http://scales-kg.org/pacer#>
SELECT ?court (COUNT(DISTINCT ?judge) AS ?n_entities)
WHERE {
  ?judge a pacer:Judge ;
         pacer:hasCode ?court .
}
GROUP BY ?court
ORDER BY DESC(?n_entities)
""",
}


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, Literal):
        return str(v.value)
    return str(v)


def run_query(store: Store, name: str, sparql: str) -> list[dict]:
    rows = []
    for row in store.query(sparql):
        # QuerySolution acts like a mapping of variable → term
        try:
            keys = list(row)
        except TypeError:
            keys = []
        if hasattr(row, "keys"):
            d = {k: _cell(row[k]) for k in row.keys()}
        else:
            # fallback: positional
            d = {f"c{i}": _cell(v) for i, v in enumerate(row)}
        rows.append(d)
    return rows


def main() -> int:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/runs/final_v3_14b_identfix"
    ttl = run / "rdf" / "judges.ttl"
    out_dir = run / "reports" / "phase_f"
    out_dir.mkdir(parents=True, exist_ok=True)

    store = Store()
    print(f"Loading {ttl} ({ttl.stat().st_size} bytes)...", flush=True)
    store.load(ttl.read_bytes(), RdfFormat.TURTLE)
    print("Loaded.", flush=True)

    results = {}
    md = ["# Phase F — Oxigraph SPARQL acceptance queries", "", f"TTL: `{ttl}`", ""]
    for name, sparql in QUERIES.items():
        print(f"Running {name}...", flush=True)
        try:
            rows = run_query(store, name, sparql)
        except Exception as e:
            rows = [{"error": str(e)}]
        results[name] = {"n": len(rows), "rows": rows[:50], "sparql": sparql.strip()}
        md.append(f"## {name}")
        md.append("")
        md.append(f"Rows: **{len(rows)}**")
        md.append("")
        md.append("```sparql")
        md.append(sparql.strip())
        md.append("```")
        md.append("")
        if rows and "error" in rows[0]:
            md.append(f"ERROR: `{rows[0]['error']}`")
        else:
            for r in rows[:15]:
                md.append(f"- `{json.dumps(r, ensure_ascii=False)[:300]}`")
        md.append("")

    (out_dir / "sparql_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "sparql_results.md").write_text("\n".join(md))
    print(json.dumps({k: v["n"] for k, v in results.items()}, indent=2))
    print("Wrote", out_dir / "sparql_results.md")

    # Tentris probe
    import urllib.request
    tentris = "http://localhost:9080/sparql"
    tentris_status = {"endpoint": tentris, "reachable": False, "blocker": None}
    try:
        req = urllib.request.Request(tentris, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            tentris_status["reachable"] = True
            tentris_status["http_status"] = resp.status
    except Exception as e:
        tentris_status["blocker"] = f"{type(e).__name__}: {e}"
    (out_dir / "tentris_probe.json").write_text(json.dumps(tentris_status, indent=2))
    print("Tentris:", json.dumps(tentris_status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
