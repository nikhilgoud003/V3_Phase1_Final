# Incremental KG insert runbook

Insert new judge RDF into an existing Tentris datastore **without** reminting
SJIDs already in the live graph. This is the measured path after the NYED
14-file smoke test. It is **not** a live-graph insert until the operator
explicitly points `--datastore-path` at the production directory.

Live graph of record: `data/tentris_judges_recall_fix_data`
(**323,032 triples / 906 judges / 1,865 LawFirm** on Tentris **v1.1.0** as of
2026-09-02). Always clone-verify first; only write the live path when the
operator explicitly approves.

**Tentris v1.1.0 (non-commercial, no license file):** bundled community license
(`limit-cores = 1`). Remove any expired `~/.config/tentris-license.toml` or
Tentris will prefer it over the bundled license. Old binary preserved at
`~/.local/bin/tentris-0.22.5-beta`; default `tentris` is v1.1.0.

**POBR migration (2026-09-02):** v1.0.0+ is **not** POBR-compatible with
0.22.5+beta. Live graph was rebuilt from TTL (`judges.ttl` + NYED incremental
`nyed_smoke_14_v2/rdf/judges.ttl` + `firms.ttl`). Old POBR archive:
Historical Tentris backups/clones (not live) were moved 2026-09-21 to
`data/reports/archive/20260921_v3_cleanup/tentris_unused/` (see that folder’s
`MOVE_MANIFEST.json`). One local safety backup remains beside live:
`data/tentris_judges_recall_fix_data_backup_20260908T134353Z_pre_punct_fold_reload`.

Datastore directory must be mode **700** (`chmod -R 700`) or v1.1.0 refuses to
start.

## Start live server (v1.1.0, no license file)

```bash
export PATH="$HOME/.local/bin:$PATH"
tentris --datastore-path data/tentris_judges_recall_fix_data serve 127.0.0.1:9080
```

No `--license` flag. If `~/.config/tentris-license.toml` exists (e.g. expired
GSU license), remove or rename it so the bundled non-commercial license is used.

## Firms bulk insert (2026-08-21)

1. Clone-verify (`data/tentris_firms_insert_clone`) then timestamped backup of live.
2. Live SPARQL preinsert safety (`scripts/preinsert_sjid_safety_check.py`) — no `--force`.
3. `tentris load --format turtle data/runs/firms_pilot/rdf/firms.ttl` with serve down (~**0.41 s**).
4. Verify on `:9080`: firms=1867, judges=906, triples=322992; four controls; Garcia three-way; Zagel trail.

Firms TTL uses **namespaced** URIs (`/id/firm/`, `/id/firm_mention/`, `/id/firm_alias/`,
`/id/firm_resolution/inc_…`) plus shared `/id/case/`. Entity = **physical office**
(city/state), not district-level USAO/FPD — see `FIRMS_KNOWN_LIMITATIONS.md`.

## Always: clone first

```bash
SRC=data/tentris_judges_recall_fix_data
DST=data/tentris_insert_clone   # throwaway
rsync -a "$SRC/" "$DST/"
chmod -R 700 "$DST"
tentris --datastore-path "$DST" serve 127.0.0.1:9081
```

Query `http://127.0.0.1:9081/sparql`. Confirm triples = 323032, judges = 906,
firms = 1865 **before** any update.

## Emit registry-aware TTL

Cascade the new JSON into an isolated output dir, then emit against the run-of-record registry:

```bash
export TIER_V3_OUTPUT_DIR="$(pwd)/data/runs/new_slice"
python3 scripts/run_pilot.py \
  --json-dir data/json/new_slice \
  --target-registry data/runs/judges_pilot_recall_fix/clusters/judges_entity_registry.jsonl
```

`--target-registry` remaps:

| Kind | REUSE | CREATE |
|------|--------|--------|
| Judge SJID | existing `SJ000xxx` from the registry | next serial after max (`SJ000896+`) |
| Alias URI | n/a (new namespace) | `{sjid}_{pref\|alt}_{sha1}` — never `{sjid}_pref_0` |
| Decision URI | n/a (new namespace) | `inc_{sha1}` — never `dec_00000…` |

Standalone emit (no `--target-registry`) is unchanged: sequential SJIDs,
`{sjid}_pref_{i}`, `dec_00000…`.

## Pre-insert safety check (live SPARQL, required)

Immediately before insert, against the **running clone**, not a TTL file:

```bash
python3 scripts/preinsert_sjid_safety_check.py \
  --assignments "$TIER_V3_OUTPUT_DIR/rdf/judges_sjid_assignments.json" \
  --sparql-endpoint http://127.0.0.1:9081/sparql
```

- CREATE URIs must **not** exist in the live graph.
- REUSE URIs must **already** exist.
- There is no `--force`. Exit 1 aborts the insert.
- The script queries Tentris SPARQL for `pacer:Judge`, `pacer:NameAssertion`,
  and `pacer:ResolutionDecision` subjects. It does **not** parse a TTL file.

## How to insert (two supported paths)

### A. SPARQL `INSERT DATA` while serve is up (preferred for incremental)

Convert Turtle → N-Triples, wrap in `INSERT DATA { … }`, POST to `/update`.
Tentris v1.1.0 expects `Accept: application/json` on `/update` (not
`application/sparql-results+json`).

```python
# sketch — see clone re-verify in data/runs/nyed_smoke_14_v2/reports/
from rdflib import Graph
g = Graph(); g.parse("judges.ttl", format="turtle")
nt = g.serialize(format="nt")
update = "INSERT DATA {\n" + nt + "\n}"
# POST urlencoded update= to http://127.0.0.1:9081/update
```

### B. `tentris load` with serve **down**

```bash
# serve must not hold this datastore
tentris --datastore-path "$DST" load --format turtle path/to/new.ttl
```

`tentris load` while serve holds the same path fails (`os error 104`).

## `/graph-store` endpoint

On Tentris **0.22.5+beta**, POSTing Turtle to `/graph-store` **SIGSEGV'd** the
server (exit 139). **v1.1.0 does not crash** — a bare Turtle POST returns HTTP
400 (`missing graph identifier` per SPARQL Graph Store protocol). Prefer
**SPARQL `INSERT DATA` via `/update`** (serve up) or **`tentris load` with serve
down** for incremental inserts.

## Known limitation: court-personnel negative-evidence coverage is incomplete

`configs/judges.yaml` `extraction.negative_evidence.court_personnel` currently
mines only the three patterns that leaked on the Adenuga docket (same drop
category as party/counsel):

- **USPO / U.S. Probation Officer** (the Yara Suarez case)
- **Court reporter**
- **Clerk signature** `(Last, First)` near `(Entered:`

**Not audited / not covered** by that YAML list (residual risk if SpaCy PERSON
emits a bare name after a role token we do not mine):

- Law clerk
- Courtroom deputy
- USMS / marshal
- Pretrial services officer (PSO) spelling variants beyond the USPO regex
- Other courtroom staff roles not listed above

Interpreters are handled separately (`name_validity` `interpreter_prefix` +
`name_compat` prose stripping), **not** via `court_personnel`. Extending the
YAML pattern list is the intended fix path — do not treat the Adenuga fix as a
full role census.

## After insert (SPARQL)

Re-run at least:

- `SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }`
- `SELECT (COUNT(DISTINCT ?j) AS ?c) WHERE { ?j a pacer:Judge }`
- Classic demos: Garcia three-way (`SJ000041` / `SJ000107` / `SJ000132`);
  Zagel evidence trail (100 bindings, first `SJ000020`)
- REUSE identity probes (Donnelly `SJ000425`, Glasser `SJ000017`,
  Strawbridge `SJ000006`, Robreno `SJ000000`) — names and NIDs must not move.
- Every CREATE SJID ≥ registry max+1 and absent from 0–895.
- No `Yara Suarez` (or other court-personnel) judge node.
- No CREATE decision URI of the form `dec_00000…`; no CREATE alias of the
  form `{sjid}_pref_0`.

Only after a **fresh clone** re-verify matches the above is a production
datastore insert in scope — and only when the operator explicitly names
`data/tentris_judges_recall_fix_data`.

## Live insert record (2026-08-17)

Operator-approved write to `data/tentris_judges_recall_fix_data` via
`INSERT DATA` on `http://127.0.0.1:9080/update` from
`data/runs/nyed_smoke_14_v2/rdf/judges.ttl` (registry-aware emit; Yara-fixed).

| | Before | After |
|--|-------:|------:|
| Triples | 137,024 | **139,131** (+2,107) |
| Judges | 896 | **906** (+10 CREATE) |
| Insert wall time | — | **~0.15 s** |
| Full reload of combined NT (clone timing) | — | ~0.03 s init + ~0.28 s load |

Evidence: `data/runs/nyed_smoke_14_v2/reports/live_insert_sparql.json`,
`live_preinsert_safety_check.json`.
