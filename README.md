# V3 Phase 1 — unified ER → RDF → Tentris

Cold-start entity resolution (judges + firms + parties) on PACER JSON files, then RDF emit and load into a **test** Tentris store.

**Not included:** PACER input JSON files. Put your own under `data/json/`.

---

## 0. One-time setup

```bash
git clone https://github.com/nikhilgoud003/V3_Phase1_Final.git
cd V3_Phase1_Final
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

```bash
# Ollama (Tier2 embeddings + Tier3 LLM). Install from https://ollama.com then:
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

```bash
# Tentris v1.1.0 binary on PATH (example location). Used only for test KG load/serve.
export PATH="$HOME/.local/bin:$PATH"
tentris --help
```

---

## 1. Place input JSON

```bash
# Copy your PACER *.json dockets here (not committed to git)
mkdir -p data/json/input
cp /path/to/your/*.json data/json/input/
```

---

## 2. Run entity resolution (start → results)

```bash
# Writes entities.jsonl, mentions.jsonl, decisions.jsonl, summary.json
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
python3 scripts/unified_5file_poc.py \
  --json-dir data/json/input \
  --output-dir data/runs/my_run
```

**What it does:** Reads each JSON once, extracts judge/firm/party mentions, runs Tier0–3 cascade on the cumulative pool, writes one flat results folder under `data/runs/my_run/`.

Optional: `--debug-steps` also writes per-file `step_XX_*` snapshots (off by default).

---

## 3. Emit RDF (.ttl)

```bash
python3 scripts/emit_rdf_from_run.py --run-dir data/runs/my_run
```

**What it does:** Runs the existing `emit_ttl` path for judges, firms, and parties; writes:

- `data/runs/my_run/rdf/judges.ttl`
- `data/runs/my_run/rdf/firms.ttl`
- `data/runs/my_run/rdf/parties.ttl`
- `data/runs/my_run/rdf/entities.ttl` (combined)

---

## 4. Load into a new Tentris test store (not live)

```bash
export PATH="$HOME/.local/bin:$PATH"

# Brand-new empty store (separate from any live :9080 graph)
rm -rf data/tentris_test_9082
mkdir -p data/tentris_test_9082
chmod 700 data/tentris_test_9082
tentris --datastore-path data/tentris_test_9082 init
chmod -R 700 data/tentris_test_9082

# Load TTL (serve must be down for this path)
tentris --datastore-path data/tentris_test_9082 load --format turtle data/runs/my_run/rdf/entities.ttl
```

**What it does:** Creates an empty Tentris database and loads your run’s Turtle. Does **not** touch any live datastore.

---

## 5. Start Tentris (test port)

```bash
export PATH="$HOME/.local/bin:$PATH"
tentris --datastore-path data/tentris_test_9082 serve 127.0.0.1:9082
```

**What it does:** Serves SPARQL on port **9082** (use this so it cannot be confused with live `:9080`).

- SPARQL: http://127.0.0.1:9082/sparql  
- UI: http://127.0.0.1:9082/ui  

Stop: `Ctrl+C`.

Quick count check (another terminal):

```bash
curl -s -H 'Accept: application/sparql-results+json' \
  --data-urlencode 'query=SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }' \
  http://127.0.0.1:9082/sparql
```
