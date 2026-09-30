# Run new PACER JSONs through V3 → check Tentris

**This is the operator cookbook** for: “I have a bunch of case JSON files →
run the full Tier_V3 cascade → see local results → optionally check / insert
into Tentris.”

## Preferred: one command for all entity types

Professor / demo expectation — upload/point at JSONs once, run ER for
**judges + firms + parties** without repeating configs:

```bash
cd Tier_V3
export KMP_DUPLICATE_LIB_OK=TRUE
export PATH="$HOME/.local/bin:$PATH"

# 1) Put PACER *.json in e.g. data/json/nyed_connectivity_test/

# 2) One shot: full ER for all three types (+ optional Tentris clone load)
python3 scripts/run_all_er.py \
  --json-dir data/json/nyed_connectivity_test \
  --load-tentris-clone
```

That writes:

```text
data/runs/nyed_connectivity_test_all_er/
  judges/   firms/   parties/   all_er_summary.json
```

and (with `--load-tentris-clone`) loads TTLs into a **clone** under
`data/tentris_nyed_connectivity_test_all_er_clone` (live KG untouched).
Then start Tentris as printed by the script (default `:9081`).

Smoke with 2 dockets: add `--limit 2`. Skip LLM: add `--skip-tier3`.

Related docs:

| Doc | Role |
|-----|------|
| [`FINAL_V3.md`](FINAL_V3.md) | Product summary + USA / parties policy |
| [`KG_INCREMENTAL_INSERT_RUNBOOK.md`](KG_INCREMENTAL_INSERT_RUNBOOK.md) | Clone → safety → insert into Tentris |
| [`FINAL_FILES/README.md`](../FINAL_FILES/README.md) | What each script/path is for |

---

## Prerequisites

```bash
cd Tier_V3
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Ollama: qwen2.5:7b + nomic-embed-text
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Advanced: single entity type only

Use only for debugging one type. Normal batches → `run_all_er.py`.

```bash
export TIER_V3_OUTPUT_DIR="$(pwd)/data/runs/my_batch_parties"
python3 scripts/run_pilot.py \
  --config configs/parties.yaml \
  --json-dir data/json/my_batch
```

| Flag / env | Meaning |
|------------|---------|
| `--json-dir PATH` | PACER JSON folder |
| `TIER_V3_OUTPUT_DIR` | Output root for that one type |
| `--limit N` | Smoke: first N dockets |
| `--skip-tier3` | No LLM |
| `--target-registry PATH` | Judges REUSE/CREATE (see KG runbook) |

---

## Check results

| Check | Where |
|-------|--------|
| All-ER summary | `data/runs/<name>_all_er/all_er_summary.json` |
| Per-type summary | `…/<type>/reports/pilot_summary.json` |
| Turtle | `…/<type>/rdf/*.ttl` |

Live Tentris read-only (`:9080`):

```bash
tentris --datastore-path data/tentris_judges_recall_fix_data serve 127.0.0.1:9080
```

Live insert (operator-approved only): [`KG_INCREMENTAL_INSERT_RUNBOOK.md`](KG_INCREMENTAL_INSERT_RUNBOOK.md).

---

## Do not

- Overwrite `data/runs/{judges_pilot_recall_fix,firms_pilot,parties_pilot}` unless intentional.
- Write the live Tentris store without clone + preinsert safety.
