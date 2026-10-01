# Tier_V3 — SCALES / OKN PACER entity resolution

**Final product doc:** [`docs/FINAL_V3.md`](docs/FINAL_V3.md)  
**One command — all ER (judges+firms+parties):** [`docs/RUN_NEW_JSONS.md`](docs/RUN_NEW_JSONS.md)  
**File / folder map (judges · firms · parties):** [`FINAL_FILES/README.md`](FINAL_FILES/README.md)

Scrap, duplicates, and superseded proofs live in  
[`data/reports/archive/20260921_v3_cleanup/`](data/reports/archive/20260921_v3_cleanup/README.md).

## Entity types

| | Config | Current run |
|--|--------|-------------|
| Judges | `configs/judges.yaml` | `data/runs/judges_pilot_recall_fix/` |
| Firms | `configs/firms.yaml` | `data/runs/firms_pilot/` |
| Parties | `configs/parties.yaml` | `data/runs/parties_pilot/` |

**Parties USA fix:** same-UCID alias merge for USA / United States / U.S.; cross-UCID pairs never go to Tier3. Details in `docs/FINAL_V3.md`.

## Quick start (one upload → full ER)

Put PACER `*.json` in a folder, then **one** command runs judges + firms + parties:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Ollama: qwen2.5:7b + nomic-embed-text

python3 scripts/run_all_er.py \
  --json-dir data/json/nyed_connectivity_test \
  --load-tentris-clone
```

Requires Ollama for Tier3. Tentris details: `docs/RUN_NEW_JSONS.md` / `docs/KG_INCREMENTAL_INSERT_RUNBOOK.md`.  
(Advanced: still can call `scripts/run_pilot.py --config configs/<type>.yaml` per type.)

## Unified one-pass run (judges + firms + parties, file by file)

`scripts/unified_5file_poc.py` reads each PACER JSON once, resolves it against
everything saved so far, and writes one output folder.

```bash
# macOS only: spaCy and FAISS each ship OpenMP; this lets both load
export KMP_DUPLICATE_LIB_OK=TRUE

# New run (refuses a non-empty folder)
caffeinate -i python3 scripts/unified_5file_poc.py \
  --json-dir /path/to/json --fresh \
  --checkpoint-every 100 \
  --output-dir data/runs/my_run

# Add new files later: same command without --fresh, same --output-dir.
# Already processed files are skipped; saved IDs are kept.
python3 scripts/unified_5file_poc.py --json-dir /path/to/json --output-dir data/runs/my_run

# RDF (Turtle) from a finished run
python3 scripts/emit_rdf_from_run.py --run-dir data/runs/my_run
```

| Option | What it does |
|---|---|
| `--fresh` | Start from file 1 in an empty `--output-dir`. |
| `--limit N` / `--files a.json b.json` | Process only the first N files / these files. |
| `--checkpoint-every N` | Also save the checkpoint every N files (default: once at the end). |
| `--workers N` | Processes for JSON reading + extraction (default `configs/unified.yaml` `extract_workers`; 0 = CPU count − 1; 1 = in-process). Results do not change. |
| `--schema-walk on\|off` | Override `configs/unified.yaml` `schema_free_walk.enabled` (default off). |
| `--debug-steps` | Keep the last work folder, the walk log and the field-type cache. |

Speed settings (results unchanged):

- Ollama health check once per run; batched embeddings with one vector cache per run (`checkpoint/embed_cache.npz`); embeddings/FAISS skipped when Tier2 cannot produce a pair.
- LLM answers cached for the whole run by model + full prompt (`checkpoint/llm_prompt_cache.jsonl`); the per-file Tier3 decision cache is unchanged.
- The LLM name check sends `name_validity.llm_validation.parallel_requests` (judges.yaml) requests at once. This only helps when Ollama serves requests in parallel: set `OLLAMA_NUM_PARALLEL` (for the Mac app: `launchctl setenv OLLAMA_NUM_PARALLEL 4`, then quit and reopen Ollama).

Matching rules added (all in config):

| Rule | Config |
|---|---|
| Same company name + same legal-form family → one ID in any court | `parties.yaml` `exact_company_name_global`, `company_legal_forms` |
| Federal agency names from the official Federal Register list → one ID | `parties.yaml` `federal_agencies`, `data/external/federal_agencies_federalregister.json` |
| Same FJC id → same judge in any court; different FJC ids never link | `scripts/incremental_resolve.py` (link step) |
| Docket prose cut from judge names (`:`, glued words, trailing lowercase words) | `judges.yaml` `normalization.name_cleaning` |
| Different names listed as separate parties in one case never merge | `parties.yaml` `coparty_barrier` |
| fka / aka / dba = same entity; successor / predecessor / alter ego / c/o = related entity | `parties.yaml` `alias_extraction`, Tier0 `explicit_alias` |

Outputs in `--output-dir`: `entities.jsonl`, `mentions.jsonl`, `decisions.jsonl`
(every merge / no-merge with its reason), `summary.json`, and `checkpoint/`
(needed to add files later).

Checking a run: `scripts/er_quality_metrics.py RUN_DIR` (FJC splits, same-name
splits, co-defendant merges, junk, aliases, plus V2 on the same files);
`scripts/compare_runs_identical.py A B`; `scripts/diff_entity_partitions.py A B`;
`scripts/compare_metrics.py A/metrics.json B/metrics.json`.
