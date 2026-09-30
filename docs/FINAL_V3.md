# Tier_V3 — Final documentation (SCALES / OKN)

**One document for the live product.** Older proof packs, logs, and phase notes were
moved to `data/reports/archive/20260921_v3_cleanup/` on 2026-09-21.

File map (what each path is for): see [`FINAL_FILES/`](../FINAL_FILES/README.md).

---

## What this is

Config-driven **4-tier early-exit entity resolution** over PACER dockets for three
entity types that share one engine:

| Type | Config | Run of record | ID prefix |
|------|--------|---------------|-----------|
| **Judges** | `configs/judges.yaml` | `data/runs/judges_pilot_recall_fix/` | `SJ` |
| **Firms** | `configs/firms.yaml` | `data/runs/firms_pilot/` | `SF` |
| **Parties** | `configs/parties.yaml` (v0.7.3) | `data/runs/parties_pilot/` | `SP` |

Cascade: **Tier0** (deterministic keys) → **Tier1** (blocking) → **Tier2**
(similarity) → **Tier3** (Ollama LLM, abstention rails) → cluster + RDF + registry.

---

## USA / United States fix (parties — ship focus)

**Problem.** The Tier3 prompt used to tell the model that USA / United States /
U.S. could “collapse to the same government party,” which invited cross-case merges
and hallucinated identity. USA mentions were also never a single global entity by
design (placeholders / single-token barrier).

**Fix (Tier0 config + engine, not the LLM prompt):**

1. **`configs/parties.yaml` → `tier0.alias_groups.united_states_government`**  
   Names: `usa`, `u s a`, `u s`, `us`, `united states`, `united states of america`.  
   **`merge_scope: same_ucid` only** — same case caption variants merge; never one
   global USA entity.

2. **`tier3.routing.skip_alias_groups_cross_ucid: [united_states_government]`**  
   Cross-UCID USA↔USA pairs are **NO_MATCH without calling Tier3**.

3. **Prompt** (`prompts/tier3_adjudicate_parties.txt`): USA collapse exception
   **removed**. Alias handling is Tier0 only.

4. **Engine** (`engine/tiers.py`): `apply_tier0_alias_groups`,
   `pair_in_skipped_alias_group_cross_ucid`.

**Tests:** `tests/test_tier0_usa_alias.py`

**Policy reminder:** USA co-occurring with a person John does **not** get a
co-party-conditioned reusable USA id today — only same-UCID alias merge +
cross-UCID Tier3 skip. That matches “don’t merge all USA into one,” not the
richer co-party identity design.

---

## Parties rails also present (v0.7.3)

Kept in the tree (supporting quality, not a substitute for the USA Tier0 fix):

- **Citation verification** — LLM must cite evidence keys; false claims → `UNCERTAIN`
  (`engine/tier3_citation.py`, `prompts/tier3_output_parties.schema.json`).
- **Asymmetric MATCH bar** — MATCH needs a verified discriminating fact
  (pacer / domain / phone / address / FJC / identical ≥2-token name + same court).
- **Punctuation fold** — parties-only near-duplicate name union
  (`name_compat.punct_fold`, clustering `union_punct_fold_variants`).

Acceptance probe on audited Tier3 pairs:
`data/runs/parties_pilot/reports/point6_tier3_acceptance_metrics.json`
(**false_MATCH rate 0** on structural-gold DIFFERENT; repeated sampling not added).

Known limitations:
`data/runs/parties_pilot/reports/PARTIES_KNOWN_LIMITATIONS.md`

---

## How to run

**Operator cookbook (new JSONs → cascade → local results → Tentris):**
[`RUN_NEW_JSONS.md`](RUN_NEW_JSONS.md).

**One command for all entity types** (what the professor asked for):

```bash
python3 scripts/run_all_er.py \
  --json-dir data/json/nyed_connectivity_test \
  --load-tentris-clone
```

```bash
cd Tier_V3
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Ollama: qwen2.5:7b + nomic-embed-text

# Advanced: single type only
export TIER_V3_OUTPUT_DIR="$(pwd)/data/runs/my_batch_parties"
python3 scripts/run_pilot.py \
  --config configs/parties.yaml \
  --json-dir data/json/my_batch
```

Useful env vars: `TIER_V3_OUTPUT_DIR`, `TIER_V3_JSON_DIR`, `OLLAMA_HOST`,
`TIER_V3_LLM_MODEL`, `KMP_DUPLICATE_LIB_OK=TRUE` (macOS FAISS).

KG insert / Tentris: `docs/KG_INCREMENTAL_INSERT_RUNBOOK.md`.

Eval / SPARQL helpers: `scripts/phase_d_harness.py`, `phase_e_eval.py`,
`phase_f_sparql.py`, `evaluate.py`.

---

## Live tree (after cleanup)

```text
Tier_V3/
├── README.md                 # pointer here
├── FINAL_FILES/              # folder/file purpose map (judges|firms|parties)
├── configs/{judges,firms,parties}.yaml
├── engine/                   # shared cascade (no type-specific forks)
├── prompts/                  # Tier3 + name-validity prompts/schemas
├── scripts/                  # entrypoints only (see FINAL_FILES)
├── tests/
├── docs/
│   ├── FINAL_V3.md           # this file
│   ├── RUN_NEW_JSONS.md      # bring-your-own JSONs → V3 → Tentris
│   └── KG_INCREMENTAL_INSERT_RUNBOOK.md
└── data/
    ├── runs/                 # ONLY current mains (4 dirs)
    ├── reports/archive/      # scrap + historical (incl. 20260921_v3_cleanup)
    └── …                     # json, gold, tentris stores, etc.
```

Everything superseded lives under **`data/reports/archive/20260921_v3_cleanup/`**
(and older material under top-level `archive/runs/`).

---

## Headline numbers (runs of record)

| | Judges (recall-fix) | Firms (pilot) | Parties (pilot) |
|--|--:|--:|--:|
| Mentions / entities | see run `pilot_summary` / clusters | see run reports | ~8.9k mentions / **4391** entities (post punct_fold) |
| Live Tentris | `data/tentris_judges_recall_fix_data` `:9080` | same graph namespaces | parties RDF inserted with SP… ids |

Exact counts: open each run’s `reports/pilot_summary.json` or
`clusters/*_entities.jsonl` line count.
