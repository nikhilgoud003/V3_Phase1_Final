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
