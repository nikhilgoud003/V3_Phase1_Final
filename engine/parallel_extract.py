"""Read PACER JSON files and run per-type extraction in worker processes.

Extraction of one file does not depend on any other file, so files are
extracted ahead in parallel and handed back in their original order; the
cumulative resolve stays sequential. workers <= 1 runs in-process.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Iterator

_CFGS: dict[str, dict] | None = None


def _init(config_paths: dict[str, str]) -> None:
    global _CFGS
    from engine.config_loader import load_config

    _CFGS = {}
    for etype, cpath in config_paths.items():
        cfg = load_config(cpath)
        _CFGS[cfg.get("entity_type") or etype] = cfg


def extract_file(fp: Path) -> dict[str, Any]:
    from engine.extract import extract_from_case

    assert _CFGS is not None, "worker not initialised"
    t_read = time.perf_counter()
    with open(fp, encoding="utf-8") as f:
        case = json.load(f)
    sec_read = time.perf_counter() - t_read
    t_extract = time.perf_counter()
    file_mentions: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
    file_xfers: dict[str, list[dict]] = {"judge": [], "firm": [], "party": []}
    for etype, cfg in _CFGS.items():
        mentions, xfers = extract_from_case(copy.deepcopy(case), cfg, source_file=fp.name)
        for t in xfers:
            t["ucid"] = case.get("ucid")
            t["source_file"] = fp.name
        file_mentions[etype] = mentions
        file_xfers[etype] = xfers
    return {
        "file": fp.name,
        "case": case,
        "mentions": file_mentions,
        "xfers": file_xfers,
        "sec_read": sec_read,
        "sec_extract": time.perf_counter() - t_extract,
    }


def iter_extracted(files: list[Path], config_paths: dict[str, str], workers: int) -> Iterator[dict[str, Any]]:
    if workers <= 1 or len(files) <= 1:
        _init(config_paths)
        for fp in files:
            yield extract_file(fp)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(config_paths,)) as pool:
        yield from pool.imap(extract_file, files, chunksize=1)
