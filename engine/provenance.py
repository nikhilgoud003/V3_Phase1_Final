"""Decision journal writer/reader — crash-safe append + fsync."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _fsync_file(f) -> None:
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        # Some network FS / special files may not support fsync
        pass


class DecisionJournal:
    def __init__(self, path: str | Path, *, fresh: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            # Truncate for a new cascade run (caller should point elsewhere if
            # another process owns the default journal path).
            with self.path.open("w", encoding="utf-8") as f:
                _fsync_file(f)
        self.n = 0
        if not fresh and self.path.exists():
            # Continue numbering after existing lines
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.n += 1

    def log(self, record: dict[str, Any]) -> None:
        self.n += 1
        if "decision_id" not in record:
            record["decision_id"] = f"dec_{self.n:08d}"
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            _fsync_file(f)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSONL record with flush+fsync (Tier3 cache, etc.)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
        _fsync_file(f)


def log_decision(record: dict, path: str) -> None:
    append_jsonl(path, record)
