"""Judges surname suffix regression — shared surname() must strip Jr/II/III."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

GOLD_MENTIONS = (
    ROOT / "data/runs/judges_pilot_recall_fix/mentions/judges_mentions.jsonl"
)


@pytest.mark.skipif(not GOLD_MENTIONS.exists(), reason="judges gold mentions missing")
def test_judges_gold_suffix_rows_recompute_to_family_name():
    """Gold run-of-record has 111 scoped rows with suffix stored as surname.

    Current engine/normalize.surname() strips generational suffixes (parties-era
    fix). Recompute on frozen normalized_name must yield the family name, not jr/ii/iii.
    """
    from engine.normalize import surname

    suffix_tokens = {"jr", "sr", "ii", "iii", "iv", "2nd", "3rd", "4th"}
    rows = []
    for line in GOLD_MENTIONS.open(encoding="utf-8"):
        if not line.strip():
            continue
        m = json.loads(line)
        if (m.get("docket_source") or "") == "line_entry":
            continue
        old = m.get("surname") or ""
        new = surname(m.get("normalized_name") or "")
        if old != new:
            rows.append((m, old, new))

    assert len(rows) == 111, f"expected 111 suffix-as-surname gold rows, got {len(rows)}"
    assert all(old in suffix_tokens for _, old, _ in rows), (
        "unexpected mismatch pattern — not suffix-as-surname"
    )
    assert all(new not in suffix_tokens for _, _, new in rows)


def test_surname_suffix_examples_recognizable_judges():
    from engine.normalize import surname

    cases = [
        ("andre birotte jr", "birotte"),
        ("joseph c wilkinson jr", "wilkinson"),
        ("w harold albritton iii", "albritton"),
        ("stephen n limbaugh jr", "limbaugh"),
        ("otis d wright ii", "wright"),
    ]
    for normalized, expected in cases:
        assert surname(normalized) == expected


def test_control_judges_unchanged():
    """Robreno/Glasser/Donnelly have no trailing suffix token — no drift."""
    from engine.normalize import surname

    for normalized, expected in [
        ("eduardo c robreno", "robreno"),
        ("i leo glasser", "glasser"),
        ("ann m donnelly", "donnelly"),
        ("david r strawbridge", "strawbridge"),
        ("marina garcia marmolejo", "marmolejo"),
    ]:
        assert surname(normalized) == expected
