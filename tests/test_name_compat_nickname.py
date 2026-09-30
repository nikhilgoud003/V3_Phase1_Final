#!/usr/bin/env python3
"""Regression: steven/steve nickname pair stays compatible."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.name_compat import names_compatible

CFG = {"name_compat": {"enabled": True, "max_surname_edit_distance": 1}}


class TestNicknameGivenGate(unittest.TestCase):
    def test_steven_steve_rau_compatible(self):
        a = {"normalized_name": "steven e rau", "surname": "rau"}
        b = {"normalized_name": "steve e rau", "surname": "rau"}
        ok, reason = names_compatible(a, b, cfg=CFG)
        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
