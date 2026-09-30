#!/usr/bin/env python3
"""Surname resolution against docket noise (A1 gate + A2 stripper).

Docket text glues procedural words onto judge names ("juan r sanchez
sentencing", "denlow s web page"). The gate compared those raw strings, so it
both blocked one judge from himself and merged two judges who shared a
trailing word. These tests pin the corpus-learned token profile that fixes it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.name_compat import build_token_profile, names_compatible, resolve_surname


# A miniature docket pool with the same shapes as the real corpus: clean names,
# names trailed by procedure, and prose-led court-staff lines.
POOL = [
    "david n hurd", "hurd advises", "hurd hears argument", "hurd directs",
    "andrew t baxter", "baxter advises", "baxter accepts", "baxter finds",
    "juan r sanchez", "juan r sanchez sentencing", "bruce w kauffman sentencing",
    "morton denlow", "denlow", "denlow s web page", "johnston s web page",
    "finnegan s web page", "gary feinerman", "feinerman s web page",
    "james knoll gardner", "john z lee", "gerald bruce lee",
    "carol sandra moore wells", "carol s wells", "k michael moore",
    "arnold c rapoport initial appearance", "david r strawbridge initial appearance",
    "interpreter juan davila-santiago", "interpreter dan decoursey",
    "interpreter deborah berry", "interpreter gloria mayne",
    "maria-elena james granting", "yvonne gonzalez rogers granting",
    "virginia k demarchi granting", "laurel beeler granting",
]
FJC = {"hurd", "baxter", "sanchez", "kauffman", "denlow", "johnston", "finnegan",
       "feinerman", "gardner", "lee", "wells", "moore", "rapoport", "strawbridge",
       "james", "rogers", "demarchi", "beeler", "page", "berry"}


def profile():
    return build_token_profile([{"normalized_name": n} for n in POOL], FJC)


class TestTokenProfile(unittest.TestCase):
    def setUp(self):
        self.prof = profile()
        self.cfg = {"name_compat": {"enabled": True}, "_token_profile": self.prof}

    def test_procedural_words_are_noise(self):
        for tok in ("sentencing", "granting", "page", "web", "initial", "appearance"):
            self.assertIn(tok, self.prof["noise"], tok)

    def test_real_surnames_are_never_noise(self):
        for tok in ("hurd", "baxter", "sanchez", "denlow", "gardner", "lee", "wells"):
            self.assertNotIn(tok, self.prof["noise"], tok)

    def test_given_names_do_not_condemn_their_surname(self):
        """"james knoll gardner" must not make "gardner" look like prose."""
        self.assertEqual(self.resolve("james knoll gardner"), "gardner")
        self.assertEqual(self.resolve("john z lee"), "lee")

    def resolve(self, name: str) -> str:
        return resolve_surname(
            name, protected=self.prof["protected"], noise=self.prof["noise"]
        )

    def test_trailing_procedure_is_stripped(self):
        self.assertEqual(self.resolve("juan r sanchez sentencing"), "sanchez")
        self.assertEqual(self.resolve("denlow s web page"), "denlow")
        self.assertEqual(self.resolve("stephen b jackson jr"), "jackson")

    def test_standalone_token_is_a_name_not_filler(self):
        self.assertEqual(self.resolve("morton denlow"), "denlow")


class TestGateRecall(unittest.TestCase):
    """Pairs the gate used to block even though they are the same judge."""

    def setUp(self):
        self.cfg = {"name_compat": {"enabled": True}, "_token_profile": profile()}

    def ok(self, a: str, b: str) -> bool:
        return names_compatible(
            {"normalized_name": a}, {"normalized_name": b}, cfg=self.cfg
        )[0]

    def test_same_judge_survives_trailing_noise(self):
        for a, b in [
            ("juan r sanchez", "juan r sanchez sentencing"),
            ("denlow s web page", "morton denlow"),
            ("hurd advises", "hurd hears argument"),
            ("baxter advises", "baxter accepts"),
            ("carlson 5/17/2016", "carlson 5/24/2016"),
            ("carol s wells", "carol sandra moore wells"),
        ]:
            self.assertTrue(self.ok(a, b), f"{a!r} | {b!r} wrongly blocked")


class TestGatePrecision(unittest.TestCase):
    """Different judges must stay apart even when they share docket prose."""

    def setUp(self):
        self.cfg = {"name_compat": {"enabled": True}, "_token_profile": profile()}

    def blocked(self, a: str, b: str) -> bool:
        return not names_compatible(
            {"normalized_name": a}, {"normalized_name": b}, cfg=self.cfg
        )[0]

    def test_shared_trailing_word_is_not_identity(self):
        for a, b in [
            ("arnold c rapoport initial appearance", "david r strawbridge initial appearance"),
            ("denlow s web page", "johnston s web page"),
            ("maria-elena james granting", "yvonne gonzalez rogers granting"),
            ("virginia k demarchi granting", "laurel beeler granting"),
            ("bruce w kauffman sentencing", "juan r sanchez sentencing"),
        ]:
            self.assertTrue(self.blocked(a, b), f"{a!r} | {b!r} wrongly allowed")

    def test_shared_leading_prose_is_not_identity(self):
        self.assertTrue(
            self.blocked("interpreter juan davila-santiago", "interpreter dan decoursey")
        )


if __name__ == "__main__":
    unittest.main()
