#!/usr/bin/env python3
"""Two-judge cases, courtroom codes, and initials-only names.

Regression cover for the CACD incident: one case listing district judge
**John F. Walter** and referred magistrate **Patrick J. Walsh** had its bare
"walter" and "walsh" mentions fused into a single entity, because the old rule
only required the two mentions to share a UCID and treated ambiguous tokens
(walter, william, robert are given names *and* surnames) as first names.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.mention_hygiene import pair_allowed
from engine.name_validity import classify_name_validity, strip_courtroom_suffix
from engine.ucid_anchor import mark_anchor_outcomes, resolve_short_mentions

CACD = "cacd;;8:16-cv-00800"
MOWD = "mowd;;2:17-cv-04052"
MDD = "mdd;;8:16-cv-00117"


def mention(mid: str, name: str, ucid: str) -> dict:
    return {"mention_id": mid, "normalized_name": name, "ucid": ucid}


class TestUcidAnchoring(unittest.TestCase):
    def setUp(self):
        self.pool = [
            mention("m1", "john f walter", CACD),
            mention("m2", "patrick j walsh", CACD),
            mention("m3", "walter", CACD),
            mention("m4", "walsh", CACD),
            mention("m5", "brian c wimes", MOWD),
            mention("m6", "wimes", MOWD),
            mention("m7", "william", MOWD),
            mention("m8", "robert", MDD),
            mention("m9", "robinson", MDD),
        ]
        self.res = resolve_short_mentions(self.pool)
        self.attached = {
            (m["short_name"], m["anchor_name"]) for m in self.res["merges"]
        }
        self.outcomes = self.res["outcomes"]

    def test_short_token_attaches_to_its_own_judge(self):
        self.assertIn(("walter", "john f walter"), self.attached)
        self.assertIn(("walsh", "patrick j walsh"), self.attached)
        self.assertIn(("wimes", "brian c wimes"), self.attached)

    def test_two_judges_in_one_case_are_never_fused(self):
        """walsh must not land on Walter, and the pair is never even proposed."""
        for short, anchor in self.attached:
            if short == "walsh":
                self.assertEqual(anchor, "patrick j walsh")
            if short == "walter":
                self.assertEqual(anchor, "john f walter")
        mark_anchor_outcomes(self.pool, self.outcomes)
        by = {m["mention_id"]: m for m in self.pool}
        self.assertFalse(pair_allowed(by["m3"], by["m4"]))
        self.assertFalse(pair_allowed(by["m4"], by["m1"]))

    def test_unmatched_short_token_abstains(self):
        """"william" matches neither Wimes nor Brian, so it stays alone."""
        self.assertEqual(self.outcomes["m7"]["status"], "abstain")
        self.assertEqual(self.outcomes["m7"]["reason"], "no_anchor_in_ucid")

    def test_two_short_tokens_with_no_anchor_both_abstain(self):
        for mid in ("m8", "m9"):
            self.assertEqual(self.outcomes[mid]["status"], "abstain")
        self.assertFalse(
            any(m["short_name"] in {"robert", "robinson"} for m in self.res["merges"])
        )

    def test_ambiguous_anchor_abstains(self):
        pool = [
            mention("a1", "john f walter", CACD),
            mention("a2", "john q walters", CACD),
            mention("a3", "walter", CACD),
        ]
        out = resolve_short_mentions(pool)["outcomes"]
        self.assertEqual(out["a3"]["status"], "abstain")
        self.assertEqual(out["a3"]["reason"], "ambiguous_anchor")


class TestCourtroomStripper(unittest.TestCase):
    def test_reporter_suffix_removed(self):
        self.assertEqual(
            strip_courtroom_suffix("timothy j sullivan ftr - singletary"),
            "timothy j sullivan",
        )
        self.assertEqual(
            strip_courtroom_suffix("gina l simms ftr - singletary"), "gina l simms"
        )

    def test_trailing_initials_code_removed(self):
        self.assertEqual(strip_courtroom_suffix("bridget s bade bsb"), "bridget s bade")
        self.assertEqual(strip_courtroom_suffix("g r smith sff"), "g r smith")

    def test_real_names_untouched(self):
        for name in (
            "john f walter",
            "patrick j walsh",
            "stacie f beckerman",
            "laurel beeler",
            "abbie crites-leoni",
            "carol sandra moore wells",
        ):
            self.assertEqual(strip_courtroom_suffix(name), name, name)


class TestInitialsOnlyValidity(unittest.TestCase):
    def _valid(self, name: str) -> bool:
        ok, _ = classify_name_validity(name, english_words=frozenset())
        return ok

    def test_initials_only_token_quarantined(self):
        for name in ("jpp", "bsb", "sff"):
            self.assertFalse(self._valid(name), name)

    def test_real_short_surnames_survive(self):
        for name in ("john f walter", "brian c wimes", "morton denlow"):
            self.assertTrue(self._valid(name), name)


if __name__ == "__main__":
    unittest.main()
