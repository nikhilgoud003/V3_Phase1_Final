"""Judge-name confirmation without an LLM (config judges.yaml
name_validity.confirmation; turned on by bulk mode).

A judge name read only from docket text becomes a judge when:
  - it is in the FJC list (incl. the fix (h) first-name-prefix link), or
  - the same name is in a case header / party-record judge field in this file, or
  - it is read directly after a judge title ("Judge X", "Magistrate Judge X"), or
  - it was already confirmed earlier in the run, or
  - it now appears in a second, different case (promotion).
Otherwise its mentions are held as "unconfirmed" (pending) and are not
resolved. Pending mentions are released into resolution when the name is
promoted, so they get an ID then. Pending state is saved in the checkpoint.
"""

from __future__ import annotations

from typing import Any

from engine.config_loader import resolve_path
from engine.fjc import fjc_first_prefix_nids, load_fjc_index
from engine.normalize import GENERATIONAL_SUFFIXES
from engine.same_case_recovery import judge_name_fits


class JudgeConfirmation:
    def __init__(self, cfg: dict, state: dict[str, Any]) -> None:
        spec = (cfg.get("name_validity") or {}).get("confirmation") or {}
        self.enabled = bool(spec.get("enabled"))
        self.confirm_sources = set(spec.get("confirm_sources") or ["case_header", "case_parties"])
        self.confirm_prefixes = set(spec.get("confirm_prefix_categories") or [])
        self.title_shape = bool(spec.get("title_name_shape_check", True))
        self.particles = {str(w).lower().rstrip(".") for w in spec.get("name_particles") or []}
        self.min_cases = int(spec.get("min_cases", 2))
        n = cfg.get("normalization") or {}
        rule = next((r for r in (cfg.get("tier0") or {}).get("rules") or [] if r.get("id") == "fjc_nid_join"), {})
        ext = rule.get("external") or {}
        self.first_prefix = rule.get("first_name_prefix_link") or {}
        self.fjc_idx = idx = load_fjc_index(
            resolve_path(cfg, ext.get("path", "data/judges_fjc.csv")),
            resolve_path(cfg, ext.get("crosswalk_path", "data/external/fjc_court_crosswalk.json")),
            n.get("strip_honorifics") or [],
            n.get("strip_chars") or "",
        )
        # FJC names and their initial forms ("paul g rosenblatt"), as the FJC linker uses.
        self.fjc_full = set(idx["name_to_nids"]) | set(idx["alias_to_nids"])
        self.fit_spec = cfg.get("same_case_name_extension") or {}
        # state["judge_confirm"] = {"confirmed": [names], "pending": {name: [mentions]}}
        jc = state.setdefault("judge_confirm", {"confirmed": [], "pending": {}})
        self.confirmed: set[str] = set(jc.get("confirmed") or [])
        # "court|surname" of judges known in this run (FJC surnames are added from the index).
        self.known_surnames: set[str] = set(jc.get("known_surnames") or [])
        self.fjc_court_surnames = {f"{c}|{s}" for (s, c) in idx.get("surname_court_to_nids") or {}}
        self.pending: dict[str, list[dict]] = jc.get("pending") or {}
        self.state = state
        self.stats = {"confirmed_names": 0, "promoted_names": 0, "released_mentions": 0, "held_mentions": 0}

    def _sync(self) -> None:
        self.state["judge_confirm"] = {
            "confirmed": sorted(self.confirmed),
            "known_surnames": sorted(self.known_surnames),
            "pending": self.pending,
        }

    def _strong(self, name: str, ms: list[dict]) -> bool:
        return (
            name in self.confirmed
            or name in self.fjc_full
            or any(m.get("docket_source") in self.confirm_sources for m in ms)
            or (
                self.first_prefix.get("enabled")
                and len(fjc_first_prefix_nids(name, ms[0].get("court") or "", self.fjc_idx, self.first_prefix)) == 1
            )
        )

    def _title_ok(self, name: str, ms: list[dict], strong: set[str]) -> bool:
        """A name read right after a judge title, with a clean name shape:
        (a) every word starts with a capital ("Hurd directs" fails), except
            lowercase name particles (de, la, del, van, von, ...: "de la Rosa");
        (b) no ALL-CAPS name followed by a normal word ("RAPOPORT Initial");
        (c) not a known judge name plus trailing words ("Ronald G. Morgan Final"),
            and not a known judge surname of this court plus a word ("Bucklo Mailed")."""
        raws = [m.get("raw_name") or "" for m in ms if m.get("prefix_category") in self.confirm_prefixes]
        if not raws:
            return False
        if not self.title_shape:
            return True
        toks = name.split()
        known = strong | self.confirmed | self.fjc_full
        if any(" ".join(toks[:-cut]) in known for cut in (1, 2) if len(toks) - cut >= 2):
            return False
        court = ms[0].get("court") or ""
        if len(toks) >= 2 and toks[0] != toks[-1]:
            key = f"{court}|{toks[0]}"
            if key in self.fjc_court_surnames or key in self.known_surnames:
                return False
        for raw in raws:
            words = [w.strip(".,;:()'\"") for w in raw.split()]
            words = [w for w in words if w]
            part = [w.lower() in self.particles for w in words]
            if not words or all(part) or not all(p or w[0].isupper() for w, p in zip(words, part)):
                continue
            core = [w for w, p in zip(words, part) if not p]
            caps_break = any(
                w.isupper() and sum(c.isalpha() for c in w) >= 3 and w.lower() not in GENERATIONAL_SUFFIXES
                and not nxt.isupper()
                for w, nxt in zip(core, core[1:])
            )
            if not caps_break:
                return True
        return False

    def filter_file(self, mentions: list[dict]) -> list[dict]:
        """Return the judge mentions to resolve now (incl. released pending ones)."""
        if not self.enabled:
            return mentions
        by_name: dict[str, list[dict]] = {}
        for m in mentions:
            by_name.setdefault((m.get("normalized_name") or "").strip(), []).append(m)
        keep: list[dict] = []
        # Names confirmed by this file's own evidence (used for same-case cut-offs).
        strong = {name for name, ms in by_name.items() if name and self._strong(name, ms)}
        file_ok = strong | {
            name for name, ms in by_name.items() if name and name not in strong and self._title_ok(name, ms, strong)
        }
        for name, ms in by_name.items():
            if not name:
                keep.extend(ms)
                continue
            ok = name in file_ok
            if not ok and self.fit_spec.get("enabled"):
                # A cut-off of a judge confirmed in this same case is that judge.
                short = name.split()
                ok = any(judge_name_fits(short, other.split(), self.fit_spec) for other in file_ok if other != name)
            held = self.pending.get(name) or []
            if not ok and held:
                cases = {m.get("ucid") for m in held} | {m.get("ucid") for m in ms}
                if len(cases) >= self.min_cases:
                    ok = True
                    self.stats["promoted_names"] += 1
            if ok:
                toks = name.split()
                if len(toks) >= 2 and ms[0].get("court"):
                    self.known_surnames.add(f"{ms[0]['court']}|{toks[-1]}")
                if name not in self.confirmed:
                    self.confirmed.add(name)
                    self.stats["confirmed_names"] += 1
                keep.extend(ms)
                if held:
                    keep.extend(held)
                    self.stats["released_mentions"] += len(held)
                    self.pending.pop(name, None)
            else:
                self.pending.setdefault(name, []).extend(ms)
                self.stats["held_mentions"] += len(ms)
        self._sync()
        return keep

    def unconfirmed_rows(self) -> list[dict]:
        rows = []
        for name, ms in sorted(self.pending.items()):
            for m in ms:
                rows.append(
                    {
                        "normalized_name": name,
                        "raw_name": m.get("raw_name"),
                        "ucid": m.get("ucid"),
                        "court": m.get("court"),
                        "source_file": m.get("source_file"),
                        "docket_index": m.get("docket_index"),
                        "status": "UNCONFIRMED_JUDGE_NAME",
                        "reason": "docket text only; not in FJC, not in a case header/party record, no judge title right before it, seen in one case",
                    }
                )
        return rows
