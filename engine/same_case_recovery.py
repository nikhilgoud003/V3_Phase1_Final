"""Same-case recovery rules (no LLM): join a cut-off or variant name to the
fuller name of the same entity inside ONE case.

Judges (config judges.yaml same_case_name_extension): in the same case, a
shorter judge name joins the single longer name it fits:
  prefix  - it is the leading words of the longer name (last word may be cut:
            "r barclay" -> "r barclay surrick", "james knoll gardne" -> "... gardner")
  suffix  - it is the trailing words ("b kim" -> "young b kim")
  surname - same surname (or a 1-letter typo in a long surname) and the other
            words match in order as equal words or initials
            ("alan b johnson" -> "alan bond johnson")
The longer name may add at most max_extra_tokens words, exactly one longer
name may fit, and two different FJC ids never join.

Firms (config firms.yaml same_case_address_prefix): in the same case, at the
same street number + ZIP, one office name is the leading words of the other
after dropping a leading "the", "and", and legal suffixes. Generic office
names (identity_exclusions.generic_office_names) never join.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _toks(s: str | None) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower().replace("&", " ")).split()


def _one_edit(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1 :] == s for i in range(len(l)))


def judge_name_fits(short: list[str], long: list[str], spec: dict) -> str | None:
    """How a shorter judge name fits a longer one, or None."""
    min_tokens = int(spec.get("min_tokens", 2))
    max_extra = int(spec.get("max_extra_tokens", 1))
    trunc_len = int(spec.get("min_truncated_token_len", 4))
    typo_len = int(spec.get("surname_typo_min_len", 5))
    if len(short) < min_tokens or short == long or len(short) > len(long):
        return None
    if len(long) - len(short) > max_extra:
        return None
    n = len(short)
    if n < len(long) and short[:-1] == long[: n - 1] and (
        short[-1] == long[n - 1] or (len(short[-1]) >= trunc_len and long[n - 1].startswith(short[-1]))
    ):
        return "prefix"
    if n < len(long) and long[-n:] == short:
        return "suffix"
    if short[-1] == long[-1] or (len(short[-1]) >= typo_len and _one_edit(short[-1], long[-1])):
        rest = long[:-1]
        j = 0
        for t in short[:-1]:
            while j < len(rest) and not (
                rest[j] == t or (len(t) == 1 and rest[j][0] == t) or (len(rest[j]) == 1 and t[0] == rest[j])
            ):
                j += 1
            if j == len(rest):
                return None
            j += 1
        return "surname"
    return None


def _address_key(addr: str | None) -> tuple[str, str] | None:
    a = (addr or "").lower()
    num = re.search(r"\b(\d{1,6})\b", a)
    z = re.search(r"\b(\d{5})(?:-\d{4})?\b\s*$", a.strip()) or re.search(r"\b(\d{5})\b", a)
    return (num.group(1), z.group(1)) if num and z else None


def _firm_key(name: str, spec: dict, suffixes: list[list[str]]) -> list[str]:
    t = _toks(name)
    lead = {w.lower() for w in spec.get("drop_leading") or []}
    drop = {w.lower() for w in spec.get("drop_tokens") or []}
    while t and t[0] in lead:
        t.pop(0)
    t = [x for x in t if x not in drop]
    changed = True
    while changed and t:
        changed = False
        for suf in suffixes:
            if len(t) > len(suf) and t[-len(suf) :] == suf:
                t = t[: -len(suf)]
                changed = True
    return t


def same_case_recovery(mentions: list[dict], cfg: dict) -> list[dict[str, Any]]:
    """Return merges [{a, b, rule, how, ucid, name_a, name_b}] for this cascade's mentions."""
    out: list[dict[str, Any]] = []
    by_case: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        if m.get("ucid"):
            by_case[m["ucid"]].append(m)

    js = cfg.get("same_case_name_extension") or {}
    if js.get("enabled"):
        for ucid, lst in by_case.items():
            names: dict[tuple, list[dict]] = defaultdict(list)
            for m in lst:
                t = tuple(_toks(m.get("normalized_name") or m.get("raw_name")))
                if t:
                    names[t].append(m)
            for st, sm in names.items():
                fits = [(lt, lm, how) for lt, lm in names.items() if (how := judge_name_fits(list(st), list(lt), js))]
                if len(fits) != 1:
                    continue
                lt, lm, how = fits[0]
                na = {str(m.get("fjc_nid")) for m in sm if m.get("fjc_nid")}
                nb = {str(m.get("fjc_nid")) for m in lm if m.get("fjc_nid")}
                if na and nb and not na & nb:
                    continue
                for m in sm:
                    out.append(
                        dict(a=lm[0]["mention_id"], b=m["mention_id"], rule="same_case_name_extension", how=how,
                             ucid=ucid, name_a=" ".join(lt), name_b=" ".join(st))
                    )

    fs = cfg.get("same_case_address_prefix") or {}
    if fs.get("enabled"):
        generic = {str(x).lower() for x in ((cfg.get("identity_exclusions") or {}).get("generic_office_names") or [])}
        sufs = [_toks(s) for s in ((cfg.get("normalization") or {}).get("strip_corp_suffixes") or [])]
        sufs += [_toks(s) for s in fs.get("extra_suffixes") or []]
        sufs = sorted({tuple(s) for s in sufs if s}, key=len, reverse=True)
        sufs = [list(s) for s in sufs]
        min_prefix = int(fs.get("min_prefix_tokens", 2))
        for ucid, lst in by_case.items():
            keyed = []
            for m in lst:
                ak = _address_key(m.get("address"))
                if ak:
                    keyed.append((m, ak, _firm_key(m.get("raw_name") or "", fs, sufs)))
            for i in range(len(keyed)):
                for j in range(i + 1, len(keyed)):
                    (ma, ka, fa), (mb, kb, fb) = keyed[i], keyed[j]
                    if ka != kb or not fa or not fb:
                        continue
                    s, l = (fa, fb) if len(fa) <= len(fb) else (fb, fa)
                    if len(s) >= min_prefix and l[: len(s)] == s and " ".join(s) not in generic:
                        out.append(
                            dict(a=ma["mention_id"], b=mb["mention_id"], rule="same_case_address_prefix", how="addr+prefix",
                                 ucid=ucid, name_a=ma.get("raw_name"), name_b=mb.get("raw_name"))
                        )
    return out
