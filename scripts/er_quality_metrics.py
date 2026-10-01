#!/usr/bin/env python3
"""Quality metrics for a unified run, and for V2 on the same files.

Definitions follow v2_vs_v3_comparison/REPORT.md (same normalization and FJC
name+court matching), restricted to the ucids the run processed.

Usage: er_quality_metrics.py RUN_DIR [--v2 V2_DIR] [--out metrics.json]
"""

from __future__ import annotations

import argparse
import collections
import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2 = ROOT.parent / "v2_pipeline_pilot1000_results"

STATES = {'ak':'Alaska','al':'Alabama','ar':'Arkansas','az':'Arizona','ca':'California','co':'Colorado','ct':'Connecticut','dc':'Columbia','fl':'Florida','ga':'Georgia','gu':'Guam','hi':'Hawaii','ia':'Iowa','id':'Idaho','il':'Illinois','in':'Indiana','ks':'Kansas','ky':'Kentucky','la':'Louisiana','ma':'Massachusetts','md':'Maryland','me':'Maine','mi':'Michigan','mn':'Minnesota','mo':'Missouri','ms':'Mississippi','mt':'Montana','nc':'North Carolina','nd':'North Dakota','nh':'New Hampshire','nj':'New Jersey','nm':'New Mexico','nv':'Nevada','ny':'New York','oh':'Ohio','ok':'Oklahoma','or':'Oregon','pa':'Pennsylvania','pr':'Puerto Rico','ri':'Rhode Island','sc':'South Carolina','sd':'South Dakota','tn':'Tennessee','tx':'Texas','ut':'Utah','va':'Virginia','vi':'Virgin Islands','wa':'Washington','wi':'Wisconsin','wv':'West Virginia','wy':'Wyoming','vt':'Vermont','ne':'Nebraska','de':'Delaware'}  # noqa: E501
DIRS = {'n': 'Northern', 's': 'Southern', 'e': 'Eastern', 'w': 'Western', 'm': 'Middle', 'c': 'Central'}
SUFF = {'jr', 'sr', 'ii', 'iii', 'iv'}
PLACEHOLDER = re.compile(r'\b(doe|does|unknown|roe|john|jane|et al|defendants?|plaintiffs?|all others)\b')
ORG = re.compile(
    r'\b(inc|incorporated|llc|l l c|lp|llp|ltd|corp|corporation|company|co|plc|ag|na|n a|bank|group|trust|'
    r'association|holdings|partners|services|industries|international|fund|insurance|department|dept|'
    r'county|city|state|commission|board|office|university|hospital|agency|authority|district|usa|'
    r'united states|america|products|systems|technologies|foundation|society|church|school)\b'
)
GLUED = re.compile(r'[A-Z]{3,}[A-Z][a-z]{3,}')


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding='utf-8') if l.strip()]


def norm(s: str | None) -> str:
    s = (s or '').lower().replace('&', ' and ')
    s = re.sub(r"[^a-z0-9 ]", ' ', s.replace("'", ''))
    return ' '.join(s.split())


def pnorm(s: str | None) -> str:
    return norm(html.unescape(html.unescape(s or '')))


def nosuff(s: str | None) -> str:
    return ' '.join(t for t in norm(s).split() if t not in SUFF)


def fjc_court(ab: str) -> str:
    st, rest = ab[:2], ab[2:]
    if rest == 'd':
        return f'U.S. District Court for the District of {STATES[st]}'
    return f'U.S. District Court for the {DIRS[rest[0]]} District of {STATES[st]}'


def fjc_index() -> tuple[dict, dict]:
    idx: dict = collections.defaultdict(set)
    info = {}
    for r in csv.DictReader((ROOT / 'data/judges_fjc.csv').open(encoding='utf-8')):
        first, mid, last = norm(r['First Name']), norm(r['Middle Name']), norm(r['Last Name'])
        info[r['nid']] = ' '.join(f"{r['First Name']} {r['Middle Name']} {r['Last Name']}".split())
        forms = {f'{first} {mid} {last}', f'{first} {last}'}
        if mid:
            forms.add(f'{first} {mid[0]} {last}')
        courts = {r[f'Court Name ({i})'] for i in range(1, 7) if r[f'Court Name ({i})'].strip()}
        for f in forms:
            for c in courts:
                idx[(' '.join(f.split()), c)].add(r['nid'])
    return idx, info


TITLES = {'honorable', 'hon', 'judge', 'magistrate', 'chief', 'senior', 'district', 'us', 'u', 's', 'united',
          'states', 'the', 'mag', 'justice', 'presiding', 'bankruptcy', 'sr'}


def jclean(s: str) -> str:
    t = nosuff(s).split()
    while t and t[0] in TITLES:
        t.pop(0)
    return ' '.join(t)


def fjc_match(name: str, court: str, idx: dict) -> str | None:
    try:
        c = fjc_court(court)
    except (KeyError, IndexError):
        return None
    s = idx.get((jclean(name), c), set())
    return next(iter(s)) if len(s) == 1 else None


def is_org(n: str) -> bool:
    return bool(ORG.search(n))


def judge_name_junk(raw: str, normalized: str) -> bool:
    """Docket prose left on a judge name: a trailing all-lowercase word in a
    name that has capitals, a ':' or a glued all-caps+Capitalized word."""
    raw = raw or ''
    toks = raw.split()
    tail_lower = (
        len(toks) >= 3
        and any(c.isupper() for c in raw)
        and toks[-1].isalpha()
        and toks[-1].islower()
    )
    return bool(tail_lower or ':' in raw or GLUED.search(raw))


def judge_metrics(rows, idf, name, court, fjc_of, idx, info):
    j2ids = collections.defaultdict(set)
    id2j = collections.defaultdict(set)
    names = collections.defaultdict(set)
    ids = set()
    for r in rows:
        i = idf(r)
        ids.add(i)
        f = fjc_of(r)
        names[i].add(name(r))
        if f:
            j2ids[f].add(i)
            id2j[i].add(f)
    splits = {f: sorted(s) for f, s in j2ids.items() if len(s) > 1}
    merges = {i: sorted(s) for i, s in id2j.items() if len(s) > 1}
    return {
        'mentions': len(rows),
        'ids': len(ids),
        'fjc_judges': len(j2ids),
        'fjc_splits': len(splits),
        'fjc_wrong_merges': len(merges),
        'split_examples': [
            {'nid': f, 'judge': info.get(f), 'ids': s, 'names': [sorted(names[i])[:4] for i in s]}
            for f, s in sorted(splits.items())
        ],
        'merge_examples': [{'id': i, 'nids': s, 'names': sorted(names[i])[:6]} for i, s in merges.items()],
    }


def party_metrics(rows, idf, name, ucid, slot):
    """slot(r) identifies the separate party entry within the case (party_enum / party_block)."""
    n2i = collections.defaultdict(set)
    ids = set()
    for r in rows:
        n = pnorm(name(r))
        ids.add(idf(r))
        if n and not PLACEHOLDER.search(n):
            n2i[n].add(idf(r))
    split_names = [n for n, v in n2i.items() if len(v) > 1]
    split_org = [n for n in split_names if is_org(n)]
    split_person = [n for n in split_names if not is_org(n)]

    ph = collections.defaultdict(set)
    for r in rows:
        if re.search(r'\b(doe|does|unknown)\b', pnorm(name(r))):
            ph[idf(r)].add(ucid(r))
    ph_x = sum(len(v) > 1 for v in ph.values())

    # Co-party merge: one id holds two differently-named parties listed as separate
    # entries in the same case.
    by_id_case = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        s = slot(r)
        if s is None:
            continue
        by_id_case[idf(r)][ucid(r)].add((s, pnorm(name(r))))
    coparty = []
    for i, cases in by_id_case.items():
        for u, entries in cases.items():
            slots = collections.defaultdict(set)
            for s, n in entries:
                slots[s].add(n)
            if len(slots) < 2:
                continue
            allnames = {n for ns in slots.values() for n in ns}
            if len(allnames) >= 2:
                coparty.append({'id': i, 'ucid': u, 'names': sorted(allnames)[:6]})
                break
    return {
        'mentions': len(rows),
        'ids': len(ids),
        'distinct_names': len(n2i),
        'same_name_splits': len(split_names),
        'same_name_splits_org': len(split_org),
        'same_name_splits_person': len(split_person),
        'split_examples_org': sorted(split_org)[:15],
        'split_examples_person': sorted(split_person)[:15],
        'placeholder_ids_across_cases': ph_x,
        'coparty_wrong_merges': len(coparty),
        'coparty_examples': coparty[:15],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--v2', default=str(DEFAULT_V2))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    run = Path(args.run_dir)
    idx, info = fjc_index()

    ents = load(run / 'entities.jsonl')
    ment = load(run / 'mentions.jsonl')
    m2e = {mid: e['entity_id'] for e in ents for mid in e.get('mention_ids') or []}
    for m in ment:
        m['eid'] = m2e.get(m['mention_id'])
    orphans = sum(1 for m in ment if m['eid'] is None)
    ment = [m for m in ment if m['eid'] is not None]
    ucids = {m['ucid'] for m in ment if m.get('ucid')}
    summ = json.loads((run / 'summary.json').read_text())
    files = {f['file'] for f in summ.get('files') or []}

    out: dict = {'run': str(run), 'files': len(files), 'ucids': len(ucids), 'orphan_mentions': orphans}
    calls_path = run.parent / f'{run.name}.ollama_calls.json'
    if calls_path.is_file():
        out['ollama_calls'] = json.loads(calls_path.read_text())
    out['elapsed_sec'] = summ.get('elapsed_sec')

    jm = [m for m in ment if m['entity_type'] == 'judge']
    out['v3_judges'] = judge_metrics(
        jm, lambda m: m['eid'], lambda m: m['raw_name'], lambda m: m['court'],
        lambda m: m.get('fjc_nid') or fjc_match(m['raw_name'], m['court'], idx), idx, info,
    )
    junk_j = [m for m in jm if judge_name_junk(m.get('raw_name'), m.get('normalized_name'))]
    out['v3_judges']['junk_name_mentions'] = len(junk_j)
    out['v3_judges']['junk_name_entities'] = len({m['eid'] for m in junk_j})
    out['v3_judges']['junk_name_examples'] = sorted({m['raw_name'] for m in junk_j})[:15]
    hdr = [m for m in jm if m.get('docket_source') in ('case_header', 'case_parties')]
    out['v3_judges_header_scope'] = judge_metrics(
        hdr, lambda m: m['eid'], lambda m: m['raw_name'], lambda m: m['court'],
        lambda m: m.get('fjc_nid') or fjc_match(m['raw_name'], m['court'], idx), idx, info,
    )

    pm = [m for m in ment if m['entity_type'] == 'party']
    out['v3_parties'] = party_metrics(
        [m for m in pm if m.get('docket_source') != 'party_alias'],
        lambda m: m['eid'], lambda m: m['raw_name'], lambda m: m['ucid'],
        lambda m: m.get('party_enum') if m.get('docket_source') == 'case_parties' else None,
    )
    walk = [m for m in ment if m.get('docket_source') == 'schema_free_walk']
    out['v3_schema_walk'] = {
        'mentions': len(walk),
        'by_type': dict(collections.Counter(m['entity_type'] for m in walk)),
        'by_path': dict(collections.Counter(m.get('discovered_path') for m in walk)),
        'examples': sorted({(m['entity_type'], m['raw_name']) for m in walk})[:25],
    }
    aliases = [m for m in pm if m.get('docket_source') == 'party_alias']
    out['v3_party_aliases'] = {
        'mentions': len(aliases),
        'by_relationship': dict(collections.Counter(m.get('relationship_type') for m in aliases)),
    }

    v2 = Path(args.v2)
    v2j = [x for x in load(v2 / 'judge_disambiguation.jsonl') if x['ucid'] in ucids]
    out['v2_judges'] = judge_metrics(
        v2j, lambda x: x['SJID'], lambda x: x['Extracted_Entity'], lambda x: x['court'],
        lambda x: fjc_match(x['Extracted_Entity'], x['court'], idx), idx, info,
    )
    v2p_all = [x for x in load(v2 / 'party_disambiguation.jsonl') if x['ucid'] in ucids]
    out['v2_parties'] = party_metrics(
        [x for x in v2p_all if x.get('entity_source') == 'party'],
        lambda x: x['SPID_Strong'], lambda x: x['party_name'], lambda x: x['ucid'],
        lambda x: x.get('party_block'),
    )
    out['v2_party_aliases'] = {
        'rows': sum(1 for x in v2p_all if x.get('entity_source') != 'party'),
        'by_relationship': dict(collections.Counter(
            x.get('relationship_type') for x in v2p_all if x.get('entity_source') != 'party')),
    }

    text = json.dumps(out, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    brief = {
        'files': out['files'],
        'elapsed_sec': out['elapsed_sec'],
        'V3 judge FJC splits': f"{out['v3_judges']['fjc_splits']}/{out['v3_judges']['fjc_judges']}",
        'V3 judge wrong merges': out['v3_judges']['fjc_wrong_merges'],
        'V3 judge junk-name entities': out['v3_judges']['junk_name_entities'],
        'V3 party same-name splits (org/person)': (
            out['v3_parties']['same_name_splits_org'], out['v3_parties']['same_name_splits_person']),
        'V3 co-party wrong merges': out['v3_parties']['coparty_wrong_merges'],
        'V3 schema-walk mentions': out['v3_schema_walk']['mentions'],
        'V3 aliases': out['v3_party_aliases']['mentions'],
        'V2 judge FJC splits': f"{out['v2_judges']['fjc_splits']}/{out['v2_judges']['fjc_judges']}",
        'V2 party same-name splits (org/person)': (
            out['v2_parties']['same_name_splits_org'], out['v2_parties']['same_name_splits_person']),
        'V2 co-party wrong merges': out['v2_parties']['coparty_wrong_merges'],
        'V2 aliases': out['v2_party_aliases']['rows'],
    }
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
