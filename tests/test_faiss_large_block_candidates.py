#!/usr/bin/env python3
"""FAISS Tier2: large same-name cluster must still surface the variant pair.

Root cause: global top_k=25 returned other instances of the giant UF root,
none of which were the block's uniq representative ID. Neighbor acceptance
now maps via UF root; small blocks also do exhaustive uniq pairwise IP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.embeddings import build_faiss_ip_index
from engine.tiers import UnionFind, tier2_candidates

N_LARGE = 500
DIM = 16
CFG = {
    "tier2": {
        "search": {
            "min_similarity": 0.72,
            "top_k": 25,
            "exhaustive_uniq_cap": 0,  # isolate UF-root membership fix
        }
    }
}


def _mentions():
    large = [
        {
            "mention_id": f"big_{i:04d}",
            "normalized_name": "george c hanks",
            "presentable_name": "George C Hanks",
            "surname": "hanks",
            "ucid": f"txsd;;3:16-cr-{i:05d}",
            "court": "txsd",
            "hygiene_scope": "global",
        }
        for i in range(N_LARGE)
    ]
    variant = {
        "mention_id": "small_0000",
        "normalized_name": "george hanks",
        "presentable_name": "George Hanks",
        "surname": "hanks",
        "ucid": "txsd;;3:18-cr-00027",
        "court": "txsd",
        "hygiene_scope": "global",
    }
    return large + [variant]


def _pack(ments):
    rng = np.random.default_rng(0)
    base = rng.standard_normal(DIM).astype(np.float32)
    base /= np.linalg.norm(base)
    # Giant cluster: near-duplicates of `base`. Variant: small perturbation, still IP>0.72.
    vectors = np.stack([base] * N_LARGE + [base], axis=0).astype(np.float32)
    noise = rng.standard_normal(DIM).astype(np.float32)
    noise /= np.linalg.norm(noise)
    vectors[-1] = base * 0.97 + noise * 0.03
    vectors[-1] /= np.linalg.norm(vectors[-1])
    ids = [m["mention_id"] for m in ments]
    index = build_faiss_ip_index(vectors)
    return {
        "vectors": vectors,
        "ids": ids,
        "id_to_row": {mid: i for i, mid in enumerate(ids)},
        "index": index,
        "backend": "faiss",
    }


def _uf(ments):
    uf = UnionFind()
    for m in ments:
        uf.add(m["mention_id"])
    root = ments[0]["mention_id"]
    for m in ments[1:-1]:
        uf.union(root, m["mention_id"])
    return uf


def test_large_cluster_variant_surfaces_via_faiss_root_map():
    ments = _mentions()
    uf = _uf(ments)
    assert len({uf.find(m["mention_id"]) for m in ments}) == 2
    mids = [m["mention_id"] for m in ments]
    blocks = {"court_surname::txsd|hanks": mids}
    mention_blocks = {mid: ["court_surname::txsd|hanks"] for mid in mids}
    pairs = tier2_candidates(ments, mention_blocks, blocks, uf, CFG, embed_pack=_pack(ments))
    names = {(a, b) for a, b, _s in pairs}
    # Representative of the giant root is big_0000; variant is small_0000.
    assert any(
        "small_0000" in (a, b) and (a.startswith("big_") or b.startswith("big_"))
        for a, b in names
    ), pairs[:10]
    sims = [s for a, b, s in pairs if "small_0000" in (a, b)]
    assert sims and max(sims) >= 0.72, sims


def test_large_cluster_also_surfaces_with_exhaustive_pairwise():
    ments = _mentions()
    uf = _uf(ments)
    mids = [m["mention_id"] for m in ments]
    blocks = {"court_surname::txsd|hanks": mids}
    mention_blocks = {mid: ["court_surname::txsd|hanks"] for mid in mids}
    cfg = {
        "tier2": {
            "search": {"min_similarity": 0.72, "top_k": 25, "exhaustive_uniq_cap": 80}
        }
    }
    pairs = tier2_candidates(ments, mention_blocks, blocks, uf, cfg, embed_pack=_pack(ments))
    assert any("small_0000" in (a, b) for a, b, _s in pairs), pairs[:10]
