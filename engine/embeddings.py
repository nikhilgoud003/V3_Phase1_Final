"""Ollama embedding client + FAISS index for Tier2.

FAISS/OpenMP note
-----------------
spaCy (thinc) and faiss-cpu each link their own libomp on macOS. With two
OpenMP runtimes in one process, multi-threaded FAISS search segfaults
(``KMP_DUPLICATE_LIB_OK=TRUE`` only suppresses the abort, turning it into a
SIGSEGV). Running FAISS single-threaded avoids the conflict; override with
``TIER_V3_FAISS_THREADS`` when FAISS is the only OpenMP user.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


def configure_faiss_threads(faiss_mod) -> int:
    """Pin FAISS OpenMP threads (default 1) and return the value used."""
    try:
        n = int(os.environ.get("TIER_V3_FAISS_THREADS", "1"))
    except ValueError:
        n = 1
    n = max(1, n)
    try:
        faiss_mod.omp_set_num_threads(n)
    except Exception:
        pass
    return n


# Process-wide vector cache: (model, text) -> L2-normalized float32 vector.
# The same text is embedded at most once per run, whichever tier asks for it.
_VEC_CACHE: dict[tuple[str, str], np.ndarray] = {}
EMBED_STATS = {"requests": 0, "texts_embedded": 0, "cache_hits": 0}


def cached_vectors(model: str) -> dict[str, np.ndarray]:
    return {t: v for (m, t), v in _VEC_CACHE.items() if m == model}


def seed_vector_cache(model: str, vectors: dict[str, np.ndarray]) -> None:
    for t, v in vectors.items():
        _VEC_CACHE[(model, t)] = np.asarray(v, dtype=np.float32)


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def _embed_uncached(texts: list[str], model: str, endpoint: str, batch_size: int) -> np.ndarray:
    base = endpoint.rstrip("/")
    rows: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        try:
            body = _post(f"{base}/api/embed", {"model": model, "input": chunk})
            vecs = body["embeddings"]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            # Older Ollama without /api/embed: one request per text.
            vecs = [_post(f"{base}/api/embeddings", {"model": model, "prompt": t})["embedding"] for t in chunk]
            EMBED_STATS["requests"] += len(chunk) - 1
        EMBED_STATS["requests"] += 1
        rows.append(np.asarray(vecs, dtype=np.float32))
    return _l2(np.vstack(rows))


def embed_texts_ollama(
    texts: list[str],
    *,
    model: str = "nomic-embed-text",
    endpoint: str = "http://localhost:11434",
    batch_log_every: int = 500,
    batch_size: int = 128,
) -> np.ndarray:
    """Embed texts via Ollama (batched /api/embed). Returns float32 matrix (n, d), L2-normalized.

    Vectors are cached per process, so repeated texts cost no request.
    """
    missing: list[str] = []
    seen: set[str] = set()
    for t in texts:
        if (model, t) not in _VEC_CACHE and t not in seen:
            seen.add(t)
            missing.append(t)
    EMBED_STATS["cache_hits"] += len(texts) - len(missing)
    if missing:
        mat = _embed_uncached(missing, model, endpoint, batch_size)
        for t, row in zip(missing, mat):
            _VEC_CACHE[(model, t)] = row
        EMBED_STATS["texts_embedded"] += len(missing)
        if batch_log_every and len(missing) >= batch_log_every:
            print(f"  embedded {len(missing)} new texts")
    return np.vstack([_VEC_CACHE[(model, t)] for t in texts]).astype(np.float32)


def build_faiss_ip_index(vectors: np.ndarray):
    import faiss

    threads = configure_faiss_threads(faiss)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"FAISS vectors must be 2-D, got shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("FAISS vectors contain NaN/Inf — refusing to build index")
    print(
        f"FAISS index: n={vectors.shape[0]} dim={vectors.shape[1]} omp_threads={threads}",
        flush=True,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    if index.ntotal != vectors.shape[0]:
        raise ValueError(f"FAISS index incomplete: {index.ntotal} != {vectors.shape[0]}")
    return index


def save_embedding_pack(emb_dir: Path, vectors: np.ndarray, ids: list[str], meta: dict[str, Any]) -> None:
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "vectors.npy", vectors)
    with open(emb_dir / "id_map.jsonl", "w", encoding="utf-8") as f:
        for i, mid in enumerate(ids):
            f.write(json.dumps({"row": i, "mention_id": mid}) + "\n")
    with open(emb_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_embedding_pack(emb_dir: Path) -> tuple[np.ndarray, list[str], dict[str, Any]] | None:
    """Load vectors.npy + id_map if present; return None if incomplete."""
    vec_path = emb_dir / "vectors.npy"
    map_path = emb_dir / "id_map.jsonl"
    if not vec_path.exists() or not map_path.exists():
        return None
    vectors = np.load(vec_path)
    ids = [json.loads(line)["mention_id"] for line in map_path.open(encoding="utf-8") if line.strip()]
    meta: dict[str, Any] = {}
    meta_path = emb_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(ids) != vectors.shape[0]:
        return None
    return vectors.astype(np.float32), ids, meta


def compact_embed_text(m: dict) -> str:
    """Profile for embedding — omit UCID so cross-case matches are possible."""
    return (
        f"Judge: {m.get('normalized_name')} | Presentable: {m.get('presentable_name')} "
        f"| Role: {m.get('role')} | Court: {m.get('court')} | CaseType: {m.get('case_type')} "
        f"| Year: {m.get('year')} | Source: {m.get('extraction_method')}"
    )
