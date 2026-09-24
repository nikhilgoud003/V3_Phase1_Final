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


def embed_texts_ollama(
    texts: list[str],
    *,
    model: str = "nomic-embed-text",
    endpoint: str = "http://localhost:11434",
    batch_log_every: int = 500,
) -> np.ndarray:
    """Embed texts via Ollama /api/embeddings. Returns float32 matrix (n, d), L2-normalized."""
    vectors = []
    url = f"{endpoint.rstrip('/')}/api/embeddings"
    for i, text in enumerate(texts):
        payload = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
        vec = np.asarray(body["embedding"], dtype=np.float32)
        vectors.append(vec)
        if batch_log_every and (i + 1) % batch_log_every == 0:
            print(f"  embedded {i+1}/{len(texts)}")
    mat = np.vstack(vectors)
    # L2 normalize for cosine via inner product
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat


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
