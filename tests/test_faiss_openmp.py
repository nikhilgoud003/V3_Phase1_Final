#!/usr/bin/env python3
"""Regression: FAISS must survive being used in the same process as spaCy.

Root cause (macOS): spaCy/thinc and faiss-cpu each link their own libomp.
Two OpenMP runtimes + multi-threaded FAISS search = SIGSEGV (masked by
KMP_DUPLICATE_LIB_OK=TRUE). engine.embeddings pins FAISS to 1 thread.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import sys
sys.path.insert(0, {root!r})
import spacy
nlp = spacy.load("en_core_web_sm")
for _ in range(25):
    nlp("Signed by Judge John W. Darrah. Referred to Magistrate Judge Susan E. Cox.")

import numpy as np
from engine.embeddings import build_faiss_ip_index

rng = np.random.default_rng(0)
v = rng.standard_normal((4000, 768)).astype("float32")
v /= np.linalg.norm(v, axis=1, keepdims=True)
index = build_faiss_ip_index(v)
sims, idxs = index.search(np.ascontiguousarray(v[:1000]), 25)
assert index.ntotal == 4000
assert sims.shape == (1000, 25)
print("FAISS_OK")
'''


def test_faiss_after_spacy_does_not_segfault():
    code = CHILD.format(root=str(ROOT))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "KMP_DUPLICATE_LIB_OK": "TRUE", "HOME": str(Path.home())},
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert "FAISS_OK" in proc.stdout, proc.stdout


def test_thread_pin_default_is_one():
    import faiss

    from engine.embeddings import configure_faiss_threads

    assert configure_faiss_threads(faiss) == 1


if __name__ == "__main__":
    test_faiss_after_spacy_does_not_segfault()
    test_thread_pin_default_is_one()
    print("all ok")
