"""Tier_V3 disambiguation engine (type-agnostic; config-driven)."""

__version__ = "0.1.0"

from .extract import extract_mentions
from .tiers import run_cascade
from .cluster import cluster_mentions
from .rdf_emit import emit_ttl

__all__ = [
    "extract_mentions",
    "run_cascade",
    "cluster_mentions",
    "emit_ttl",
    "__version__",
]
