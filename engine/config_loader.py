"""Load and resolve paths for entity-type YAML configs.

Path roots (all optional; defaults preserve Mac repo-relative behavior):

  TIER_V3_DATA_DIR    — replaces the ``data/`` prefix for any relative ``data/...`` path
  TIER_V3_OUTPUT_DIR  — redirects run outputs (decisions, cache, reports, clusters,
                        rdf, mentions, embeddings, gold) under this directory
  TIER_V3_JSON_DIR    — default PACER JSON directory for scripts (if set)
  OLLAMA_HOST         — Ollama base URL (e.g. http://127.0.0.1:11434)
  TIER_V3_LLM_MODEL   — Tier3 model name (overrides configs/judges.yaml)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Relative paths under data/ that are run outputs (may live on Drive via TIER_V3_OUTPUT_DIR)
_OUTPUT_DATA_PREFIXES = (
    "data/decisions/",
    "data/reports/",
    "data/clusters/",
    "data/rdf/",
    "data/mentions/",
    "data/embeddings/",
    "data/gold/",
    "data/incremental/",
)


def repo_root() -> Path:
    return ROOT


def _norm_rel(rel: str) -> str:
    return str(rel).replace("\\", "/").lstrip("./")


def is_output_rel(rel: str) -> bool:
    r = _norm_rel(rel)
    if r in {
        "data/decisions",
        "data/reports",
        "data/clusters",
        "data/rdf",
        "data/mentions",
        "data/embeddings",
        "data/gold",
        "data/incremental",
    }:
        return True
    return any(r.startswith(p) for p in _OUTPUT_DATA_PREFIXES)


def apply_runtime_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply env overrides onto a loaded config (mutates and returns cfg)."""
    host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_ENDPOINT")
    if host:
        host = host.rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        t3 = cfg.setdefault("tier3", {})
        t3["endpoint"] = host
        t2 = cfg.setdefault("tier2", {})
        t2["ollama_endpoint"] = host

    llm = os.environ.get("TIER_V3_LLM_MODEL")
    if llm:
        cfg.setdefault("tier3", {})["model"] = llm.strip()

    t2_backend = os.environ.get("TIER_V3_TIER2_BACKEND")
    if t2_backend:
        cfg.setdefault("tier2", {})["backend"] = t2_backend.strip()

    json_dir = os.environ.get("TIER_V3_JSON_DIR")
    if json_dir:
        cfg["_json_dir"] = str(Path(json_dir).expanduser())

    data_dir = os.environ.get("TIER_V3_DATA_DIR")
    if data_dir:
        cfg["_data_dir"] = str(Path(data_dir).expanduser().resolve())

    out_dir = os.environ.get("TIER_V3_OUTPUT_DIR")
    if out_dir:
        cfg["_output_dir"] = str(Path(out_dir).expanduser().resolve())

    return cfg


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(ROOT)
    return apply_runtime_overrides(cfg)


def resolve_path(cfg: dict[str, Any], rel: str) -> Path:
    """Resolve a config-relative path with optional DATA/OUTPUT env remapping."""
    p = Path(rel)
    if p.is_absolute():
        return p

    rel_s = _norm_rel(rel)
    out_dir = cfg.get("_output_dir") or os.environ.get("TIER_V3_OUTPUT_DIR")
    if out_dir and is_output_rel(rel_s):
        suffix = rel_s[len("data/") :] if rel_s.startswith("data/") else rel_s
        return Path(out_dir).expanduser().resolve() / suffix

    data_dir = cfg.get("_data_dir") or os.environ.get("TIER_V3_DATA_DIR")
    if data_dir and rel_s.startswith("data/"):
        return Path(data_dir).expanduser().resolve() / rel_s[len("data/") :]

    return Path(cfg.get("_root") or ROOT) / rel_s


def default_json_dir(cfg: dict[str, Any] | None = None) -> Path:
    """PACER JSON directory: TIER_V3_JSON_DIR > cfg[_json_dir] > data/json/pilot_1000."""
    env = os.environ.get("TIER_V3_JSON_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if cfg and cfg.get("_json_dir"):
        return Path(cfg["_json_dir"])
    if cfg:
        return resolve_path(cfg, "data/json/pilot_1000")
    return ROOT / "data/json/pilot_1000"


def ollama_endpoint(cfg: dict[str, Any] | None = None, default: str = "http://localhost:11434") -> str:
    host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_ENDPOINT")
    if host:
        host = host.rstrip("/")
        return host if host.startswith("http") else "http://" + host
    if cfg:
        t3 = cfg.get("tier3") or {}
        if t3.get("endpoint"):
            return str(t3["endpoint"]).rstrip("/")
        t2 = cfg.get("tier2") or {}
        if t2.get("ollama_endpoint"):
            return str(t2["ollama_endpoint"]).rstrip("/")
    return default.rstrip("/")
