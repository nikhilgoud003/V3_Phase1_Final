"""Ollama / Tier3 preflight — cascade must refuse to start if this fails."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class PreflightError(RuntimeError):
    """Raised when mandatory infrastructure checks fail."""


class CascadeFailedError(RuntimeError):
    """Raised when a run must abort (e.g. Tier3 error rate > 1%)."""

    def __init__(self, message: str, *, stats: dict | None = None):
        super().__init__(message)
        self.stats = stats or {}
        self.status = "FAILED"


def _http_json(url: str, payload: dict | None = None, timeout: float = 60.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def preflight_ollama(
    *,
    endpoint: str = "http://127.0.0.1:11434",
    llm_model: str = "qwen2.5:7b",
    embed_model: str = "nomic-embed-text",
) -> dict[str, Any]:
    """
    Verify Ollama is up, required models are present, and a tiny generate + embed works.
    Raises PreflightError on any failure.
    """
    base = endpoint.rstrip("/")
    report: dict[str, Any] = {"endpoint": base, "ok": False}

    try:
        tags = _http_json(f"{base}/api/tags", timeout=10.0)
    except Exception as e:
        raise PreflightError(f"Ollama /api/tags failed at {base}: {e}") from e

    names = {m.get("name") or m.get("model") for m in (tags.get("models") or [])}
    # allow :latest aliases
    def has_model(want: str) -> bool:
        if want in names:
            return True
        base_name = want.split(":")[0]
        return any((n or "").startswith(base_name) for n in names)

    if not has_model(llm_model):
        raise PreflightError(f"Required LLM model not pulled: {llm_model} (have {sorted(names)[:12]}…)")
    if not has_model(embed_model):
        raise PreflightError(f"Required embed model not pulled: {embed_model}")

    report["models_ok"] = True
    report["models"] = sorted(n for n in names if n)

    # 1-token / short generation.
    # Colab cold-load of 14B can exceed 2–5+ minutes; override with TIER_V3_PREFLIGHT_TIMEOUT.
    gen_timeout = float(os.environ.get("TIER_V3_PREFLIGHT_TIMEOUT", "300"))
    try:
        gen = _http_json(
            f"{base}/api/generate",
            {
                "model": llm_model,
                "prompt": "Reply with exactly: OK",
                "stream": False,
                "options": {"temperature": 0, "num_predict": 8},
            },
            timeout=gen_timeout,
        )
    except Exception as e:
        raise PreflightError(
            f"Ollama generate ping failed for {llm_model} "
            f"(timeout={gen_timeout}s — first load on Colab can be slow; "
            f"warm with `ollama run {llm_model}` or use qwen2.5:7b): {e}"
        ) from e

    text = (gen.get("response") or "").strip()
    if not text:
        raise PreflightError(f"Ollama generate ping returned empty response for {llm_model}")
    report["generate_ping"] = text[:80]
    report["generate_ok"] = True

    # Embedding ping
    try:
        emb = _http_json(
            f"{base}/api/embeddings",
            {"model": embed_model, "prompt": "ping"},
            timeout=60.0,
        )
    except Exception as e:
        raise PreflightError(f"Ollama embeddings ping failed for {embed_model}: {e}") from e

    vec = emb.get("embedding") or []
    if len(vec) < 8:
        raise PreflightError(f"Ollama embeddings ping returned bad vector (len={len(vec)})")
    report["embed_dim"] = len(vec)
    report["embed_ok"] = True
    report["ok"] = True
    return report


def check_tier3_error_rate(
    *,
    llm_calls: int,
    errors: int,
    consecutive_errors: int,
    max_error_rate: float = 0.01,
    min_attempts_for_rate: int = 100,
    max_consecutive_errors: int = 3,
) -> None:
    """Abort if Tier3 error rate is too high or Ollama appears down.

    A single isolated failure (common on Ollama reconnect) must not abort a
    long resume: require either consecutive errors or a sustained high rate
    over enough fresh attempts.
    """
    attempts = llm_calls + errors
    if consecutive_errors >= max_consecutive_errors:
        raise CascadeFailedError(
            f"Tier3 consecutive errors={consecutive_errors} "
            f"(>= {max_consecutive_errors}) — aborting (likely Ollama down).",
            stats={"llm_calls": llm_calls, "errors": errors, "consecutive_errors": consecutive_errors},
        )
    # One isolated blip is OK; rate gate needs a real sample.
    if errors <= 1:
        return
    if attempts >= min_attempts_for_rate:
        rate = errors / attempts
        if rate > max_error_rate:
            raise CascadeFailedError(
                f"Tier3 error rate {rate:.2%} exceeds {max_error_rate:.0%} "
                f"({errors}/{attempts}) — aborting. Do not use entity counts from this run.",
                stats={"llm_calls": llm_calls, "errors": errors, "error_rate": rate},
            )
