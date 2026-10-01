"""Process-wide memo for pure file loaders (FJC CSV, surname lists, YAML).

A loader wrapped with ``file_memo`` runs once per distinct argument tuple and
file mtime. Callers must treat the returned object as read-only.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable


def _freeze(x: Any) -> Any:
    if isinstance(x, (list, tuple)):
        return tuple(_freeze(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
    if isinstance(x, set):
        return frozenset(x)
    if isinstance(x, Path):
        try:
            return (str(x), x.stat().st_mtime_ns)
        except OSError:
            return (str(x), None)
    return x


def file_memo(fn: Callable) -> Callable:
    cache: dict[Any, Any] = {}

    @functools.wraps(fn)
    def wrapper(*args: Any) -> Any:
        key = tuple(_freeze(a) for a in args)
        if key not in cache:
            cache[key] = fn(*args)
        return cache[key]

    wrapper.cache = cache  # type: ignore[attr-defined]
    return wrapper
