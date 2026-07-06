"""Local JSON cache for provider API responses."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .utils import ensure_dir, read_json, write_json


def cache_key(provider: str, query: str, max_results: int, from_year: int | None) -> str:
    """Create a deterministic cache key for one provider query."""

    raw = f"{provider}|{query}|{max_results}|{from_year or ''}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    provider_prefix = provider.replace(" ", "_").lower()
    return f"{provider_prefix}_{digest}.json"


def cache_path(
    cache_dir: str | Path,
    provider: str,
    query: str,
    max_results: int,
    from_year: int | None,
) -> Path:
    """Return the cache file path for a provider query."""

    return Path(cache_dir) / cache_key(provider, query, max_results, from_year)


def load_cached_response(
    cache_dir: str | Path,
    provider: str,
    query: str,
    max_results: int,
    from_year: int | None,
) -> Any | None:
    """Load a cached provider response, or return None if absent."""

    path = cache_path(cache_dir, provider, query, max_results, from_year)
    if not path.exists():
        return None
    return read_json(path)


def save_cached_response(
    cache_dir: str | Path,
    provider: str,
    query: str,
    max_results: int,
    from_year: int | None,
    data: Any,
) -> None:
    """Persist a raw provider response as JSON."""

    ensure_dir(cache_dir)
    write_json(cache_path(cache_dir, provider, query, max_results, from_year), data)
