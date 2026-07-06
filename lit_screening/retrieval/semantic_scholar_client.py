"""Semantic Scholar Graph API client."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from lit_screening.cache import load_cached_response, save_cached_response
from lit_screening.dedup import normalize_doi
from lit_screening.models import Paper
from lit_screening.retrieval.base import RetrievalResult
from lit_screening.utils import safe_int, stable_id


class SemanticScholarClient:
    """Small Semantic Scholar search client with optional API-key header."""

    provider_name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        timeout: float = 20.0,
        retries: int = 2,
        sleep_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("S2_API_KEY")
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.timeout = timeout
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    def search(
        self,
        query: str,
        max_results: int,
        from_year: int | None = None,
    ) -> RetrievalResult:
        """Search Semantic Scholar and normalize results into Paper objects."""

        if max_results <= 0:
            return RetrievalResult(raw={"data": []}, papers=[])

        if self.use_cache:
            cached = load_cached_response(
                self.cache_dir, self.provider_name, query, max_results, from_year
            )
            if cached is not None:
                return RetrievalResult(raw=cached, papers=self._normalize_many(cached))

        params: dict[str, Any] = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,venue,citationCount,externalIds,url",
        }
        if from_year:
            params["year"] = f"{from_year}-"

        headers = {"x-api-key": self.api_key} if self.api_key else None
        raw: dict[str, Any] = {"data": []}
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(self.sleep_seconds * (attempt + 1))
                    continue
                response.raise_for_status()
                raw = response.json()
                if self.use_cache:
                    save_cached_response(
                        self.cache_dir,
                        self.provider_name,
                        query,
                        max_results,
                        from_year,
                        raw,
                    )
                break
            except requests.RequestException as exc:
                last_error = exc.__class__.__name__
                if attempt < self.retries:
                    time.sleep(self.sleep_seconds * (attempt + 1))
                    continue
                raw = {
                    "data": [],
                    "error": last_error,
                    "provider": self.provider_name,
                }

        return RetrievalResult(raw=raw, papers=self._normalize_many(raw))

    def _normalize_many(self, raw: dict[str, Any]) -> list[Paper]:
        return [self._normalize_result(item) for item in raw.get("data", [])]

    def _normalize_result(self, item: dict[str, Any]) -> Paper:
        external_ids = item.get("externalIds") or {}
        doi = normalize_doi(external_ids.get("DOI"))
        semantic_id = item.get("paperId") or ""
        authors = [
            author.get("name", "")
            for author in item.get("authors", [])
            if author.get("name")
        ]
        title = item.get("title") or ""
        paper_id = stable_id(doi or semantic_id or title)

        provider_ids = {"semantic_scholar": semantic_id} if semantic_id else {}
        for key, value in external_ids.items():
            if value:
                provider_ids[key.lower()] = str(value)

        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=item.get("abstract") or "",
            authors=authors,
            year=safe_int(item.get("year"), default=0) or None,
            venue=item.get("venue") or "",
            doi=doi,
            url=item.get("url") or "",
            source_provider=self.provider_name,
            provider_ids=provider_ids,
            citation_count=safe_int(item.get("citationCount"), default=0),
            raw=item,
        )
