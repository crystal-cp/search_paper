"""OpenAlex metadata client."""

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


class OpenAlexClient:
    """Small OpenAlex search client with caching and graceful failures."""

    provider_name = "openalex"
    base_url = "https://api.openalex.org/works"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        timeout: float = 20.0,
        retries: int = 2,
        sleep_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENALEX_API_KEY")
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
        sort_mode: str = "relevance",
    ) -> RetrievalResult:
        """Search OpenAlex and normalize results into Paper objects."""

        if max_results <= 0:
            return RetrievalResult(raw={"results": []}, papers=[])

        if self.use_cache:
            cached = load_cached_response(
                self.cache_dir,
                f"{self.provider_name}_{sort_mode}",
                query,
                max_results,
                from_year,
            )
            if cached is not None:
                return RetrievalResult(raw=cached, papers=self._normalize_many(cached))

        params: dict[str, Any] = {
            "search": query,
            "per-page": max_results,
        }
        if from_year:
            params["filter"] = f"from_publication_date:{from_year}-01-01"
        if sort_mode in {"recent", "recent_then_relevant"}:
            params["sort"] = "publication_date:desc"
        elif sort_mode == "cited":
            params["sort"] = "cited_by_count:desc"
        if self.api_key:
            params["api_key"] = self.api_key

        raw: dict[str, Any] = {"results": []}
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(self.base_url, params=params, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(self.sleep_seconds * (attempt + 1))
                    continue
                response.raise_for_status()
                raw = response.json()
                if self.use_cache:
                    save_cached_response(
                        self.cache_dir,
                        f"{self.provider_name}_{sort_mode}",
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
                    "results": [],
                    "error": last_error,
                    "provider": self.provider_name,
                }

        return RetrievalResult(raw=raw, papers=self._normalize_many(raw))

    def _normalize_many(self, raw: dict[str, Any]) -> list[Paper]:
        return [self._normalize_result(item) for item in raw.get("results", [])]

    def _normalize_result(self, item: dict[str, Any]) -> Paper:
        title = item.get("title") or item.get("display_name") or ""
        doi = normalize_doi(item.get("doi"))
        openalex_id = item.get("id") or ""
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
        authors = [
            authorship.get("author", {}).get("display_name", "")
            for authorship in item.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ]
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = source.get("display_name") or item.get("host_venue", {}).get("display_name", "")
        url = item.get("doi") or item.get("landing_page_url") or openalex_id
        paper_id = stable_id(doi or openalex_id or title)
        relevance_score = safe_float(item.get("relevance_score"))
        concepts = [
            concept.get("display_name", "")
            for concept in item.get("concepts", [])
            if concept.get("display_name")
        ]
        topics = [
            topic.get("display_name", "")
            for topic in item.get("topics", [])
            if topic.get("display_name")
        ]
        open_access = item.get("open_access") or {}
        best_location = item.get("best_oa_location") or {}
        open_access_pdf_url = (
            best_location.get("pdf_url")
            or primary_location.get("pdf_url")
            or open_access.get("oa_url")
            or ""
        )
        publication_types = [
            str(value)
            for value in [item.get("type"), item.get("type_crossref")]
            if value
        ]

        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=safe_int(item.get("publication_year"), default=0) or None,
            venue=venue,
            doi=doi,
            url=url,
            source_provider=self.provider_name,
            provider_ids={"openalex": openalex_id} if openalex_id else {},
            citation_count=safe_int(item.get("cited_by_count"), default=0),
            api_relevance_score=relevance_score,
            openalex_relevance_score=relevance_score,
            publication_date=item.get("publication_date") or "",
            publication_types=publication_types,
            fields_of_study=list(dict.fromkeys([*concepts, *topics])),
            reference_count=safe_int(item.get("referenced_works_count"), default=0),
            open_access_pdf_url=open_access_pdf_url,
            raw=item,
        )


def reconstruct_abstract(abstract_inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct OpenAlex abstracts from an inverted index."""

    if not abstract_inverted_index:
        return ""
    all_positions = [
        position
        for positions in abstract_inverted_index.values()
        for position in positions
    ]
    if not all_positions:
        return ""
    max_position = max(all_positions)
    words = [""] * (max_position + 1)
    for word, positions in abstract_inverted_index.items():
        for position in positions:
            words[position] = word
    return " ".join(word for word in words if word)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without raising."""

    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
