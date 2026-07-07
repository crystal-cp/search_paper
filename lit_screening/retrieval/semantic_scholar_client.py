"""Semantic Scholar Graph API client."""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

import requests

from lit_screening.cache import load_cached_response, save_cached_response
from lit_screening.dedup import normalize_doi
from lit_screening.models import Paper
from lit_screening.retrieval.base import RetrievalResult
from lit_screening.utils import safe_int, stable_id


DEFAULT_SEMANTIC_SCHOLAR_FIELDS = (
    "paperId,title,abstract,authors,year,publicationDate,publicationTypes,venue,"
    "citationCount,influentialCitationCount,referenceCount,externalIds,url,"
    "openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,tldr"
)


class SemanticScholarClient:
    """Small Semantic Scholar search client with optional API-key header."""

    provider_name = "semantic_scholar"
    api_base_url = "https://api.semanticscholar.org/graph/v1"
    recommendations_base_url = "https://api.semanticscholar.org/recommendations/v1"
    base_url = f"{api_base_url}/paper/search"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        timeout: float = 20.0,
        retries: int = 2,
        sleep_seconds: float = 1.0,
        fields: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("S2_API_KEY")
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.timeout = timeout
        self.retries = retries
        self.sleep_seconds = sleep_seconds
        self.fields = fields or DEFAULT_SEMANTIC_SCHOLAR_FIELDS

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
            "fields": self.fields,
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
                    **request_error_details(exc),
                    "provider": self.provider_name,
                }

        return RetrievalResult(raw=raw, papers=self._normalize_many(raw))

    def get_paper(self, paper_id: str) -> RetrievalResult:
        """Fetch one paper by Semantic Scholar ID or prefixed external ID."""

        if not paper_id:
            return RetrievalResult(raw={"data": []}, papers=[])
        raw = self._get_json(
            f"{self.api_base_url}/paper/{quote(paper_id, safe=':')}",
            {"fields": self.fields},
        )
        papers = [] if raw.get("error") else [self._normalize_result(raw, rank=1)]
        return RetrievalResult(raw=raw, papers=papers)

    def get_references(self, paper_id: str, limit: int = 10) -> RetrievalResult:
        """Fetch papers referenced by a seed paper."""

        return self._paper_links(
            paper_id,
            endpoint="references",
            nested_key="citedPaper",
            limit=limit,
        )

    def get_citations(self, paper_id: str, limit: int = 10) -> RetrievalResult:
        """Fetch papers that cite a seed paper."""

        return self._paper_links(
            paper_id,
            endpoint="citations",
            nested_key="citingPaper",
            limit=limit,
        )

    def get_recommendations(self, paper_id: str, limit: int = 10) -> RetrievalResult:
        """Fetch recommended papers for a seed paper."""

        if not paper_id:
            return RetrievalResult(raw={"data": []}, papers=[])
        raw = self._get_json(
            f"{self.recommendations_base_url}/papers/forpaper/{quote(paper_id, safe=':')}",
            {"limit": limit, "fields": self.fields},
        )
        items = as_list(raw.get("recommendedPapers")) or as_list(raw.get("data"))
        papers = [
            self._normalize_result(item, rank=index)
            for index, item in enumerate(items, start=1)
            if isinstance(item, dict)
        ]
        return RetrievalResult(raw=raw, papers=papers)

    def _paper_links(
        self,
        paper_id: str,
        endpoint: str,
        nested_key: str,
        limit: int,
    ) -> RetrievalResult:
        """Fetch nested citation/reference records and normalize child papers."""

        if not paper_id:
            return RetrievalResult(raw={"data": []}, papers=[])
        nested_fields = ",".join(
            f"{nested_key}.{field.strip()}"
            for field in self.fields.split(",")
            if field.strip()
        )
        raw = self._get_json(
            f"{self.api_base_url}/paper/{quote(paper_id, safe=':')}/{endpoint}",
            {"limit": limit, "fields": nested_fields},
        )
        papers = []
        for index, item in enumerate(as_list(raw.get("data")), start=1):
            child = as_dict(item).get(nested_key)
            if isinstance(child, dict):
                papers.append(self._normalize_result(child, rank=index))
        return RetrievalResult(raw=raw, papers=papers)

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET JSON with retry handling and no API-key leakage."""

        headers = {"x-api-key": self.api_key} if self.api_key else None
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(self.sleep_seconds * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc.__class__.__name__
                if attempt < self.retries:
                    time.sleep(self.sleep_seconds * (attempt + 1))
                    continue
                return {
                    "data": [],
                    "error": last_error,
                    **request_error_details(exc),
                    "provider": self.provider_name,
                }

    def _normalize_many(self, raw: dict[str, Any]) -> list[Paper]:
        return [
            self._normalize_result(item, rank=index)
            for index, item in enumerate(as_list(raw.get("data")), start=1)
            if isinstance(item, dict)
        ]

    def _normalize_result(self, item: dict[str, Any], rank: int = 0) -> Paper:
        external_ids = as_dict(item.get("externalIds"))
        doi = normalize_doi(external_ids.get("DOI"))
        semantic_id = item.get("paperId") or ""
        authors = [
            as_dict(author).get("name", "")
            for author in as_list(item.get("authors"))
            if as_dict(author).get("name")
        ]
        title = item.get("title") or ""
        paper_id = stable_id(doi or semantic_id or title)

        provider_ids = {"semantic_scholar": semantic_id} if semantic_id else {}
        for key, value in external_ids.items():
            if value:
                provider_ids[key.lower()] = str(value)
        open_access_pdf = as_dict(item.get("openAccessPdf"))
        tldr = as_dict(item.get("tldr"))
        s2_fields = [
            as_dict(field).get("category", "")
            for field in as_list(item.get("s2FieldsOfStudy"))
            if as_dict(field).get("category")
        ]
        fields_of_study = [
            *[str(value) for value in as_list(item.get("fieldsOfStudy")) if value],
            *s2_fields,
        ]

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
            api_relevance_score=1.0 / rank if rank else 0.0,
            semantic_scholar_rank=rank,
            publication_date=item.get("publicationDate") or "",
            publication_types=[
                str(value) for value in as_list(item.get("publicationTypes")) if value
            ],
            fields_of_study=list(dict.fromkeys(fields_of_study)),
            influential_citation_count=safe_int(
                item.get("influentialCitationCount"),
                default=0,
            ),
            reference_count=safe_int(item.get("referenceCount"), default=0),
            open_access_pdf_url=open_access_pdf.get("url") or "",
            tldr=tldr.get("text") or "",
            raw=item,
        )


def as_list(value: Any) -> list[Any]:
    """Return a list for provider fields that may be null."""

    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    """Return a dict for provider fields that may be null."""

    return value if isinstance(value, dict) else {}


def request_error_details(exc: requests.RequestException) -> dict[str, Any]:
    """Return safe request-error details without leaking request headers or keys."""

    response = getattr(exc, "response", None)
    if response is None:
        return {"status_code": None, "error_message": str(exc)[:300]}
    return {
        "status_code": response.status_code,
        "error_message": response.text[:300],
    }
