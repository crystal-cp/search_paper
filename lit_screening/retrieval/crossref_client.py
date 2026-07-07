"""Crossref works API client skeleton."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import requests

from lit_screening.dedup import normalize_doi
from lit_screening.models import Paper
from lit_screening.retrieval.base import RetrievalResult
from lit_screening.utils import safe_int, stable_id


class CrossrefClient:
    """Small Crossref client for future DOI metadata completion."""

    provider_name = "crossref"
    base_url = "https://api.crossref.org/works"

    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 1,
        sleep_seconds: float = 1.0,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    def works(self, query: str, rows: int = 20) -> RetrievalResult:
        """Search Crossref works and normalize metadata into Paper objects."""

        if rows <= 0:
            return RetrievalResult(raw=self._raw_payload(query=query, items=[]), papers=[])
        params = {"query": query, "rows": rows}
        raw = self._get_json(self.base_url, params=params, query=query)
        return RetrievalResult(raw=raw, papers=self._normalize_many(raw))

    def resolve_doi(self, doi: str) -> RetrievalResult:
        """Resolve one DOI through Crossref and return a one-paper result."""

        normalized_doi = normalize_doi(doi)
        if not normalized_doi:
            raw = self._raw_payload(
                query=doi,
                items=[],
                error="missing_doi",
                error_details={
                    "status_code": None,
                    "error_message": "A DOI is required.",
                },
            )
            return RetrievalResult(raw=raw, papers=[])
        raw = self._get_json(
            f"{self.base_url}/{quote(normalized_doi, safe='')}",
            params={},
            query=normalized_doi,
        )
        if raw.get("error"):
            return RetrievalResult(raw=raw, papers=[])
        message = as_dict(raw.get("message"))
        if not message:
            return RetrievalResult(raw=raw, papers=[])
        return RetrievalResult(raw=raw, papers=[self._normalize_item(message)])

    def _get_json(
        self,
        url: str,
        params: dict[str, Any],
        query: str,
    ) -> dict[str, Any]:
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(self.sleep_seconds * (attempt + 1))
                    continue
                response.raise_for_status()
                raw = response.json()
                return self._with_provider_metadata(raw, query=query)
            except requests.RequestException as exc:
                last_error = exc.__class__.__name__
                if attempt < self.retries:
                    time.sleep(self.sleep_seconds * (attempt + 1))
                    continue
                return self._raw_payload(
                    query=query,
                    items=[],
                    error=last_error,
                    error_details=request_error_details(exc),
                )
        return self._raw_payload(query=query, items=[], error=last_error)

    def _normalize_many(self, raw: dict[str, Any]) -> list[Paper]:
        message = as_dict(raw.get("message"))
        return [
            self._normalize_item(item)
            for item in as_list(message.get("items"))
            if isinstance(item, dict)
        ]

    def _normalize_item(self, item: dict[str, Any]) -> Paper:
        title = first_text(item.get("title"))
        doi = normalize_doi(item.get("DOI"))
        year = issued_year(item)
        venue = first_text(item.get("container-title"))
        url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
        paper_id = stable_id(doi or url or title)
        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=clean_text(item.get("abstract", "")),
            authors=authors_from_crossref(item.get("author")),
            year=year,
            venue=venue,
            doi=doi,
            url=url,
            source_provider=self.provider_name,
            retrieval_provider=self.provider_name,
            retrieval_stage=self.provider_name,
            retrieval_query=str(item.get("_retrieval_query") or ""),
            source_stage="metadata",
            provider_ids={"crossref_doi": doi} if doi else {},
            publication_types=[
                str(item.get("type"))
            ]
            if item.get("type")
            else [],
            reference_count=safe_int(item.get("reference-count"), default=0),
            raw={
                **item,
                "source": self.provider_name,
                "_retrieval": {
                    "retrieval_provider": self.provider_name,
                    "retrieval_stage": self.provider_name,
                    "retrieval_query": str(item.get("_retrieval_query") or ""),
                },
            },
        )

    def _with_provider_metadata(
        self,
        raw: dict[str, Any],
        query: str,
    ) -> dict[str, Any]:
        message = as_dict(raw.get("message"))
        if isinstance(message.get("items"), list):
            for item in message["items"]:
                if isinstance(item, dict):
                    item["_retrieval_query"] = query
        elif message:
            message["_retrieval_query"] = query
        return {
            **raw,
            "provider": raw.get("provider") or self.provider_name,
            "source": raw.get("source") or self.provider_name,
            "retrieval_provider": self.provider_name,
            "retrieval_stage": self.provider_name,
            "retrieval_query": query,
        }

    def _raw_payload(
        self,
        query: str,
        items: list[dict[str, Any]],
        error: str = "",
        error_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "provider": self.provider_name,
            "source": self.provider_name,
            "message": {"items": items},
            "retrieval_provider": self.provider_name,
            "retrieval_stage": self.provider_name,
            "retrieval_query": query,
        }
        if error:
            payload["error"] = error
            payload.update(error_details or {})
        return payload


def first_text(value: Any) -> str:
    if isinstance(value, list) and value:
        return clean_text(value[0])
    return clean_text(value)


def authors_from_crossref(value: Any) -> list[str]:
    authors: list[str] = []
    for author in as_list(value):
        if not isinstance(author, dict):
            continue
        given = clean_text(author.get("given", ""))
        family = clean_text(author.get("family", ""))
        name = clean_text(author.get("name", ""))
        full_name = " ".join(part for part in [given, family] if part).strip()
        if full_name:
            authors.append(full_name)
        elif name:
            authors.append(name)
    return authors


def issued_year(item: dict[str, Any]) -> int | None:
    for key in ["published-print", "published-online", "published", "issued"]:
        date_parts = as_list(as_dict(item.get(key)).get("date-parts"))
        if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            year = safe_int(date_parts[0][0], default=0)
            if year:
                return year
    return None


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def request_error_details(exc: requests.RequestException) -> dict[str, Any]:
    """Return safe request-error details without leaking request headers."""

    response = getattr(exc, "response", None)
    if response is None:
        return {"status_code": None, "error_message": str(exc)[:300]}
    return {
        "status_code": response.status_code,
        "error_message": response.text[:300],
    }
