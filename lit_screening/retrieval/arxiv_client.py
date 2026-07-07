"""arXiv Atom API client skeleton."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from lit_screening.models import Paper
from lit_screening.retrieval.base import RetrievalResult
from lit_screening.utils import safe_int, stable_id


ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivClient:
    """Small arXiv search client for future optional preprint retrieval."""

    provider_name = "arxiv"
    base_url = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        timeout: float = 20.0,
        retries: int = 1,
        sleep_seconds: float = 1.0,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    def search(self, query: str, max_results: int = 20) -> RetrievalResult:
        """Search arXiv and normalize Atom entries into Paper objects."""

        if max_results <= 0:
            return RetrievalResult(
                raw=self._raw_payload(query=query, entries=[]),
                papers=[],
            )

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        }
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(self.sleep_seconds * (attempt + 1))
                    continue
                response.raise_for_status()
                raw = self._parse_feed(response.text, query=query)
                return RetrievalResult(raw=raw, papers=self._normalize_many(raw))
            except (requests.RequestException, ET.ParseError) as exc:
                last_error = exc.__class__.__name__
                if attempt < self.retries:
                    time.sleep(self.sleep_seconds * (attempt + 1))
                    continue
                raw = self._raw_payload(
                    query=query,
                    entries=[],
                    error=last_error,
                    error_details=request_error_details(exc),
                )
                return RetrievalResult(raw=raw, papers=[])

        raw = self._raw_payload(query=query, entries=[], error=last_error)
        return RetrievalResult(raw=raw, papers=[])

    def _parse_feed(self, text: str, query: str) -> dict[str, Any]:
        root = ET.fromstring(text)
        entries = [
            atom_entry_to_dict(entry)
            for entry in root.findall("atom:entry", ARXIV_ATOM_NS)
        ]
        return self._raw_payload(
            query=query,
            entries=entries,
            feed_title=atom_text(root, "atom:title"),
        )

    def _normalize_many(self, raw: dict[str, Any]) -> list[Paper]:
        return [
            self._normalize_entry(
                item,
                retrieval_query=str(raw.get("retrieval_query") or ""),
            )
            for item in as_list(raw.get("entries"))
            if isinstance(item, dict)
        ]

    def _normalize_entry(self, item: dict[str, Any], retrieval_query: str = "") -> Paper:
        title = clean_text(item.get("title", ""))
        arxiv_id = arxiv_id_from_url(item.get("id", ""))
        published_year = safe_int(str(item.get("published", ""))[:4], default=0) or None
        url = item.get("html_url") or item.get("id") or ""
        paper_id = stable_id(arxiv_id or title)
        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=clean_text(item.get("summary", "")),
            authors=[
                clean_text(author)
                for author in as_list(item.get("authors"))
                if clean_text(author)
            ],
            year=published_year,
            venue="arXiv",
            url=url,
            source_provider=self.provider_name,
            retrieval_provider=self.provider_name,
            retrieval_stage=self.provider_name,
            retrieval_query=retrieval_query,
            source_stage="preprint",
            provider_ids={"arxiv": arxiv_id} if arxiv_id else {},
            publication_date=item.get("published", ""),
            publication_types=["preprint"],
            fields_of_study=[
                str(category)
                for category in as_list(item.get("categories"))
                if category
            ],
            raw={
                **item,
                "source": self.provider_name,
                "_retrieval": {
                    "retrieval_provider": self.provider_name,
                    "retrieval_stage": self.provider_name,
                    "retrieval_query": retrieval_query,
                },
            },
        )

    def _raw_payload(
        self,
        query: str,
        entries: list[dict[str, Any]],
        feed_title: str = "",
        error: str = "",
        error_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "provider": self.provider_name,
            "source": self.provider_name,
            "feed_title": feed_title,
            "entries": entries,
            "retrieval_provider": self.provider_name,
            "retrieval_stage": self.provider_name,
            "retrieval_query": query,
        }
        if error:
            payload["error"] = error
            payload.update(error_details or {})
        return payload


def atom_entry_to_dict(entry: ET.Element) -> dict[str, Any]:
    """Convert one arXiv Atom entry into a JSON-like dict."""

    links = []
    for link in entry.findall("atom:link", ARXIV_ATOM_NS):
        links.append(dict(link.attrib))
    html_url = ""
    pdf_url = ""
    for link in links:
        href = link.get("href", "")
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = href
        elif link.get("rel") == "alternate":
            html_url = href
    categories = [
        category.attrib.get("term", "")
        for category in entry.findall("atom:category", ARXIV_ATOM_NS)
        if category.attrib.get("term")
    ]
    authors = [
        atom_text(author, "atom:name")
        for author in entry.findall("atom:author", ARXIV_ATOM_NS)
        if atom_text(author, "atom:name")
    ]
    return {
        "id": atom_text(entry, "atom:id"),
        "title": atom_text(entry, "atom:title"),
        "summary": atom_text(entry, "atom:summary"),
        "published": atom_text(entry, "atom:published"),
        "updated": atom_text(entry, "atom:updated"),
        "authors": authors,
        "categories": categories,
        "html_url": html_url,
        "pdf_url": pdf_url,
        "links": links,
    }


def atom_text(element: ET.Element, path: str) -> str:
    child = element.find(path, ARXIV_ATOM_NS)
    return clean_text(child.text if child is not None else "")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def arxiv_id_from_url(url: str) -> str:
    value = str(url or "").rstrip("/")
    if not value:
        return ""
    return value.rsplit("/", 1)[-1]


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def request_error_details(exc: Exception) -> dict[str, Any]:
    """Return safe request or parse error details."""

    response = getattr(exc, "response", None)
    if response is None:
        return {"status_code": None, "error_message": str(exc)[:300]}
    return {
        "status_code": response.status_code,
        "error_message": response.text[:300],
    }
