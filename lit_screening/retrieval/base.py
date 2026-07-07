"""Base interfaces for retrieval clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lit_screening.models import Paper


@dataclass
class RetrievalResult:
    """Raw provider payload plus normalized papers."""

    raw: dict
    papers: list[Paper]


class RetrievalClient(Protocol):
    """Minimal protocol implemented by provider clients and test fakes."""

    provider_name: str

    def search(
        self,
        query: str,
        max_results: int,
        from_year: int | None = None,
        sort_mode: str = "relevance",
        search_mode: str = "keyword",
    ) -> RetrievalResult:
        """Search one provider and return normalized papers."""
