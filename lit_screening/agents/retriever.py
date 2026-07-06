"""Retriever agent that coordinates provider clients."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from lit_screening.config import PipelineConfig
from lit_screening.models import Paper
from lit_screening.retrieval.base import RetrievalClient, RetrievalResult
from lit_screening.retrieval.openalex_client import OpenAlexClient
from lit_screening.retrieval.semantic_scholar_client import SemanticScholarClient
from lit_screening.utils import write_json


class RetrieverAgent:
    """Run planned queries through selected metadata providers."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        clients: dict[str, RetrievalClient] | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.clients = clients or self._default_clients()

    def _default_clients(self) -> dict[str, RetrievalClient]:
        return {
            "openalex": OpenAlexClient(
                cache_dir=self.config.cache_dir,
                use_cache=self.config.use_cache,
                timeout=self.config.request_timeout,
                retries=self.config.request_retries,
                sleep_seconds=self.config.rate_limit_sleep,
            ),
            "semantic_scholar": SemanticScholarClient(
                cache_dir=self.config.cache_dir,
                use_cache=self.config.use_cache,
                timeout=self.config.request_timeout,
                retries=self.config.request_retries,
                sleep_seconds=self.config.rate_limit_sleep,
            ),
        }

    def retrieve(
        self,
        queries: list[str] | dict[str, list[str]],
        providers: list[str],
        max_per_query: int,
        from_year: int | None,
        output_dir: str | Path | None = None,
        progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
        sort_mode: str = "relevance",
    ) -> tuple[list[Paper], dict[str, list[dict]], dict[str, int]]:
        """Retrieve papers and optionally save raw provider result bundles."""

        all_papers: list[Paper] = []
        raw_by_provider: dict[str, list[dict]] = {provider: [] for provider in providers}
        retrieval_counts: dict[str, int] = {provider: 0 for provider in providers}
        if max_per_query <= 0:
            if output_dir:
                out = Path(output_dir)
                for provider, raw_items in raw_by_provider.items():
                    write_json(out / f"raw_{provider}_results.json", raw_items)
            return all_papers, raw_by_provider, retrieval_counts

        for provider in providers:
            if provider not in self.clients:
                raise ValueError(f"Unknown provider: {provider}")
            client = self.clients[provider]
            provider_queries = queries.get(provider, []) if isinstance(queries, dict) else queries
            for index, query in enumerate(provider_queries, start=1):
                if progress_callback:
                    progress_callback(
                        "retrieval",
                        f"Searching {provider} ({index}/{len(provider_queries)})",
                        {
                            "provider": provider,
                            "query": query,
                            "current_query": index,
                            "total_queries": len(provider_queries),
                        },
                    )
                try:
                    if provider == "openalex":
                        result = client.search(
                            query,
                            max_per_query,
                            from_year,
                            sort_mode=sort_mode,
                        )
                    else:
                        result = client.search(query, max_per_query, from_year)
                except Exception as exc:
                    error_raw = {
                        "error": exc.__class__.__name__,
                        "error_message": str(exc)[:500],
                        "provider": provider,
                        "stage": "client_search",
                    }
                    raw_by_provider[provider].append(
                        {"query": query, "response": error_raw}
                    )
                    if output_dir:
                        write_json(
                            Path(output_dir) / f"raw_{provider}_results.json",
                            raw_by_provider[provider],
                        )
                    if progress_callback:
                        progress_callback(
                            "retrieval_error",
                            f"{provider} failed while searching",
                            {
                                "provider": provider,
                                "query": query,
                                "error": exc.__class__.__name__,
                                "message": str(exc)[:240],
                            },
                        )
                    raise
                if not isinstance(result, RetrievalResult):
                    result = RetrievalResult(raw=result[0], papers=result[1])
                raw_by_provider[provider].append({"query": query, "response": result.raw})
                all_papers.extend(result.papers)
                retrieval_counts[provider] += len(result.papers)
                if output_dir:
                    write_json(
                        Path(output_dir) / f"raw_{provider}_results.json",
                        raw_by_provider[provider],
                    )
                if progress_callback:
                    progress_callback(
                        "retrieval",
                        f"{provider} returned {len(result.papers)} papers",
                        {
                            "provider": provider,
                            "query": query,
                            "returned": len(result.papers),
                            "provider_total": retrieval_counts[provider],
                        },
                    )
                if should_stop_provider_after_error(result.raw):
                    if progress_callback:
                        progress_callback(
                            "retrieval",
                            f"Stopping {provider} after provider rate-limit or budget error",
                            {
                                "provider": provider,
                                "error": result.raw.get("error", ""),
                                "status_code": result.raw.get("status_code", ""),
                                "message": result.raw.get("error_message", "")[:180],
                            },
                        )
                    break

        if output_dir:
            out = Path(output_dir)
            for provider, raw_items in raw_by_provider.items():
                write_json(out / f"raw_{provider}_results.json", raw_items)

        return all_papers, raw_by_provider, retrieval_counts


def should_stop_provider_after_error(raw: dict[str, Any]) -> bool:
    """Return True when more queries to this provider would likely waste quota."""

    if raw.get("status_code") != 429:
        return False
    message = str(raw.get("error_message") or "").lower()
    return any(
        marker in message
        for marker in ["insufficient budget", "rate limit", "too many requests"]
    )
