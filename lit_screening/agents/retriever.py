"""Retriever agent that coordinates provider clients."""

from __future__ import annotations

import inspect
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
                min_interval_seconds=self.config.semantic_scholar_min_interval_seconds,
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
        openalex_mode: str = "keyword",
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
            search_stages = retrieval_stages_for_provider(provider, openalex_mode)
            for index, query in enumerate(provider_queries, start=1):
                stop_provider = False
                for search_mode, retrieval_stage in search_stages:
                    if progress_callback:
                        progress_callback(
                            "retrieval",
                            f"Searching {provider} ({index}/{len(provider_queries)})",
                            {
                                "provider": provider,
                                "query": query,
                                "current_query": index,
                                "total_queries": len(provider_queries),
                                "search_mode": search_mode,
                                "retrieval_stage": retrieval_stage,
                            },
                        )
                    try:
                        if provider == "openalex":
                            result = call_client_search(
                                client,
                                query,
                                max_per_query,
                                from_year,
                                sort_mode=sort_mode,
                                search_mode=search_mode,
                            )
                        else:
                            result = client.search(query, max_per_query, from_year)
                    except Exception as exc:
                        error_raw = {
                            "error": exc.__class__.__name__,
                            "error_message": str(exc)[:500],
                            "provider": provider,
                            "stage": "client_search",
                            "search_mode": search_mode,
                            "retrieval_stage": retrieval_stage,
                            "retrieval_query": query,
                        }
                        raw_by_provider[provider].append(
                            {
                                "query": query,
                                "search_mode": search_mode,
                                "retrieval_stage": retrieval_stage,
                                "paper_count": 0,
                                "missing_abstract_count": 0,
                                "response": error_raw,
                            }
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
                                    "search_mode": search_mode,
                                    "retrieval_stage": retrieval_stage,
                                    "error": exc.__class__.__name__,
                                    "message": str(exc)[:240],
                                },
                            )
                        raise
                    if not isinstance(result, RetrievalResult):
                        result = RetrievalResult(raw=result[0], papers=result[1])
                    for paper in result.papers:
                        if not paper.retrieval_provider:
                            paper.retrieval_provider = provider
                        if not paper.retrieval_stage:
                            paper.retrieval_stage = retrieval_stage
                        if not paper.retrieval_query:
                            paper.retrieval_query = query
                        if not paper.source_stage:
                            paper.source_stage = (
                                "semantic"
                                if search_mode == "semantic"
                                else "keyword"
                            )
                    missing_abstract_count = sum(1 for paper in result.papers if not paper.abstract)
                    raw_by_provider[provider].append(
                        {
                            "query": query,
                            "search_mode": search_mode,
                            "retrieval_stage": retrieval_stage,
                            "paper_count": len(result.papers),
                            "paper_ids": [paper.paper_id for paper in result.papers],
                            "missing_abstract_count": missing_abstract_count,
                            "response": {
                                **result.raw,
                                "search_mode": result.raw.get("search_mode") or search_mode,
                                "retrieval_stage": result.raw.get("retrieval_stage")
                                or retrieval_stage,
                                "retrieval_query": result.raw.get("retrieval_query")
                                or query,
                            },
                        }
                    )
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
                                "search_mode": search_mode,
                                "retrieval_stage": retrieval_stage,
                                "returned": len(result.papers),
                                "missing_abstract_count": missing_abstract_count,
                                "provider_total": retrieval_counts[provider],
                            },
                        )
                    if should_stop_provider_after_error(result.raw):
                        stop_provider = True
                        if progress_callback:
                            progress_callback(
                                "retrieval",
                                f"Stopping {provider} after provider rate-limit or budget error",
                                {
                                    "provider": provider,
                                    "error": result.raw.get("error", ""),
                                    "status_code": result.raw.get("status_code", ""),
                                    "message": result.raw.get("error_message", "")[:180],
                                    "search_mode": search_mode,
                                    "retrieval_stage": retrieval_stage,
                                },
                            )
                        break
                if stop_provider:
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


def retrieval_stages_for_provider(
    provider: str,
    openalex_mode: str,
) -> list[tuple[str, str]]:
    """Return retrieval stages for a provider and OpenAlex mode."""

    if provider != "openalex":
        return [("keyword", provider)]
    mode = (openalex_mode or "keyword").strip().lower()
    if mode == "keyword+semantic":
        return [("keyword", "openalex_keyword"), ("semantic", "openalex_semantic")]
    if mode in {"keyword", "exact", "semantic"}:
        return [(mode, f"openalex_{mode}")]
    return [("keyword", "openalex_keyword")]


def call_client_search(
    client: RetrievalClient,
    query: str,
    max_results: int,
    from_year: int | None,
    sort_mode: str,
    search_mode: str,
) -> RetrievalResult:
    """Call a retrieval client while preserving compatibility with simple fakes."""

    search = client.search
    signature = inspect.signature(search)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "sort_mode" in signature.parameters:
        kwargs["sort_mode"] = sort_mode
    if accepts_kwargs or "search_mode" in signature.parameters:
        kwargs["search_mode"] = search_mode
    return search(query, max_results, from_year, **kwargs)
