"""Optional seed-paper citation snowballing."""

from __future__ import annotations

import csv
import inspect
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from lit_screening.dedup import normalize_doi, normalize_title
from lit_screening.models import Paper, RankedPaper, RetrievalPath, SeedPaper
from lit_screening.retrieval.arxiv_client import ArxivClient
from lit_screening.retrieval.base import RetrievalResult
from lit_screening.retrieval.crossref_client import CrossrefClient
from lit_screening.utils import stable_id


SEED_FIELDS = ["seed_id", "seed_type", "title", "doi", "note"]


class CitationSnowballAgent:
    """Expand candidate papers through references, citations, and recommendations."""

    def __init__(
        self,
        semantic_scholar_client: Any | None = None,
        enabled: bool = False,
        require_api_key: bool = True,
    ) -> None:
        self.semantic_scholar_client = semantic_scholar_client
        self.enabled = enabled
        self.require_api_key = require_api_key

    def expand(
        self,
        existing_papers: list[Paper],
        ranked_papers: list[RankedPaper],
        seed_papers: list[SeedPaper] | None = None,
        top_n: int = 3,
    ) -> tuple[list[Paper], list[RetrievalPath], list[SeedPaper]]:
        """Return expanded papers, retrieval paths, and resolved seed list."""

        seeds = seed_papers or auto_select_seed_papers(ranked_papers, top_n)
        if not self.enabled or not seeds:
            return [], [], seeds
        client = self.semantic_scholar_client
        if client is None:
            return [], [], seeds
        if self.require_api_key and not getattr(client, "api_key", ""):
            return [], [], seeds

        seen_keys = paper_keys(existing_papers)
        expanded: list[Paper] = []
        paths: list[RetrievalPath] = []
        self.last_expansion_report = empty_seed_expansion_report(
            seed_input_count=len(seeds),
            enabled=True,
        )
        for seed in seeds[:top_n]:
            resolved = resolve_seed(seed, client)
            if not resolved:
                continue
            seed_paper = resolved[0]
            seed_identifier = semantic_identifier(seed_paper) or seed_identifier_for_client(seed)
            if not seed_identifier:
                continue
            for source_stage, getter_name in [
                ("seed_reference", "get_references"),
                ("seed_citation", "get_citations"),
                ("seed_recommendation", "get_recommendations"),
            ]:
                getter = getattr(client, getter_name, None)
                if getter is None:
                    continue
                result = getter(seed_identifier, limit=top_n)
                expansion_count_key = {
                    "seed_reference": "references_retrieved",
                    "seed_citation": "citations_retrieved",
                    "seed_recommendation": "recommendations_retrieved",
                }[source_stage]
                self.last_expansion_report[expansion_count_key] += len(result.papers)
                raw = result.raw if isinstance(result, RetrievalResult) else {}
                if isinstance(raw, dict) and raw.get("error"):
                    self.last_expansion_report["provider_errors"].append(
                        provider_error_record("semantic_scholar", seed, raw)
                    )
                for paper in result.papers[:top_n]:
                    key = best_paper_key(paper)
                    if key and key in seen_keys:
                        continue
                    if key:
                        seen_keys.add(key)
                    reason = (
                        f"Found via {source_stage} expansion from seed "
                        f"'{seed_paper.title or seed.title or seed.seed_id}'."
                    )
                    enriched = replace(
                        paper,
                        retrieval_provider="semantic_scholar",
                        retrieval_stage=f"snowball_{source_stage}",
                        retrieval_query=seed.seed_id or seed.title or seed.doi,
                        source_stage=source_stage,
                        seed_paper_id=seed_paper.paper_id,
                        seed_title=seed_paper.title or seed.title,
                        seed_reason=reason,
                        seed_relation=source_stage.replace("seed_", ""),
                        seed_confidence=0.8,
                    )
                    expanded.append(enriched)
                    paths.append(
                        RetrievalPath(
                            paper_id=enriched.paper_id,
                            source_stage=source_stage,
                            seed_paper_id=seed_paper.paper_id,
                            seed_title=seed_paper.title or seed.title,
                            reason=reason,
                            seed_relation=enriched.seed_relation,
                            seed_confidence=enriched.seed_confidence,
                        )
                    )
        self.last_expansion_report["expanded_paper_count"] = len(expanded)
        self.last_expansion_report["retrieval_path_count"] = len(paths)
        return expanded, paths, seeds


def parse_seed_file(path: str | Path | None) -> list[SeedPaper]:
    """Read seed papers from a CSV file."""

    if not path:
        return []
    seed_path = Path(path)
    if not seed_path.exists():
        return []
    seeds: list[SeedPaper] = []
    with seed_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seed_id = (row.get("seed_id") or row.get("doi") or row.get("title") or "").strip()
            title = (row.get("title") or "").strip()
            doi = normalize_doi(row.get("doi") or "")
            if not doi and looks_like_doi(seed_id):
                doi = normalize_doi(seed_id)
            seed_type = (row.get("seed_type") or infer_seed_type(seed_id, doi, title)).strip().lower()
            if not seed_id and not title and not doi:
                continue
            seeds.append(
                SeedPaper(
                    seed_id=seed_id or doi or title,
                    seed_type=seed_type,
                    title=title,
                    doi=doi,
                    note=row.get("note") or "",
                )
            )
    return seeds


def parse_seed_values(values: list[str] | None) -> list[SeedPaper]:
    """Parse CLI seed-paper values."""

    seeds: list[SeedPaper] = []
    for value in values or []:
        cleaned = " ".join(str(value).split())
        if not cleaned:
            continue
        doi = normalize_doi(cleaned) if looks_like_doi(cleaned) else ""
        seeds.append(
            SeedPaper(
                seed_id=doi or cleaned,
                seed_type=infer_seed_type(cleaned, doi, ""),
                title="" if doi else cleaned,
                doi=doi,
                note="CLI seed paper",
            )
        )
    return seeds


def resolve_seed_inputs(
    seeds: list[SeedPaper],
    clients: dict[str, Any] | None = None,
    max_results: int = 3,
) -> tuple[list[Paper], list[SeedPaper], dict[str, Any]]:
    """Resolve user seeds through multiple metadata sources with manual fallback."""

    clients = seed_resolution_clients(clients or {})
    report = empty_seed_resolution_report(len(seeds))
    resolved_papers: list[Paper] = []
    resolved_seeds: list[SeedPaper] = []

    for seed in seeds:
        record: dict[str, Any] = {
            "seed_id": seed.seed_id,
            "seed_type": seed.seed_type,
            "input_title": seed.title,
            "input_doi": seed.doi,
            "status": "unresolved",
            "provider": "",
            "paper_id": "",
            "resolved_title": "",
            "resolved_doi": "",
        }
        paper: Paper | None = None
        provider = ""
        for provider_name in ["semantic_scholar", "openalex", "crossref", "arxiv"]:
            client = clients.get(provider_name)
            if client is None:
                continue
            try:
                result = resolve_seed_with_provider(
                    seed,
                    provider_name,
                    client,
                    max_results=max_results,
                )
            except Exception as exc:
                report["provider_errors"].append(
                    {
                        "provider": provider_name,
                        "seed_id": seed.seed_id,
                        "seed_title": seed.title,
                        "error": exc.__class__.__name__,
                        "error_message": str(exc)[:240],
                    }
                )
                continue
            raw = result.raw if isinstance(result, RetrievalResult) else {}
            if isinstance(raw, dict) and raw.get("error"):
                report["provider_errors"].append(provider_error_record(provider_name, seed, raw))
            paper = select_seed_candidate(seed, result.papers)
            if paper is not None:
                provider = provider_name
                break

        if paper is None:
            paper = manual_seed_paper(seed)
            record["status"] = "manual_seed"
            record["provider"] = "manual_seed"
            report["seed_unresolved_count"] += 1
        else:
            paper = mark_seed_exact_paper(paper, seed, provider)
            record["status"] = "resolved"
            record["provider"] = provider
            report["seed_resolved_count"] += 1

        record["paper_id"] = paper.paper_id
        record["resolved_title"] = paper.title
        record["resolved_doi"] = paper.doi
        record["source_stage"] = paper.source_stage
        record["seed_reason"] = paper.seed_reason
        record["seed_confidence"] = paper.seed_confidence
        report["records"].append(record)
        resolved_papers.append(paper)
        resolved_seeds.append(
            SeedPaper(
                seed_id=seed.seed_id or paper.doi or paper.title,
                seed_type=seed.seed_type,
                title=seed.title or paper.title,
                doi=seed.doi or paper.doi,
                note=seed.note,
            )
        )

    return resolved_papers, resolved_seeds, report


def seed_resolution_clients(clients: dict[str, Any]) -> dict[str, Any]:
    """Return clients used for seed resolution without changing main providers."""

    resolved = dict(clients)
    if "semantic_scholar" in resolved or "openalex" in resolved:
        resolved.setdefault("crossref", CrossrefClient())
        resolved.setdefault("arxiv", ArxivClient())
    return resolved


def resolve_seed_with_provider(
    seed: SeedPaper,
    provider_name: str,
    client: Any,
    max_results: int = 3,
) -> RetrievalResult:
    """Resolve one seed against one provider with DOI priority."""

    doi = normalize_doi(seed.doi)
    if doi:
        doi_result = resolve_seed_doi(seed, provider_name, client, doi, max_results)
        if doi_result.papers:
            return doi_result
    title_query = seed.title or (seed.seed_id if seed.seed_type == "title" else "")
    if not title_query:
        return RetrievalResult(raw={"provider": provider_name, "data": []}, papers=[])
    return resolve_seed_title(provider_name, client, title_query, max_results)


def resolve_seed_doi(
    seed: SeedPaper,
    provider_name: str,
    client: Any,
    doi: str,
    max_results: int,
) -> RetrievalResult:
    """Resolve a DOI through provider-specific lookup where available."""

    if provider_name == "semantic_scholar" and hasattr(client, "get_paper"):
        return coerce_retrieval_result(client.get_paper(f"DOI:{doi}"))
    if provider_name == "crossref" and hasattr(client, "resolve_doi"):
        return coerce_retrieval_result(client.resolve_doi(doi))
    if hasattr(client, "search"):
        return call_search(client, doi, max_results=max_results)
    if hasattr(client, "works"):
        return coerce_retrieval_result(client.works(doi, rows=max_results))
    return RetrievalResult(raw={"provider": provider_name, "data": []}, papers=[])


def resolve_seed_title(
    provider_name: str,
    client: Any,
    title: str,
    max_results: int,
) -> RetrievalResult:
    """Resolve an exact title through provider-specific search."""

    if provider_name == "crossref" and hasattr(client, "works"):
        return coerce_retrieval_result(client.works(title, rows=max_results))
    if hasattr(client, "search"):
        return call_search(client, title, max_results=max_results)
    return RetrievalResult(raw={"provider": provider_name, "data": []}, papers=[])


def call_search(client: Any, query: str, max_results: int) -> RetrievalResult:
    """Call heterogeneous search clients while preserving fake-client support."""

    search = getattr(client, "search")
    parameters = inspect.signature(search).parameters
    kwargs: dict[str, Any] = {}
    if "max_results" in parameters:
        kwargs["max_results"] = max_results
    elif "rows" in parameters:
        kwargs["rows"] = max_results
    if "search_mode" in parameters:
        kwargs["search_mode"] = "exact"
    if "sort_mode" in parameters:
        kwargs["sort_mode"] = "relevance"
    try:
        if kwargs:
            return coerce_retrieval_result(search(query, **kwargs))
        return coerce_retrieval_result(search(query, max_results))
    except TypeError:
        return coerce_retrieval_result(search(query, max_results))


def coerce_retrieval_result(value: Any) -> RetrievalResult:
    """Coerce provider return values into RetrievalResult."""

    if isinstance(value, RetrievalResult):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return RetrievalResult(raw=value[0], papers=value[1])
    return RetrievalResult(raw={"data": []}, papers=[])


def select_seed_candidate(seed: SeedPaper, papers: list[Paper]) -> Paper | None:
    """Return the first DOI or exact-title candidate for a seed."""

    seed_doi = normalize_doi(seed.doi)
    seed_title = normalize_title(seed.title or (seed.seed_id if seed.seed_type == "title" else ""))
    for paper in papers:
        if seed_doi and normalize_doi(paper.doi) == seed_doi:
            return paper
        if seed_title and normalize_title(paper.title) == seed_title:
            return paper
    if seed_doi and papers:
        return papers[0]
    return None


def mark_seed_exact_paper(paper: Paper, seed: SeedPaper, provider_name: str) -> Paper:
    """Attach exact seed provenance to a resolved paper."""

    seed_title = seed.title or paper.title or seed.seed_id
    provider = paper.source_provider or provider_name
    reason = f"User-provided seed exact match resolved via {provider_name}."
    return replace(
        paper,
        source_provider=provider,
        retrieval_provider=paper.retrieval_provider or provider_name,
        retrieval_stage="seed_exact",
        retrieval_query=seed.doi or seed.title or seed.seed_id,
        source_stage="seed_exact",
        seed_paper_id=paper.paper_id or stable_id(seed.doi or seed_title),
        seed_title=seed_title,
        seed_reason=reason,
        seed_relation="self",
        seed_confidence=1.0,
    )


def manual_seed_paper(seed: SeedPaper) -> Paper:
    """Create a retained manual seed record when providers cannot resolve it."""

    title = seed.title or seed.seed_id or seed.doi
    paper_id = stable_id(f"manual-seed:{seed.doi or title}")
    reason = "User-provided seed retained because external providers did not resolve it."
    return Paper(
        paper_id=paper_id,
        title=title,
        doi=seed.doi,
        source_provider="manual_seed",
        retrieval_provider="manual_seed",
        retrieval_stage="manual_seed",
        retrieval_query=seed.doi or seed.title or seed.seed_id,
        source_stage="manual_seed",
        seed_paper_id=paper_id,
        seed_title=title,
        seed_reason=reason,
        seed_relation="self",
        seed_confidence=0.75,
        raw={"source": "manual_seed", "seed": seed},
    )


def empty_seed_resolution_report(seed_input_count: int = 0) -> dict[str, Any]:
    """Return the seed-resolution diagnostics payload."""

    return {
        "seed_input_count": seed_input_count,
        "seed_resolved_count": 0,
        "seed_unresolved_count": 0,
        "records": [],
        "provider_errors": [],
    }


def empty_seed_expansion_report(
    seed_input_count: int = 0,
    enabled: bool = False,
) -> dict[str, Any]:
    """Return the seed-expansion diagnostics payload."""

    return {
        "enabled": enabled,
        "seed_input_count": seed_input_count,
        "seed_resolved_count": 0,
        "seed_unresolved_count": 0,
        "references_retrieved": 0,
        "citations_retrieved": 0,
        "recommendations_retrieved": 0,
        "expanded_paper_count": 0,
        "retrieval_path_count": 0,
        "provider_errors": [],
    }


def provider_error_record(provider_name: str, seed: SeedPaper, raw: dict[str, Any]) -> dict[str, Any]:
    """Return a compact provider-error record without secrets."""

    return {
        "provider": provider_name,
        "seed_id": seed.seed_id,
        "seed_title": seed.title,
        "error": raw.get("error", ""),
        "status_code": raw.get("status_code"),
        "error_message": str(raw.get("error_message") or "")[:240],
    }


def resolve_seed(seed: SeedPaper, client: Any) -> list[Paper]:
    """Resolve a seed paper through Semantic Scholar."""

    identifier = seed_identifier_for_client(seed)
    if identifier and seed.seed_type != "title" and hasattr(client, "get_paper"):
        result = client.get_paper(identifier)
        if result.papers:
            return result.papers
    query = seed.title or seed.doi or seed.seed_id
    if query and hasattr(client, "search"):
        return client.search(query, max_results=1).papers
    return []


def seed_identifier_for_client(seed: SeedPaper) -> str:
    """Return a Semantic Scholar-compatible seed identifier."""

    seed_type = seed.seed_type.lower()
    if seed_type == "doi" or seed.doi:
        return f"DOI:{seed.doi or normalize_doi(seed.seed_id)}"
    if seed_type in {"semantic_scholar", "semantic_scholar_id", "s2", "paper_id"}:
        return seed.seed_id
    if seed_type == "openalex":
        return ""
    return ""


def semantic_identifier(paper: Paper) -> str:
    """Return a Semantic Scholar identifier from a normalized paper."""

    return paper.provider_ids.get("semantic_scholar") or paper.raw.get("paperId", "")


def auto_select_seed_papers(
    ranked_papers: list[RankedPaper],
    top_n: int,
) -> list[SeedPaper]:
    """Choose high-confidence ranked papers as seeds."""

    seeds: list[SeedPaper] = []
    for item in ranked_papers:
        if item.verification.support_level not in {"strict_support", "weak_support"}:
            continue
        if item.domain_assessment and item.domain_assessment.domain_decision == "out_of_scope":
            continue
        seed_id = semantic_identifier(item.paper) or item.paper.doi or item.paper.title
        seed_type = "semantic_scholar" if semantic_identifier(item.paper) else "doi" if item.paper.doi else "title"
        seeds.append(
            SeedPaper(
                seed_id=seed_id,
                seed_type=seed_type,
                title=item.paper.title,
                doi=item.paper.doi,
                note="Auto-selected from top high-confidence ranked papers.",
            )
        )
        if len(seeds) >= top_n:
            break
    return seeds


def infer_seed_type(seed_id: str, doi: str = "", title: str = "") -> str:
    """Infer seed type from a CLI or CSV row."""

    value = (seed_id or doi or title or "").strip()
    lowered = value.lower()
    if doi or looks_like_doi(value):
        return "doi"
    if "openalex.org" in lowered or re.fullmatch(r"w\d+", lowered):
        return "openalex"
    if re.fullmatch(r"[a-f0-9]{40}", lowered):
        return "semantic_scholar"
    if lowered.startswith("s2:"):
        return "semantic_scholar"
    return "title"


def looks_like_doi(value: str) -> bool:
    """Return True if a value resembles a DOI."""

    cleaned = normalize_doi(value)
    return bool(re.match(r"^10\.\d{4,9}/\S+$", cleaned))


def paper_keys(papers: list[Paper]) -> set[str]:
    """Return DOI/title keys for deduplication."""

    return {key for paper in papers for key in [best_paper_key(paper)] if key}


def best_paper_key(paper: Paper) -> str:
    """Return the strongest available dedup key for a paper."""

    doi = normalize_doi(paper.doi)
    if doi:
        return f"doi:{doi}"
    title = normalize_title(paper.title)
    return f"title:{title}" if title else ""
