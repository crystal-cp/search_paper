"""Optional seed-paper citation snowballing."""

from __future__ import annotations

import csv
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from lit_screening.dedup import normalize_doi, normalize_title
from lit_screening.models import Paper, RankedPaper, RetrievalPath, SeedPaper


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
        for seed in seeds[:top_n]:
            resolved = resolve_seed(seed, client)
            if not resolved:
                continue
            seed_paper = resolved[0]
            seed_identifier = semantic_identifier(seed_paper) or seed_identifier_for_client(seed)
            if not seed_identifier:
                continue
            for source_stage, getter_name in [
                ("reference", "get_references"),
                ("citation", "get_citations"),
                ("recommendation", "get_recommendations"),
            ]:
                getter = getattr(client, getter_name, None)
                if getter is None:
                    continue
                result = getter(seed_identifier, limit=top_n)
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
                    )
                    expanded.append(enriched)
                    paths.append(
                        RetrievalPath(
                            paper_id=enriched.paper_id,
                            source_stage=source_stage,
                            seed_paper_id=seed_paper.paper_id,
                            seed_title=seed_paper.title or seed.title,
                            reason=reason,
                        )
                    )
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
