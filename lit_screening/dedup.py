"""Metadata normalization and deduplication utilities."""

from __future__ import annotations

import re
import string
from dataclasses import replace

from .models import Paper


def normalize_doi(doi: str | None) -> str:
    """Normalize DOI strings for matching."""

    if not doi:
        return ""
    value = doi.strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    value = value.strip().strip(".")
    return value


def normalize_title(title: str | None) -> str:
    """Normalize titles by lowercasing and removing punctuation."""

    if not title:
        return ""
    translator = str.maketrans("", "", string.punctuation)
    no_punctuation = title.lower().translate(translator)
    return " ".join(no_punctuation.split())


def _completeness_score(paper: Paper) -> float:
    score = 0.0
    score += 2.0 if paper.title else 0.0
    score += min(len(paper.abstract) / 500.0, 3.0)
    score += min(len(paper.authors), 5) * 0.2
    score += 1.0 if paper.doi else 0.0
    score += 0.5 if paper.url else 0.0
    score += 0.5 if paper.venue else 0.0
    score += 0.5 if paper.year else 0.0
    score += min(paper.citation_count / 100.0, 1.0)
    return score


def merge_papers(left: Paper, right: Paper) -> Paper:
    """Merge duplicate paper records, keeping the more complete base record."""

    base, other = (left, right)
    if _completeness_score(right) > _completeness_score(left):
        base, other = right, left

    authors = list(dict.fromkeys([*base.authors, *other.authors]))
    provider_ids = {**other.provider_ids, **base.provider_ids}
    providers = sorted(
        {
            provider
            for provider in [base.source_provider, other.source_provider]
            if provider
            for provider in provider.split(";")
        }
    )
    publication_types = list(
        dict.fromkeys([*base.publication_types, *other.publication_types])
    )
    fields_of_study = list(dict.fromkeys([*base.fields_of_study, *other.fields_of_study]))
    semantic_ranks = [
        rank
        for rank in [base.semantic_scholar_rank, other.semantic_scholar_rank]
        if rank
    ]
    return replace(
        base,
        abstract=base.abstract or other.abstract,
        authors=authors,
        year=base.year or other.year,
        venue=base.venue or other.venue,
        doi=base.doi or other.doi,
        url=base.url or other.url,
        source_provider=";".join(providers),
        provider_ids=provider_ids,
        citation_count=max(base.citation_count, other.citation_count),
        api_relevance_score=max(base.api_relevance_score, other.api_relevance_score),
        openalex_relevance_score=max(
            base.openalex_relevance_score,
            other.openalex_relevance_score,
        ),
        semantic_scholar_rank=min(semantic_ranks) if semantic_ranks else 0,
        publication_date=base.publication_date or other.publication_date,
        publication_types=publication_types,
        fields_of_study=fields_of_study,
        influential_citation_count=max(
            base.influential_citation_count,
            other.influential_citation_count,
        ),
        reference_count=max(base.reference_count, other.reference_count),
        open_access_pdf_url=base.open_access_pdf_url or other.open_access_pdf_url,
        tldr=base.tldr or other.tldr,
    )


def deduplicate_with_stats(papers: list[Paper]) -> tuple[list[Paper], int]:
    """Deduplicate papers by DOI first, then normalized title."""

    deduped: list[Paper] = []
    doi_index: dict[str, int] = {}
    title_index: dict[str, int] = {}
    duplicate_count = 0

    for paper in papers:
        doi_key = normalize_doi(paper.doi)
        title_key = normalize_title(paper.title)
        match_index: int | None = None

        if doi_key and doi_key in doi_index:
            match_index = doi_index[doi_key]
        elif title_key and title_key in title_index:
            match_index = title_index[title_key]

        if match_index is None:
            deduped.append(paper)
            new_index = len(deduped) - 1
            if doi_key:
                doi_index[doi_key] = new_index
            if title_key:
                title_index[title_key] = new_index
            continue

        duplicate_count += 1
        merged = merge_papers(deduped[match_index], paper)
        deduped[match_index] = merged

        merged_doi_key = normalize_doi(merged.doi)
        merged_title_key = normalize_title(merged.title)
        if merged_doi_key:
            doi_index[merged_doi_key] = match_index
        if merged_title_key:
            title_index[merged_title_key] = match_index

    return deduped, duplicate_count


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Return deduplicated papers without stats."""

    deduped, _ = deduplicate_with_stats(papers)
    return deduped
