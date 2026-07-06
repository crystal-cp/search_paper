"""CSV importer for user-curated literature libraries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .base import ImportResult, clean_metadata_value, paper_from_metadata


FIELD_ALIASES = {
    "title": ["title", "paper_title", "article_title"],
    "abstract": ["abstract", "summary", "description"],
    "author": ["authors", "author", "creators"],
    "year": ["year", "publication_year", "pub_year"],
    "date": ["date", "publication_date"],
    "journal": ["journal", "journal_name", "source_title", "publication"],
    "venue": ["venue", "booktitle", "conference"],
    "doi": ["doi", "DOI"],
    "url": ["url", "link"],
    "citation_count": ["citation_count", "citations", "cited_by_count"],
    "keywords": ["keywords", "tags", "fields_of_study"],
    "publication_type": ["publication_type", "type", "item_type"],
}


def _value(row: dict[str, Any], aliases: list[str]) -> str:
    """Return the first non-empty CSV value matching any alias."""

    lower_map = {key.lower(): key for key in row}
    for alias in aliases:
        key = lower_map.get(alias.lower())
        if key is not None:
            cleaned = clean_metadata_value(row.get(key))
            if cleaned:
                return cleaned
    return ""


def _metadata_from_row(row: dict[str, Any]) -> dict[str, str]:
    """Normalize a CSV row into common metadata fields."""

    return {
        field: _value(row, aliases)
        for field, aliases in FIELD_ALIASES.items()
    }


def import_csv(path: str | Path) -> ImportResult:
    """Import a CSV file into Paper records."""

    source = Path(path)
    papers = []
    skipped = 0
    errors: list[str] = []
    with source.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for index, row in enumerate(rows, start=1):
        metadata = _metadata_from_row(row)
        paper = paper_from_metadata(
            metadata,
            source_provider="imported_csv",
            raw={"csv_row_index": index, **row},
        )
        if paper is None:
            skipped += 1
            errors.append(f"Skipped CSV row without title at index {index}")
            continue
        papers.append(paper)
    return ImportResult(
        papers=papers,
        raw_count=len(rows),
        skipped_count=skipped,
        errors=errors,
        source_path=str(source),
        detected_format="csv",
    )
