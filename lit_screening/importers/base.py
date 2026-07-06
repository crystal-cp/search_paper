"""Shared helpers for importing external literature-library files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lit_screening.dedup import normalize_doi, normalize_title
from lit_screening.models import Paper


SUPPORTED_IMPORT_FORMATS = {"auto", "bibtex", "bib", "ris", "csv"}


@dataclass
class ImportResult:
    """Result of importing papers from an external library file."""

    papers: list[Paper]
    raw_count: int
    skipped_count: int = 0
    errors: list[str] = field(default_factory=list)
    source_path: str = ""
    detected_format: str = ""

    def diagnostics(self) -> dict[str, Any]:
        """Return a compact JSON-serializable import diagnostics block."""

        return {
            "source_path": self.source_path,
            "detected_format": self.detected_format,
            "raw_count": self.raw_count,
            "paper_count": len(self.papers),
            "skipped_count": self.skipped_count,
            "errors": self.errors[:20],
        }


def detect_import_format(path: str | Path, requested_format: str = "auto") -> str:
    """Detect import format from an explicit value or file extension."""

    requested = (requested_format or "auto").strip().lower()
    if requested not in SUPPORTED_IMPORT_FORMATS:
        raise ValueError(f"Unsupported import format: {requested_format}")
    if requested != "auto":
        return "bibtex" if requested == "bib" else requested

    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in {"bib", "bibtex"}:
        return "bibtex"
    if suffix == "ris":
        return "ris"
    if suffix == "csv":
        return "csv"
    raise ValueError(
        "Could not auto-detect import format. Use bibtex, ris, or csv explicitly."
    )


def split_authors(value: str) -> list[str]:
    """Split common author-list formats into a clean list."""

    if not value:
        return []
    normalized = value.replace("\n", " ")
    if " and " in normalized:
        parts = re.split(r"\s+\band\b\s+", normalized)
    elif ";" in normalized:
        parts = normalized.split(";")
    else:
        parts = normalized.split("|")
    return [" ".join(part.split()) for part in parts if " ".join(part.split())]


def parse_year(value: Any) -> int | None:
    """Extract a four-digit year from a metadata value."""

    if value is None:
        return None
    match = re.search(r"(18|19|20|21)\d{2}", str(value))
    return int(match.group(0)) if match else None


def parse_int(value: Any, default: int = 0) -> int:
    """Parse an integer metadata value with a safe default."""

    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def clean_metadata_value(value: Any) -> str:
    """Normalize whitespace and strip common BibTeX/RIS wrappers."""

    if value is None:
        return ""
    text = str(value).strip()
    text = text.strip("{}").strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    return text


def imported_paper_id(title: str, doi: str = "", year: int | None = None) -> str:
    """Build a stable local paper id for imported records."""

    normalized_doi = normalize_doi(doi)
    if normalized_doi:
        key = normalized_doi
    else:
        key = f"{normalize_title(title)}::{year or ''}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"imported:{digest}"


def paper_from_metadata(
    metadata: dict[str, Any],
    source_provider: str,
    raw: dict[str, Any] | None = None,
) -> Paper | None:
    """Convert common imported metadata into a Paper."""

    title = clean_metadata_value(metadata.get("title"))
    if not title:
        return None
    doi = normalize_doi(clean_metadata_value(metadata.get("doi")))
    year = parse_year(metadata.get("year") or metadata.get("date"))
    authors_value = metadata.get("authors") or metadata.get("author") or ""
    authors = (
        list(authors_value)
        if isinstance(authors_value, list)
        else split_authors(clean_metadata_value(authors_value))
    )
    venue = clean_metadata_value(
        metadata.get("venue")
        or metadata.get("journal")
        or metadata.get("booktitle")
        or metadata.get("publication")
    )
    return Paper(
        paper_id=imported_paper_id(title, doi, year),
        title=title,
        abstract=clean_metadata_value(metadata.get("abstract")),
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        url=clean_metadata_value(metadata.get("url")),
        source_provider=source_provider,
        citation_count=parse_int(metadata.get("citation_count"), 0),
        publication_date=clean_metadata_value(metadata.get("publication_date")),
        publication_types=[
            item
            for item in [
                clean_metadata_value(metadata.get("publication_type")),
                clean_metadata_value(metadata.get("entry_type")),
            ]
            if item
        ],
        fields_of_study=[
            item.strip()
            for item in clean_metadata_value(
                metadata.get("fields_of_study") or metadata.get("keywords")
            ).split(";")
            if item.strip()
        ],
        raw=raw or dict(metadata),
    )


def import_papers_from_file(
    path: str | Path,
    input_format: str = "auto",
) -> ImportResult:
    """Import papers from a BibTeX, RIS, or CSV file."""

    detected = detect_import_format(path, input_format)
    if detected == "bibtex":
        from .bibtex import import_bibtex

        return import_bibtex(path)
    if detected == "ris":
        from .ris import import_ris

        return import_ris(path)
    if detected == "csv":
        from .csv_importer import import_csv

        return import_csv(path)
    raise ValueError(f"Unsupported import format: {detected}")
