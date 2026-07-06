"""RIS importer for Zotero, Web of Science, Scopus, and similar exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ImportResult, clean_metadata_value, paper_from_metadata, parse_year


def _parse_ris_records(text: str) -> list[dict[str, list[str]]]:
    """Parse RIS records into tag -> values dictionaries."""

    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    last_tag = ""
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if len(raw_line) >= 6 and raw_line[2:6] == "  - ":
            tag = raw_line[:2].strip().upper()
            value = raw_line[6:].strip()
            if tag == "TY" and current:
                records.append(current)
                current = {}
            current.setdefault(tag, []).append(value)
            last_tag = tag
            if tag == "ER":
                records.append(current)
                current = {}
                last_tag = ""
        elif last_tag and current:
            current[last_tag][-1] = f"{current[last_tag][-1]} {raw_line.strip()}"
    if current:
        records.append(current)
    return records


def _first(record: dict[str, list[str]], tags: list[str]) -> str:
    """Return the first non-empty RIS value for any tag."""

    for tag in tags:
        for value in record.get(tag, []):
            cleaned = clean_metadata_value(value)
            if cleaned:
                return cleaned
    return ""


def _metadata_from_record(record: dict[str, list[str]]) -> dict[str, Any]:
    """Map RIS tags into common metadata fields."""

    date_value = _first(record, ["PY", "Y1", "DA"])
    return {
        "title": _first(record, ["TI", "T1", "CT", "BT"]),
        "abstract": _first(record, ["AB", "N2"]),
        "authors": [clean_metadata_value(value) for value in record.get("AU", []) if value],
        "year": parse_year(date_value),
        "publication_date": date_value,
        "journal": _first(record, ["JO", "JF", "JA", "T2"]),
        "doi": _first(record, ["DO"]),
        "url": _first(record, ["UR", "L1", "L2"]),
        "keywords": "; ".join(record.get("KW", [])),
        "publication_type": _first(record, ["TY"]),
    }


def import_ris(path: str | Path) -> ImportResult:
    """Import a RIS file into Paper records."""

    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    records = _parse_ris_records(text)
    papers = []
    skipped = 0
    errors: list[str] = []
    for index, record in enumerate(records, start=1):
        metadata = _metadata_from_record(record)
        paper = paper_from_metadata(
            metadata,
            source_provider="imported_ris",
            raw={"ris_record_index": index, **record},
        )
        if paper is None:
            skipped += 1
            errors.append(f"Skipped RIS record without title at index {index}")
            continue
        papers.append(paper)

    return ImportResult(
        papers=papers,
        raw_count=len(records),
        skipped_count=skipped,
        errors=errors,
        source_path=str(source),
        detected_format="ris",
    )
