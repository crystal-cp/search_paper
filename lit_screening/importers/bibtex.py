"""Small BibTeX importer for literature-library exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import ImportResult, clean_metadata_value, paper_from_metadata


def _find_bibtex_entries(text: str) -> list[dict[str, Any]]:
    """Extract BibTeX entries with a permissive brace-depth parser."""

    entries: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("@", index)
        if start == -1:
            break
        header_match = re.match(r"@(?P<type>[A-Za-z]+)\s*[{(]", text[start:])
        if not header_match:
            index = start + 1
            continue
        entry_type = header_match.group("type").lower()
        body_start = start + header_match.end()
        opener = text[start + header_match.end() - 1]
        closer = "}" if opener == "{" else ")"
        depth = 1
        cursor = body_start
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
            cursor += 1
        if depth != 0:
            break
        body = text[body_start : cursor - 1]
        entries.append({"entry_type": entry_type, "body": body})
        index = cursor
    return entries


def _split_citation_key(body: str) -> tuple[str, str]:
    """Return citation key and field body."""

    parts = body.split(",", 1)
    if len(parts) == 1:
        return clean_metadata_value(parts[0]), ""
    return clean_metadata_value(parts[0]), parts[1]


def _parse_value(text: str, start: int) -> tuple[str, int]:
    """Parse a BibTeX value starting after the equals sign."""

    cursor = start
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        return "", cursor

    opener = text[cursor]
    if opener in {"{", '"'}:
        closer = "}" if opener == "{" else '"'
        depth = 1 if opener == "{" else 0
        cursor += 1
        value_start = cursor
        while cursor < len(text):
            char = text[cursor]
            if opener == "{" and char == "{":
                depth += 1
            elif opener == "{" and char == "}":
                depth -= 1
                if depth == 0:
                    value = text[value_start:cursor]
                    cursor += 1
                    return clean_metadata_value(value), cursor
            elif opener == '"' and char == '"':
                value = text[value_start:cursor]
                cursor += 1
                return clean_metadata_value(value), cursor
            cursor += 1
        return clean_metadata_value(text[value_start:cursor]), cursor

    value_start = cursor
    while cursor < len(text) and text[cursor] != ",":
        cursor += 1
    return clean_metadata_value(text[value_start:cursor]), cursor


def _parse_fields(field_text: str) -> dict[str, str]:
    """Parse BibTeX fields into a dictionary."""

    fields: dict[str, str] = {}
    cursor = 0
    while cursor < len(field_text):
        while cursor < len(field_text) and field_text[cursor] in {",", " ", "\n", "\t"}:
            cursor += 1
        match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=", field_text[cursor:])
        if not match:
            cursor += 1
            continue
        key = match.group(1).lower()
        cursor += match.end()
        value, cursor = _parse_value(field_text, cursor)
        fields[key] = value
        while cursor < len(field_text) and field_text[cursor] != ",":
            cursor += 1
    return fields


def import_bibtex(path: str | Path) -> ImportResult:
    """Import a BibTeX file into Paper records."""

    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    raw_entries = _find_bibtex_entries(text)
    papers = []
    errors: list[str] = []
    skipped = 0
    for raw_entry in raw_entries:
        citation_key, fields_text = _split_citation_key(raw_entry["body"])
        fields = _parse_fields(fields_text)
        fields["entry_type"] = raw_entry["entry_type"]
        metadata = {
            "title": fields.get("title", ""),
            "abstract": fields.get("abstract", ""),
            "author": fields.get("author", ""),
            "year": fields.get("year", ""),
            "journal": fields.get("journal", ""),
            "booktitle": fields.get("booktitle", ""),
            "doi": fields.get("doi", ""),
            "url": fields.get("url", ""),
            "keywords": fields.get("keywords", ""),
            "entry_type": fields.get("entry_type", ""),
        }
        paper = paper_from_metadata(
            metadata,
            source_provider="imported_bibtex",
            raw={"citation_key": citation_key, **fields},
        )
        if paper is None:
            skipped += 1
            errors.append(f"Skipped BibTeX entry without title: {citation_key}")
            continue
        papers.append(paper)

    return ImportResult(
        papers=papers,
        raw_count=len(raw_entries),
        skipped_count=skipped,
        errors=errors,
        source_path=str(source),
        detected_format="bibtex",
    )
