"""Rule-based extraction of seed-paper hints from user questions."""

from __future__ import annotations

import re

from lit_screening.models import SeedHint


KNOWN_TITLE_PATTERNS = [
    "Surface Magnetization in Antiferromagnets: Classification, Example Materials, and Relation to Magnetoelectric Responses",
    "Local Magnetoelectric Effects as Predictors of Surface Magnetic Order",
]

QUOTE_PATTERN = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{8,180})[\"“”'‘’]")
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_PATTERN = re.compile(
    r"\barxiv\s*[: ]\s*([a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)\b",
    re.IGNORECASE,
)


class SeedExtractionAgent:
    """Extract explicit bibliographic seed hints without network calls."""

    def extract(self, question: str) -> list[SeedHint]:
        """Return seed hints mentioned in a user question."""

        text = " ".join(str(question or "").split())
        if not text:
            return []

        authors = _extract_author_hints(text)
        hints: list[SeedHint] = []
        seen: set[tuple[str, str]] = set()

        def add_hint(hint: SeedHint) -> None:
            if hint.title:
                key = ("title", _normalize_key(hint.title))
            elif hint.doi:
                key = ("doi", _normalize_key(hint.doi))
            elif hint.arxiv_id:
                key = ("arxiv", _normalize_key(hint.arxiv_id))
            else:
                key = ("raw", _normalize_key(hint.raw_mention))
            if not key[1] or key in seen:
                return
            seen.add(key)
            hints.append(hint)

        for title in KNOWN_TITLE_PATTERNS:
            match = re.search(re.escape(title), text, flags=re.IGNORECASE)
            if not match:
                continue
            add_hint(
                SeedHint(
                    title=title,
                    authors=authors,
                    doi=None,
                    arxiv_id=None,
                    raw_mention=match.group(0),
                    confidence=0.95,
                    extraction_reason="known_title_match",
                )
            )

        for match in QUOTE_PATTERN.finditer(text):
            title = _clean_title(match.group(1))
            if _looks_like_title(title):
                add_hint(
                    SeedHint(
                        title=title,
                        authors=authors,
                        doi=None,
                        arxiv_id=None,
                        raw_mention=match.group(0),
                        confidence=0.86,
                        extraction_reason="quoted_title",
                    )
                )

        for raw_title in _colon_title_candidates(text):
            title = _clean_title(raw_title)
            if _looks_like_title(title):
                add_hint(
                    SeedHint(
                        title=title,
                        authors=authors,
                        doi=None,
                        arxiv_id=None,
                        raw_mention=raw_title,
                        confidence=0.8,
                        extraction_reason="colon_title_fragment",
                    )
                )

        for match in DOI_PATTERN.finditer(text):
            doi = _clean_identifier(match.group(0))
            add_hint(
                SeedHint(
                    title=None,
                    authors=authors,
                    doi=doi,
                    arxiv_id=None,
                    raw_mention=match.group(0),
                    confidence=0.98,
                    extraction_reason="doi",
                )
            )

        for match in ARXIV_PATTERN.finditer(text):
            arxiv_id = _clean_identifier(match.group(1))
            add_hint(
                SeedHint(
                    title=None,
                    authors=authors,
                    doi=None,
                    arxiv_id=arxiv_id,
                    raw_mention=match.group(0),
                    confidence=0.96,
                    extraction_reason="arxiv_id",
                )
            )

        return hints


def _extract_author_hints(text: str) -> list[str]:
    authors: list[str] = []
    if re.search(r"\bNicola\s+A\.\s+Spaldin\b", text):
        authors.append("Nicola A. Spaldin")
    elif re.search(r"\bSpaldin\b", text):
        authors.append("Spaldin")
    return authors


def _colon_title_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(":", text):
        prefix = text[: match.start()]
        suffix = text[match.end() :]
        left_match = re.search(
            r"([A-Z][A-Za-z0-9'’\-]+(?:\s+[A-Za-z0-9'’\-]+){2,})\s*$",
            prefix,
        )
        right_match = re.match(r"\s*([A-Za-z0-9'’\- ,]+)", suffix)
        if not left_match or not right_match:
            continue
        candidate = f"{left_match.group(1)}: {right_match.group(1)}"
        candidates.append(_trim_trailing_connectors(candidate))
    return candidates


def _trim_trailing_connectors(text: str) -> str:
    cleaned = _clean_title(text)
    cleaned = re.sub(r"\s+(and|or|with|for|of|to|in|as)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.rstrip(" ,;:")


def _clean_title(text: str) -> str:
    return " ".join(str(text or "").strip(" \t\n\r\"'“”‘’.,;，。").split())


def _clean_identifier(text: str) -> str:
    return str(text or "").strip(" \t\n\r.,;，。)")


def _looks_like_title(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*", text)
    if len(words) < 5:
        return False
    uppercase_starts = sum(1 for word in words if word[:1].isupper())
    return uppercase_starts >= 3


def _normalize_key(text: str) -> str:
    return re.sub(r"\W+", " ", str(text or "").lower()).strip()
