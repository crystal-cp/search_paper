"""Small utility helpers shared by the MVP pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value to a closed interval."""

    return max(low, min(high, value))


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int, returning a default on failure."""

    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def current_year() -> int:
    """Return the current local year."""

    return datetime.now().year


def stable_id(*parts: str) -> str:
    """Build a stable short identifier from text parts."""

    text = "||".join(part.strip().lower() for part in parts if part)
    if not text:
        text = "missing"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase keyword-like tokens."""

    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def overlap_terms(text: str, question: str) -> list[str]:
    """Return sorted shared keyword tokens between text and a question."""

    return sorted(set(tokenize(text)) & set(tokenize(question)))


def keyword_overlap_score(text: str, question: str) -> float:
    """Score keyword overlap between a candidate text and research question."""

    question_terms = set(tokenize(question))
    if not question_terms:
        return 0.0
    shared = set(tokenize(text)) & question_terms
    return clamp(len(shared) / len(question_terms))


def split_sentences(text: str) -> list[str]:
    """Split an abstract into simple sentence-like chunks."""

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def to_plain_data(value: Any) -> Any:
    """Convert dataclasses and Paths into JSON-serializable values."""

    if is_dataclass(value):
        return to_plain_data(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    """Write pretty JSON to disk."""

    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(
        json.dumps(to_plain_data(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file with stable columns."""

    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
