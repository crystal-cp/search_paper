"""Validate that evidence text is actually present in the abstract."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .utils import split_sentences


@dataclass
class SpanValidationResult:
    """Evidence-to-abstract span match result."""

    is_valid: bool
    match_type: str
    confidence: float
    matched_text: str = ""


def normalize_for_span_match(text: str) -> str:
    """Normalize text for robust exact and fuzzy span matching."""

    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'")
    return normalized


def validate_evidence_span(
    evidence_sentence: str,
    abstract: str,
    fuzzy_threshold: float = 0.92,
) -> SpanValidationResult:
    """Require exact or high-confidence fuzzy evidence match in the abstract."""

    evidence = normalize_for_span_match(evidence_sentence)
    abstract_normalized = normalize_for_span_match(abstract)
    if not evidence or not abstract_normalized:
        return SpanValidationResult(False, "none", 0.0)

    if evidence in abstract_normalized:
        return SpanValidationResult(True, "exact", 1.0, evidence_sentence)

    best_sentence = ""
    best_score = 0.0
    for sentence in split_sentences(abstract):
        candidate = normalize_for_span_match(sentence)
        if not candidate:
            continue
        score = difflib.SequenceMatcher(None, evidence, candidate).ratio()
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_score >= fuzzy_threshold:
        return SpanValidationResult(True, "fuzzy", best_score, best_sentence)

    return SpanValidationResult(False, "none", best_score, best_sentence)
