"""Hybrid relevance features for reranking retrieved papers."""

from __future__ import annotations

import math
from collections import Counter

try:  # pragma: no cover - exercised when scikit-learn is installed.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:  # pragma: no cover - fallback is tested locally.
    TfidfVectorizer = None
    cosine_similarity = None

from lit_screening.models import EvidenceRecord, Paper, QueryPlan
from lit_screening.utils import clamp, tokenize


def _planning_question(plan: QueryPlan) -> str:
    """Return the English question-like query used for reranking."""

    return plan.translated_question or plan.original_question


def _fallback_cosine_similarity(query: str, text: str) -> float:
    """Small cosine similarity fallback when scikit-learn is unavailable."""

    query_counts = Counter(tokenize(query))
    text_counts = Counter(tokenize(text))
    if not query_counts or not text_counts:
        return 0.0
    shared = set(query_counts) & set(text_counts)
    numerator = sum(query_counts[token] * text_counts[token] for token in shared)
    query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    text_norm = math.sqrt(sum(value * value for value in text_counts.values()))
    if not query_norm or not text_norm:
        return 0.0
    return clamp(numerator / (query_norm * text_norm))


def tfidf_cosine_similarity(query: str, text: str) -> float:
    """Compute TF-IDF cosine similarity between a query and candidate text."""

    if not query or not text:
        return 0.0
    if TfidfVectorizer is None or cosine_similarity is None:
        return _fallback_cosine_similarity(query, text)
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([query, text])
    except ValueError:
        return 0.0
    return clamp(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]))


def _contains_term(text: str, term: str) -> bool:
    """Return True when all tokens in a required term appear in text."""

    text_tokens = set(tokenize(text))
    term_tokens = set(tokenize(term))
    return bool(term_tokens) and term_tokens <= text_tokens


def must_term_coverage(text: str, plan: QueryPlan) -> float:
    """Measure how many must terms are covered by a paper text."""

    groups = _required_term_groups(plan)
    if groups:
        hits = sum(
            1
            for group in groups
            if any(_contains_term(text, term) for term in group)
        )
        return clamp(hits / len(groups))
    if not plan.must_terms:
        return 1.0
    hits = sum(1 for term in plan.must_terms if _contains_term(text, term))
    return clamp(hits / len(plan.must_terms))


def _required_term_groups(plan: QueryPlan) -> list[list[str]]:
    raw_groups = plan.filters.get("must_term_groups", [])
    groups: list[list[str]] = []
    if not isinstance(raw_groups, list):
        return groups
    for group in raw_groups:
        if isinstance(group, list):
            terms = [" ".join(str(term).split()) for term in group if str(term).strip()]
            if terms:
                groups.append(terms)
    return groups


def field_match_score(paper: Paper, plan: QueryPlan) -> float:
    """Score whether provider fields of study match the query topic."""

    if not paper.fields_of_study:
        return 0.0
    field_text = " ".join(paper.fields_of_study)
    query_terms = [*plan.core_terms, *plan.must_terms, *plan.optional_terms]
    if not query_terms:
        return 0.0
    query_text = " ".join(query_terms)
    return tfidf_cosine_similarity(query_text, field_text)


def _normalized_api_relevance(score: float) -> float:
    """Normalize provider relevance scores into [0, 1]."""

    score = max(0.0, float(score or 0.0))
    if score <= 1.0:
        return score
    return clamp(score / (score + 1.0))


def compute_hybrid_relevance_features(
    paper: Paper,
    evidence: EvidenceRecord,
    plan: QueryPlan,
) -> dict[str, float]:
    """Compute reusable hybrid relevance components for one paper."""

    question = _planning_question(plan)
    evidence_text = " ".join([evidence.claim, evidence.evidence_sentence])
    combined_text = " ".join([paper.title, paper.abstract, evidence_text])
    features = {
        "title_similarity": tfidf_cosine_similarity(question, paper.title),
        "abstract_similarity": tfidf_cosine_similarity(question, paper.abstract),
        "evidence_similarity": tfidf_cosine_similarity(question, evidence_text),
        "api_relevance_score": _normalized_api_relevance(paper.api_relevance_score),
        "must_term_coverage": must_term_coverage(combined_text, plan),
        "field_match_score": field_match_score(paper, plan),
    }
    features["hybrid_relevance_score"] = hybrid_relevance_score_from_features(features)
    return features


def hybrid_relevance_score_from_features(features: dict[str, float]) -> float:
    """Combine hybrid relevance components using the project formula."""

    return clamp(
        0.30 * clamp(features.get("title_similarity", 0.0))
        + 0.25 * clamp(features.get("abstract_similarity", 0.0))
        + 0.15 * clamp(features.get("evidence_similarity", 0.0))
        + 0.10 * clamp(features.get("api_relevance_score", 0.0))
        + 0.10 * clamp(features.get("must_term_coverage", 0.0))
        + 0.10 * clamp(features.get("field_match_score", 0.0))
    )


def hybrid_relevance_score(
    paper: Paper,
    evidence: EvidenceRecord,
    plan: QueryPlan,
) -> float:
    """Return the final hybrid relevance score for one paper."""

    return compute_hybrid_relevance_features(paper, evidence, plan)[
        "hybrid_relevance_score"
    ]
