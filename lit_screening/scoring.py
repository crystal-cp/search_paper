"""Transparent scoring functions for ranking screened literature."""

from __future__ import annotations

import math

from .models import (
    DomainAssessment,
    EvidenceRecord,
    Paper,
    QueryPlan,
    ScoreBreakdown,
    VerificationResult,
)
from .reranking import compute_hybrid_relevance_features
from .utils import clamp, current_year, keyword_overlap_score


DEFAULT_SCORE_WEIGHTS = {
    "relevance": 0.40,
    "evidence": 0.25,
    "recency": 0.15,
    "quality": 0.15,
    "diversity": 0.05,
}
RANKING_PROFILES = {
    "balanced": DEFAULT_SCORE_WEIGHTS,
    "relevance_first": {
        "relevance": 0.55,
        "evidence": 0.25,
        "recency": 0.08,
        "quality": 0.08,
        "diversity": 0.04,
    },
    "high_quality_review": {
        "relevance": 0.30,
        "evidence": 0.20,
        "recency": 0.10,
        "quality": 0.35,
        "diversity": 0.05,
    },
    "overview": {
        "relevance": 0.35,
        "evidence": 0.20,
        "recency": 0.10,
        "quality": 0.25,
        "diversity": 0.10,
    },
    "frontier": {
        "relevance": 0.42,
        "evidence": 0.18,
        "recency": 0.25,
        "quality": 0.10,
        "diversity": 0.05,
    },
    "implementation": {
        "relevance": 0.42,
        "evidence": 0.25,
        "recency": 0.10,
        "quality": 0.15,
        "diversity": 0.08,
    },
    "proposal": {
        "relevance": 0.40,
        "evidence": 0.20,
        "recency": 0.15,
        "quality": 0.15,
        "diversity": 0.10,
    },
    "systematic_review": {
        "relevance": 0.34,
        "evidence": 0.28,
        "recency": 0.08,
        "quality": 0.25,
        "diversity": 0.05,
    },
}


def sanitize_score_weights(
    weights: dict[str, float] | None = None,
    ranking_profile: str = "balanced",
) -> dict[str, float]:
    """Return complete non-negative scoring weights with defaults filled in."""

    merged = dict(RANKING_PROFILES.get(ranking_profile, DEFAULT_SCORE_WEIGHTS))
    if weights:
        for key in merged:
            if key in weights:
                merged[key] = max(0.0, float(weights[key]))
    return merged


def score_relevance(
    paper: Paper,
    evidence: EvidenceRecord,
    question: str,
    query_plan: QueryPlan | None = None,
) -> float:
    """Estimate topical relevance from title, abstract, and extracted evidence."""

    if query_plan is not None:
        features = compute_hybrid_relevance_features(paper, evidence, query_plan)
        paper.raw.setdefault("hybrid_reranking", features)
        return features["hybrid_relevance_score"]
    text = " ".join([paper.title, paper.abstract, evidence.claim, evidence.evidence_sentence])
    return keyword_overlap_score(text, question)


def score_evidence(
    verification: VerificationResult,
    evidence: EvidenceRecord | None = None,
    question: str = "",
) -> float:
    """Score evidence from grounding confidence and question relevance."""

    evidence_text = ""
    if evidence is not None:
        evidence_text = " ".join([evidence.claim, evidence.evidence_sentence])
    evidence_question_relevance = keyword_overlap_score(evidence_text, question)
    groundedness = clamp(verification.confidence) if verification.supported else 0.0
    return clamp(0.60 * groundedness + 0.40 * evidence_question_relevance)


def score_recency(year: int | None, now_year: int | None = None) -> float:
    """Prefer recent work while keeping older papers visible."""

    if not year:
        return 0.3
    now = now_year or current_year()
    age = max(0, now - year)
    if age <= 1:
        return 1.0
    if age <= 3:
        return 0.9
    if age <= 5:
        return 0.75
    if age <= 10:
        return 0.5
    return 0.25


def score_quality(citation_count: int, venue: str = "") -> float:
    """Estimate quality from citations and whether venue metadata exists."""

    citations = max(0, citation_count)
    citation_score = clamp(math.log10(citations + 1) / 3.0)
    venue_score = 1.0 if venue else 0.3
    return clamp(0.75 * citation_score + 0.25 * venue_score)


def compute_final_score(
    relevance_score: float,
    evidence_score: float,
    recency_score: float,
    quality_score: float,
    diversity_score: float,
    human_feedback_adjustment: float = 0.0,
    weights: dict[str, float] | None = None,
    ranking_profile: str = "balanced",
    aspect_coverage_score: float = 0.0,
    domain_penalty_multiplier: float = 1.0,
    preference_score: float | None = None,
    preference_weight: float = 0.10,
) -> ScoreBreakdown:
    """Low-level formula helper for combining already-computed score components."""

    score_weights = sanitize_score_weights(weights, ranking_profile=ranking_profile)
    relevance = clamp(relevance_score)
    evidence = clamp(evidence_score)
    recency = clamp(recency_score)
    quality = clamp(quality_score)
    diversity = clamp(diversity_score)
    preference = clamp(preference_score) if preference_score is not None else 0.0
    preference_adjustment = (
        preference_weight * (preference - 0.5)
        if preference_score is not None
        else 0.0
    )
    pre_domain_final = (
        score_weights["relevance"] * relevance
        + score_weights["evidence"] * evidence
        + score_weights["recency"] * recency
        + score_weights["quality"] * quality
        + score_weights["diversity"] * diversity
        + human_feedback_adjustment
        + preference_adjustment
    )
    penalty = clamp(domain_penalty_multiplier, 0.0, 1.0)
    final = pre_domain_final * penalty
    return ScoreBreakdown(
        relevance_score=relevance,
        evidence_score=evidence,
        recency_score=recency,
        quality_score=quality,
        diversity_score=diversity,
        human_feedback_adjustment=human_feedback_adjustment,
        final_score=final,
        aspect_coverage_score=clamp(aspect_coverage_score),
        domain_penalty_multiplier=penalty,
        pre_domain_final_score=pre_domain_final,
        preference_score=preference,
        preference_adjustment=preference_adjustment,
    )


def compute_score_breakdown(
    paper: Paper,
    evidence: EvidenceRecord,
    verification: VerificationResult,
    question: str,
    diversity_score: float = 0.5,
    human_feedback_adjustment: float = 0.0,
    weights: dict[str, float] | None = None,
    query_plan: QueryPlan | None = None,
    ranking_profile: str = "balanced",
    aspect_coverage_score: float = 0.0,
    domain_assessment: DomainAssessment | None = None,
    preference_score: float | None = None,
) -> ScoreBreakdown:
    """Main scoring entrypoint for one paper.

    This is the implementation corresponding to the README formulas:
    hybrid relevance is computed in reranking.py, evidence score combines
    grounding confidence and evidence-question relevance, and final_score is
    produced by compute_final_score().
    """

    relevance = score_relevance(paper, evidence, question, query_plan=query_plan)
    if aspect_coverage_score:
        relevance = clamp(0.85 * relevance + 0.15 * aspect_coverage_score)
    return compute_final_score(
        relevance_score=relevance,
        evidence_score=score_evidence(verification, evidence, question),
        recency_score=score_recency(paper.year),
        quality_score=score_quality(paper.citation_count, paper.venue),
        diversity_score=diversity_score,
        human_feedback_adjustment=human_feedback_adjustment,
        weights=weights,
        ranking_profile=ranking_profile,
        aspect_coverage_score=aspect_coverage_score,
        domain_penalty_multiplier=domain_penalty_multiplier(domain_assessment),
        preference_score=preference_score,
    )


def score_paper(
    paper: Paper,
    evidence: EvidenceRecord,
    verification: VerificationResult,
    question: str,
    diversity_score: float = 0.5,
    human_feedback_adjustment: float = 0.0,
    weights: dict[str, float] | None = None,
    query_plan: QueryPlan | None = None,
    ranking_profile: str = "balanced",
    aspect_coverage_score: float = 0.0,
    domain_assessment: DomainAssessment | None = None,
    preference_score: float | None = None,
) -> ScoreBreakdown:
    """Backward-compatible alias for compute_score_breakdown().

    New ranking code should call compute_score_breakdown() so there is one
    obvious scoring entrypoint.
    """

    return compute_score_breakdown(
        paper=paper,
        evidence=evidence,
        verification=verification,
        question=question,
        diversity_score=diversity_score,
        human_feedback_adjustment=human_feedback_adjustment,
        weights=weights,
        query_plan=query_plan,
        ranking_profile=ranking_profile,
        aspect_coverage_score=aspect_coverage_score,
        domain_assessment=domain_assessment,
        preference_score=preference_score,
    )


def domain_penalty_multiplier(domain_assessment: DomainAssessment | None) -> float:
    """Return transparent hard-demotion multiplier for domain guardrails."""

    if domain_assessment is None:
        return 1.0
    if domain_assessment.domain_decision == "out_of_scope":
        return 0.3
    if domain_assessment.domain_decision == "borderline":
        return 0.7
    return 1.0
