"""PRISMA-like screening-flow summary for the prototype."""

from __future__ import annotations

from collections import Counter
from typing import Any

from lit_screening.models import (
    Paper,
    RankedPaper,
    ScreeningDecision,
    VerificationResult,
)


def build_prisma_like_flow(
    retrieval_counts: dict[str, int],
    raw_paper_count: int,
    merged_papers: list[Paper],
    duplicate_count: int,
    verification_results: list[VerificationResult],
    ranked_papers: list[RankedPaper],
    screening_decisions: list[ScreeningDecision] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build PRISMA-like screening counts from pipeline state."""

    missing_abstract_count = sum(1 for paper in merged_papers if not paper.abstract)
    low_confidence = [
        result
        for result in verification_results
        if result.support_level in {"missing_abstract", "missing_evidence", "unverified", "llm_invalid_evidence"}
    ]
    low_confidence_reasons = Counter(
        result.error_type or result.support_level for result in low_confidence
    )
    decisions = screening_decisions or [
        item.screening_decision
        for item in ranked_papers
        if item.screening_decision is not None
    ]
    decision_counts = Counter(record.decision for record in decisions)
    decision_reasons = Counter(
        reason
        for record in decisions
        for reason in record.exclusion_reasons
    )
    reasons = decision_reasons or low_confidence_reasons
    out_of_domain_records = sum(
        1
        for item in ranked_papers
        if item.domain_assessment
        and item.domain_assessment.domain_decision == "out_of_scope"
    )
    if decisions:
        out_of_domain_records = max(
            out_of_domain_records,
            sum(1 for record in decisions if record.domain_decision == "out_of_scope"),
        )
    return {
        "records_identified_by_openalex": retrieval_counts.get("openalex", 0),
        "records_identified_by_semantic_scholar": retrieval_counts.get("semantic_scholar", 0),
        "records_identified": raw_paper_count,
        "records_identified_total": raw_paper_count,
        "duplicate_records_removed": duplicate_count,
        "duplicates_removed": duplicate_count,
        "records_with_missing_abstracts": missing_abstract_count,
        "missing_abstracts": missing_abstract_count,
        "records_screened": len(merged_papers),
        "records_included": decision_counts.get("include", 0),
        "records_maybe": decision_counts.get("maybe", 0),
        "records_excluded": decision_counts.get("exclude", 0),
        "out_of_domain_records": out_of_domain_records,
        "records_included_in_top_ranked_results": len(ranked_papers[:top_k]),
        "records_excluded_or_low_confidence": len(low_confidence),
        "common_exclusion_reasons": dict(reasons.most_common(10)),
    }
