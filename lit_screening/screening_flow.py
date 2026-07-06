"""PRISMA-like screening-flow summary for the prototype."""

from __future__ import annotations

from collections import Counter
from typing import Any

from lit_screening.models import Paper, RankedPaper, VerificationResult


def build_prisma_like_flow(
    retrieval_counts: dict[str, int],
    raw_paper_count: int,
    merged_papers: list[Paper],
    duplicate_count: int,
    verification_results: list[VerificationResult],
    ranked_papers: list[RankedPaper],
    top_k: int = 10,
) -> dict[str, Any]:
    """Build PRISMA-like screening counts from pipeline state."""

    missing_abstract_count = sum(1 for paper in merged_papers if not paper.abstract)
    low_confidence = [
        result
        for result in verification_results
        if result.support_level in {"missing_abstract", "missing_evidence", "unverified", "llm_invalid_evidence"}
    ]
    reasons = Counter(result.error_type or result.support_level for result in low_confidence)
    return {
        "records_identified_by_openalex": retrieval_counts.get("openalex", 0),
        "records_identified_by_semantic_scholar": retrieval_counts.get("semantic_scholar", 0),
        "records_identified_total": raw_paper_count,
        "duplicate_records_removed": duplicate_count,
        "records_with_missing_abstracts": missing_abstract_count,
        "records_screened": len(merged_papers),
        "records_included_in_top_ranked_results": len(ranked_papers[:top_k]),
        "records_excluded_or_low_confidence": len(low_confidence),
        "common_exclusion_reasons": dict(reasons.most_common(10)),
    }
