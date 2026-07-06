"""Group ranked papers into sensemaking result lists."""

from __future__ import annotations

from lit_screening.models import AspectCoverageRecord, RankedPaper, SearchBrief


GROUP_NAMES = [
    "must_read",
    "recent_frontier",
    "implementation_relevant",
    "evaluation_relevant",
    "background_or_survey",
    "peripheral",
    "excluded_or_low_confidence",
]


def group_ranked_papers(
    ranked_papers: list[RankedPaper],
    aspect_records: list[AspectCoverageRecord] | None = None,
    search_brief: SearchBrief | None = None,
) -> dict[str, list[dict]]:
    """Group ranked papers by likely role in the user's reading task."""

    aspect_by_id = {record.paper_id: record for record in aspect_records or []}
    groups: dict[str, list[dict]] = {name: [] for name in GROUP_NAMES}
    intent = search_brief.search_intent if search_brief else "overview"
    for item in ranked_papers:
        aspect = aspect_by_id.get(item.paper.paper_id)
        aspect_score = aspect.aspect_coverage_score if aspect else 0.0
        role = recommended_role(item, aspect_score, intent)
        row = {
            "rank": item.rank,
            "paper_id": item.paper.paper_id,
            "title": item.paper.title,
            "year": item.paper.year,
            "final_score": item.scores.final_score,
            "support_level": item.verification.support_level,
            "aspect_coverage_score": aspect_score,
            "recommended_role": role,
        }
        groups[_group_name(item, aspect_score, role)].append(row)
    return groups


def recommended_role(item: RankedPaper, aspect_score: float, intent: str = "overview") -> str:
    """Recommend how a paper should be used by the reader."""

    text = " ".join([item.paper.title, item.paper.venue, item.evidence.claim]).lower()
    if item.verification.support_level in {"unverified", "missing_abstract", "llm_invalid_evidence"}:
        return "uncertain"
    if "review" in text or "survey" in text:
        return "background"
    if intent in {"implementation", "proposal"} and any(word in text for word in ["method", "system", "framework"]):
        return "method"
    if any(word in text for word in ["evaluation", "benchmark", "experiment"]):
        return "evaluation"
    if item.paper.year and item.paper.year >= 2022:
        return "future work"
    if aspect_score >= 0.6:
        return "motivation"
    return "background"


def _group_name(item: RankedPaper, aspect_score: float, role: str) -> str:
    if item.verification.support_level in {"missing_abstract", "llm_invalid_evidence"}:
        return "excluded_or_low_confidence"
    if item.scores.final_score >= 0.65 and aspect_score >= 0.5:
        return "must_read"
    if role == "future work":
        return "recent_frontier"
    if role == "method":
        return "implementation_relevant"
    if role == "evaluation":
        return "evaluation_relevant"
    if role == "background":
        return "background_or_survey"
    if item.scores.final_score < 0.25:
        return "excluded_or_low_confidence"
    return "peripheral"
