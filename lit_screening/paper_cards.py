"""Generate Markdown paper evidence cards."""

from __future__ import annotations

from pathlib import Path

from lit_screening.models import AspectCoverageRecord, RankedPaper
from lit_screening.result_groups import recommended_role
from lit_screening.utils import ensure_dir


def generate_paper_cards(
    path: str | Path,
    ranked_papers: list[RankedPaper],
    aspect_records: list[AspectCoverageRecord] | None = None,
    limit: int = 12,
) -> None:
    """Write Markdown evidence cards for top ranked papers."""

    aspect_by_id = {record.paper_id: record for record in aspect_records or []}
    lines = ["# Paper Evidence Cards", ""]
    for item in ranked_papers[:limit]:
        aspect = aspect_by_id.get(item.paper.paper_id)
        covered = ", ".join(aspect.covered_aspects) if aspect else ""
        missing = ", ".join(aspect.missing_aspects) if aspect else ""
        role = recommended_role(
            item,
            aspect.aspect_coverage_score if aspect else 0.0,
        )
        action = suggested_action(item, aspect)
        lines.extend(
            [
                f"## {item.rank}. {item.paper.title or 'Untitled'}",
                "",
                f"- Year / venue / citations: {item.paper.year or ''} | {item.paper.venue or ''} | {item.paper.citation_count}",
                f"- DOI or URL: {item.paper.doi or item.paper.url or ''}",
                f"- Recommended role: {role}",
                f"- Why relevant: score={item.scores.final_score:.3f}, relevance={item.scores.relevance_score:.3f}, evidence={item.scores.evidence_score:.3f}",
                f"- Covered aspects: {covered}",
                f"- Missing aspects: {missing}",
                f"- Claim: {item.evidence.claim}",
                f"- Evidence sentence: {item.evidence.evidence_sentence}",
                f"- Verification result: {item.verification.support_level} ({item.verification.span_match_type}, confidence={item.verification.confidence:.2f})",
                f"- Limitations: {item.evidence.limitation or item.verification.rationale}",
                f"- Suggested action: {action}",
                "",
            ]
        )
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text("\n".join(lines), encoding="utf-8")


def suggested_action(
    item: RankedPaper,
    aspect: AspectCoverageRecord | None,
) -> str:
    """Suggest include/exclude/uncertain based on score and grounding."""

    aspect_score = aspect.aspect_coverage_score if aspect else 0.0
    if item.verification.support_level in {"missing_abstract", "llm_invalid_evidence"}:
        return "uncertain"
    if item.scores.final_score >= 0.5 and aspect_score >= 0.4:
        return "include"
    if item.scores.final_score < 0.2:
        return "exclude"
    return "uncertain"
