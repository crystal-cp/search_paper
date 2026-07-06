"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import AspectCoverageRecord, EvidenceRecord, QueryPlan, RankedPaper, SearchBrief
from .utils import ensure_dir


SCORING_FORMULA = (
    "final_score = 0.40 * relevance_score + 0.25 * evidence_score "
    "+ 0.15 * recency_score + 0.15 * quality_score "
    "+ 0.05 * diversity_score + human_feedback_adjustment"
)


def _escape_table(text: Any) -> str:
    value = str(text or "").replace("\n", " ").replace("|", "\\|")
    return value[:300]


def generate_report(
    path: str | Path,
    research_question: str,
    planned_queries: list[str],
    retrieval_statistics: dict[str, Any],
    ranked_papers: list[RankedPaper],
    evidence_records: list[EvidenceRecord],
    evaluation_metrics: dict[str, Any],
    feedback_applied: bool = False,
    search_brief: SearchBrief | None = None,
    question_refinement: dict[str, Any] | None = None,
    query_plan: QueryPlan | None = None,
    aspect_coverage_records: list[AspectCoverageRecord] | None = None,
    result_groups: dict[str, Any] | None = None,
    reading_path_path: str | Path | None = None,
    paper_cards_path: str | Path | None = None,
    prisma_like_flow: dict[str, Any] | None = None,
) -> None:
    """Generate a human-readable Markdown report."""

    destination = Path(path)
    ensure_dir(destination.parent)
    planner_metadata = retrieval_statistics.get("llm", {}).get("planner", {})
    planning_question = planner_metadata.get("planning_question", research_question)
    translated_question = planner_metadata.get("translated_question", "")

    lines: list[str] = [
        "# Literature Screening Report",
        "",
        "## Research Question",
        "",
        research_question,
        "",
    ]
    if planning_question and planning_question != research_question:
        lines.extend(
            [
                "## Question Preprocessing",
                "",
                f"Planning question: {planning_question}",
                "",
            ]
        )
    if translated_question:
        lines.extend(
            [
                f"Translated question: {translated_question}",
                "",
            ]
        )
    if search_brief:
        lines.extend(
            [
                "## Search Brief",
                "",
                f"- Refined question: {search_brief.refined_question}",
                f"- Search intent: {search_brief.search_intent}",
                f"- User goal: {search_brief.user_goal}",
                f"- Inclusion criteria: {', '.join(search_brief.inclusion_criteria)}",
                f"- Exclusion criteria: {', '.join(search_brief.exclusion_criteria)}",
                f"- Required aspects: {', '.join(search_brief.required_aspects)}",
                f"- Preferred paper types: {', '.join(search_brief.preferred_paper_types)}",
                f"- Time window: {search_brief.time_window}",
                f"- Success definition: {search_brief.success_definition}",
                "",
            ]
        )
    if question_refinement:
        lines.extend(["## Refined Subquestions", ""])
        subquestions = question_refinement.get("subquestions", [])
        if subquestions:
            lines.extend([f"- {item}" for item in subquestions])
        else:
            lines.append("- No subquestions were needed for this run.")
        lines.append("")
    if query_plan:
        lines.extend(
            [
                "## Query Strategy",
                "",
                f"- Core terms: {', '.join(query_plan.core_terms)}",
                f"- Must terms: {', '.join(query_plan.must_terms)}",
                f"- Optional terms: {', '.join(query_plan.optional_terms)}",
                f"- Exclude terms: {', '.join(query_plan.exclude_terms)}",
                f"- Required aspects: {', '.join(query_plan.required_aspects)}",
                f"- Filters: {query_plan.filters}",
                "",
            ]
        )
    lines.extend(
        [
            "## Planned Queries",
            "",
        ]
    )
    lines.extend([f"- {query}" for query in planned_queries])
    lines.extend(
        [
            "",
            "## Retrieval Statistics",
            "",
            f"- Raw retrieved paper count: {retrieval_statistics.get('raw_retrieved_paper_count', 0)}",
            f"- Merged paper count: {retrieval_statistics.get('merged_paper_count', 0)}",
            f"- Duplicate count: {retrieval_statistics.get('duplicate_count', 0)}",
            f"- Counts by provider: {retrieval_statistics.get('retrieval_counts_by_provider', {})}",
            "",
            "## PRISMA-like Screening Flow",
            "",
            f"- Records identified by OpenAlex: {(prisma_like_flow or {}).get('records_identified_by_openalex', 0)}",
            f"- Records identified by Semantic Scholar: {(prisma_like_flow or {}).get('records_identified_by_semantic_scholar', 0)}",
            f"- Duplicate records removed: {(prisma_like_flow or {}).get('duplicate_records_removed', 0)}",
            f"- Records with missing abstracts: {(prisma_like_flow or {}).get('records_with_missing_abstracts', 0)}",
            f"- Records screened: {(prisma_like_flow or {}).get('records_screened', 0)}",
            f"- Included in top ranked results: {(prisma_like_flow or {}).get('records_included_in_top_ranked_results', 0)}",
            f"- Excluded or low confidence: {(prisma_like_flow or {}).get('records_excluded_or_low_confidence', 0)}",
            f"- Common exclusion reasons: {(prisma_like_flow or {}).get('common_exclusion_reasons', {})}",
            "",
            "## LLM Settings",
            "",
            f"- Backend requested: {retrieval_statistics.get('llm', {}).get('backend_requested', 'none')}",
            f"- Backend active: {retrieval_statistics.get('llm', {}).get('backend_active', 'none')}",
            f"- Planner mode: {retrieval_statistics.get('llm', {}).get('planner_mode', 'rule')}",
            f"- Extractor mode: {retrieval_statistics.get('llm', {}).get('extractor_mode', 'rule')}",
            f"- Verifier mode: {retrieval_statistics.get('llm', {}).get('verifier_mode', 'rule')}",
            f"- Invalid LLM output count: {retrieval_statistics.get('llm', {}).get('invalid_llm_output_count', 0)}",
            "",
            "## Evidence Validation",
            "",
            f"- Strict supported count: {retrieval_statistics.get('strict_supported_count', 0)}",
            f"- Weak support count: {retrieval_statistics.get('weak_support_count', 0)}",
            f"- Unverified count: {retrieval_statistics.get('unverified_count', 0)}",
            f"- LLM invalid evidence count: {retrieval_statistics.get('llm_invalid_evidence_count', 0)}",
            f"- Grounding accuracy: {retrieval_statistics.get('grounding_accuracy', 0):.3f}",
            f"- Strict support rate: {retrieval_statistics.get('strict_support_rate', 0):.3f}",
            f"- Weak support rate: {retrieval_statistics.get('weak_support_rate', 0):.3f}",
            f"- LLM invalid evidence rate: {retrieval_statistics.get('llm_invalid_evidence_rate', 0):.3f}",
            f"- Average aspect coverage: {retrieval_statistics.get('average_aspect_coverage', 0):.3f}",
            "",
            "## Scoring Formula",
            "",
            f"`{SCORING_FORMULA}`",
            "",
            "Current weights:",
            "",
            f"- Relevance: {evaluation_metrics.get('scoring_weights', {}).get('relevance', 0.40)}",
            f"- Evidence: {evaluation_metrics.get('scoring_weights', {}).get('evidence', 0.25)}",
            f"- Recency: {evaluation_metrics.get('scoring_weights', {}).get('recency', 0.15)}",
            f"- Quality: {evaluation_metrics.get('scoring_weights', {}).get('quality', 0.15)}",
            f"- Diversity: {evaluation_metrics.get('scoring_weights', {}).get('diversity', 0.05)}",
            "",
            "## Top 10 Ranked Papers",
            "",
            "| Rank | Score | Year | Title | Venue | DOI |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )

    for item in ranked_papers[:10]:
        lines.append(
            "| "
            f"{item.rank} | "
            f"{item.scores.final_score:.3f} | "
            f"{item.paper.year or ''} | "
            f"{_escape_table(item.paper.title)} | "
            f"{_escape_table(item.paper.venue)} | "
            f"{_escape_table(item.paper.doi)} |"
        )

    lines.extend(["", "## Aspect Coverage Summary", ""])
    aspect_records = aspect_coverage_records or []
    if aspect_records:
        lines.extend(
            [
                "| Paper | Covered aspects | Missing aspects | Score |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for record in aspect_records[:10]:
            lines.append(
                "| "
                f"{_escape_table(record.title)} | "
                f"{_escape_table(', '.join(record.covered_aspects))} | "
                f"{_escape_table(', '.join(record.missing_aspects))} | "
                f"{record.aspect_coverage_score:.2f} |"
            )
    else:
        lines.append("- No required aspects were classified.")

    lines.extend(["", "## Recommended Reading Path", ""])
    if reading_path_path:
        lines.append(f"- See `{Path(reading_path_path).name}`.")
    else:
        lines.append("- No reading path was generated.")

    lines.extend(["", "## Grouped Result Lists", ""])
    for group_name, rows in (result_groups or {}).items():
        lines.append(f"- {group_name}: {len(rows)} papers")

    lines.extend(["", "## Top Paper Evidence Cards", ""])
    if paper_cards_path:
        lines.append(f"- See `{Path(paper_cards_path).name}`.")
    else:
        lines.append("- No paper cards were generated.")

    lines.extend(
        [
            "",
            "## Evidence Chain Table",
            "",
            "| Rank | Support level | Span match | Span confidence | LLM invalid | Missing abstract | Claim | Evidence | Matched text |",
            "| ---: | --- | --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    evidence_by_id = {record.paper_id: record for record in evidence_records}
    for item in ranked_papers[:10]:
        evidence = evidence_by_id.get(item.paper.paper_id, item.evidence)
        lines.append(
            "| "
            f"{item.rank} | "
            f"{item.verification.support_level} | "
            f"{item.verification.span_match_type} | "
            f"{item.verification.span_match_confidence:.2f} | "
            f"{item.verification.support_level == 'llm_invalid_evidence'} | "
            f"{item.verification.support_level == 'missing_abstract'} | "
            f"{_escape_table(evidence.claim)} | "
            f"{_escape_table(evidence.evidence_sentence)} | "
            f"{_escape_table(item.verification.matched_text)} |"
        )

    lines.extend(["", "## Human Feedback Effects", ""])
    if feedback_applied:
        changes = evaluation_metrics.get("ranking_changes", {})
        lines.append(f"- Feedback applied: yes")
        lines.append(f"- Papers with rank changes: {changes.get('moved_count', 0)}")
    else:
        lines.append("- Feedback applied: no")

    lines.extend(
        [
            "",
            "## Evaluation Metrics",
            "",
            f"- Missing abstract ratio: {evaluation_metrics.get('missing_abstract_ratio', 0):.3f}",
            f"- Unsupported claim rate: {evaluation_metrics.get('unsupported_claim_rate', 0):.3f}",
            f"- Precision@10: {evaluation_metrics.get('precision_at_10')}",
            f"- nDCG@10: {evaluation_metrics.get('ndcg_at_10')}",
            f"- MAP: {evaluation_metrics.get('map')}",
            f"- Recall@10: {evaluation_metrics.get('recall_at_10')}",
            f"- Feedback mean absolute rank delta: {evaluation_metrics.get('feedback_before_after_ranking_delta', {}).get('mean_abs_rank_delta', 0)}",
            "",
            "## Limitations",
            "",
            "- Retrieval depends on provider metadata quality and availability.",
            "- Evidence extraction is lexical and abstract-only.",
            "- Verification checks grounding in abstracts, not full-text claims.",
            "- Ranking weights are transparent but manually chosen for the MVP.",
            "",
            "## Future Work",
            "",
            "- Add richer query expansion and provider diagnostics.",
            "- Compare rule-based extraction with optional LLM extraction.",
            "- Add a small human-labeling loop for iterative ranking studies.",
            "- Add full-text or PDF support only after the abstract-level baseline is validated.",
            "",
        ]
    )

    destination.write_text("\n".join(lines), encoding="utf-8")
