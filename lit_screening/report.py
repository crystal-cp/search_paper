"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import (
    AspectCoverageRecord,
    DomainAssessment,
    EvidenceRecord,
    PreferenceLearningResult,
    QueryPlan,
    RankedPaper,
    RetrievalPath,
    SearchBrief,
    SearchContract,
    SeedPaper,
    ScreeningDecision,
)
from .utils import ensure_dir


SCORING_FORMULA = (
    "final_score = 0.40 * relevance_score + 0.25 * evidence_score "
    "+ 0.15 * recency_score + 0.15 * quality_score "
    "+ 0.05 * diversity_score + human_feedback_adjustment "
    "+ preference_adjustment, then domain penalty is applied"
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
    search_contract: SearchContract | None = None,
    ambiguity_analysis: list[dict[str, Any]] | None = None,
    domain_assessments: list[DomainAssessment] | None = None,
    query_pilot_diagnostics: dict[str, Any] | None = None,
    query_repair_suggestions: dict[str, Any] | None = None,
    question_refinement: dict[str, Any] | None = None,
    query_plan: QueryPlan | None = None,
    aspect_coverage_records: list[AspectCoverageRecord] | None = None,
    result_groups: dict[str, Any] | None = None,
    reading_path_path: str | Path | None = None,
    paper_cards_path: str | Path | None = None,
    prisma_like_flow: dict[str, Any] | None = None,
    screening_decisions: list[ScreeningDecision] | None = None,
    method_comparison_matrix: list[dict[str, Any]] | None = None,
    research_gap_matrix: list[dict[str, Any]] | None = None,
    suggested_next_searches: list[dict[str, Any]] | None = None,
    preference_learning: PreferenceLearningResult | None = None,
    feedback_query_refinement: dict[str, Any] | None = None,
    seed_papers: list[SeedPaper] | None = None,
    retrieval_paths: list[RetrievalPath] | None = None,
    citation_expansion_papers: list[Any] | None = None,
) -> None:
    """Generate a human-readable Markdown report."""

    destination = Path(path)
    ensure_dir(destination.parent)
    planner_metadata = retrieval_statistics.get("llm", {}).get("planner", {})
    planning_question = planner_metadata.get("planning_question", research_question)
    translated_question = planner_metadata.get("translated_question", "")

    lines: list[str] = [
        "# Literature Screening Decision Report",
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
                "## What the System Thinks the User Is Looking For",
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
    else:
        lines.extend(
            [
                "## What the System Thinks the User Is Looking For",
                "",
                "- No search brief was generated for this run.",
                "",
            ]
        )
    if search_contract:
        domain = search_contract.domain_profile
        lines.extend(
            [
                "## Search Contract",
                "",
                f"- Domain: {domain.domain_name}",
                f"- Positive domains: {', '.join(domain.positive_domains)}",
                f"- Negative domains: {', '.join(domain.negative_domains)}",
                f"- Must include concepts: {', '.join(search_contract.must_include_concepts)}",
                f"- Must exclude concepts: {', '.join(search_contract.must_exclude_concepts)}",
                f"- Required aspects: {', '.join(search_contract.required_aspects)}",
                f"- Field whitelist: {', '.join(domain.field_of_study_whitelist)}",
                f"- Field blacklist: {', '.join(domain.field_of_study_blacklist)}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Search Contract",
                "",
                "- No Search Contract was generated for this run.",
                "",
            ]
        )
    if ambiguity_analysis:
        lines.extend(
            [
                "## Ambiguity Handling",
                "",
                "| Term | Selected meaning | Recommended excludes |",
                "| --- | --- | --- |",
            ]
        )
        for record in ambiguity_analysis:
            lines.append(
                "| "
                f"{_escape_table(record.get('term', ''))} | "
                f"{_escape_table(record.get('selected_meaning', ''))} | "
                f"{_escape_table(', '.join(record.get('recommended_exclude_terms', [])))} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Ambiguity Handling",
                "",
                "- No ambiguous terms were detected.",
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
    year_filter = retrieval_statistics.get("year_filter", {})
    imported_library = retrieval_statistics.get("imported_library", {})
    lines.extend(
        [
            "## Planned Queries",
            "",
        ]
    )
    lines.extend([f"- {query}" for query in planned_queries])
    domain_guardrails = retrieval_statistics.get("domain_guardrails", {})
    domain_counts = domain_guardrails.get("counts", {})
    demoted_examples = domain_guardrails.get("demoted_examples", [])
    common_reasons = domain_guardrails.get("common_off_topic_reasons", [])
    lines.extend(
        [
            "",
            "## Domain Guardrails",
            "",
            f"- In scope: {domain_counts.get('in_scope', 0)}",
            f"- Borderline: {domain_counts.get('borderline', 0)}",
            f"- Out of scope: {domain_counts.get('out_of_scope', 0)}",
            "",
        ]
    )
    if demoted_examples:
        lines.extend(
            [
                "| Rank | Decision | Penalty | Domain score | Paper | Reason |",
                "| ---: | --- | ---: | ---: | --- | --- |",
            ]
        )
        for example in demoted_examples[:8]:
            lines.append(
                "| "
                f"{example.get('rank', '')} | "
                f"{_escape_table(example.get('domain_decision', ''))} | "
                f"{example.get('domain_penalty_multiplier', '')} | "
                f"{example.get('domain_match_score', '')} | "
                f"{_escape_table(example.get('title', ''))} | "
                f"{_escape_table(example.get('off_topic_reason', ''))} |"
            )
        lines.append("")
    else:
        lines.extend(["- No papers were demoted by domain guardrails.", ""])
    if common_reasons:
        lines.append("Common off-topic reasons:")
        for item in common_reasons[:5]:
            lines.append(f"- {item.get('reason', '')} ({item.get('count', 0)})")
        lines.append("")

    pilot = query_pilot_diagnostics or {}
    repairs = query_repair_suggestions or {}
    lines.extend(["## Query Pilot Diagnostics", ""])
    if pilot.get("enabled"):
        summary = pilot.get("summary", {})
        lines.extend(
            [
                f"- Pilot max per query: {pilot.get('pilot_max_per_query', '')}",
                f"- Pilot records: {len(pilot.get('results', []))}",
                f"- Mean off-topic rate: {summary.get('mean_off_topic_rate', 0)}",
                f"- Recommendation counts: {summary.get('recommendation_counts', {})}",
                f"- Detected drift counts: {summary.get('detected_drift_counts', {})}",
                "",
                "| Provider | Stage | Recommendation | Off-topic rate | Query | Drift |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for result in pilot.get("results", [])[:12]:
            lines.append(
                "| "
                f"{_escape_table(result.get('provider', ''))} | "
                f"{_escape_table(result.get('retrieval_stage', ''))} | "
                f"{_escape_table(result.get('recommendation', ''))} | "
                f"{result.get('off_topic_rate_estimate', 0)} | "
                f"{_escape_table(result.get('query', ''))} | "
                f"{_escape_table(', '.join(result.get('detected_drift', [])))} |"
            )
        lines.append("")
    else:
        lines.append("- Pilot search was not run for this report.")
        lines.append("")

    lines.extend(["## Query Repairs Applied", ""])
    if repairs.get("enabled"):
        lines.append(f"- Auto repair applied: {repairs.get('applied', False)}")
        suggestions = repairs.get("suggestions", [])
        if suggestions:
            lines.extend(
                [
                    "",
                    "| Provider | Recommendation | Original query | Repaired query | Reason |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for suggestion in suggestions[:12]:
                lines.append(
                    "| "
                    f"{_escape_table(suggestion.get('provider', ''))} | "
                    f"{_escape_table(suggestion.get('recommendation', ''))} | "
                    f"{_escape_table(suggestion.get('original_query', ''))} | "
                    f"{_escape_table(suggestion.get('repaired_query', ''))} | "
                    f"{_escape_table(suggestion.get('reason', ''))} |"
                )
            lines.append("")
        else:
            lines.append("- No query repairs were suggested.")
            lines.append("")
    else:
        lines.append("- Query repair was not run.")
        lines.append("")

    lines.extend(["## Seed Paper Expansion", ""])
    seed_rows = seed_papers or []
    path_rows = retrieval_paths or []
    expansion_rows = citation_expansion_papers or []
    if seed_rows:
        lines.extend(
            [
                f"- Seed papers: {len(seed_rows)}",
                f"- Expanded papers kept for ranking: {len(expansion_rows)}",
                f"- Retrieval paths recorded: {len(path_rows)}",
                "",
                "| Seed type | Seed ID | Title | Note |",
                "| --- | --- | --- | --- |",
            ]
        )
        for seed in seed_rows[:10]:
            lines.append(
                "| "
                f"{_escape_table(seed.seed_type)} | "
                f"{_escape_table(seed.seed_id)} | "
                f"{_escape_table(seed.title)} | "
                f"{_escape_table(seed.note)} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "- No user-provided seed papers were supplied. If snowballing was enabled, seeds were selected from high-confidence ranked papers when possible.",
                "",
            ]
        )
    if path_rows:
        lines.extend(
            [
                "| Stage | Expanded paper ID | Seed | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for path in path_rows[:20]:
            lines.append(
                "| "
                f"{_escape_table(path.source_stage)} | "
                f"{_escape_table(path.paper_id)} | "
                f"{_escape_table(path.seed_title or path.seed_paper_id)} | "
                f"{_escape_table(path.reason)} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "- No citation/reference/recommendation expansion paths were recorded for this run.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Retrieval Summary",
            "",
            f"- Raw retrieved paper count: {retrieval_statistics.get('raw_retrieved_paper_count', 0)}",
            f"- Merged paper count: {retrieval_statistics.get('merged_paper_count', 0)}",
            f"- Duplicate count: {retrieval_statistics.get('duplicate_count', 0)}",
            f"- Counts by provider: {retrieval_statistics.get('retrieval_counts_by_provider', {})}",
            f"- Imported library papers: {imported_library.get('paper_count', 0)}",
            f"- Imported library format: {imported_library.get('detected_format', 'none')}",
            f"- Year filter enabled: {year_filter.get('enabled', False)}",
            f"- From year: {year_filter.get('from_year')}",
            f"- Records kept after year filter: {year_filter.get('kept_count', retrieval_statistics.get('raw_retrieved_paper_count', 0))}",
            f"- Records excluded before from-year: {year_filter.get('excluded_before_year_count', 0)}",
            f"- Records excluded because year is missing: {year_filter.get('excluded_missing_year_count', 0)}",
            "",
            "## PRISMA-like Screening Flow",
            "",
            f"- Records identified by OpenAlex: {(prisma_like_flow or {}).get('records_identified_by_openalex', 0)}",
            f"- Records identified by Semantic Scholar: {(prisma_like_flow or {}).get('records_identified_by_semantic_scholar', 0)}",
            f"- Duplicate records removed: {(prisma_like_flow or {}).get('duplicate_records_removed', 0)}",
            f"- Records with missing abstracts: {(prisma_like_flow or {}).get('records_with_missing_abstracts', 0)}",
            f"- Records screened: {(prisma_like_flow or {}).get('records_screened', 0)}",
            f"- Records included: {(prisma_like_flow or {}).get('records_included', 0)}",
            f"- Records maybe: {(prisma_like_flow or {}).get('records_maybe', 0)}",
            f"- Records excluded: {(prisma_like_flow or {}).get('records_excluded', 0)}",
            f"- Out-of-domain records: {(prisma_like_flow or {}).get('out_of_domain_records', 0)}",
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
            "- Domain penalty: in_scope x1.0, borderline x0.7, out_of_scope x0.3",
            "",
            "## Top 10 Ranked Papers",
            "",
            "| Rank | Decision | Score | Domain | Year | Title | Venue | DOI |",
            "| --- | --- | ---: | --- | ---: | --- | --- | --- |",
        ]
    )

    for item in ranked_papers[:10]:
        domain = item.domain_assessment
        decision = item.screening_decision
        lines.append(
            "| "
            f"{item.rank} | "
            f"{_escape_table(decision.decision if decision else '')} | "
            f"{item.scores.final_score:.3f} | "
            f"{_escape_table(domain.domain_decision if domain else '')} | "
            f"{item.paper.year or ''} | "
            f"{_escape_table(item.paper.title)} | "
            f"{_escape_table(item.paper.venue)} | "
            f"{_escape_table(item.paper.doi)} |"
        )

    lines.extend(["", "## Included Papers", ""])
    decision_records = screening_decisions or [
        item.screening_decision
        for item in ranked_papers
        if item.screening_decision is not None
    ]
    ranked_by_id = {item.paper.paper_id: item for item in ranked_papers}
    included = [record for record in decision_records if record.decision == "include"]
    maybe = [record for record in decision_records if record.decision == "maybe"]
    excluded = [record for record in decision_records if record.decision == "exclude"]
    if included:
        lines.extend(
            [
                "| Rank | Score | Paper | Primary reason | Reading priority |",
                "| ---: | ---: | --- | --- | --- |",
            ]
        )
        for record in included[:10]:
            item = ranked_by_id.get(record.paper_id)
            score_text = f"{item.scores.final_score:.3f}" if item else ""
            lines.append(
                "| "
                f"{item.rank if item else ''} | "
                f"{score_text} | "
                f"{_escape_table(item.paper.title if item else record.paper_id)} | "
                f"{_escape_table(record.primary_reason)} | "
                f"{_escape_table(record.reading_priority)} |"
            )
    else:
        lines.append("- No papers were automatically marked include.")

    lines.extend(["", "## Maybe / Needs Human Inspection", ""])
    if maybe:
        lines.extend(
            [
                "| Rank | Score | Paper | Primary reason | Suggested action |",
                "| ---: | ---: | --- | --- | --- |",
            ]
        )
        for record in maybe[:15]:
            item = ranked_by_id.get(record.paper_id)
            score_text = f"{item.scores.final_score:.3f}" if item else ""
            lines.append(
                "| "
                f"{item.rank if item else ''} | "
                f"{score_text} | "
                f"{_escape_table(item.paper.title if item else record.paper_id)} | "
                f"{_escape_table(record.primary_reason)} | "
                f"{_escape_table(record.suggested_action)} |"
            )
    else:
        lines.append("- No papers were marked maybe.")

    lines.extend(["", "## Excluded Papers And Reasons", ""])
    if excluded:
        lines.extend(
            [
                "| Rank | Score | Paper | Primary reason | Exclusion reasons |",
                "| ---: | ---: | --- | --- | --- |",
            ]
        )
        for record in excluded[:20]:
            item = ranked_by_id.get(record.paper_id)
            score_text = f"{item.scores.final_score:.3f}" if item else ""
            lines.append(
                "| "
                f"{item.rank if item else ''} | "
                f"{score_text} | "
                f"{_escape_table(item.paper.title if item else record.paper_id)} | "
                f"{_escape_table(record.primary_reason)} | "
                f"{_escape_table(', '.join(record.exclusion_reasons))} |"
            )
    else:
        lines.append("- No papers were automatically excluded.")

    lines.extend(["", "## Common Exclusion Reasons", ""])
    reason_counts = (
        evaluation_metrics.get("screening_decisions", {})
        .get("common_exclusion_reasons", {})
    )
    if not reason_counts:
        reason_counts = (prisma_like_flow or {}).get("common_exclusion_reasons", {})
    if reason_counts:
        for reason, count in reason_counts.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- No exclusion reasons were recorded.")

    method_rows = method_comparison_matrix or []
    lines.extend(["", "## Method Comparison Matrix", ""])
    if method_rows:
        lines.extend(
            [
                "| Paper | Method | Human role | Agent design | Evidence verification | Evaluation | Recommended use |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in method_rows[:12]:
            lines.append(
                "| "
                f"{_escape_table(row.get('title', ''))} | "
                f"{_escape_table(row.get('method', ''))} | "
                f"{_escape_table(row.get('human_role', ''))} | "
                f"{_escape_table(row.get('agent_design', ''))} | "
                f"{_escape_table(row.get('evidence_verification', ''))} | "
                f"{_escape_table(row.get('evaluation', ''))} | "
                f"{_escape_table(row.get('recommended_use', ''))} |"
            )
        lines.append("")
        lines.append("- Full matrix: `method_comparison_matrix.csv` and `method_comparison_matrix.md`.")
    else:
        lines.append("- No method comparison rows were generated.")

    gap_rows = research_gap_matrix or []
    lines.extend(["", "## Research Gap Matrix", ""])
    if gap_rows:
        lines.extend(
            [
                "| Gap | Supporting papers | Why gap remains | Possible project idea | Related aspects |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in gap_rows[:10]:
            lines.append(
                "| "
                f"{_escape_table(row.get('gap', ''))} | "
                f"{_escape_table(row.get('supporting_papers', ''))} | "
                f"{_escape_table(row.get('why_gap_remains', ''))} | "
                f"{_escape_table(row.get('possible_project_idea', ''))} | "
                f"{_escape_table(row.get('related_aspects', ''))} |"
            )
        lines.append("")
        lines.append("- Full matrix: `research_gap_matrix.csv` and `research_gap_matrix.md`.")
    else:
        lines.append("- No research gaps were generated from this run.")

    next_searches = suggested_next_searches or []
    lines.extend(["", "## Suggested Next Searches", ""])
    if next_searches:
        lines.extend(
            [
                "| Query | Reason | Source |",
                "| --- | --- | --- |",
            ]
        )
        for row in next_searches[:12]:
            lines.append(
                "| "
                f"{_escape_table(row.get('query', ''))} | "
                f"{_escape_table(row.get('reason', ''))} | "
                f"{_escape_table(row.get('source', ''))} |"
            )
        lines.append("")
        lines.append("- Full list: `suggested_next_searches.json` and `suggested_next_searches.md`.")
    else:
        lines.append("- No follow-up searches were suggested.")

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

    lines.extend(["", "## Result Groups", ""])
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

    lines.extend(["", "## Human Feedback Preference Learning", ""])
    if preference_learning and preference_learning.enabled:
        lines.extend(
            [
                f"- Model type: {preference_learning.model_type}",
                f"- Labeled papers: {preference_learning.labeled_paper_count}",
                f"- Include labels: {preference_learning.include_count}",
                f"- Exclude labels: {preference_learning.exclude_count}",
                f"- Learned positive terms: {', '.join(preference_learning.positive_terms[:12])}",
                f"- Learned negative terms: {', '.join(preference_learning.negative_terms[:12])}",
                f"- Note: {preference_learning.note}",
                "",
            ]
        )
    else:
        note = preference_learning.note if preference_learning else "No feedback model was available."
        lines.extend([f"- Preference learning inactive: {note}", ""])

    lines.extend(["## How feedback changed ranking", ""])
    if feedback_applied:
        changes = evaluation_metrics.get("ranking_changes", {})
        lines.append(f"- Feedback applied: yes")
        lines.append(f"- Papers with rank changes: {changes.get('moved_count', 0)}")
    else:
        lines.append("- Feedback applied: no")

    lines.extend(["", "## Suggested query refinements from feedback", ""])
    refinement = feedback_query_refinement or {}
    if refinement.get("enabled"):
        lines.extend(
            [
                f"- Suggested must terms: {', '.join(refinement.get('suggested_must_terms', []))}",
                f"- Suggested optional terms: {', '.join(refinement.get('suggested_optional_terms', []))}",
                f"- Suggested exclude terms: {', '.join(refinement.get('suggested_exclude_terms', []))}",
            ]
        )
    else:
        lines.append("- No feedback-derived query refinements were generated.")

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
            "## Agent Trace Summary",
            "",
            f"- Intent agent: {search_brief.search_intent if search_brief else 'not available'}",
            f"- Planner: {len(planned_queries)} planned query strings.",
            f"- Retriever: {retrieval_statistics.get('raw_retrieved_paper_count', 0)} raw records, {retrieval_statistics.get('merged_paper_count', 0)} merged records.",
            f"- Domain guardrail: {domain_counts.get('in_scope', 0)} in_scope, {domain_counts.get('borderline', 0)} borderline, {domain_counts.get('out_of_scope', 0)} out_of_scope.",
            f"- Screening decision agent: {(prisma_like_flow or {}).get('records_included', 0)} include, {(prisma_like_flow or {}).get('records_maybe', 0)} maybe, {(prisma_like_flow or {}).get('records_excluded', 0)} exclude.",
            "- Full trace: `agent_trace.json`.",
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
