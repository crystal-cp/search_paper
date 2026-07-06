"""Command-line pipeline for the literature-screening MVP."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .agents.aspect_classifier import AspectCoverageAgent
from .agents.extractor import ExtractorAgent
from .agents.human_feedback import HumanFeedbackAgent
from .agents.planner import PlannerAgent
from .agents.question_refiner import QuestionRefinementAgent
from .agents.ranker import RankerAgent
from .agents.research_intent import ResearchIntentAgent
from .agents.retriever import RetrieverAgent
from .agents.verifier import VerifierAgent
from .config import PipelineConfig
from .dedup import deduplicate_with_stats
from .evaluation import compute_evaluation, save_evaluation
from .llm_client import GenericLLMClient
from .models import (
    AspectCoverageRecord,
    EvidenceRecord,
    FeedbackRecord,
    Paper,
    PipelineResult,
    QueryPlan,
    RankedPaper,
    SearchBrief,
    VerificationResult,
)
from .paper_cards import generate_paper_cards
from .reading_path import generate_reading_path
from .report import generate_report
from .result_groups import group_ranked_papers
from .run_logging import ScreeningRunLogger
from .scoring import sanitize_score_weights
from .screening_flow import build_prisma_like_flow
from .utils import ensure_dir, write_csv, write_json


PAPER_FIELDS = [
    "paper_id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_provider",
    "citation_count",
]

EVIDENCE_FIELDS = [
    "paper_id",
    "title",
    "claim",
    "evidence_sentence",
    "relevance_reason",
    "limitation",
    "keyword_overlap",
    "supported",
    "confidence",
    "error_type",
    "support_level",
    "span_match_type",
    "span_match_confidence",
    "matched_text",
    "strict_span_validated",
    "llm_invalid_evidence",
    "missing_abstract",
    "rationale",
    "extraction_mode",
    "evidence_llm_used",
    "evidence_invalid_llm_output",
    "evidence_llm_error_type",
    "verification_mode",
    "verification_llm_used",
    "verification_invalid_llm_output",
    "verification_llm_error_type",
]

RANKED_FIELDS = [
    "rank",
    "paper_id",
    "title",
    "year",
    "venue",
    "doi",
    "url",
    "source_provider",
    "citation_count",
    "relevance_score",
    "evidence_score",
    "recency_score",
    "quality_score",
    "diversity_score",
    "aspect_coverage_score",
    "human_feedback_adjustment",
    "final_score",
    "supported",
    "confidence",
    "error_type",
    "support_level",
    "span_match_type",
    "span_match_confidence",
    "matched_text",
    "strict_span_validated",
    "llm_invalid_evidence",
    "missing_abstract",
    "claim",
    "evidence_sentence",
    "feedback_label",
    "feedback_note",
    "extraction_mode",
    "evidence_llm_used",
    "evidence_invalid_llm_output",
    "evidence_llm_error_type",
    "verification_mode",
    "verification_llm_used",
    "verification_invalid_llm_output",
    "verification_llm_error_type",
]

ASPECT_COVERAGE_FIELDS = [
    "paper_id",
    "title",
    "covered_aspects",
    "missing_aspects",
    "aspect_coverage_score",
]


def paper_to_row(paper: Paper) -> dict[str, Any]:
    """Convert a Paper into a flat CSV row."""

    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": "; ".join(paper.authors),
        "year": paper.year or "",
        "venue": paper.venue,
        "doi": paper.doi,
        "url": paper.url,
        "source_provider": paper.source_provider,
        "citation_count": paper.citation_count,
    }


def evidence_to_row(
    evidence: EvidenceRecord,
    verification: VerificationResult,
) -> dict[str, Any]:
    """Convert evidence plus verification into a flat CSV row."""

    return {
        "paper_id": evidence.paper_id,
        "title": evidence.title,
        "claim": evidence.claim,
        "evidence_sentence": evidence.evidence_sentence,
        "relevance_reason": evidence.relevance_reason,
        "limitation": evidence.limitation,
        "keyword_overlap": f"{evidence.keyword_overlap:.4f}",
        "supported": verification.supported,
        "confidence": f"{verification.confidence:.4f}",
        "error_type": verification.error_type,
        "support_level": verification.support_level,
        "span_match_type": verification.span_match_type,
        "span_match_confidence": f"{verification.span_match_confidence:.4f}",
        "matched_text": verification.matched_text,
        "strict_span_validated": verification.support_level == "strict_support",
        "llm_invalid_evidence": verification.support_level == "llm_invalid_evidence",
        "missing_abstract": verification.support_level == "missing_abstract",
        "rationale": verification.rationale,
        "extraction_mode": evidence.extraction_mode,
        "evidence_llm_used": evidence.llm_used,
        "evidence_invalid_llm_output": evidence.invalid_llm_output,
        "evidence_llm_error_type": evidence.llm_error_type,
        "verification_mode": verification.verification_mode,
        "verification_llm_used": verification.llm_used,
        "verification_invalid_llm_output": verification.invalid_llm_output,
        "verification_llm_error_type": verification.llm_error_type,
    }


def ranked_to_row(item: RankedPaper) -> dict[str, Any]:
    """Convert a RankedPaper into a flat CSV row."""

    feedback = item.feedback
    return {
        "rank": item.rank,
        "paper_id": item.paper.paper_id,
        "title": item.paper.title,
        "year": item.paper.year or "",
        "venue": item.paper.venue,
        "doi": item.paper.doi,
        "url": item.paper.url,
        "source_provider": item.paper.source_provider,
        "citation_count": item.paper.citation_count,
        "relevance_score": f"{item.scores.relevance_score:.4f}",
        "evidence_score": f"{item.scores.evidence_score:.4f}",
        "recency_score": f"{item.scores.recency_score:.4f}",
        "quality_score": f"{item.scores.quality_score:.4f}",
        "diversity_score": f"{item.scores.diversity_score:.4f}",
        "aspect_coverage_score": f"{item.scores.aspect_coverage_score:.4f}",
        "human_feedback_adjustment": f"{item.scores.human_feedback_adjustment:.4f}",
        "final_score": f"{item.scores.final_score:.4f}",
        "supported": item.verification.supported,
        "confidence": f"{item.verification.confidence:.4f}",
        "error_type": item.verification.error_type,
        "support_level": item.verification.support_level,
        "span_match_type": item.verification.span_match_type,
        "span_match_confidence": f"{item.verification.span_match_confidence:.4f}",
        "matched_text": item.verification.matched_text,
        "strict_span_validated": item.verification.support_level == "strict_support",
        "llm_invalid_evidence": item.verification.support_level == "llm_invalid_evidence",
        "missing_abstract": not bool(item.paper.abstract),
        "claim": item.evidence.claim,
        "evidence_sentence": item.evidence.evidence_sentence,
        "feedback_label": feedback.label if feedback else "",
        "feedback_note": feedback.note if feedback else "",
        "extraction_mode": item.evidence.extraction_mode,
        "evidence_llm_used": item.evidence.llm_used,
        "evidence_invalid_llm_output": item.evidence.invalid_llm_output,
        "evidence_llm_error_type": item.evidence.llm_error_type,
        "verification_mode": item.verification.verification_mode,
        "verification_llm_used": item.verification.llm_used,
        "verification_invalid_llm_output": item.verification.invalid_llm_output,
        "verification_llm_error_type": item.verification.llm_error_type,
    }


def aspect_coverage_to_row(record: AspectCoverageRecord) -> dict[str, Any]:
    """Convert an AspectCoverageRecord into a CSV row."""

    return {
        "paper_id": record.paper_id,
        "title": record.title,
        "covered_aspects": "; ".join(record.covered_aspects),
        "missing_aspects": "; ".join(record.missing_aspects),
        "aspect_coverage_score": f"{record.aspect_coverage_score:.4f}",
    }


def build_llm_client(
    config: PipelineConfig,
    llm_backend: str,
    llm_client: GenericLLMClient | None = None,
) -> GenericLLMClient | None:
    """Create an optional LLM client from config."""

    if llm_client is not None:
        return llm_client if llm_client.is_available else None
    if llm_backend == "none":
        return None
    if llm_backend == "deepseek":
        client = GenericLLMClient(
            provider_name="deepseek",
            api_key_env_var=config.deepseek_api_key_env,
            base_url=config.deepseek_base_url,
            model=config.deepseek_model,
            timeout=config.llm_timeout,
        )
        return client if client.is_available else None
    raise ValueError(f"Unknown LLM backend: {llm_backend}")


def write_pipeline_csvs(
    output_dir: Path,
    merged_papers: list[Paper],
    evidence_records: list[EvidenceRecord],
    verification_results: list[VerificationResult],
    aspect_coverage_records: list[AspectCoverageRecord],
    ranked_before_feedback: list[RankedPaper],
    ranked_final: list[RankedPaper],
    ranked_after_feedback: list[RankedPaper] | None,
) -> None:
    """Write all required tabular outputs."""

    write_csv(
        output_dir / "merged_papers.csv",
        [paper_to_row(paper) for paper in merged_papers],
        PAPER_FIELDS,
    )
    verification_by_id = {result.paper_id: result for result in verification_results}
    write_csv(
        output_dir / "evidence_table.csv",
        [
            evidence_to_row(record, verification_by_id[record.paper_id])
            for record in evidence_records
            if record.paper_id in verification_by_id
        ],
        EVIDENCE_FIELDS,
    )
    write_csv(
        output_dir / "aspect_coverage.csv",
        [aspect_coverage_to_row(record) for record in aspect_coverage_records],
        ASPECT_COVERAGE_FIELDS,
    )
    write_csv(
        output_dir / "ranked_papers_before_feedback.csv",
        [ranked_to_row(item) for item in ranked_before_feedback],
        RANKED_FIELDS,
    )
    if ranked_after_feedback is not None:
        write_csv(
            output_dir / "ranked_papers_after_feedback.csv",
            [ranked_to_row(item) for item in ranked_after_feedback],
            RANKED_FIELDS,
        )
    write_csv(
        output_dir / "ranked_papers.csv",
        [ranked_to_row(item) for item in ranked_final],
        RANKED_FIELDS,
    )


def _llm_metrics(
    llm_backend: str,
    active_llm_backend: str,
    planner_mode: str,
    extractor_mode: str,
    verifier_mode: str,
    planner_metadata: dict[str, Any],
    evidence_records: list[EvidenceRecord],
    verification_results: list[VerificationResult],
) -> dict[str, Any]:
    """Build a compact diagnostics block for optional LLM usage."""

    planner_invalid = bool(planner_metadata.get("invalid_llm_output"))
    evidence_invalid_count = sum(1 for record in evidence_records if record.invalid_llm_output)
    verification_invalid_count = sum(
        1 for result in verification_results if result.invalid_llm_output
    )
    return {
        "backend_requested": llm_backend,
        "backend_active": active_llm_backend,
        "planner_mode": planner_mode,
        "extractor_mode": extractor_mode,
        "verifier_mode": verifier_mode,
        "planner": planner_metadata,
        "evidence_invalid_llm_output_count": evidence_invalid_count,
        "verification_invalid_llm_output_count": verification_invalid_count,
        "invalid_llm_output_count": int(planner_invalid)
        + evidence_invalid_count
        + verification_invalid_count,
    }


def build_agent_trace(
    question: str,
    planning_question: str,
    queries: list[str],
    query_plan: QueryPlan | None,
    search_brief: SearchBrief | None,
    question_refinement: dict[str, Any] | None,
    planner_metadata: dict[str, Any],
    retrieval_counts: dict[str, int],
    raw_paper_count: int,
    merged_papers: list[Paper],
    duplicate_count: int,
    evidence_records: list[EvidenceRecord],
    verification_results: list[VerificationResult],
    aspect_coverage_records: list[AspectCoverageRecord],
    ranked_papers: list[RankedPaper],
    scoring_weights: dict[str, float],
    result_groups: dict[str, Any] | None = None,
    year_filter_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an inspectable trace of agent decisions for demos and audits."""

    verification_by_id = {result.paper_id: result for result in verification_results}
    return {
        "question": question,
        "planning_question": planning_question,
        "research_intent": {
            "search_brief": search_brief,
            "decision": "Interpreted the user's search goal before query planning.",
        },
        "question_refiner": {
            "refinement": question_refinement or {},
            "decision": "Flagged broad or mixed questions and suggested subquestions.",
        },
        "planner": {
            "queries": queries,
            "query_plan": query_plan,
            "metadata": planner_metadata,
            "decision": "Generated scholarly queries from the user's actual topic.",
        },
        "retriever": {
            "retrieval_counts_by_provider": retrieval_counts,
            "raw_paper_count": raw_paper_count,
            "year_filter": year_filter_stats or {},
        },
        "deduplicator": {
            "merged_paper_count": len(merged_papers),
            "duplicate_count": duplicate_count,
            "decision": "Deduplicated by normalized DOI first, then normalized title.",
        },
        "extractor": [
            {
                "paper_id": record.paper_id,
                "title": record.title,
                "mode": record.extraction_mode,
                "keyword_overlap": record.keyword_overlap,
                "evidence_sentence": record.evidence_sentence,
                "limitation": record.limitation,
                "llm_used": record.llm_used,
                "invalid_llm_output": record.invalid_llm_output,
                "llm_error_type": record.llm_error_type,
            }
            for record in evidence_records
        ],
        "verifier": [
            {
                "paper_id": result.paper_id,
                "supported": result.supported,
                "support_level": result.support_level,
                "span_match_type": result.span_match_type,
                "span_match_confidence": result.span_match_confidence,
                "strict_span_validated": result.support_level == "strict_support",
                "llm_invalid_evidence": result.support_level == "llm_invalid_evidence",
                "missing_abstract": result.support_level == "missing_abstract",
                "error_type": result.error_type,
                "rationale": result.rationale,
                "matched_text": result.matched_text,
            }
            for result in verification_results
        ],
        "aspect_coverage": [
            {
                "paper_id": record.paper_id,
                "title": record.title,
                "covered_aspects": record.covered_aspects,
                "missing_aspects": record.missing_aspects,
                "aspect_coverage_score": record.aspect_coverage_score,
            }
            for record in aspect_coverage_records[:50]
        ],
        "ranker": [
            {
                "rank": item.rank,
                "paper_id": item.paper.paper_id,
                "title": item.paper.title,
                "final_score": item.scores.final_score,
                "support_level": verification_by_id[item.paper.paper_id].support_level
                if item.paper.paper_id in verification_by_id
                else "",
                "strict_span_validated": verification_by_id[
                    item.paper.paper_id
                ].support_level
                == "strict_support"
                if item.paper.paper_id in verification_by_id
                else False,
            }
            for item in ranked_papers[:50]
        ],
        "result_groups": {
            name: len(rows) for name, rows in (result_groups or {}).items()
        },
        "scoring_weights": scoring_weights,
    }


def apply_feedback_to_pipeline_result(
    result: PipelineResult,
    feedback_records: dict[str, FeedbackRecord],
    gold_labels_path: str | None = None,
) -> PipelineResult:
    """Apply feedback to an in-memory pipeline result without rerunning retrieval."""

    feedback_agent = HumanFeedbackAgent()
    ranked_after_feedback = feedback_agent.apply(
        result.ranked_before_feedback,
        feedback_records,
        scoring_weights=result.scoring_weights,
    )
    metrics = compute_evaluation(
        retrieval_counts=result.retrieval_counts,
        original_paper_count=result.raw_paper_count,
        merged_papers=result.merged_papers,
        evidence_records=result.evidence_records,
        verification_results=result.verification_results,
        ranked_before_feedback=result.ranked_before_feedback,
        ranked_after_feedback=ranked_after_feedback,
        gold_labels_path=gold_labels_path,
    )
    metrics["duplicate_count"] = result.duplicate_count
    metrics["query_controls"] = result.evaluation_metrics.get("query_controls", {})
    metrics["search_intent"] = (
        result.search_brief.search_intent if result.search_brief else result.evaluation_metrics.get("search_intent", "")
    )
    metrics["average_aspect_coverage"] = (
        sum(record.aspect_coverage_score for record in result.aspect_coverage_records)
        / len(result.aspect_coverage_records)
        if result.aspect_coverage_records
        else 0.0
    )
    if "llm" in result.evaluation_metrics:
        metrics["llm"] = result.evaluation_metrics["llm"]
    if "year_filter" in result.evaluation_metrics:
        metrics["year_filter"] = result.evaluation_metrics["year_filter"]
    metrics["scoring_weights"] = result.scoring_weights
    result_groups = group_ranked_papers(
        ranked_after_feedback,
        result.aspect_coverage_records,
        result.search_brief,
    )
    prisma_like_flow = build_prisma_like_flow(
        retrieval_counts=result.retrieval_counts,
        raw_paper_count=result.raw_paper_count,
        merged_papers=result.merged_papers,
        duplicate_count=result.duplicate_count,
        verification_results=result.verification_results,
        ranked_papers=ranked_after_feedback,
    )
    metrics["prisma_like_flow"] = prisma_like_flow
    trace = dict(result.agent_trace)
    trace["result_groups"] = {
        name: len(rows) for name, rows in result_groups.items()
    }
    trace["feedback"] = [
        {
            "paper_id": record.paper_id,
            "label": record.label,
            "adjustment": record.adjustment,
            "note": record.note,
        }
        for record in feedback_records.values()
    ]

    output_dir = Path(result.output_dir)
    save_evaluation(output_dir / "evaluation.json", metrics)
    write_json(output_dir / "agent_trace.json", trace)
    write_json(output_dir / "result_groups.json", result_groups)
    write_json(output_dir / "prisma_like_flow.json", prisma_like_flow)
    generate_paper_cards(
        output_dir / "paper_cards.md",
        ranked_after_feedback,
        result.aspect_coverage_records,
    )
    generate_reading_path(
        output_dir / "reading_path.md",
        ranked_after_feedback,
        result_groups,
    )
    write_pipeline_csvs(
        output_dir,
        result.merged_papers,
        result.evidence_records,
        result.verification_results,
        result.aspect_coverage_records,
        result.ranked_before_feedback,
        ranked_after_feedback,
        ranked_after_feedback,
    )
    generate_report(
        path=output_dir / "report.md",
        research_question=result.question,
        planned_queries=result.planned_queries,
        retrieval_statistics=metrics,
        ranked_papers=ranked_after_feedback,
        evidence_records=result.evidence_records,
        evaluation_metrics=metrics,
        feedback_applied=bool(feedback_records),
        search_brief=result.search_brief,
        question_refinement=result.question_refinement,
        query_plan=result.query_plan,
        aspect_coverage_records=result.aspect_coverage_records,
        result_groups=result_groups,
        reading_path_path=output_dir / "reading_path.md",
        paper_cards_path=output_dir / "paper_cards.md",
        prisma_like_flow=prisma_like_flow,
    )

    return replace(
        result,
        ranked_after_feedback=ranked_after_feedback,
        ranked_final=ranked_after_feedback,
        evaluation_metrics=metrics,
        agent_trace=trace,
        result_groups=result_groups,
    )


def _query_plan_payload(
    question: str,
    query_plan: QueryPlan,
    planner_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the serializable query-plan payload used by CLI and UI."""

    planning_question = query_plan.translated_question or query_plan.original_question or question
    queries = _combined_queries(query_plan)
    return {
        "question": question,
        "planning_question": planning_question,
        "translated_question": query_plan.translated_question,
        "queries": queries,
        "queries_by_provider": {
            "openalex": query_plan.openalex_queries,
            "semantic_scholar": query_plan.semantic_scholar_queries,
        },
        "query_plan": query_plan,
        "planner_metadata": planner_metadata,
        "llm": planner_metadata,
    }


def _combined_queries(query_plan: QueryPlan) -> list[str]:
    """Flatten provider-specific queries into a backward-compatible query list."""

    return _unique_strings(
        [
            query_plan.translated_question or query_plan.original_question,
            *query_plan.openalex_queries,
            *query_plan.semantic_scholar_queries,
        ]
    )


def _unique_strings(values: list[str]) -> list[str]:
    """Return unique non-empty strings while preserving order."""

    unique: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def _coerce_query_plan(value: QueryPlan | dict[str, Any] | None) -> QueryPlan | None:
    """Convert UI/API query-plan payloads back into a QueryPlan dataclass."""

    if value is None:
        return None
    if isinstance(value, QueryPlan):
        return value
    data = value.get("query_plan", value)
    return QueryPlan(
        original_question=data.get("original_question", ""),
        detected_language=data.get("detected_language", "en"),
        translated_question=data.get("translated_question", ""),
        core_terms=list(data.get("core_terms", [])),
        must_terms=list(data.get("must_terms", [])),
        optional_terms=list(data.get("optional_terms", [])),
        exclude_terms=list(data.get("exclude_terms", [])),
        required_aspects=list(data.get("required_aspects", [])),
        openalex_queries=list(data.get("openalex_queries", [])),
        semantic_scholar_queries=list(data.get("semantic_scholar_queries", [])),
        filters=dict(data.get("filters", {})),
    )


def _coerce_search_brief(
    value: SearchBrief | dict[str, Any] | None,
    fallback_question: str,
) -> SearchBrief | None:
    """Convert UI/API search-brief payloads back into a SearchBrief dataclass."""

    if value is None:
        return None
    if isinstance(value, SearchBrief):
        return value
    data = value.get("search_brief", value)
    return SearchBrief(
        original_question=str(data.get("original_question") or fallback_question),
        refined_question=str(
            data.get("refined_question")
            or data.get("translated_question")
            or fallback_question
        ),
        search_intent=str(data.get("search_intent") or "overview"),
        user_goal=str(data.get("user_goal") or "Find papers aligned with the research question."),
        inclusion_criteria=list(data.get("inclusion_criteria", [])),
        exclusion_criteria=list(data.get("exclusion_criteria", [])),
        required_aspects=list(data.get("required_aspects", [])),
        preferred_paper_types=list(data.get("preferred_paper_types", [])),
        time_window=str(data.get("time_window") or ""),
        success_definition=str(data.get("success_definition") or ""),
    )


def _manual_planner_metadata(
    question: str,
    planner_mode: str,
    planner_metadata_override: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return planner metadata for user-confirmed or user-edited queries."""

    if planner_metadata_override:
        metadata = dict(planner_metadata_override)
    else:
        cleaned_question = " ".join(question.split())
        metadata = {
            "planner_mode": planner_mode,
            "llm_used": False,
            "invalid_llm_output": False,
            "llm_error_type": "",
            "original_question": cleaned_question,
            "question_language": "en_or_other",
            "translation_used": False,
            "translation_mode": "none",
            "translated_question": "",
            "planning_question": cleaned_question,
            "translation_warning": "",
        }
    metadata["query_source"] = "user_confirmed"
    return metadata


def _make_query_plan(
    question: str,
    planner_mode: str,
    active_llm_client: GenericLLMClient | None,
    strictness: str = "balanced",
    openalex_mode: str = "keyword+semantic",
    sort_preference: str = "relevance",
    ranking_profile: str = "balanced",
    search_brief: Any | None = None,
) -> dict[str, Any]:
    """Run the planner and return a structured query plan."""

    planner = PlannerAgent(mode=planner_mode, llm_client=active_llm_client)
    query_plan = planner.plan_structured(
        question,
        strictness=strictness,
        openalex_mode=openalex_mode,
        sort_preference=sort_preference,
        ranking_profile=ranking_profile,
        search_brief=search_brief,
    )
    return _query_plan_payload(question, query_plan, planner.last_llm_metadata)


def plan_screening_queries(
    question: str,
    llm_backend: str = "none",
    planner_mode: str = "rule",
    strictness: str = "balanced",
    openalex_mode: str = "keyword+semantic",
    sort_preference: str = "relevance",
    ranking_profile: str = "balanced",
    llm_client: GenericLLMClient | None = None,
) -> dict[str, Any]:
    """Plan English scholarly queries without running retrieval.

    This supports a human checkpoint in the UI: users can inspect and edit the
    generated queries before spending provider requests or LLM extraction calls.
    """

    config = PipelineConfig(llm_backend=llm_backend)
    active_llm_client = build_llm_client(config, llm_backend, llm_client)
    search_brief = ResearchIntentAgent(
        mode=planner_mode,
        llm_client=active_llm_client,
    ).analyze(question)
    refinement = QuestionRefinementAgent().refine(question, search_brief)
    payload = _make_query_plan(
        question,
        planner_mode,
        active_llm_client,
        strictness=strictness,
        openalex_mode=openalex_mode,
        sort_preference=sort_preference,
        ranking_profile=ranking_profile,
        search_brief=search_brief,
    )
    payload["search_brief"] = search_brief
    payload["question_refinement"] = refinement
    return payload


def _queries_by_provider(
    providers: list[str],
    query_plan: QueryPlan,
    combined_queries: list[str],
) -> dict[str, list[str]]:
    """Return provider-specific queries with a fallback for custom test providers."""

    query_map = {
        "openalex": query_plan.openalex_queries,
        "semantic_scholar": query_plan.semantic_scholar_queries,
    }
    fallback_queries = combined_queries
    if query_plan.filters.get("query_source") == "user_confirmed":
        fallback_queries = query_plan.openalex_queries or query_plan.semantic_scholar_queries
    return {
        provider: query_map.get(provider) or fallback_queries
        for provider in providers
    }


def _raw_items_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return provider result items from a raw response."""

    items = response.get("results")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    items = response.get("data")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def _raw_title(item: dict[str, Any]) -> str:
    return str(item.get("title") or item.get("display_name") or "")


def build_retrieval_diagnostics(
    question: str,
    query_plan: QueryPlan,
    queries_by_provider: dict[str, list[str]],
    raw_by_provider: dict[str, list[dict]],
    merged_papers: list[Paper],
    duplicate_count: int,
    ranked_papers: list[RankedPaper],
    year_filter_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build transparent retrieval and reranking diagnostics."""

    raw_count_per_query: dict[str, dict[str, int]] = {}
    top_titles_per_query: dict[str, dict[str, list[str]]] = {}
    provider_errors: dict[str, list[dict[str, Any]]] = {}
    for provider, bundles in raw_by_provider.items():
        raw_count_per_query[provider] = {}
        top_titles_per_query[provider] = {}
        provider_errors[provider] = []
        for bundle in bundles:
            query = str(bundle.get("query") or "")
            response = bundle.get("response") or {}
            items = _raw_items_from_response(response)
            raw_count_per_query[provider][query] = len(items)
            top_titles_per_query[provider][query] = [
                title
                for title in [_raw_title(item) for item in items[:5]]
                if title
            ]
            if response.get("error"):
                provider_errors[provider].append(
                    {
                        "query": query,
                        "error": response.get("error"),
                        "status_code": response.get("status_code"),
                        "error_message": response.get("error_message", ""),
                    }
                )

    return {
        "question": question,
        "query_plan": query_plan,
        "queries_per_provider": queries_by_provider,
        "raw_count_per_query": raw_count_per_query,
        "provider_errors": provider_errors,
        "merged_count": len(merged_papers),
        "duplicate_count": duplicate_count,
        "year_filter": year_filter_stats or {},
        "missing_abstract_count": sum(1 for paper in merged_papers if not paper.abstract),
        "top_titles_per_query": top_titles_per_query,
        "top_10_score_breakdown": [
            {
                "rank": item.rank,
                "paper_id": item.paper.paper_id,
                "title": item.paper.title,
                "final_score": item.scores.final_score,
                "relevance_score": item.scores.relevance_score,
                "evidence_score": item.scores.evidence_score,
                "recency_score": item.scores.recency_score,
                "quality_score": item.scores.quality_score,
                "diversity_score": item.scores.diversity_score,
                "api_relevance_score": item.paper.api_relevance_score,
                "hybrid_reranking": item.paper.raw.get("hybrid_reranking", {}),
                "support_level": item.verification.support_level,
            }
            for item in ranked_papers[:10]
        ],
    }


def filter_papers_by_from_year(
    papers: list[Paper],
    from_year: int | None,
) -> tuple[list[Paper], dict[str, Any]]:
    """Apply a hard local publication-year filter after provider retrieval."""

    if not from_year:
        return papers, {
            "from_year": None,
            "enabled": False,
            "input_count": len(papers),
            "kept_count": len(papers),
            "excluded_before_year_count": 0,
            "excluded_missing_year_count": 0,
        }
    kept: list[Paper] = []
    excluded_before = 0
    excluded_missing = 0
    examples: list[dict[str, Any]] = []
    for paper in papers:
        if paper.year is None:
            excluded_missing += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "year": None,
                        "reason": "missing_year",
                    }
                )
            continue
        if paper.year < from_year:
            excluded_before += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "year": paper.year,
                        "reason": "before_from_year",
                    }
                )
            continue
        kept.append(paper)
    return kept, {
        "from_year": from_year,
        "enabled": True,
        "input_count": len(papers),
        "kept_count": len(kept),
        "excluded_before_year_count": excluded_before,
        "excluded_missing_year_count": excluded_missing,
        "excluded_examples": examples,
    }


def run_pipeline(
    question: str,
    providers: list[str],
    max_per_query: int = 10,
    from_year: int | None = None,
    feedback_path: str | None = None,
    gold_labels_path: str | None = None,
    output_dir: str = "outputs",
    use_cache: bool = True,
    llm_backend: str = "none",
    planner_mode: str = "rule",
    extractor_mode: str = "rule",
    verifier_mode: str = "rule",
    scoring_weights: dict[str, float] | None = None,
    search_brief_override: SearchBrief | dict[str, Any] | None = None,
    planned_queries_override: list[str] | None = None,
    query_plan_override: QueryPlan | dict[str, Any] | None = None,
    planner_metadata_override: dict[str, Any] | None = None,
    strictness: str = "balanced",
    openalex_mode: str = "keyword+semantic",
    sort_preference: str = "relevance",
    ranking_profile: str = "balanced",
    llm_client: GenericLLMClient | None = None,
    retriever_agent: RetrieverAgent | None = None,
    progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> PipelineResult:
    """Run the full MVP pipeline and write output artifacts."""

    out = ensure_dir(output_dir)
    ensure_dir("data/cache")
    run_logger = ScreeningRunLogger(out)
    external_progress_callback = progress_callback

    def log_progress(stage: str, message: str, details: dict[str, Any]) -> None:
        run_logger.log(stage, message, details)
        if external_progress_callback:
            external_progress_callback(stage, message, details)

    progress_callback = log_progress
    progress_callback(
        "start",
        "Pipeline started",
        {
            "providers": providers,
            "max_per_query": max_per_query,
            "from_year": from_year,
            "output_dir": str(out),
            "llm_backend": llm_backend,
            "planner_mode": planner_mode,
            "extractor_mode": extractor_mode,
            "verifier_mode": verifier_mode,
            "use_cache": use_cache,
        },
    )

    config = PipelineConfig(
        providers=providers,
        max_per_query=max_per_query,
        from_year=from_year,
        output_dir=str(out),
        use_cache=use_cache,
        llm_backend=llm_backend,
    )

    active_llm_client = build_llm_client(config, llm_backend, llm_client)
    active_scoring_weights = sanitize_score_weights(
        scoring_weights,
        ranking_profile=ranking_profile,
    )
    search_brief = _coerce_search_brief(search_brief_override, question)
    if search_brief is None:
        search_brief = ResearchIntentAgent(
            mode=planner_mode,
            llm_client=active_llm_client,
        ).analyze(question)
    question_refinement = QuestionRefinementAgent().refine(question, search_brief)
    write_json(out / "search_brief.json", search_brief)
    write_json(out / "question_refinement.json", question_refinement)

    if progress_callback:
        progress_callback(
            "planning",
            "Preparing query plan",
            {
                "planner_mode": planner_mode,
                "llm_backend": llm_backend,
                "query_override": planned_queries_override is not None
                or query_plan_override is not None,
            },
        )
    if planned_queries_override is None and query_plan_override is None:
        query_plan_payload = _make_query_plan(
            question,
            planner_mode,
            active_llm_client,
            strictness=strictness,
            openalex_mode=openalex_mode,
            sort_preference=sort_preference,
            ranking_profile=ranking_profile,
            search_brief=search_brief,
        )
    else:
        planner_metadata = _manual_planner_metadata(
            question,
            planner_mode,
            planner_metadata_override,
        )
        structured_override = _coerce_query_plan(query_plan_override)
        if structured_override is None:
            cleaned_queries = [
                " ".join(query.split())
                for query in planned_queries_override or []
                if isinstance(query, str) and " ".join(query.split())
            ]
            structured_override = QueryPlan(
                original_question=question,
                detected_language="zh"
                if planner_metadata.get("question_language") == "zh"
                else "en",
                translated_question=str(planner_metadata.get("translated_question") or ""),
                required_aspects=search_brief.required_aspects,
                openalex_queries=cleaned_queries,
                semantic_scholar_queries=cleaned_queries,
                filters={
                    "strictness": strictness,
                    "openalex_mode": openalex_mode,
                    "sort_preference": sort_preference,
                    "ranking_profile": ranking_profile,
                    "query_source": "user_confirmed",
                },
            )
            if not structured_override.translated_question and search_brief.refined_question:
                structured_override.translated_question = search_brief.refined_question
        else:
            if not structured_override.required_aspects:
                structured_override.required_aspects = search_brief.required_aspects
            if not structured_override.translated_question and search_brief.refined_question:
                structured_override.translated_question = search_brief.refined_question
            structured_override.filters = {
                **structured_override.filters,
                "strictness": strictness,
                "openalex_mode": openalex_mode,
                "sort_preference": sort_preference,
                "ranking_profile": ranking_profile,
                "query_source": "user_confirmed",
            }
        query_plan_payload = _query_plan_payload(
            question,
            structured_override,
            planner_metadata,
        )

    query_plan = query_plan_payload["query_plan"]
    query_plan_payload["search_brief"] = search_brief
    query_plan_payload["question_refinement"] = question_refinement
    queries = query_plan_payload["queries"]
    planner_metadata = query_plan_payload["planner_metadata"]
    planning_question = str(query_plan_payload.get("planning_question") or question)
    queries_by_provider = _queries_by_provider(providers, query_plan, queries)
    write_json(out / "planned_queries.json", query_plan_payload)
    if progress_callback:
        progress_callback(
            "planning",
            "Query plan ready",
            {
                "query_count": len(queries),
                "planning_question": planning_question,
                "translation_mode": planner_metadata.get("translation_mode", "none"),
                "queries_by_provider": {
                    provider: len(provider_queries)
                    for provider, provider_queries in queries_by_provider.items()
                },
            },
        )

    retriever = retriever_agent or RetrieverAgent(config=config)
    if progress_callback:
        progress_callback(
            "retrieval",
            "Starting literature metadata retrieval",
            {
                "providers": providers,
                "query_count": sum(len(items) for items in queries_by_provider.values()),
                "max_per_query": max_per_query,
                "from_year": from_year,
                "use_cache": use_cache,
            },
        )
    raw_papers, raw_by_provider, retrieval_counts = retriever.retrieve(
        queries=queries_by_provider,
        providers=providers,
        max_per_query=max_per_query,
        from_year=from_year,
        output_dir=out,
        progress_callback=progress_callback,
        sort_mode=sort_preference,
    )
    raw_paper_count_before_year_filter = len(raw_papers)
    if progress_callback:
        progress_callback(
            "retrieval",
            "Retrieval finished",
            {
                "raw_paper_count": raw_paper_count_before_year_filter,
                "retrieval_counts_by_provider": retrieval_counts,
            },
        )
    raw_papers, year_filter_stats = filter_papers_by_from_year(raw_papers, from_year)
    if progress_callback:
        progress_callback(
            "retrieval",
            "Local year filter applied"
            if year_filter_stats.get("enabled")
            else "Local year filter skipped",
            year_filter_stats,
        )

    if progress_callback:
        progress_callback(
            "dedup",
            "Merging duplicate records by DOI and title",
            {"raw_paper_count": len(raw_papers)},
        )
    merged_papers, duplicate_count = deduplicate_with_stats(raw_papers)
    if progress_callback:
        progress_callback(
            "dedup",
            "Deduplication finished",
            {
                "merged_paper_count": len(merged_papers),
                "duplicate_count": duplicate_count,
            },
        )

    extractor = ExtractorAgent(mode=extractor_mode, llm_client=active_llm_client)
    if progress_callback:
        progress_callback(
            "extraction",
            "Extracting evidence sentences from abstracts",
            {
                "paper_count": len(merged_papers),
                "extractor_mode": extractor_mode,
            },
        )
    evidence_records = extractor.extract_many(merged_papers, planning_question)
    if progress_callback:
        missing_abstract_count = sum(1 for paper in merged_papers if not paper.abstract)
        progress_callback(
            "extraction",
            "Evidence extraction finished",
            {
                "evidence_record_count": len(evidence_records),
                "missing_abstract_count": missing_abstract_count,
            },
        )

    verifier = VerifierAgent(mode=verifier_mode, llm_client=active_llm_client)
    if progress_callback:
        progress_callback(
            "verification",
            "Validating evidence spans against abstracts",
            {
                "evidence_record_count": len(evidence_records),
                "verifier_mode": verifier_mode,
            },
        )
    verification_results = verifier.verify_many(merged_papers, evidence_records)
    if progress_callback:
        support_counts: dict[str, int] = {}
        for result in verification_results:
            support_counts[result.support_level] = support_counts.get(result.support_level, 0) + 1
        progress_callback(
            "verification",
            "Evidence grounding verification finished",
            {
                "verification_count": len(verification_results),
                "support_level_counts": support_counts,
            },
        )

    aspect_agent = AspectCoverageAgent()
    required_aspects = query_plan.required_aspects or search_brief.required_aspects
    if progress_callback:
        progress_callback(
            "aspect_coverage",
            "Classifying required-aspect coverage",
            {"required_aspects": required_aspects},
        )
    aspect_coverage_records = aspect_agent.classify_many(
        merged_papers,
        evidence_records,
        required_aspects,
    )
    if progress_callback:
        average_coverage = (
            sum(record.aspect_coverage_score for record in aspect_coverage_records)
            / len(aspect_coverage_records)
            if aspect_coverage_records
            else 0.0
        )
        progress_callback(
            "aspect_coverage",
            "Aspect coverage classification finished",
            {
                "paper_count": len(aspect_coverage_records),
                "average_aspect_coverage": average_coverage,
            },
        )

    ranker = RankerAgent()
    if progress_callback:
        progress_callback(
            "ranking",
            "Ranking papers with configured scoring weights",
            {"scoring_weights": active_scoring_weights},
        )
    ranked_before_feedback = ranker.rank(
        merged_papers,
        evidence_records,
        verification_results,
        planning_question,
        scoring_weights=active_scoring_weights,
        query_plan=query_plan,
        ranking_profile=ranking_profile,
        aspect_coverage_records=aspect_coverage_records,
    )
    if progress_callback:
        progress_callback(
            "ranking",
            "Initial ranking finished",
            {"ranked_paper_count": len(ranked_before_feedback)},
        )

    feedback_agent = HumanFeedbackAgent()
    ranked_after_feedback: list[RankedPaper] | None = None
    ranked_final = ranked_before_feedback
    feedback_applied = False
    if feedback_path:
        if progress_callback:
            progress_callback(
                "feedback",
                "Applying human feedback and reranking",
                {"feedback_path": feedback_path},
            )
        feedback_records = feedback_agent.read_feedback(feedback_path)
        ranked_after_feedback = feedback_agent.apply(
            ranked_before_feedback,
            feedback_records,
            scoring_weights=active_scoring_weights,
        )
        ranked_final = ranked_after_feedback
        feedback_applied = bool(feedback_records)
        if progress_callback:
            progress_callback(
                "feedback",
                "Feedback reranking finished",
                {"feedback_record_count": len(feedback_records)},
            )

    result_groups = group_ranked_papers(
        ranked_final,
        aspect_coverage_records,
        search_brief,
    )
    prisma_like_flow = build_prisma_like_flow(
        retrieval_counts=retrieval_counts,
        raw_paper_count=raw_paper_count_before_year_filter,
        merged_papers=merged_papers,
        duplicate_count=duplicate_count,
        verification_results=verification_results,
        ranked_papers=ranked_final,
    )

    if progress_callback:
        progress_callback(
            "evaluation",
            "Computing evaluation metrics",
            {"gold_labels_path": gold_labels_path or ""},
        )
    metrics = compute_evaluation(
        retrieval_counts=retrieval_counts,
        original_paper_count=raw_paper_count_before_year_filter,
        merged_papers=merged_papers,
        evidence_records=evidence_records,
        verification_results=verification_results,
        ranked_before_feedback=ranked_before_feedback,
        ranked_after_feedback=ranked_after_feedback,
        gold_labels_path=gold_labels_path,
    )
    metrics["duplicate_count"] = duplicate_count
    metrics["year_filter"] = year_filter_stats
    metrics["scoring_weights"] = active_scoring_weights
    metrics["query_controls"] = {
        "strictness": strictness,
        "openalex_mode": openalex_mode,
        "sort_preference": sort_preference,
        "ranking_profile": ranking_profile,
    }
    metrics["search_intent"] = search_brief.search_intent
    metrics["average_aspect_coverage"] = (
        sum(record.aspect_coverage_score for record in aspect_coverage_records)
        / len(aspect_coverage_records)
        if aspect_coverage_records
        else 0.0
    )
    metrics["prisma_like_flow"] = prisma_like_flow
    metrics["llm"] = _llm_metrics(
        llm_backend=llm_backend,
        active_llm_backend=active_llm_client.provider_name if active_llm_client else "none",
        planner_mode=planner_mode,
        extractor_mode=extractor_mode,
        verifier_mode=verifier_mode,
        planner_metadata=planner_metadata,
        evidence_records=evidence_records,
        verification_results=verification_results,
    )
    save_evaluation(out / "evaluation.json", metrics)
    if progress_callback:
        progress_callback(
            "artifacts",
            "Writing trace, CSV files, and Markdown report",
            {"output_dir": str(out)},
        )
    trace = build_agent_trace(
        question=question,
        planning_question=planning_question,
        queries=queries,
        query_plan=query_plan,
        search_brief=search_brief,
        question_refinement=question_refinement,
        planner_metadata=planner_metadata,
        retrieval_counts=retrieval_counts,
        raw_paper_count=raw_paper_count_before_year_filter,
        merged_papers=merged_papers,
        duplicate_count=duplicate_count,
        evidence_records=evidence_records,
        verification_results=verification_results,
        aspect_coverage_records=aspect_coverage_records,
        ranked_papers=ranked_final,
        scoring_weights=active_scoring_weights,
        result_groups=result_groups,
        year_filter_stats=year_filter_stats,
    )
    write_json(out / "agent_trace.json", trace)
    write_json(out / "result_groups.json", result_groups)
    write_json(out / "prisma_like_flow.json", prisma_like_flow)
    retrieval_diagnostics = build_retrieval_diagnostics(
        question=question,
        query_plan=query_plan,
        queries_by_provider=queries_by_provider,
        raw_by_provider=raw_by_provider,
        merged_papers=merged_papers,
        duplicate_count=duplicate_count,
        ranked_papers=ranked_final,
        year_filter_stats=year_filter_stats,
    )
    write_json(out / "retrieval_diagnostics.json", retrieval_diagnostics)
    generate_paper_cards(
        out / "paper_cards.md",
        ranked_final,
        aspect_coverage_records,
    )
    generate_reading_path(
        out / "reading_path.md",
        ranked_final,
        result_groups,
    )

    write_pipeline_csvs(
        out,
        merged_papers,
        evidence_records,
        verification_results,
        aspect_coverage_records,
        ranked_before_feedback,
        ranked_final,
        ranked_after_feedback,
    )

    generate_report(
        path=out / "report.md",
        research_question=question,
        planned_queries=queries,
        retrieval_statistics=metrics,
        ranked_papers=ranked_final,
        evidence_records=evidence_records,
        evaluation_metrics=metrics,
        feedback_applied=feedback_applied,
        search_brief=search_brief,
        question_refinement=question_refinement,
        query_plan=query_plan,
        aspect_coverage_records=aspect_coverage_records,
        result_groups=result_groups,
        reading_path_path=out / "reading_path.md",
        paper_cards_path=out / "paper_cards.md",
        prisma_like_flow=prisma_like_flow,
    )
    if progress_callback:
        progress_callback(
            "complete",
            "Literature screening complete",
            {
                "output_dir": str(out),
                "ranked_papers_path": str(out / "ranked_papers.csv"),
                "report_path": str(out / "report.md"),
            },
        )

    return PipelineResult(
        output_dir=str(out),
        planned_queries=queries,
        retrieval_counts=retrieval_counts,
        merged_paper_count=len(merged_papers),
        duplicate_count=duplicate_count,
        report_path=str(out / "report.md"),
        evaluation_path=str(out / "evaluation.json"),
        ranked_papers_path=str(out / "ranked_papers.csv"),
        question=question,
        planning_question=planning_question,
        translated_question=str(planner_metadata.get("translated_question") or ""),
        raw_paper_count=raw_paper_count_before_year_filter,
        merged_papers=merged_papers,
        evidence_records=evidence_records,
        verification_results=verification_results,
        ranked_before_feedback=ranked_before_feedback,
        ranked_after_feedback=ranked_after_feedback,
        ranked_final=ranked_final,
        evaluation_metrics=metrics,
        agent_trace=trace,
        scoring_weights=active_scoring_weights,
        query_plan=query_plan,
        search_brief=search_brief,
        question_refinement=question_refinement,
        aspect_coverage_records=aspect_coverage_records,
        result_groups=result_groups,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Run the literature-screening MVP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the screening pipeline.")
    run.add_argument("--question", required=True, help="Research question to screen for.")
    run.add_argument(
        "--providers",
        nargs="+",
        default=["openalex", "semantic_scholar"],
        help="Retrieval providers to use.",
    )
    run.add_argument("--max-per-query", type=int, default=10)
    run.add_argument("--from-year", type=int, default=None)
    run.add_argument("--feedback", dest="feedback_path", default=None)
    run.add_argument("--gold-labels", dest="gold_labels_path", default=None)
    run.add_argument("--output-dir", default="outputs")
    run.add_argument(
        "--llm-backend",
        choices=["none", "deepseek"],
        default="none",
        help="Optional OpenAI-compatible LLM backend.",
    )
    run.add_argument(
        "--planner-mode",
        choices=["rule", "llm"],
        default="rule",
        help="Use rule-based or LLM-enhanced query planning.",
    )
    run.add_argument(
        "--extractor-mode",
        choices=["rule", "llm"],
        default="rule",
        help="Use rule-based or LLM-enhanced evidence extraction.",
    )
    run.add_argument(
        "--verifier-mode",
        choices=["rule", "llm"],
        default="rule",
        help="Use rule-based or LLM-enhanced evidence verification.",
    )
    run.add_argument(
        "--strictness",
        choices=["strict", "balanced", "broad"],
        default="balanced",
        help="How narrowly the planner should constrain search terms.",
    )
    run.add_argument(
        "--openalex-mode",
        choices=["keyword", "semantic", "keyword+semantic"],
        default="keyword+semantic",
    )
    run.add_argument(
        "--sort-preference",
        choices=["relevance", "recent", "cited"],
        default="relevance",
    )
    run.add_argument(
        "--ranking-profile",
        choices=["relevance_first", "balanced", "high_quality_review"],
        default="balanced",
    )
    run.add_argument("--weight-relevance", type=float, default=None)
    run.add_argument("--weight-evidence", type=float, default=None)
    run.add_argument("--weight-recency", type=float, default=None)
    run.add_argument("--weight-quality", type=float, default=None)
    run.add_argument("--weight-diversity", type=float, default=None)
    run.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable local API response cache.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        weight_values = {
            "relevance": args.weight_relevance,
            "evidence": args.weight_evidence,
            "recency": args.weight_recency,
            "quality": args.weight_quality,
            "diversity": args.weight_diversity,
        }
        scoring_weights = {
            key: value for key, value in weight_values.items() if value is not None
        } or None
        try:
            result = run_pipeline(
                question=args.question,
                providers=args.providers,
                max_per_query=args.max_per_query,
                from_year=args.from_year,
                feedback_path=args.feedback_path,
                gold_labels_path=args.gold_labels_path,
                output_dir=args.output_dir,
                use_cache=not args.disable_cache,
                llm_backend=args.llm_backend,
                planner_mode=args.planner_mode,
                extractor_mode=args.extractor_mode,
                verifier_mode=args.verifier_mode,
                scoring_weights=scoring_weights,
                strictness=args.strictness,
                openalex_mode=args.openalex_mode,
                sort_preference=args.sort_preference,
                ranking_profile=args.ranking_profile,
            )
        except Exception as exc:
            ScreeningRunLogger(args.output_dir).log_exception(
                "fatal",
                exc,
                {"output_dir": args.output_dir},
            )
            raise
        print(f"Report: {result.report_path}")
        print(f"Ranking: {result.ranked_papers_path}")
        print(f"Evaluation: {result.evaluation_path}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
