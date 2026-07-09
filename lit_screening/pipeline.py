"""Command-line pipeline for the literature-screening MVP."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .agents.aspect_classifier import AspectCoverageAgent
from .agents.ambiguity_detector import AmbiguityDetectorAgent
from .agents.concept_mapper import ConceptMapper
from .agents.controversy_boundary import ControversyAndBoundaryAgent
from .agents.domain_guardrail import DomainGuardrailAgent
from .agents.evidence_function_classifier import classify_evidence_function
from .agents.extractor import ExtractorAgent
from .agents.generic_intent import is_single_acronym_query
from .agents.human_feedback import HumanFeedbackAgent
from .agents.intent_repair import NoviceIntentInterpreter
from .agents.llm_enhancement import (
    LLMFeedbackInterpreter,
    LLMIntentFrameEnhancer as AdvisoryLLMIntentFrameEnhancer,
    LLMQueryPlanCritic,
    LLMUserReportAdapter,
)
from .agents.planner import PlannerAgent
from .agents.paper_role_classifier import PaperRoleClassifier
from .agents.preference_learning import PreferenceLearningAgent
from .agents.query_family_planner import QueryFamilyPlanner
from .agents.query_pilot import QueryPilotAgent
from .agents.query_repair import QueryRepairAgent
from .agents.question_refiner import QuestionRefinementAgent
from .agents.ranker import RankerAgent
from .agents.research_intent import ResearchIntentAgent
from .agents.retriever import RetrieverAgent
from .agents.search_contract import SearchContractAgent
from .agents.screening_decision import (
    ScreeningDecisionAgent,
    summarize_reading_priority_policy,
    summarize_screening_decisions,
)
from .agents.seed_extraction import SeedExtractionAgent
from .agents.snowball import (
    CitationSnowballAgent,
    empty_seed_expansion_report,
    parse_seed_file,
    parse_seed_values,
    resolve_seed_inputs,
)
from .agents.verifier import VerifierAgent
from .config import PipelineConfig
from .dedup import deduplicate_with_stats
from .decision_artifacts import write_decision_artifacts
from .evaluation import compute_evaluation, save_evaluation
from .evaluation.exploration_quality import (
    compute_exploration_quality,
    save_exploration_quality,
)
from .importers import ImportResult, import_papers_from_file
from .llm_client import GenericLLMClient
from .llm.intent_frame_enhancer import (
    FakeLLMIntentProvider,
    GenericLLMIntentProvider,
    LLMIntentFrameEnhancer as OptionalLLMIntentFrameEnhancer,
    LLMIntentProvider,
    apply_verified_suggestions_to_search_contract,
    write_intent_enhancement_artifacts,
)
from .llm.query_plan_critic import (
    FakeLLMQueryPlanCriticProvider,
    LLMQueryPlanCritic as OptionalLLMQueryPlanCritic,
    LLMQueryPlanCriticProvider,
    apply_verified_query_critic_issues_to_query_plan,
    empty_query_critic_repair_result,
    write_query_plan_critic_artifacts,
)
from .models import (
    AspectCoverageRecord,
    DomainAssessment,
    EvidenceRecord,
    ExpertResearchIntent,
    FeedbackRecord,
    Paper,
    PaperRoleRecord,
    PipelineResult,
    PreferenceLearningResult,
    QueryFamily,
    QueryPlan,
    QueryFamilyPlan,
    RankedPaper,
    RetrievalPath,
    ResearchTension,
    ResearchLensPlan,
    SearchBrief,
    SearchContract,
    SeedHint,
    SeedPaper,
    ScreeningDecision,
    VerificationResult,
    is_user_seed_paper,
)
from .paper_cards import generate_paper_cards
from .reading_path import generate_reading_path
from .report import generate_report
from .result_groups import group_ranked_papers
from .run_logging import ScreeningRunLogger
from .scoring import sanitize_score_weights
from .screening_flow import build_prisma_like_flow
from .utils import ensure_dir, tokenize, write_csv, write_json


PAPER_FIELDS = [
    "paper_id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_provider",
    "retrieval_provider",
    "retrieval_stage",
    "retrieval_query",
    "source_stage",
    "seed_paper_id",
    "seed_title",
    "seed_reason",
    "seed_relation",
    "seed_confidence",
    "citation_count",
]

EVIDENCE_FIELDS = [
    "paper_id",
    "title",
    "claim",
    "evidence_sentence",
    "evidence_function",
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
    "retrieval_provider",
    "retrieval_stage",
    "retrieval_query",
    "source_stage",
    "seed_paper_id",
    "seed_title",
    "seed_reason",
    "seed_relation",
    "seed_confidence",
    "citation_count",
    "relevance_score",
    "evidence_score",
    "recency_score",
    "quality_score",
    "diversity_score",
    "aspect_coverage_score",
    "intent_centrality_score",
    "required_group_coverage_score",
    "missing_required_group_count",
    "group_coverage_explanation",
    "matched_groups",
    "missing_groups",
    "target_context_match",
    "negative_context_match",
    "topic_focus_score",
    "aspect_match_score",
    "target_context_score",
    "negative_context_penalty",
    "peripheral_context_reason",
    "domain_match_score",
    "domain_decision",
    "off_topic_reason",
    "domain_penalty_multiplier",
    "pre_domain_final_score",
    "human_feedback_adjustment",
    "preference_score",
    "preference_adjustment",
    "final_score",
    "decision",
    "decision_confidence",
    "primary_reason",
    "reading_priority",
    "suggested_action",
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
    "evidence_function",
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

SCREENING_DECISION_FIELDS = [
    "paper_id",
    "decision",
    "decision_confidence",
    "primary_reason",
    "exclusion_reasons",
    "required_aspects_covered",
    "required_aspects_missing",
    "domain_match_score",
    "domain_decision",
    "reading_priority",
    "suggested_action",
]

RETRIEVAL_PATH_FIELDS = [
    "paper_id",
    "source_stage",
    "seed_paper_id",
    "seed_title",
    "seed_relation",
    "seed_confidence",
    "reason",
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
        "retrieval_provider": paper.retrieval_provider,
        "retrieval_stage": paper.retrieval_stage,
        "retrieval_query": paper.retrieval_query,
        "source_stage": paper.source_stage,
        "seed_paper_id": paper.seed_paper_id,
        "seed_title": paper.seed_title,
        "seed_reason": paper.seed_reason,
        "seed_relation": paper.seed_relation,
        "seed_confidence": f"{paper.seed_confidence:.4f}" if paper.seed_confidence else "",
        "citation_count": paper.citation_count,
    }


def evidence_function_value(evidence: EvidenceRecord) -> str:
    """Return evidence function as a stable string."""

    value = evidence.evidence_function
    return getattr(value, "value", str(value or "unknown"))


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
        "evidence_function": evidence_function_value(evidence),
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
    domain = item.domain_assessment
    decision = item.screening_decision
    return {
        "rank": item.rank,
        "paper_id": item.paper.paper_id,
        "title": item.paper.title,
        "year": item.paper.year or "",
        "venue": item.paper.venue,
        "doi": item.paper.doi,
        "url": item.paper.url,
        "source_provider": item.paper.source_provider,
        "retrieval_provider": item.paper.retrieval_provider,
        "retrieval_stage": item.paper.retrieval_stage,
        "retrieval_query": item.paper.retrieval_query,
        "source_stage": item.paper.source_stage,
        "seed_paper_id": item.paper.seed_paper_id,
        "seed_title": item.paper.seed_title,
        "seed_reason": item.paper.seed_reason,
        "seed_relation": item.paper.seed_relation,
        "seed_confidence": f"{item.paper.seed_confidence:.4f}" if item.paper.seed_confidence else "",
        "citation_count": item.paper.citation_count,
        "relevance_score": f"{item.scores.relevance_score:.4f}",
        "evidence_score": f"{item.scores.evidence_score:.4f}",
        "recency_score": f"{item.scores.recency_score:.4f}",
        "quality_score": f"{item.scores.quality_score:.4f}",
        "diversity_score": f"{item.scores.diversity_score:.4f}",
        "aspect_coverage_score": f"{item.scores.aspect_coverage_score:.4f}",
        "intent_centrality_score": f"{item.scores.intent_centrality_score:.4f}",
        "required_group_coverage_score": (
            f"{domain.required_group_coverage_score:.4f}" if domain else ""
        ),
        "missing_required_group_count": (
            domain.missing_required_group_count if domain else ""
        ),
        "group_coverage_explanation": (
            "; ".join(domain.group_coverage_explanation) if domain else ""
        ),
        "matched_groups": "; ".join(domain.matched_groups) if domain else "",
        "missing_groups": "; ".join(domain.missing_groups) if domain else "",
        "target_context_match": (
            "; ".join(domain.target_context_match) if domain else ""
        ),
        "negative_context_match": (
            "; ".join(domain.negative_context_match) if domain else ""
        ),
        "topic_focus_score": f"{domain.topic_focus_score:.4f}" if domain else "",
        "aspect_match_score": f"{domain.aspect_match_score:.4f}" if domain else "",
        "target_context_score": f"{domain.target_context_score:.4f}" if domain else "",
        "negative_context_penalty": (
            f"{domain.negative_context_penalty:.4f}" if domain else ""
        ),
        "peripheral_context_reason": domain.peripheral_context_reason if domain else "",
        "domain_match_score": f"{domain.domain_match_score:.4f}" if domain else "",
        "domain_decision": domain.domain_decision if domain else "",
        "off_topic_reason": domain.off_topic_reason if domain else "",
        "domain_penalty_multiplier": f"{item.scores.domain_penalty_multiplier:.4f}",
        "pre_domain_final_score": f"{item.scores.pre_domain_final_score:.4f}",
        "human_feedback_adjustment": f"{item.scores.human_feedback_adjustment:.4f}",
        "preference_score": f"{item.scores.preference_score:.4f}",
        "preference_adjustment": f"{item.scores.preference_adjustment:.4f}",
        "final_score": f"{item.scores.final_score:.4f}",
        "decision": decision.decision if decision else "",
        "decision_confidence": f"{decision.decision_confidence:.4f}" if decision else "",
        "primary_reason": decision.primary_reason if decision else "",
        "reading_priority": decision.reading_priority if decision else "",
        "suggested_action": decision.suggested_action if decision else "",
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
        "evidence_function": evidence_function_value(item.evidence),
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


def screening_decision_to_row(record: ScreeningDecision) -> dict[str, Any]:
    """Convert a ScreeningDecision into a flat CSV row."""

    return {
        "paper_id": record.paper_id,
        "decision": record.decision,
        "decision_confidence": f"{record.decision_confidence:.4f}",
        "primary_reason": record.primary_reason,
        "exclusion_reasons": "; ".join(record.exclusion_reasons),
        "required_aspects_covered": "; ".join(record.required_aspects_covered),
        "required_aspects_missing": "; ".join(record.required_aspects_missing),
        "domain_match_score": f"{record.domain_match_score:.4f}",
        "domain_decision": record.domain_decision,
        "reading_priority": record.reading_priority,
        "suggested_action": record.suggested_action,
    }


def retrieval_path_to_row(record: RetrievalPath) -> dict[str, Any]:
    """Convert a RetrievalPath into a flat CSV row."""

    return {
        "paper_id": record.paper_id,
        "source_stage": record.source_stage,
        "seed_paper_id": record.seed_paper_id,
        "seed_title": record.seed_title,
        "seed_relation": record.seed_relation,
        "seed_confidence": f"{record.seed_confidence:.4f}" if record.seed_confidence else "",
        "reason": record.reason,
    }


def write_preference_learning_outputs(
    output_dir: Path,
    preference_learning: PreferenceLearningResult,
    feedback_query_refinement: dict[str, Any],
) -> None:
    """Write learned preference and feedback query-refinement artifacts."""

    write_json(output_dir / "preference_learning.json", preference_learning)
    write_json(output_dir / "feedback_query_refinement.json", feedback_query_refinement)


def preference_learning_metrics(
    preference_learning: PreferenceLearningResult,
) -> dict[str, Any]:
    """Build compact metrics for learned feedback preferences."""

    return {
        "enabled": preference_learning.enabled,
        "model_type": preference_learning.model_type,
        "labeled_paper_count": preference_learning.labeled_paper_count,
        "include_count": preference_learning.include_count,
        "exclude_count": preference_learning.exclude_count,
        "positive_terms": preference_learning.positive_terms,
        "negative_terms": preference_learning.negative_terms,
        "suggested_must_terms": preference_learning.suggested_must_terms,
        "suggested_optional_terms": preference_learning.suggested_optional_terms,
        "suggested_exclude_terms": preference_learning.suggested_exclude_terms,
        "note": preference_learning.note,
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
    screening_decisions: list[ScreeningDecision],
    seed_papers: list[SeedPaper],
    citation_expansion_papers: list[Paper],
    retrieval_paths: list[RetrievalPath],
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
        output_dir / "screening_decisions.csv",
        [screening_decision_to_row(record) for record in screening_decisions],
        SCREENING_DECISION_FIELDS,
    )
    write_json(output_dir / "screening_decisions.json", screening_decisions)
    write_json(output_dir / "seed_papers.json", seed_papers)
    write_csv(
        output_dir / "citation_expansion.csv",
        [paper_to_row(paper) for paper in citation_expansion_papers],
        PAPER_FIELDS,
    )
    write_csv(
        output_dir / "retrieval_paths.csv",
        [retrieval_path_to_row(record) for record in retrieval_paths],
        RETRIEVAL_PATH_FIELDS,
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
    search_contract: SearchContract | None,
    ambiguity_analysis: list[dict[str, Any]] | None,
    question_refinement: dict[str, Any] | None,
    planner_metadata: dict[str, Any],
    retrieval_counts: dict[str, int],
    raw_paper_count: int,
    merged_papers: list[Paper],
    duplicate_count: int,
    evidence_records: list[EvidenceRecord],
    verification_results: list[VerificationResult],
    aspect_coverage_records: list[AspectCoverageRecord],
    domain_assessments: list[DomainAssessment],
    screening_decisions: list[ScreeningDecision] | None,
    preference_learning: PreferenceLearningResult | None,
    feedback_query_refinement: dict[str, Any] | None,
    ranked_papers: list[RankedPaper],
    scoring_weights: dict[str, float],
    result_groups: dict[str, Any] | None = None,
    year_filter_stats: dict[str, Any] | None = None,
    import_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an inspectable trace of agent decisions for demos and audits."""

    verification_by_id = {result.paper_id: result for result in verification_results}
    domain_by_id = {result.paper_id: result for result in domain_assessments}
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
        "ambiguity_detector": {
            "analysis": ambiguity_analysis or [],
            "decision": "Identified ambiguous terms before retrieval.",
        },
        "search_contract": {
            "contract": search_contract,
            "decision": "Bound retrieval to the intended domain, required concepts, and exclusions.",
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
            "imported_library": import_diagnostics or {},
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
        "domain_guardrail": [
            {
                "paper_id": record.paper_id,
                "domain_match_score": record.domain_match_score,
                "domain_decision": record.domain_decision,
                "off_topic_reason": record.off_topic_reason,
                "positive_domain_matches": record.positive_domain_matches,
                "negative_domain_matches": record.negative_domain_matches,
                "missing_required_concepts": record.missing_required_concepts,
                "forbidden_concepts_found": record.forbidden_concepts_found,
            }
            for record in domain_assessments[:50]
        ],
        "screening_decision": [
            {
                "paper_id": record.paper_id,
                "decision": record.decision,
                "decision_confidence": record.decision_confidence,
                "primary_reason": record.primary_reason,
                "exclusion_reasons": record.exclusion_reasons,
                "required_aspects_covered": record.required_aspects_covered,
                "required_aspects_missing": record.required_aspects_missing,
                "domain_match_score": record.domain_match_score,
                "domain_decision": record.domain_decision,
                "reading_priority": record.reading_priority,
                "suggested_action": record.suggested_action,
            }
            for record in (screening_decisions or [])[:50]
        ],
        "preference_learning": {
            "learning": preference_learning,
            "feedback_query_refinement": feedback_query_refinement or {},
            "decision": "Learned positive and negative preference terms from human feedback.",
        },
        "ranker": [
            {
                "rank": item.rank,
                "paper_id": item.paper.paper_id,
                "title": item.paper.title,
                "final_score": item.scores.final_score,
                "pre_domain_final_score": item.scores.pre_domain_final_score,
                "domain_penalty_multiplier": item.scores.domain_penalty_multiplier,
                "preference_score": item.scores.preference_score,
                "preference_adjustment": item.scores.preference_adjustment,
                "domain_decision": domain_by_id[item.paper.paper_id].domain_decision
                if item.paper.paper_id in domain_by_id
                else "",
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

    screening_decision_agent = ScreeningDecisionAgent()
    _, ranked_before_feedback = screening_decision_agent.decide_many(
        result.ranked_before_feedback,
        result.aspect_coverage_records,
    )
    preference_agent = PreferenceLearningAgent()
    preference_learning = preference_agent.learn(
        ranked_before_feedback,
        feedback_records,
        result.search_contract,
    )
    feedback_query_refinement = preference_agent.query_refinement_payload(
        preference_learning
    )
    feedback_agent = HumanFeedbackAgent()
    ranked_after_feedback = feedback_agent.apply(
        ranked_before_feedback,
        feedback_records,
        scoring_weights=result.scoring_weights,
        preference_scores=preference_learning.preference_scores
        if preference_learning.enabled
        else None,
    )
    screening_decisions, ranked_after_feedback = screening_decision_agent.decide_many(
        ranked_after_feedback,
        result.aspect_coverage_records,
    )
    metrics = compute_evaluation(
        retrieval_counts=result.retrieval_counts,
        original_paper_count=result.raw_paper_count,
        merged_papers=result.merged_papers,
        evidence_records=result.evidence_records,
        verification_results=result.verification_results,
        ranked_before_feedback=ranked_before_feedback,
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
    if "imported_library" in result.evaluation_metrics:
        metrics["imported_library"] = result.evaluation_metrics["imported_library"]
    metrics["scoring_weights"] = result.scoring_weights
    metrics["domain_guardrails"] = build_domain_guardrail_summary(
        result.domain_assessments,
        ranked_after_feedback,
    )
    metrics["query_pilot"] = result.evaluation_metrics.get("query_pilot", {})
    metrics["query_repair"] = result.evaluation_metrics.get("query_repair", {})
    metrics["seed_paper_expansion"] = result.evaluation_metrics.get(
        "seed_paper_expansion",
        {},
    )
    metrics["screening_decisions"] = summarize_screening_decisions(
        screening_decisions
    )
    reading_priority_policy = summarize_reading_priority_policy(ranked_after_feedback)
    metrics.update(reading_priority_policy)
    metrics["reading_priority_policy"] = reading_priority_policy
    metrics["paper_roles"] = summarize_paper_roles(result.paper_role_records)
    metrics["preference_learning"] = preference_learning_metrics(preference_learning)
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
        screening_decisions=screening_decisions,
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
    trace["preference_learning"] = {
        "learning": preference_learning,
        "feedback_query_refinement": feedback_query_refinement,
        "decision": "Learned relevance preferences from human feedback without rerunning retrieval.",
    }
    trace["screening_decision"] = [
        {
            "paper_id": record.paper_id,
            "decision": record.decision,
            "decision_confidence": record.decision_confidence,
            "primary_reason": record.primary_reason,
            "exclusion_reasons": record.exclusion_reasons,
            "required_aspects_covered": record.required_aspects_covered,
            "required_aspects_missing": record.required_aspects_missing,
            "domain_match_score": record.domain_match_score,
            "domain_decision": record.domain_decision,
            "reading_priority": record.reading_priority,
            "suggested_action": record.suggested_action,
        }
        for record in screening_decisions[:50]
    ]

    output_dir = Path(result.output_dir)
    save_evaluation(output_dir / "evaluation.json", metrics)
    write_json(output_dir / "agent_trace.json", trace)
    write_json(output_dir / "result_groups.json", result_groups)
    write_json(output_dir / "prisma_like_flow.json", prisma_like_flow)
    write_json(output_dir / "domain_assessments.json", result.domain_assessments)
    write_preference_learning_outputs(
        output_dir,
        preference_learning,
        feedback_query_refinement,
    )
    generate_paper_cards(
        output_dir / "paper_cards.md",
        ranked_after_feedback,
        result.aspect_coverage_records,
    )
    reading_path_diagnostics = generate_reading_path(
        output_dir / "reading_path.md",
        ranked_after_feedback,
        result_groups,
    )
    metrics.update(reading_path_diagnostics)
    metrics["reading_path_diagnostics"] = reading_path_diagnostics
    decision_artifacts = write_decision_artifacts(
        output_dir,
        ranked_after_feedback,
        result.aspect_coverage_records,
        search_contract=result.search_contract,
        query_plan=result.query_plan,
        query_pilot_diagnostics=result.query_pilot_diagnostics,
        prisma_like_flow=prisma_like_flow,
    )
    metrics["decision_artifacts"] = {
        "method_comparison_rows": len(decision_artifacts["method_comparison_matrix"]),
        "research_gap_rows": len(decision_artifacts["research_gap_matrix"]),
        "suggested_next_search_count": len(decision_artifacts["suggested_next_searches"]),
    }
    exploration_quality = compute_exploration_quality(
        concept_map=result.concept_map,
        query_families=result.query_family_plan,
        paper_roles=result.paper_role_records,
        evidence_functions=result.evidence_records,
        gap_matrix=decision_artifacts["research_gap_matrix"],
        research_tensions=result.research_tensions,
        seed_hints=result.seed_hints,
        query_provenance=result.query_provenance,
        aspect_coverage_records=result.aspect_coverage_records,
        ranking_diagnostics=result.evaluation_metrics.get("ranking_diagnostics", {}),
    )
    save_exploration_quality(
        output_dir / "exploration_quality.json",
        exploration_quality,
    )
    save_evaluation(output_dir / "evaluation.json", metrics)
    write_pipeline_csvs(
        output_dir,
        result.merged_papers,
        result.evidence_records,
        result.verification_results,
        result.aspect_coverage_records,
        screening_decisions,
        result.seed_papers,
        result.citation_expansion_papers,
        result.retrieval_paths,
        ranked_before_feedback,
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
        search_contract=result.search_contract,
        ambiguity_analysis=result.ambiguity_analysis,
        domain_assessments=result.domain_assessments,
        query_pilot_diagnostics=result.query_pilot_diagnostics,
        query_repair_suggestions=result.query_repair_suggestions,
        question_refinement=result.question_refinement,
        query_plan=result.query_plan,
        aspect_coverage_records=result.aspect_coverage_records,
        result_groups=result_groups,
        reading_path_path=output_dir / "reading_path.md",
        paper_cards_path=output_dir / "paper_cards.md",
        prisma_like_flow=prisma_like_flow,
        screening_decisions=screening_decisions,
        method_comparison_matrix=decision_artifacts["method_comparison_matrix"],
        research_gap_matrix=decision_artifacts["research_gap_matrix"],
        suggested_next_searches=decision_artifacts["suggested_next_searches"],
        preference_learning=preference_learning,
        feedback_query_refinement=feedback_query_refinement,
        seed_papers=result.seed_papers,
        retrieval_paths=result.retrieval_paths,
        citation_expansion_papers=result.citation_expansion_papers,
    )

    return replace(
        result,
        ranked_before_feedback=ranked_before_feedback,
        ranked_after_feedback=ranked_after_feedback,
        ranked_final=ranked_after_feedback,
        evaluation_metrics=metrics,
        agent_trace=trace,
        result_groups=result_groups,
        screening_decisions=screening_decisions,
        preference_learning=preference_learning,
        feedback_query_refinement=feedback_query_refinement,
    )


def _query_plan_payload(
    question: str,
    query_plan: QueryPlan,
    planner_metadata: dict[str, Any],
    search_contract: SearchContract | None = None,
    expert_research_intent: ExpertResearchIntent | None = None,
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
        "search_contract": search_contract,
        "expert_research_intent": expert_research_intent,
        "selected_domain": search_contract.domain_profile.domain_name
        if search_contract
        else "",
        "candidate_domains": search_contract.domain_profile.candidate_domains
        if search_contract
        else [],
        "generic_intent_frame": search_contract.generic_intent_frame
        if search_contract
        else None,
        "expert_rewritten_question": query_plan.expert_rewritten_question,
        "intent_assumptions": query_plan.intent_assumptions,
        "downweighted_user_terms": query_plan.downweighted_user_terms,
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
        expert_rewritten_question=str(data.get("expert_rewritten_question") or ""),
        intent_assumptions=list(data.get("intent_assumptions", [])),
        downweighted_user_terms=list(data.get("downweighted_user_terms", [])),
        query_families_applied=bool(data.get("query_families_applied", False)),
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


def _apply_expert_intent_to_search_brief(
    search_brief: SearchBrief,
    expert_research_intent: ExpertResearchIntent | None,
) -> SearchBrief:
    """Use repaired intent as the planning-facing brief while preserving provenance."""

    if expert_research_intent is None:
        return search_brief
    structured_concepts = expert_research_intent.structured_concepts
    inclusion = _unique_strings(
        [
            *search_brief.inclusion_criteria,
            *expert_research_intent.target_objects[:4],
            *expert_research_intent.mechanisms[:4],
            *expert_research_intent.methods[:4],
        ]
    )
    active_categories = {
        concept.category
        for concept in structured_concepts
        if concept.should_use_in_provider_query
        and concept.query_role in {"must", "optional"}
    }
    generic_aspects = []
    if {"object", "property", "mechanism"} & active_categories:
        generic_aspects.append("core concept coverage")
    if "method" in active_categories:
        generic_aspects.append("method or measurement evidence")
    if "material" in active_categories:
        generic_aspects.append("representative case coverage")
    if "application" in active_categories:
        generic_aspects.append("application or motivation bridge")
    required_aspects = _unique_strings(
        [*search_brief.required_aspects, *generic_aspects]
    )
    preferred_types = _unique_strings(
        [
            *search_brief.preferred_paper_types,
            *(
                ["methodological paper"]
                if "method" in active_categories
                else []
            ),
            *(
                ["theoretical paper"]
                if {"mechanism", "property"} & active_categories
                else []
            ),
        ]
    )
    success_definition = (
        search_brief.success_definition
        or "A useful result set should cover the selected concept groups while treating uncertain or broad motivation terms as assumptions, not hard query constraints."
    )
    return replace(
        search_brief,
        refined_question=expert_research_intent.expert_rewritten_question
        or search_brief.refined_question,
        user_goal=expert_research_intent.inferred_goal or search_brief.user_goal,
        inclusion_criteria=inclusion,
        required_aspects=required_aspects,
        preferred_paper_types=preferred_types,
        success_definition=success_definition,
    )


def build_research_lens_artifacts(
    question: str,
    search_brief: SearchBrief,
    search_contract: SearchContract,
    output_dir: Path,
    seed_hints: list[SeedHint] | None = None,
    progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> tuple[ResearchLensPlan | None, QueryFamilyPlan | None, dict[str, Any]]:
    """Optionally write concept-map and query-family artifacts.

    These artifacts are explanatory only. They do not replace the existing
    QueryPlan or alter retrieval queries.
    """

    domain = search_contract.domain_profile.domain_name if search_contract else ""
    trace_summary: dict[str, Any] = {
        "domain": domain,
        "generic_fallback_used": domain in {"", "general_science", "generic"},
        "domain_pack_enhancement_used": domain not in {"", "general_science", "generic"},
        "concept_mapper": {
            "executed": False,
            "skipped": False,
            "lens_count": 0,
            "warning": "",
        },
        "query_family_planner": {
            "executed": False,
            "skipped": False,
            "query_family_count": 0,
            "warning": "",
        },
    }

    try:
        concept_map = ConceptMapper().map_question(
            question=question,
            domain=domain,
            search_brief=search_brief,
            search_contract=search_contract,
            seed_hints=seed_hints,
        )
        write_json(output_dir / "concept_map.json", concept_map)
        trace_summary["concept_mapper"]["executed"] = True
        trace_summary["concept_mapper"]["lens_count"] = len(concept_map.lenses)
    except Exception as exc:
        warning = str(exc)[:240]
        trace_summary["concept_mapper"]["skipped"] = True
        trace_summary["concept_mapper"]["warning"] = warning
        trace_summary["query_family_planner"]["skipped"] = True
        trace_summary["query_family_planner"]["warning"] = (
            "Skipped because ConceptMapper failed."
        )
        trace_summary["reason"] = "concept_mapper_failed"
        if progress_callback:
            progress_callback(
                "research_lens",
                "ConceptMapper failed; continuing with the existing query planner",
                {
                    "domain": domain,
                    "warning": warning,
                },
            )
        return None, None, trace_summary

    try:
        query_family_plan = QueryFamilyPlanner().plan(
            concept_map,
            seed_hints=seed_hints,
        )
        write_json(output_dir / "query_families.json", query_family_plan)
        trace_summary["query_family_planner"]["executed"] = True
        trace_summary["query_family_planner"]["query_family_count"] = len(
            query_family_plan.families
        )
        return concept_map, query_family_plan, trace_summary
    except Exception as exc:
        warning = str(exc)[:240]
        trace_summary["query_family_planner"]["skipped"] = True
        trace_summary["query_family_planner"]["warning"] = warning
        trace_summary["reason"] = "query_family_planner_failed"
        if progress_callback:
            progress_callback(
                "research_lens",
                "QueryFamilyPlanner failed; continuing with the existing query planner",
                {
                    "domain": domain,
                    "warning": warning,
                },
            )
        return concept_map, None, trace_summary


def _make_query_plan(
    question: str,
    planner_mode: str,
    active_llm_client: GenericLLMClient | None,
    strictness: str = "balanced",
    openalex_mode: str = "keyword+semantic",
    sort_preference: str = "relevance",
    ranking_profile: str = "balanced",
    search_brief: Any | None = None,
    search_contract: SearchContract | None = None,
    expert_research_intent: ExpertResearchIntent | None = None,
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
        search_contract=search_contract,
        expert_intent=expert_research_intent,
    )
    return _query_plan_payload(
        question,
        query_plan,
        planner.last_llm_metadata,
        search_contract=search_contract,
        expert_research_intent=expert_research_intent,
    )


def plan_screening_queries(
    question: str,
    llm_backend: str = "none",
    planner_mode: str = "rule",
    strictness: str = "balanced",
    openalex_mode: str = "keyword+semantic",
    sort_preference: str = "relevance",
    ranking_profile: str = "balanced",
    llm_client: GenericLLMClient | None = None,
    intent_repair: bool = True,
    legacy_query_planning: bool = False,
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
    seed_hints = SeedExtractionAgent().extract(question)
    expert_research_intent = None
    if intent_repair and not legacy_query_planning:
        expert_research_intent = NoviceIntentInterpreter().repair(
            question,
            seed_hints=seed_hints,
            llm_client=active_llm_client,
            use_llm=True,
        )
        search_brief = _apply_expert_intent_to_search_brief(
            search_brief,
            expert_research_intent,
        )
    refinement = QuestionRefinementAgent().refine(question, search_brief)
    ambiguity_analysis = AmbiguityDetectorAgent().analyze(question)
    search_contract = SearchContractAgent(
        mode=planner_mode,
        llm_client=active_llm_client,
    ).build(
        question,
        search_brief=search_brief,
        ambiguity_analysis=ambiguity_analysis,
        expert_intent=expert_research_intent,
    )
    payload = _make_query_plan(
        question,
        planner_mode,
        active_llm_client,
        strictness=strictness,
        openalex_mode=openalex_mode,
        sort_preference=sort_preference,
        ranking_profile=ranking_profile,
        search_brief=search_brief,
        search_contract=search_contract,
        expert_research_intent=expert_research_intent,
    )
    payload["search_brief"] = search_brief
    payload["search_contract"] = search_contract
    payload["ambiguity_analysis"] = ambiguity_analysis
    payload["question_refinement"] = refinement
    payload["seed_hints"] = seed_hints
    payload["expert_research_intent"] = expert_research_intent
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


def _query_provenance_record(
    provider: str,
    raw_query: str,
    source: str,
    family_name: str = "",
    lens_name: str = "",
    purpose: str = "",
    priority: int | None = None,
    budget: int | None = None,
    concepts: list[dict[str, Any]] | None = None,
    provider_serializer_changes: list[str] | None = None,
    domain_pack_enhanced: bool = False,
) -> dict[str, Any]:
    """Return one serializable query provenance record."""

    record = {
        "provider": provider,
        "raw_query": " ".join(str(raw_query).split()),
        "source": source,
        "family_name": family_name,
        "lens_name": lens_name,
        "purpose": purpose,
        "concepts": concepts or [],
        "concept_sources": _unique_strings(
            [
                source
                for concept in concepts or []
                for source in concept.get("sources", [])
                if isinstance(concept, dict)
            ]
        ),
        "provider_serializer_changes": provider_serializer_changes or [],
        "domain_pack_enhanced": bool(domain_pack_enhanced),
    }
    if priority is not None:
        record["priority"] = priority
    if budget is not None:
        record["budget"] = budget
    return record


def _family_queries_for_provider(family: QueryFamily, provider: str) -> list[str]:
    """Return provider-specific family queries with a fallback for fake clients."""

    provider_queries = family.queries_by_provider.get(provider)
    if provider_queries:
        return _unique_strings(provider_queries)
    fallback_queries: list[str] = []
    for fallback_provider in ["openalex", "semantic_scholar"]:
        fallback_queries.extend(family.queries_by_provider.get(fallback_provider, []))
    for family_provider, queries in family.queries_by_provider.items():
        if family_provider not in {"openalex", "semantic_scholar"}:
            fallback_queries.extend(queries)
    return _unique_strings(fallback_queries)


def _remove_single_acronym_queries(queries: list[str]) -> tuple[list[str], list[str]]:
    """Remove provider queries that are only one ambiguous acronym/short token."""

    kept: list[str] = []
    removed: list[str] = []
    for query in _unique_strings(queries):
        if is_single_acronym_query(query):
            removed.append(query)
            continue
        kept.append(query)
    return kept, removed


def _final_query_quality(
    query: str,
    search_contract: SearchContract | None,
    source: str,
) -> dict[str, Any]:
    """Return a deterministic keep/drop decision for final provider queries."""

    cleaned = " ".join(str(query or "").split())
    if not cleaned:
        return {"keep": False, "reason": "empty_query"}
    if source == "seed_context":
        return {"keep": True, "reason": "seed_context"}
    if is_single_acronym_query(cleaned):
        return {"keep": False, "reason": "single_acronym_only_query"}
    frame = search_contract.generic_intent_frame if search_contract else None
    if frame is None:
        return {"keep": True, "reason": "no_generic_frame_available"}
    critic_issue = _rule_query_critic_issue(cleaned, search_contract)
    if critic_issue:
        return {
            "keep": False,
            "reason": "accepted_rule_query_critic_issue",
            "accepted_critic_issue": critic_issue,
        }
    core_terms = _anchor_candidates(
        [
            *frame.research_object,
            *frame.core_terms,
        ],
        drop_aspect_only=True,
    )
    secondary_terms = _anchor_candidates(
        [
            *frame.domain_context,
            *frame.target_process_or_property,
            *frame.material_or_case_terms,
            *frame.application_or_metric_terms,
            *frame.failure_or_limitation_terms,
            *frame.controversy_terms,
        ],
        drop_aspect_only=False,
    )
    core_lower = {_normalized_query_text(term) for term in core_terms}
    secondary_terms = [
        term
        for term in secondary_terms
        if _normalized_query_text(term) not in core_lower
    ]
    generic_need_terms = _anchor_candidates(
        [
            *frame.method_terms,
            *frame.mechanism_terms,
            *frame.relation_or_effect,
            *frame.downweighted_terms,
        ],
        drop_aspect_only=False,
    )
    has_core = _query_contains_any(cleaned, core_terms)
    has_secondary = _query_contains_any(cleaned, secondary_terms)
    core_hit_count = _query_anchor_hit_count(cleaned, core_terms)
    has_need = _query_contains_any(cleaned, generic_need_terms)
    has_only_need = _query_contains_any(cleaned, generic_need_terms) and not has_core
    missing_required_query_groups = [
        group_name
        for group_name, group_terms in _required_context_or_process_groups(search_contract)
        if group_terms and not _query_contains_any(cleaned, group_terms)
    ]
    if missing_required_query_groups:
        return {
            "keep": False,
            "reason": "missing_required_context_or_process_anchor",
            "missing_required_query_groups": missing_required_query_groups,
            "core_anchor_terms": core_terms,
            "secondary_anchor_terms": secondary_terms,
        }
    if has_only_need:
        return {
            "keep": False,
            "reason": "method_or_need_only_query",
            "core_anchor_terms": core_terms,
            "secondary_anchor_terms": secondary_terms,
        }
    if core_terms and not has_core:
        return {
            "keep": False,
            "reason": "missing_core_object_anchor",
            "core_anchor_terms": core_terms,
            "secondary_anchor_terms": secondary_terms,
        }
    if secondary_terms and not has_secondary:
        return {
            "keep": False,
            "reason": "missing_context_or_process_anchor",
            "core_anchor_terms": core_terms,
            "secondary_anchor_terms": secondary_terms,
        }
    if _is_thin_film_method_comparison_query(cleaned, search_contract):
        return {
            "keep": True,
            "reason": "thin_film_method_comparison_anchor_passed",
            "core_anchor_terms": core_terms,
            "secondary_anchor_terms": secondary_terms,
        }
    if not secondary_terms and core_hit_count < 2 and not has_need:
        return {
            "keep": False,
            "reason": "missing_second_anchor",
            "core_anchor_terms": core_terms,
            "secondary_anchor_terms": secondary_terms,
        }
    if not secondary_terms and _looks_broad_standalone_query(cleaned):
        return {
            "keep": False,
            "reason": "broad_standalone_query",
            "core_anchor_terms": core_terms,
            "secondary_anchor_terms": secondary_terms,
        }
    return {
        "keep": True,
        "reason": "anchor_rule_passed",
        "core_anchor_terms": core_terms,
        "secondary_anchor_terms": secondary_terms,
    }


def _anchor_candidates(values: list[str], drop_aspect_only: bool) -> list[str]:
    terms: list[str] = []
    for value in values:
        cleaned = " ".join(str(value or "").split())
        if not cleaned:
            continue
        if drop_aspect_only and _looks_like_need_or_aspect_term(cleaned):
            continue
        if cleaned.lower() in {"importance", "significance", "background", "review"}:
            continue
        terms.append(cleaned)
    return _unique_strings(terms)


def _rule_query_critic_issue(
    query: str,
    search_contract: SearchContract | None,
) -> str:
    """Return a deterministic critic issue that should affect final selection."""

    if not search_contract or search_contract.generic_intent_frame is None:
        return ""
    frame = search_contract.generic_intent_frame
    lowered = _normalized_query_text(query)
    tokens = tokenize(lowered)
    research_blob = _normalized_query_text(
        " ".join(
            [
                *frame.research_object,
                *frame.domain_context,
                *frame.target_process_or_property,
                *frame.method_terms,
                *frame.application_or_metric_terms,
            ]
        )
    )
    if "llm" in research_blob or "systematic review" in research_blob or "literature screening" in research_blob:
        if "representative material" in lowered or "experimental characterization" in lowered or "materials" in lowered:
            return "cross_domain_pollution:ai_screening_material_template_terms"
    if "mof" in research_blob and (
        "co2" in research_blob
        or "carbon dioxide" in research_blob
        or "capture" in research_blob
        or "adsorption" in research_blob
    ):
        has_mof = "mof" in tokens or "metal organic framework" in lowered
        has_context = any(
            marker in lowered
            for marker in ["co2", "carbon dioxide", "carbon capture", "co2 capture", "co2 adsorption", "adsorption"]
        )
        has_specific_aspect = any(
            marker in lowered
            for marker in [
                "pore",
                "functional",
                "water stability",
                "adsorption performance",
                "structure property",
                "review",
                "framework",
                "case stud",
            ]
        )
        if "representative materials" in lowered:
            return "cross_domain_pollution:mof_representative_materials_template"
        if has_mof and len(tokens) <= 3:
            return "overbroad_queries:mof_short_anchor_only"
        if has_mof and not has_context:
            return "overbroad_queries:mof_missing_co2_adsorption_context"
        if has_mof and not has_specific_aspect and len(tokens) <= 4:
            return "overbroad_queries:mof_missing_aspect_or_review_intent"
    if (
        "thin film deposition" in research_blob
        or "atomic layer deposition" in research_blob
        or "sputtering" in research_blob
    ):
        method_hits = sum(
            1
            for marker in [
                "ald",
                "atomic layer deposition",
                "pld",
                "pulsed laser deposition",
                "sputtering",
                "cvd",
                "chemical vapor deposition",
            ]
            if marker in lowered
        )
        has_thin_anchor = any(
            marker in lowered
            for marker in [
                "thin film",
                "deposition",
                "fabrication",
                "physical vapor deposition",
                "chemical vapor deposition",
            ]
        )
        has_comparison = any(
            marker in lowered
            for marker in ["comparison", "review", "advantages", "disadvantages", "pros cons", "techniques", "methods"]
        )
        ald_only_bias = (
            ("atomic layer deposition" in lowered or "ald" in tokens)
            and method_hits < 3
        )
        if ald_only_bias:
            return "diversity_suggestions:thin_film_ald_bias"
        if method_hits >= 2 and not has_comparison and "thin film" not in lowered:
            return "diversity_suggestions:thin_film_missing_comparison_intent"
        if method_hits < 2 and not has_comparison:
            return "overbroad_queries:thin_film_single_method_query"
        if not has_thin_anchor and not has_comparison:
            return "overbroad_queries:thin_film_missing_method_comparison_anchor"
    return ""


def _is_thin_film_method_comparison_query(
    query: str,
    search_contract: SearchContract | None,
) -> bool:
    if not search_contract or search_contract.generic_intent_frame is None:
        return False
    frame = search_contract.generic_intent_frame
    research_blob = _normalized_query_text(
        " ".join([*frame.research_object, *frame.domain_context, *frame.method_terms])
    )
    if "thin film" not in research_blob and "atomic layer deposition" not in research_blob:
        return False
    lowered = _normalized_query_text(query)
    has_anchor = "thin film" in lowered or "deposition" in lowered or "fabrication" in lowered
    has_comparison = any(
        marker in lowered
        for marker in [
            "comparison",
            "review",
            "advantages",
            "disadvantages",
            "pros cons",
            "techniques",
            "methods",
        ]
    )
    method_hits = sum(
        1
        for marker in [
            "ald",
            "atomic layer deposition",
            "pld",
            "pulsed laser deposition",
            "sputtering",
            "cvd",
            "chemical vapor deposition",
            "physical vapor deposition",
        ]
        if marker in lowered
    )
    return has_anchor and (has_comparison or method_hits >= 4)


def _required_context_or_process_groups(
    search_contract: SearchContract | None,
) -> list[tuple[str, list[str]]]:
    if not search_contract:
        return []
    groups: list[tuple[str, list[str]]] = []
    for group in search_contract.constraint_groups:
        group_name = str(getattr(group, "group_name", "") or "")
        lowered = group_name.lower()
        if not getattr(group, "required", False):
            continue
        if not any(token in lowered for token in ("context", "process", "property")):
            continue
        terms = _anchor_candidates(list(getattr(group, "terms", []) or []), drop_aspect_only=False)
        if terms:
            groups.append((group_name, terms))
    return groups


def _query_contains_any(query: str, terms: list[str]) -> bool:
    query_lower = _normalized_query_text(query)
    query_tokens = set(tokenize(query_lower))
    for term in terms:
        term_lower = _normalized_query_text(term)
        if not term_lower:
            continue
        if term_lower in query_lower:
            return True
        term_tokens = set(tokenize(term_lower))
        if term_tokens and term_tokens <= query_tokens:
            return True
        if len(term_tokens) > 2 and len(term_tokens & query_tokens) >= 2:
            return True
    return False


def _query_anchor_hit_count(query: str, terms: list[str]) -> int:
    query_lower = _normalized_query_text(query)
    query_tokens = set(tokenize(query_lower))
    count = 0
    seen: set[str] = set()
    for term in terms:
        term_lower = _normalized_query_text(term)
        if not term_lower or term_lower in seen:
            continue
        term_tokens = set(tokenize(term_lower))
        if term_lower in query_lower or (term_tokens and term_tokens <= query_tokens):
            count += 1
            seen.add(term_lower)
    return count


def _normalized_query_text(value: str) -> str:
    return " ".join(
        str(value or "")
        .lower()
        .replace('"', " ")
        .replace("+", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )


def _looks_like_need_or_aspect_term(value: str) -> bool:
    lowered = _normalized_query_text(value)
    if lowered in {
        "in situ",
        "ex situ",
        "operando",
        "post mortem",
        "failure mechanism",
        "theoretical mechanism",
        "experimental characterization",
        "characterization",
        "comparison",
        "advantages disadvantages",
        "review",
        "background",
    }:
        return True
    return any(
        marker in lowered
        for marker in [
            "characterization",
            "mechanism",
            "measurement",
            "importance",
            "significance",
        ]
    ) and not any(
        anchor in lowered
        for anchor in [
            "reaction",
            "interface",
            "cell",
            "screening",
            "deposition",
            "capture",
            "passivation",
        ]
    )


def _looks_broad_standalone_query(query: str) -> bool:
    lowered = _normalized_query_text(query)
    tokens = tokenize(lowered)
    if len(tokens) <= 2:
        return True
    broad_markers = {
        "review",
        "background",
        "comparison",
        "characterization",
        "mechanism",
        "battery",
        "thin",
        "film",
        "carbon",
        "dioxide",
    }
    return len(tokens) <= 4 and bool(set(tokens) & broad_markers)


def _query_concepts_for_provenance(
    query: str,
    search_contract: SearchContract | None = None,
    lens: Any | None = None,
) -> list[dict[str, Any]]:
    """Explain which structured or generic concepts contributed to a query."""

    query_lower = " ".join(str(query or "").lower().replace('"', " ").split())
    candidates: list[tuple[str, list[str]]] = []
    frame = search_contract.generic_intent_frame if search_contract else None
    if frame is not None:
        for term, sources in frame.term_sources.items():
            candidates.append((term, list(sources)))
        for term in frame.relation_or_effect:
            candidates.append((term, ["generic_template"]))
    if search_contract is not None:
        for group in search_contract.constraint_groups:
            for term in group.terms:
                candidates.append((term, [group.source or "search_contract"]))
    if lens is not None:
        for term in [
            *getattr(lens, "core_concepts", []),
            *getattr(lens, "synonyms", []),
            *getattr(lens, "materials", []),
            *getattr(lens, "methods", []),
            *getattr(lens, "applications", []),
            *getattr(lens, "expected_evidence_types", []),
        ]:
            candidates.append((term, ["generic_template"]))
    concepts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term, sources in candidates:
        cleaned = " ".join(str(term or "").split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        if key in query_lower or _term_tokens_overlap(key, query_lower):
            concepts.append(
                {
                    "term": cleaned,
                    "sources": _unique_strings([str(source) for source in sources]),
                }
            )
            seen.add(key)
    if not concepts and query_lower:
        concepts.append({"term": query_lower, "sources": ["generic_template"]})
    return concepts[:12]


def _term_tokens_overlap(term_lower: str, query_lower: str) -> bool:
    term_tokens = {token for token in tokenize(term_lower) if len(token) > 2}
    query_tokens = {token for token in tokenize(query_lower) if len(token) > 2}
    return bool(term_tokens and len(term_tokens & query_tokens) >= min(2, len(term_tokens)))


def _provider_serializer_changes(provider: str, query: str, source: str) -> list[str]:
    changes: list[str] = []
    if provider == "semantic_scholar":
        changes.append("provider_neutral_query_no_mechanical_plus_terms")
    if source == "query_family":
        changes.append("context_combined_query_family_template")
    if is_single_acronym_query(query):
        changes.append("single_acronym_query_should_be_removed")
    return changes


def build_query_provenance(
    providers: list[str],
    queries_by_provider: dict[str, list[str]],
    query_family_plan: QueryFamilyPlan | None,
    use_query_families: bool = False,
    max_family_queries_per_provider: int | None = 18,
    search_contract: SearchContract | None = None,
    concept_map: ResearchLensPlan | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Merge optional query-family queries and record retrieval provenance."""

    merged_queries: dict[str, list[str]] = {}
    records: list[dict[str, Any]] = []
    duplicate_family_records: list[dict[str, Any]] = []
    query_family_queries: list[dict[str, Any]] = []
    provider_queries_by_family: dict[str, dict[str, list[str]]] = {}
    removed_single_acronym_queries: list[dict[str, str]] = []
    dropped_queries: list[dict[str, Any]] = []
    delayed_old_planner_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    domain_name = query_family_plan.domain if query_family_plan else (
        search_contract.domain_profile.domain_name if search_contract else ""
    )
    domain_pack_enhanced = domain_name not in {"", "general_science", "generic"}
    lens_by_name = {
        lens.name: lens
        for lens in (concept_map.lenses if concept_map else [])
    }
    old_planner_queries: dict[str, list[str]] = {}
    for provider in providers:
        kept, removed = _remove_single_acronym_queries(
            queries_by_provider.get(provider, [])
        )
        old_planner_queries[provider] = kept
        for query in removed:
            removed_single_acronym_queries.append(
                {"provider": provider, "query": query, "source": "old_planner"}
            )

    for provider in providers:
        provider_queries = list(old_planner_queries.get(provider, []))
        merged_queries[provider] = []
        strict_old_planner_filter = (
            use_query_families
            and query_family_plan is not None
            and provider in {"openalex", "semantic_scholar"}
        )
        for query in provider_queries:
            quality = (
                _final_query_quality(query, search_contract, source="old_planner")
                if strict_old_planner_filter
                else {"keep": True, "reason": "legacy_or_custom_provider_query_preserved"}
            )
            record = _query_provenance_record(
                provider=provider,
                raw_query=query,
                source="old_planner",
                concepts=_query_concepts_for_provenance(
                    query,
                    search_contract=search_contract,
                    lens=None,
                ),
                provider_serializer_changes=_provider_serializer_changes(
                    provider,
                    query,
                    source="old_planner",
                ),
                domain_pack_enhanced=domain_pack_enhanced,
            )
            record["source_stage"] = "old_planner"
            record["query_quality"] = quality
            if not quality.get("keep"):
                dropped_queries.append(
                    {
                        "provider": provider,
                        "query": query,
                        "source_stage": "old_planner",
                        "reason": quality.get("reason", "anchor_rule_failed"),
                        "accepted_critic_issue": quality.get(
                            "accepted_critic_issue",
                            "",
                        ),
                    }
                )
                continue
            if strict_old_planner_filter:
                delayed_old_planner_records.append(record)
                continue
            seen.add((provider, query))
            merged_queries.setdefault(provider, []).append(query)
            records.append(record)

    family_candidate_count = 0
    family_added_count = 0
    family_skipped_by_cap_count = 0
    family_added_by_provider = {provider: 0 for provider in providers}
    reason = "disabled"
    if use_query_families:
        if query_family_plan is None:
            reason = "no_query_family_plan"
        else:
            reason = ""
            sorted_families = sorted(
                query_family_plan.families,
                key=lambda item: (-int(getattr(item, "priority", 50)), item.name),
            )
            for family in sorted_families:
                family_budget = max(0, int(getattr(family, "budget", 0) or 0))
                for provider in providers:
                    family_provider_queries = _family_queries_for_provider(family, provider)
                    if family_budget:
                        family_provider_queries = family_provider_queries[:family_budget]
                    kept_family_queries, removed_family_queries = _remove_single_acronym_queries(
                        family_provider_queries
                    )
                    for query in removed_family_queries:
                        removed_single_acronym_queries.append(
                            {
                                "provider": provider,
                                "query": query,
                                "source": "query_family",
                                "family_name": family.name,
                            }
                        )
                    family_provider_queries = kept_family_queries
                    provider_queries_by_family.setdefault(family.name, {})[
                        provider
                    ] = list(family_provider_queries)
                    for query in family_provider_queries:
                        family_candidate_count += 1
                        if (
                            max_family_queries_per_provider is not None
                            and max_family_queries_per_provider >= 0
                            and family_added_by_provider.get(provider, 0)
                            >= max_family_queries_per_provider
                        ):
                            family_skipped_by_cap_count += 1
                            continue
                        key = (provider, query)
                        quality = (
                            {
                                "keep": True,
                                "reason": "domain_pack_query_template",
                            }
                            if domain_pack_enhanced and family.name != "seed_context"
                            else _final_query_quality(
                                query,
                                search_contract,
                                source=(
                                    family.name
                                    if family.name == "seed_context"
                                    else "query_family"
                                ),
                            )
                        )
                        family_record = _query_provenance_record(
                            provider=provider,
                            raw_query=query,
                            source="query_family",
                            family_name=family.name,
                            lens_name=family.lens_name,
                            purpose=family.purpose,
                            priority=family.priority,
                            budget=family.budget,
                            concepts=_query_concepts_for_provenance(
                                query,
                                search_contract=search_contract,
                                lens=lens_by_name.get(family.lens_name),
                            ),
                            provider_serializer_changes=_provider_serializer_changes(
                                provider,
                                query,
                                source="query_family",
                            ),
                            domain_pack_enhanced=domain_pack_enhanced,
                        )
                        family_record["source_stage"] = "query_family"
                        family_record["query_quality"] = quality
                        query_family_queries.append(family_record)
                        if not quality.get("keep"):
                            dropped_queries.append(
                                {
                                    "provider": provider,
                                    "query": query,
                                    "source_stage": "query_family",
                                    "family_name": family.name,
                                    "reason": quality.get("reason", "anchor_rule_failed"),
                                    "accepted_critic_issue": quality.get(
                                        "accepted_critic_issue",
                                        "",
                                    ),
                                }
                            )
                            continue
                        if key in seen:
                            duplicate_family_records.append(family_record)
                            continue
                        seen.add(key)
                        merged_queries.setdefault(provider, []).append(query)
                        records.append(family_record)
                        family_added_count += 1
                        family_added_by_provider[provider] = (
                            family_added_by_provider.get(provider, 0) + 1
                        )
            if family_candidate_count == 0:
                reason = "no_family_queries"
            elif family_added_count == 0:
                reason = (
                    "family_query_cap_reached"
                    if family_skipped_by_cap_count
                    else "all_family_queries_already_present"
                )
    if use_query_families and query_family_plan is not None:
        for record in delayed_old_planner_records:
            query = str(record.get("raw_query") or "")
            provider = str(record.get("provider") or "")
            key = (provider, query)
            if not query or key in seen:
                continue
            if family_added_by_provider.get(provider, 0) >= 6:
                dropped_queries.append(
                    {
                        "provider": provider,
                        "query": query,
                        "source_stage": "old_planner",
                        "reason": "query_family_queries_sufficient",
                        "accepted_critic_issue": "",
                    }
                )
                continue
            seen.add(key)
            merged_queries.setdefault(provider, []).append(query)
            record["source_stage"] = "old_planner_supplemental"
            records.append(record)

    payload = {
        "enabled": bool(use_query_families),
        "applied": bool(use_query_families and family_added_count > 0),
        "reason": reason,
        "domain": domain_name,
        "generic_fallback_used": domain_name in {"general_science", "generic"},
        "domain_pack_enhancement_used": domain_pack_enhanced,
        "old_planner_queries": old_planner_queries,
        "query_family_queries": query_family_queries,
        "final_openalex_queries": merged_queries.get("openalex", []),
        "final_semantic_scholar_queries": merged_queries.get("semantic_scholar", []),
        "provider_queries_by_family": provider_queries_by_family,
        "removed_single_acronym_queries": removed_single_acronym_queries,
        "dropped_queries": dropped_queries,
        "repaired_queries": [],
        "dropped_query_count": len(dropped_queries),
        "single_acronym_query_count": len(removed_single_acronym_queries),
        "records": records,
        "duplicate_family_records": duplicate_family_records,
        "provider_query_counts": {
            provider: len(queries) for provider, queries in merged_queries.items()
        },
        "old_planner_query_count": sum(
            1 for record in records if record["source"] == "old_planner"
        ),
        "family_candidate_query_count": family_candidate_count,
        "family_query_count": family_added_count,
        "duplicate_family_query_count": len(duplicate_family_records),
        "family_query_cap_per_provider": max_family_queries_per_provider,
        "family_queries_skipped_by_cap_count": family_skipped_by_cap_count,
    }
    return merged_queries, payload


def build_auto_query_repair_suggestions(
    query_plan: QueryPlan,
    query_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic query-plan repair diagnostics from provenance."""

    provider_counts = query_provenance.get("provider_query_counts", {})
    provider_query_count = sum(int(count or 0) for count in provider_counts.values())
    trigger_reasons: list[str] = []
    if not query_provenance.get("applied"):
        trigger_reasons.append("query_family_applied_false")
    if provider_query_count < 6:
        trigger_reasons.append("provider_query_count_lt_6")
    if query_provenance.get("single_acronym_query_count", 0):
        trigger_reasons.append("single_acronym_query_removed")
    if query_provenance.get("dropped_query_count", 0):
        trigger_reasons.append("static_query_quality_repair")
    if not trigger_reasons:
        return {
            "enabled": True,
            "applied": False,
            "trigger_reasons": [],
            "suggestions": [],
            "repaired_query_plan": None,
        }
    final_openalex = _unique_strings(query_provenance.get("final_openalex_queries", []))
    final_semantic = _unique_strings(
        query_provenance.get("final_semantic_scholar_queries", [])
    )
    original_queries = _unique_strings(
        [*query_plan.openalex_queries, *query_plan.semantic_scholar_queries]
    )
    repaired_queries = _unique_strings([*final_openalex, *final_semantic])
    removed = [
        record.get("query", "")
        for record in query_provenance.get("removed_single_acronym_queries", [])
        if isinstance(record, dict)
    ]
    dropped = [
        record.get("query", "")
        for record in query_provenance.get("dropped_queries", [])
        if isinstance(record, dict)
    ]
    added = [query for query in repaired_queries if query not in original_queries]
    repaired_plan = replace(
        query_plan,
        openalex_queries=final_openalex or query_plan.openalex_queries,
        semantic_scholar_queries=final_semantic or query_plan.semantic_scholar_queries,
    )
    repair_applied = bool(removed or dropped or added)
    return {
        "enabled": True,
        "applied": repair_applied,
        "trigger_reasons": trigger_reasons,
        "original_query_count": len(original_queries),
        "repaired_query_count": len(repaired_queries),
        "removed_queries": _unique_strings([str(query) for query in [*removed, *dropped]]),
        "dropped_queries": query_provenance.get("dropped_queries", []),
        "added_queries": added,
        "acronym_expansions": [],
        "context_terms_added": _context_terms_added(original_queries, added),
        "provider_specific_changes": {
            "openalex": {
                "final_query_count": len(final_openalex),
            },
            "semantic_scholar": {
                "final_query_count": len(final_semantic),
                "serializer": "provider-neutral generic combinations; no mechanical +term fallback",
            },
        },
        "query_actions": _query_repair_actions(query_provenance),
        "expected_effect": (
            "Replace isolated acronym/short-token queries with context-combined "
            "generic or domain-enhanced query families."
        ),
        "suggestions": [
            {
                "reason": reason,
                "action": "rewrite_query_plan_with_context_combined_families",
            }
            for reason in trigger_reasons
        ],
        "repaired_query_plan": repaired_plan,
    }


def flattened_provider_queries(queries_by_provider: dict[str, list[str]]) -> list[str]:
    """Return stable unique query strings from provider query mapping."""

    return _unique_strings(
        [
            query
            for provider_queries in queries_by_provider.values()
            for query in provider_queries
        ]
    )


def build_query_repair_stage_status(
    *,
    repair_enabled: bool,
    disabled_by_ablation: bool,
    raw_candidate_queries_by_provider: dict[str, list[str]],
    final_queries_by_provider: dict[str, list[str]],
    query_provenance: dict[str, Any],
    query_repair_suggestions: dict[str, Any],
) -> dict[str, Any]:
    """Summarize whether query repair changed final provider queries."""

    raw_flat = flattened_provider_queries(raw_candidate_queries_by_provider)
    final_flat = flattened_provider_queries(final_queries_by_provider)
    raw_set = set(raw_flat)
    final_set = set(final_flat)
    added_queries = [
        str(query)
        for query in query_repair_suggestions.get("added_queries", []) or []
        if str(query).strip()
    ]
    dropped_queries = query_provenance.get("dropped_queries", []) or []
    removed_queries = [
        str(query)
        for query in query_repair_suggestions.get("removed_queries", []) or []
        if str(query).strip()
    ]
    action_changes = [
        action
        for action in query_repair_suggestions.get("query_actions", []) or []
        if isinstance(action, dict)
        and str(action.get("action") or "").strip().lower() not in {"", "keep"}
    ]
    repaired_query_count = len(_unique_strings([*removed_queries, *added_queries]))
    if not repaired_query_count:
        repaired_query_count = len(action_changes)

    reason = ""
    if raw_set == final_set:
        if disabled_by_ablation:
            reason = "repair stage disabled but upstream sanitizer still applied"
        elif not query_repair_suggestions.get("trigger_reasons"):
            reason = "no repair issues detected"
        elif query_provenance.get("applied"):
            reason = "query family templates already generated clean queries"
        else:
            reason = "ablation partially_supported"
    return {
        "repair_enabled": repair_enabled,
        "deterministic_query_repair_enabled": repair_enabled,
        "deterministic_query_repair_applied": bool(query_repair_suggestions.get("applied")),
        "llm_query_critic_repair_enabled": False,
        "llm_query_critic_repair_applied": False,
        "llm_query_critic_verified_issue_count": 0,
        "llm_query_critic_applied_issue_count": 0,
        "llm_query_critic_reason_if_no_change": "",
        "repair_applied": bool(query_repair_suggestions.get("applied")),
        "disabled_by_ablation": disabled_by_ablation,
        "raw_candidate_query_count": len(raw_flat),
        "final_query_count": len(final_flat),
        "dropped_query_count": len(dropped_queries),
        "repaired_query_count": repaired_query_count,
        "added_query_count": len(added_queries),
        "query_added_count": len(added_queries),
        "query_dropped_count": len(dropped_queries),
        "query_modified_count": 0,
        "reason_if_no_difference": reason,
        "raw_only_queries": sorted(raw_set - final_set),
        "final_only_queries": sorted(final_set - raw_set),
    }


def llm_query_critic_repair_summary(
    repair_result: Any,
) -> dict[str, Any]:
    """Return a compact serializable summary of applied LLM critic repairs."""

    records = list(getattr(repair_result, "applied_issue_records", []) or [])
    applied_terms = _unique_strings(
        [
            str(term)
            for record in records
            for term in (record.get("applied_terms", []) or [])
            if str(term).strip()
        ]
    )
    rejected_terms = _unique_strings(
        [
            str(term.get("term") if isinstance(term, dict) else term)
            for record in records
            for term in (record.get("rejected_terms", []) or [])
            if str(term.get("term") if isinstance(term, dict) else term).strip()
        ]
    )
    term_grounding = [
        grounding
        for record in records
        for grounding in (record.get("term_grounding", []) or [])
        if isinstance(grounding, dict)
    ]
    original_examples = [
        str(record.get("original_query") or "")
        for record in records
        if str(record.get("original_query") or "").strip()
    ]
    repaired_examples = [
        str(record.get("new_query") or record.get("added_query") or "")
        for record in records
        if str(record.get("new_query") or record.get("added_query") or "").strip()
    ]
    return {
        "enabled": bool(getattr(repair_result, "apply_enabled", False)),
        "applied": bool(getattr(repair_result, "applied_issue_count", 0)),
        "applied_issue_count": int(getattr(repair_result, "applied_issue_count", 0) or 0),
        "rejected_for_application_count": int(
            getattr(repair_result, "rejected_for_application_count", 0) or 0
        ),
        "query_added_count": int(getattr(repair_result, "query_added_count", 0) or 0),
        "query_dropped_count": int(getattr(repair_result, "query_dropped_count", 0) or 0),
        "query_modified_count": int(getattr(repair_result, "query_modified_count", 0) or 0),
        "repaired_query_count": int(
            (getattr(repair_result, "query_added_count", 0) or 0)
            + (getattr(repair_result, "query_dropped_count", 0) or 0)
            + (getattr(repair_result, "query_modified_count", 0) or 0)
        ),
        "provenance_count": len(records),
        "original_query_example": original_examples[0] if original_examples else "",
        "repaired_query_example": repaired_examples[0] if repaired_examples else "",
        "applied_terms": applied_terms,
        "rejected_terms": rejected_terms,
        "term_grounding": term_grounding,
    }


def update_query_repair_stage_status_with_llm_critic(
    *,
    query_repair_stage_status: dict[str, Any],
    query_critic_trace: Any,
    query_repair_after_llm_critic: Any,
    apply_llm_query_critic_repairs: bool,
) -> dict[str, Any]:
    """Record LLM critic repair enablement even when no repair is applied."""

    summary = llm_query_critic_repair_summary(query_repair_after_llm_critic)
    verified_issue_count = int(
        getattr(query_critic_trace, "verified_issue_count", 0) or 0
    )
    reason = str(getattr(query_critic_trace, "reason_if_no_change", "") or "")
    if apply_llm_query_critic_repairs and not summary["applied"]:
        if not verified_issue_count:
            reason = "no_verified_grounded_issue_available_clean_plan_unchanged"
        elif summary["rejected_for_application_count"]:
            reason = reason or "verified_issue_available_but_no_safe_grounded_repair_applied"
        else:
            reason = reason or "no_llm_query_critic_repair_applied"

    query_repair_stage_status.update(
        {
            "llm_query_critic_repair_enabled": bool(apply_llm_query_critic_repairs),
            "llm_query_critic_repair_applied": bool(summary["applied"]),
            "llm_query_critic_verified_issue_count": verified_issue_count,
            "llm_query_critic_applied_issue_count": summary["applied_issue_count"],
            "llm_query_critic_reason_if_no_change": reason,
            "llm_query_critic_repaired_query_count": summary["repaired_query_count"],
            "llm_query_critic_original_query_example": summary["original_query_example"],
            "llm_query_critic_repaired_query_example": summary["repaired_query_example"],
            "llm_query_critic_applied_terms": summary["applied_terms"],
            "llm_query_critic_rejected_terms": summary["rejected_terms"],
            "llm_query_critic_repair_provenance_count": summary["provenance_count"],
            "repair_applied": bool(
                query_repair_stage_status.get("deterministic_query_repair_applied")
                or summary["applied"]
            ),
        }
    )
    if apply_llm_query_critic_repairs and not summary["applied"]:
        query_repair_stage_status["reason_if_no_difference"] = reason
    return query_repair_stage_status


def provider_query_signature(queries_by_provider: dict[str, list[str]]) -> tuple[str, ...]:
    """Return a normalized signature for provider-query artifact comparison."""

    return tuple(
        sorted(
            {
                " ".join(str(query).split()).lower()
                for query in flattened_provider_queries(queries_by_provider)
                if str(query).strip()
            }
        )
    )


def provider_queries_from_artifact(payload: Any) -> dict[str, list[str]]:
    """Extract provider query mappings from common query artifacts."""

    if not isinstance(payload, dict):
        return {}
    for key in ("final_provider_queries", "queries_by_provider"):
        value = payload.get(key)
        if isinstance(value, dict):
            return {
                str(provider): [str(query) for query in queries or []]
                for provider, queries in value.items()
                if isinstance(queries, list)
            }
    return {}


def provider_query_artifacts_consistent(
    artifacts: list[dict[str, list[str]]],
) -> bool:
    """Return whether non-empty provider-query artifacts agree."""

    signatures = [
        provider_query_signature(artifact)
        for artifact in artifacts
        if artifact and provider_query_signature(artifact)
    ]
    if len(signatures) <= 1:
        return True
    first = signatures[0]
    return all(signature == first for signature in signatures[1:])


def synchronize_final_query_artifacts_after_llm_critic_repair(
    *,
    providers: list[str],
    query_plan_payload: dict[str, Any],
    query_plan: QueryPlan,
    queries_by_provider: dict[str, list[str]],
    raw_candidate_queries_before_repair: dict[str, list[str]],
    query_provenance: dict[str, Any],
    final_queries_after_repair: dict[str, list[str]],
    query_repair_stage_status: dict[str, Any],
    query_repair_suggestions: dict[str, Any],
    query_repair_after_llm_critic: Any,
    disable_query_repair: bool,
    apply_llm_query_critic_repairs: bool,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Synchronize final query artifacts after explicit LLM critic repairs."""

    final_queries = {
        provider: list(queries_by_provider.get(provider, []))
        for provider in providers
    }
    flat_queries = flattened_provider_queries(final_queries)
    query_plan.openalex_queries = list(final_queries.get("openalex", []))
    query_plan.semantic_scholar_queries = list(final_queries.get("semantic_scholar", []))
    query_plan_payload["query_plan"] = query_plan
    query_plan_payload["queries"] = _unique_strings(flat_queries)
    query_plan_payload["queries_by_provider"] = final_queries
    query_plan_payload["final_provider_queries"] = final_queries
    query_plan_payload["final_openalex_queries"] = query_plan.openalex_queries
    query_plan_payload["final_semantic_scholar_queries"] = query_plan.semantic_scholar_queries

    query_provenance["final_provider_queries"] = final_queries
    query_provenance["final_openalex_queries"] = query_plan.openalex_queries
    query_provenance["final_semantic_scholar_queries"] = query_plan.semantic_scholar_queries
    query_provenance["provider_query_counts"] = {
        provider: len(queries) for provider, queries in final_queries.items()
    }
    summary = llm_query_critic_repair_summary(query_repair_after_llm_critic)
    query_provenance["llm_query_critic_repair_summary"] = summary

    final_queries_after_repair = final_queries
    raw_flat = flattened_provider_queries(raw_candidate_queries_before_repair)
    raw_set = set(raw_flat)
    raw_candidate_count = len(raw_flat)
    final_set = set(flat_queries)
    query_repair_stage_status.update(
        {
            "deterministic_query_repair_enabled": not disable_query_repair,
            "deterministic_query_repair_applied": bool(query_repair_suggestions.get("applied")),
            "llm_query_critic_repair_enabled": bool(apply_llm_query_critic_repairs),
            "llm_query_critic_repair_applied": bool(summary["applied"]),
            "llm_query_critic_verified_issue_count": int(
                query_repair_stage_status.get("llm_query_critic_verified_issue_count", 0)
                or 0
            ),
            "repair_applied": bool(
                query_repair_suggestions.get("applied") or summary["applied"]
            ),
            "disabled_by_ablation": bool(disable_query_repair),
            "final_query_count": len(flat_queries),
            "repaired_query_count": summary["repaired_query_count"],
            "query_added_count": summary["query_added_count"],
            "query_dropped_count": summary["query_dropped_count"],
            "query_modified_count": summary["query_modified_count"],
            "llm_query_critic_applied_issue_count": summary["applied_issue_count"],
            "llm_query_critic_reason_if_no_change": ""
            if summary["applied"]
            else query_repair_stage_status.get(
                "llm_query_critic_reason_if_no_change",
                "",
            ),
            "llm_query_critic_repaired_query_count": summary["repaired_query_count"],
            "llm_query_critic_original_query_example": summary["original_query_example"],
            "llm_query_critic_repaired_query_example": summary["repaired_query_example"],
            "llm_query_critic_applied_terms": summary["applied_terms"],
            "llm_query_critic_rejected_terms": summary["rejected_terms"],
            "llm_query_critic_repair_provenance_count": summary["provenance_count"],
            "reason_if_no_difference": ""
            if summary["applied"]
            else query_repair_stage_status.get("reason_if_no_difference", ""),
            "raw_only_queries": sorted(raw_set - final_set),
            "final_only_queries": sorted(final_set - raw_set),
        }
    )
    if raw_candidate_count:
        query_repair_stage_status["raw_to_final_query_change_count"] = len(
            raw_set.symmetric_difference(final_set)
        )
    return final_queries_after_repair, query_repair_stage_status


def sync_query_critic_with_repair(
    critic: dict[str, Any],
    repair: dict[str, Any],
) -> dict[str, Any]:
    """Record repair-accepted critic issues in the query critic artifact."""

    if not repair:
        return critic
    updated = dict(critic)
    accepted_actions: list[dict[str, Any]] = []
    issue_fields = {
        "missing_user_aspects",
        "overbroad_queries",
        "overconstrained_queries",
        "repetition_issues",
        "cross_domain_pollution",
        "diversity_suggestions",
    }
    for action in repair.get("query_actions", []) or []:
        if not isinstance(action, dict):
            continue
        issue = str(action.get("accepted_critic_issue") or "").strip()
        if not issue:
            continue
        field, _, reason = issue.partition(":")
        if field not in issue_fields:
            field = "overbroad_queries"
        value = reason or issue
        original_query = str(action.get("original_query") or "").strip()
        if original_query:
            value = f"{value} | {original_query}"
        values = list(updated.get(field, []) or [])
        if value not in values:
            values.append(value)
        updated[field] = values
        accepted_actions.append(
            {
                "type": field,
                "value": value,
                "accepted_reason": "accepted_by_query_repair",
                "action": action.get("action", ""),
            }
        )
    if accepted_actions:
        accepted = list(updated.get("accepted_suggestions", []) or [])
        accepted.extend(accepted_actions)
        updated["accepted_suggestions"] = accepted
        updated["accepted_repair_issues"] = accepted_actions
    return updated


def _query_repair_actions(query_provenance: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for record in query_provenance.get("dropped_queries", []):
        if not isinstance(record, dict):
            continue
        query = str(record.get("query") or "")
        if not query:
            continue
        actions.append(
            {
                "original_query": query,
                "action": "drop",
                "repaired_query": "",
                "reason": record.get("reason", "dropped_by_query_quality_gate"),
                "accepted_critic_issue": record.get("accepted_critic_issue", ""),
            }
        )
    for record in query_provenance.get("records", []):
        if not isinstance(record, dict):
            continue
        query = str(record.get("raw_query") or "")
        if not query:
            continue
        quality = record.get("query_quality", {}) if isinstance(record.get("query_quality"), dict) else {}
        actions.append(
            {
                "original_query": query,
                "action": "keep",
                "repaired_query": query,
                "reason": quality.get("reason", "kept_final_query"),
                "accepted_critic_issue": quality.get("accepted_critic_issue", ""),
            }
        )
    return actions


def _context_terms_added(original_queries: list[str], added_queries: list[str]) -> list[str]:
    original_text = " ".join(original_queries).lower()
    terms: list[str] = []
    for query in added_queries:
        for token in tokenize(query):
            if token not in original_text and len(token) > 3:
                terms.append(token)
    return _unique_strings(terms)[:24]


def annotate_papers_with_query_provenance(
    papers: list[Paper],
    query_provenance: dict[str, Any],
) -> None:
    """Attach matched-query metadata to retrieved papers without changing schema."""

    records = query_provenance.get("records", [])
    by_provider_query = {
        (str(record.get("provider") or ""), str(record.get("raw_query") or "")): record
        for record in records
        if isinstance(record, dict)
    }
    by_query: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_query = str(record.get("raw_query") or "")
        if raw_query and raw_query not in by_query:
            by_query[raw_query] = record

    for paper in papers:
        query = paper.retrieval_query
        if not query:
            continue
        provider_candidates = _unique_strings(
            [paper.retrieval_provider, paper.source_provider]
        )
        record = None
        for provider in provider_candidates:
            record = by_provider_query.get((provider, query))
            if record:
                break
        if record is None:
            record = by_query.get(query)
        if record is None:
            continue
        paper.raw.update(
            {
                "matched_query": query,
                "matched_query_source": record.get("source", ""),
                "matched_query_family": record.get("family_name", ""),
                "matched_lens": record.get("lens_name", ""),
                "matched_query_purpose": record.get("purpose", ""),
            }
        )
        if record.get("source") == "query_family" and not is_user_seed_paper(paper):
            paper.source_stage = "query_family"


def enrich_query_provenance_with_results(
    query_provenance: dict[str, Any],
    raw_by_provider: dict[str, list[dict]],
) -> dict[str, Any]:
    """Add returned paper ids/counts to query provenance after retrieval."""

    records = query_provenance.get("records", [])
    lookup = {
        (str(record.get("provider") or ""), str(record.get("raw_query") or "")): record
        for record in records
        if isinstance(record, dict)
    }
    for provider, bundles in raw_by_provider.items():
        for bundle in bundles:
            query = str(bundle.get("query") or "")
            record = lookup.get((provider, query))
            if record is None:
                continue
            paper_ids = [
                str(paper_id)
                for paper_id in bundle.get("paper_ids", [])
                if str(paper_id)
            ]
            record["paper_count"] = int(record.get("paper_count") or 0) + int(
                bundle.get("paper_count") or 0
            )
            record["paper_ids"] = _unique_strings(
                [*list(record.get("paper_ids", [])), *paper_ids]
            )
    query_provenance["returned_paper_count"] = sum(
        int(record.get("paper_count") or 0)
        for record in records
        if isinstance(record, dict)
    )
    return query_provenance


def run_query_pilot_workflow(
    query_plan: QueryPlan,
    search_contract: SearchContract,
    ambiguity_analysis: list[dict[str, Any]],
    providers: list[str],
    max_per_query: int = 5,
    from_year: int | None = None,
    use_cache: bool = True,
    cache_dir: str = "data/cache",
    openalex_mode: str = "keyword+semantic",
    sort_preference: str = "relevance",
    retriever_agent: RetrieverAgent | None = None,
    progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the optional pilot-search and query-repair workflow."""

    combined_queries = _combined_queries(query_plan)
    queries_by_provider = _queries_by_provider(providers, query_plan, combined_queries)
    diagnostics = QueryPilotAgent(retriever_agent=retriever_agent).run(
        queries=queries_by_provider,
        providers=providers,
        search_contract=search_contract,
        query_plan=query_plan,
        max_per_query=max_per_query,
        from_year=from_year,
        use_cache=use_cache,
        cache_dir=cache_dir,
        openalex_mode=openalex_mode,
        sort_mode=sort_preference,
        progress_callback=progress_callback,
    )
    repair_suggestions = QueryRepairAgent().suggest(
        query_plan=query_plan,
        search_contract=search_contract,
        ambiguity_analysis=ambiguity_analysis,
        pilot_diagnostics=diagnostics,
    )
    return {
        "diagnostics": diagnostics,
        "repair_suggestions": repair_suggestions,
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
    import_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build transparent retrieval and reranking diagnostics."""

    raw_count_per_query: dict[str, dict[str, int]] = {}
    top_titles_per_query: dict[str, dict[str, list[str]]] = {}
    provider_errors: dict[str, list[dict[str, Any]]] = {}
    retrieval_stages: list[dict[str, Any]] = []
    for provider, bundles in raw_by_provider.items():
        raw_count_per_query[provider] = {}
        top_titles_per_query[provider] = {}
        provider_errors[provider] = []
        for bundle in bundles:
            query = str(bundle.get("query") or "")
            search_mode = str(bundle.get("search_mode") or "")
            retrieval_stage = str(bundle.get("retrieval_stage") or provider)
            response = bundle.get("response") or {}
            items = _raw_items_from_response(response)
            stage_key = query
            if stage_key in raw_count_per_query[provider]:
                stage_key = f"{retrieval_stage}:{query}"
            top_titles = [
                title
                for title in [_raw_title(item) for item in items[:5]]
                if title
            ]
            raw_count_per_query[provider][stage_key] = len(items)
            top_titles_per_query[provider][stage_key] = top_titles
            retrieval_stages.append(
                {
                    "provider": provider,
                    "query": query,
                    "search_mode": search_mode or response.get("search_mode", ""),
                    "retrieval_stage": retrieval_stage
                    or response.get("retrieval_stage", provider),
                    "raw_count": len(items),
                    "kept_count": int(bundle.get("paper_count") or 0),
                    "top_titles": top_titles,
                    "missing_abstract_count": int(
                        bundle.get("missing_abstract_count") or 0
                    ),
                }
            )
            if response.get("error"):
                provider_errors[provider].append(
                    {
                        "query": query,
                        "search_mode": search_mode,
                        "retrieval_stage": retrieval_stage,
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
        "retrieval_stages": retrieval_stages,
        "provider_errors": provider_errors,
        "merged_count": len(merged_papers),
        "duplicate_count": duplicate_count,
        "year_filter": year_filter_stats or {},
        "imported_library": import_diagnostics or {},
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


def build_provider_status(
    providers: list[str],
    queries_by_provider: dict[str, list[str]],
    raw_by_provider: dict[str, list[dict]],
    retrieval_counts: dict[str, int],
) -> dict[str, Any]:
    """Summarize provider-level success, partial failure, and rate limits."""

    status: dict[str, Any] = {}
    for provider in providers:
        bundles = raw_by_provider.get(provider, [])
        requested = len(queries_by_provider.get(provider, []))
        attempted = len(bundles)
        errors: list[dict[str, Any]] = []
        rate_limited = False
        for bundle in bundles:
            response = bundle.get("response") or {}
            if response.get("error") or response.get("status_code"):
                error = {
                    "query": bundle.get("query", ""),
                    "retrieval_stage": bundle.get("retrieval_stage", provider),
                    "error": response.get("error", ""),
                    "status_code": response.get("status_code", ""),
                    "error_message": response.get("error_message", ""),
                }
                errors.append(error)
                if response.get("status_code") == 429:
                    rate_limited = True
        returned = int(retrieval_counts.get(provider, 0) or 0)
        stopped_early = requested > 0 and attempted < requested
        if rate_limited:
            state = "partial_success_rate_limited" if returned else "rate_limited"
        elif errors:
            state = "partial_success" if returned else "failed"
        elif returned:
            state = "success"
        elif attempted:
            state = "success_no_results"
        else:
            state = "not_attempted"
        status[provider] = {
            "status": state,
            "requested_query_count": requested,
            "attempted_query_count": attempted,
            "returned_paper_count": returned,
            "stopped_after_query_count": attempted if stopped_early or rate_limited else None,
            "rate_limited": rate_limited,
            "errors": errors,
        }
    return status


def build_retrieval_context(
    provider_status: dict[str, Any],
    merged_paper_count: int,
    ranked_paper_count: int,
    max_per_query: int,
) -> dict[str, Any]:
    """Summarize whether downstream artifacts are based on screened papers."""

    states = [
        str(status.get("status") or "")
        for status in provider_status.values()
        if isinstance(status, dict)
    ]
    attempted = [
        status
        for status in provider_status.values()
        if isinstance(status, dict)
        and int(status.get("attempted_query_count", 0) or 0) > 0
    ]
    successes = [
        state
        for state in states
        if state
        in {"success", "success_no_results", "partial_success", "partial_success_rate_limited"}
    ]
    failures = [
        state
        for state in states
        if state in {"failed", "rate_limited", "partial_success", "partial_success_rate_limited"}
    ]
    if max_per_query <= 0 or (states and all(state == "not_attempted" for state in states)):
        retrieval_status = "planning_only"
        reason = "retrieval_not_performed"
    elif not attempted and not states:
        retrieval_status = "planning_only"
        reason = "retrieval_not_performed"
    elif successes and failures:
        retrieval_status = "partial_success"
        reason = ""
    elif successes:
        retrieval_status = "success" if merged_paper_count else "failed"
        reason = "" if merged_paper_count else "insufficient_screened_papers"
    else:
        retrieval_status = "failed"
        reason = "insufficient_screened_papers"
    provider_bits = [
        f"{provider}:{status.get('status', 'unknown')}({status.get('returned_paper_count', 0)} papers)"
        for provider, status in provider_status.items()
        if isinstance(status, dict)
    ]
    return {
        "retrieval_status": retrieval_status,
        "provider_summary": "; ".join(provider_bits) or "no providers requested",
        "ranked_papers_based_on_real_retrieval": retrieval_status
        in {"success", "partial_success"}
        and merged_paper_count > 0
        and ranked_paper_count > 0,
        "merged_paper_count": merged_paper_count,
        "ranked_paper_count": ranked_paper_count,
        "reason": reason,
    }


def build_domain_guardrail_summary(
    assessments: list[DomainAssessment],
    ranked_papers: list[RankedPaper],
) -> dict[str, Any]:
    """Summarize domain guardrail decisions for reporting."""

    counts = {"in_scope": 0, "borderline": 0, "out_of_scope": 0}
    reasons: dict[str, int] = {}
    assessment_by_id = {item.paper_id: item for item in assessments}
    for assessment in assessments:
        counts[assessment.domain_decision] = counts.get(assessment.domain_decision, 0) + 1
        if assessment.domain_decision != "in_scope":
            reasons[assessment.off_topic_reason] = reasons.get(assessment.off_topic_reason, 0) + 1
    examples = []
    for item in ranked_papers:
        assessment = assessment_by_id.get(item.paper.paper_id)
        if not assessment or assessment.domain_decision == "in_scope":
            continue
        examples.append(
            {
                "rank": item.rank,
                "paper_id": item.paper.paper_id,
                "title": item.paper.title,
                "domain_decision": assessment.domain_decision,
                "domain_match_score": assessment.domain_match_score,
                "domain_penalty_multiplier": item.scores.domain_penalty_multiplier,
                "off_topic_reason": assessment.off_topic_reason,
            }
        )
        if len(examples) >= 10:
            break
    common_reasons = [
        {"reason": reason, "count": count}
        for reason, count in sorted(
            reasons.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    ]
    return {
        "counts": counts,
        "demoted_examples": examples,
        "common_off_topic_reasons": common_reasons,
    }


def summarize_paper_roles(records: list[PaperRoleRecord]) -> dict[str, Any]:
    """Summarize paper-role labels for metrics and trace artifacts."""

    primary_role_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for record in records:
        primary_role_counts[record.primary_role] = (
            primary_role_counts.get(record.primary_role, 0) + 1
        )
        for role in record.roles:
            role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "record_count": len(records),
        "primary_role_counts": primary_role_counts,
        "role_counts": role_counts,
    }


ABLATION_MODULES = {
    "query_family": "QueryFamily retrieval expansion",
    "query_repair": "Query repair suggestions and auto-repair",
    "domain_guardrail": "Domain guardrail scoring and demotion",
    "intent_repair": "Novice intent repair",
    "llm_intent_frame_enhancer": "Optional LLM intent-frame enhancement",
    "llm_query_plan_critic": "Optional LLM query-plan critique artifacts",
    "intent_centrality": "Intent centrality score blending",
    "group_coverage_ranking": "Required-group coverage ranking and priority limits",
}


def build_ablation_config(
    *,
    ablation_config_name: str,
    cli_flags_used: list[str] | None = None,
    legacy_query_planning: bool = False,
    use_query_families: bool = True,
    intent_repair: bool = True,
    disable_query_repair: bool = False,
    disable_domain_guardrail: bool = False,
    disable_intent_centrality: bool = False,
    disable_group_coverage_ranking: bool = False,
    enable_llm_intent_enhancer: bool = False,
    enable_llm_query_critic: bool = False,
    apply_llm_query_critic_repairs: bool = False,
) -> dict[str, Any]:
    """Return a lightweight record of pilot ablation switches."""

    disabled: list[str] = []
    support: dict[str, str] = {}
    if legacy_query_planning:
        disabled.extend(["intent_repair", "query_family"])
        support["legacy_query_planning"] = "supported"
    if not use_query_families and "query_family" not in disabled:
        disabled.append("query_family")
        support["query_family"] = "supported"
    if not intent_repair and "intent_repair" not in disabled:
        disabled.append("intent_repair")
        support["intent_repair"] = "supported"
    if disable_query_repair:
        disabled.append("query_repair")
        support["query_repair"] = "supported"
    if disable_domain_guardrail:
        disabled.append("domain_guardrail")
        support["domain_guardrail"] = "supported"
    if disable_intent_centrality:
        disabled.append("intent_centrality")
        support["intent_centrality"] = "supported"
    if disable_group_coverage_ranking:
        disabled.append("group_coverage_ranking")
        support["group_coverage_ranking"] = "partially_supported"
    if enable_llm_intent_enhancer:
        support["llm_intent_frame_enhancer"] = "supported"
    if enable_llm_query_critic:
        support["llm_query_plan_critic"] = "diagnostic_artifacts_only"
    if enable_llm_query_critic and apply_llm_query_critic_repairs:
        support["llm_query_plan_critic_repairs"] = "rule_applier_enabled"

    disabled = _unique_strings(disabled)
    enabled = [
        module
        for module in ABLATION_MODULES
        if module not in set(disabled)
        and (
            module != "llm_intent_frame_enhancer"
            or enable_llm_intent_enhancer
        )
        and (
            module != "llm_query_plan_critic"
            or enable_llm_query_critic
        )
    ]
    return {
        "ablation_config_name": ablation_config_name or "custom_debug",
        "disabled_modules": disabled,
        "enabled_modules": enabled,
        "cli_flags_used": cli_flags_used or [],
        "support_status": support,
        "notes": [
            "Pilot ablation switches are for debug/evaluation only and do not change default behavior.",
            "group_coverage_ranking is partially supported because group coverage also affects upstream domain assessment.",
        ],
    }


def neutral_domain_assessments_for_ablation(
    papers: list[Paper],
) -> list[DomainAssessment]:
    """Return pass-through domain assessments for no_domain_guardrail runs."""

    return [
        DomainAssessment(
            paper_id=paper.paper_id,
            domain_match_score=1.0,
            domain_decision="in_scope",
            off_topic_reason="domain_guardrail_disabled_for_ablation",
            positive_domain_matches=["ablation_pass_through"],
            negative_domain_matches=[],
            missing_required_concepts=[],
            forbidden_concepts_found=[],
            required_group_matches={},
            optional_group_matches={},
            intent_centrality_score=1.0,
            required_group_coverage_score=1.0,
            missing_required_group_count=0,
            group_coverage_explanation=["domain_guardrail_disabled_for_ablation"],
            matched_groups=["ablation_pass_through"],
            missing_groups=[],
            target_context_match=[],
            negative_context_match=[],
            topic_focus_score=1.0,
            aspect_match_score=1.0,
            target_context_score=1.0,
            negative_context_penalty=0.0,
            peripheral_context_reason="",
        )
        for paper in papers
    ]


def neutralize_group_coverage_for_ablation(
    assessments: list[DomainAssessment],
) -> list[DomainAssessment]:
    """Remove required-group ranking limits while preserving hard negatives."""

    neutralized: list[DomainAssessment] = []
    for assessment in assessments:
        required_matches = {
            key: True for key in assessment.required_group_matches
        }
        decision = assessment.domain_decision
        score = assessment.domain_match_score
        reason = assessment.off_topic_reason
        if decision == "out_of_scope" and not assessment.forbidden_concepts_found:
            decision = "borderline" if assessment.negative_domain_matches else "in_scope"
            score = max(score, 0.55 if assessment.negative_domain_matches else 0.72)
            reason = "group_coverage_ranking_disabled_for_ablation"
        neutralized.append(
            replace(
                assessment,
                domain_match_score=score,
                domain_decision=decision,
                off_topic_reason=reason,
                required_group_matches=required_matches,
                required_group_coverage_score=1.0,
                missing_required_group_count=0,
                group_coverage_explanation=[
                    "group_coverage_ranking_disabled_for_ablation"
                ],
                missing_groups=[],
            )
        )
    return neutralized


def apply_research_engine_ranking_adjustments(
    ranked_papers: list[RankedPaper],
    paper_roles: list[PaperRoleRecord],
    query_provenance: dict[str, Any],
    seed_hints: list[SeedHint],
    search_contract: SearchContract | None = None,
    aspect_coverage_records: list[AspectCoverageRecord] | None = None,
) -> tuple[list[RankedPaper], dict[str, Any]]:
    """Apply small transparent research-process adjustments after base scoring."""

    domain = search_contract.domain_profile.domain_name if search_contract else ""
    role_by_id = {record.paper_id: record for record in paper_roles}
    adjusted: list[RankedPaper] = []
    role_adjustments: dict[str, float] = {}
    lane_adjustments: dict[str, float] = {}
    seed_boosts: dict[str, float] = {}
    false_positive_penalties: dict[str, float] = {}
    for item in ranked_papers:
        role_record = role_by_id.get(item.paper.paper_id)
        role_adjustment = _role_adjustment(role_record, domain)
        lane_adjustment = _lane_adjustment(item.paper, query_provenance, domain)
        seed_boost = _seed_or_title_mention_boost(item.paper, seed_hints)
        false_positive_penalty = _false_positive_penalty(item, domain)
        final_score = _bounded_score(
            item.scores.final_score
            + role_adjustment
            + lane_adjustment
            + seed_boost
            - false_positive_penalty
        )
        scores = replace(
            item.scores,
            final_score=final_score,
            role_adjustment=role_adjustment,
            lane_adjustment=lane_adjustment,
            seed_or_title_mention_boost=seed_boost,
            false_positive_penalty=false_positive_penalty,
        )
        role_adjustments[item.paper.paper_id] = role_adjustment
        lane_adjustments[item.paper.paper_id] = lane_adjustment
        seed_boosts[item.paper.paper_id] = seed_boost
        false_positive_penalties[item.paper.paper_id] = false_positive_penalty
        adjusted.append(replace(item, scores=scores))
    adjusted.sort(key=lambda item: item.scores.final_score, reverse=True)
    adjusted = [replace(item, rank=index + 1) for index, item in enumerate(adjusted)]
    diagnostics = build_ranking_diagnostics(
        adjusted,
        paper_roles,
        query_provenance,
        seed_hints,
        role_adjustments,
        lane_adjustments,
        seed_boosts,
        false_positive_penalties,
        domain=domain,
        aspect_coverage_records=aspect_coverage_records or [],
    )
    return adjusted, diagnostics


def build_ranking_diagnostics(
    ranked_papers: list[RankedPaper],
    paper_roles: list[PaperRoleRecord],
    query_provenance: dict[str, Any],
    seed_hints: list[SeedHint],
    role_adjustments: dict[str, float],
    lane_adjustments: dict[str, float],
    seed_boosts: dict[str, float],
    false_positive_penalties: dict[str, float],
    domain: str = "",
    aspect_coverage_records: list[AspectCoverageRecord] | None = None,
) -> dict[str, Any]:
    """Return diagnostics for research-engine ranking adjustments."""

    role_summary = summarize_paper_roles(paper_roles)
    aspect_records = aspect_coverage_records or []
    aspect_scores = [record.aspect_coverage_score for record in aspect_records]
    overbroad_warnings = [
        {
            "paper_id": record.paper_id,
            "title": record.title,
            "warning": record.overbroad_role_warning,
        }
        for record in paper_roles
        if record.overbroad_role_warning
    ]
    top20 = ranked_papers[:20]
    top20_by_lane: dict[str, list[dict[str, Any]]] = {}
    for item in top20:
        lane = _paper_lane(item.paper, query_provenance)
        top20_by_lane.setdefault(lane, []).append(
            {
                "rank": item.rank,
                "paper_id": item.paper.paper_id,
                "title": item.paper.title,
                "final_score": item.scores.final_score,
            }
        )
    seed_titles = [str(hint.title or "") for hint in seed_hints if hint.title]
    gold_paper_ranks: dict[str, int] = {}
    for title in seed_titles:
        normalized_title = _normalize_for_match(title)
        for item in ranked_papers:
            if normalized_title and normalized_title in _normalize_for_match(item.paper.title):
                gold_paper_ranks[title] = item.rank
                break
    return {
        "domain_pack_used": domain,
        "query_family_applied": bool(query_provenance.get("applied")),
        "semantic_scholar_zero_result_warning": _semantic_scholar_zero_result_warning(query_provenance),
        "role_assignment_evidence": [
            {
                "paper_id": record.paper_id,
                "primary_role": record.primary_role,
                "roles": record.roles,
                "content_role": getattr(record, "content_role", ""),
                "retrieval_lane": getattr(record, "retrieval_lane", ""),
                "evidence": getattr(record, "evidence", []),
                "matched_query_family": getattr(record, "matched_query_family", ""),
            }
            for record in paper_roles
        ],
        "role_counts": role_summary["role_counts"],
        "primary_role_counts": role_summary["primary_role_counts"],
        "overbroad_role_warnings": overbroad_warnings,
        "role_adjustments": role_adjustments,
        "lane_adjustments": lane_adjustments,
        "lane_aspect_contribution": {
            record.paper_id: {
                "aspect_coverage_score": record.aspect_coverage_score,
                "covered_aspects": record.covered_aspects,
                "missing_aspects": record.missing_aspects,
                "lane_adjustment": lane_adjustments.get(record.paper_id, 0.0),
            }
            for record in aspect_records
        },
        "group_coverage_by_paper": {
            item.paper.paper_id: {
                "intent_centrality_score": item.scores.intent_centrality_score,
                "required_group_coverage_score": item.scores.required_group_coverage_score,
                "missing_required_group_count": item.scores.missing_required_group_count,
                "required_group_matches": (
                    item.domain_assessment.required_group_matches
                    if item.domain_assessment
                    else {}
                ),
                "optional_group_matches": (
                    item.domain_assessment.optional_group_matches
                    if item.domain_assessment
                    else {}
                ),
                "group_coverage_explanation": (
                    item.domain_assessment.group_coverage_explanation
                    if item.domain_assessment
                    else []
                ),
                "matched_groups": (
                    item.domain_assessment.matched_groups
                    if item.domain_assessment
                    else []
                ),
                "missing_groups": (
                    item.domain_assessment.missing_groups
                    if item.domain_assessment
                    else []
                ),
                "target_context_match": (
                    item.domain_assessment.target_context_match
                    if item.domain_assessment
                    else []
                ),
                "negative_context_match": (
                    item.domain_assessment.negative_context_match
                    if item.domain_assessment
                    else []
                ),
                "topic_focus_score": (
                    item.domain_assessment.topic_focus_score
                    if item.domain_assessment
                    else 0.0
                ),
                "aspect_match_score": (
                    item.domain_assessment.aspect_match_score
                    if item.domain_assessment
                    else 0.0
                ),
                "target_context_score": (
                    item.domain_assessment.target_context_score
                    if item.domain_assessment
                    else 0.0
                ),
                "negative_context_penalty": (
                    item.domain_assessment.negative_context_penalty
                    if item.domain_assessment
                    else 0.0
                ),
                "peripheral_context_reason": (
                    item.domain_assessment.peripheral_context_reason
                    if item.domain_assessment
                    else ""
                ),
            }
            for item in ranked_papers
        },
        "seed_or_title_mention_boosts": seed_boosts,
        "false_positive_penalties": false_positive_penalties,
        "acronym_disambiguation_penalties": {
            "single_acronym_query_count": query_provenance.get(
                "single_acronym_query_count",
                0,
            ),
            "removed_single_acronym_queries": query_provenance.get(
                "removed_single_acronym_queries",
                [],
            ),
        },
        "gold_paper_ranks": gold_paper_ranks,
        "top20_by_lane": top20_by_lane,
        "false_positive_top20_count": sum(
            1 for item in top20 if _false_positive_penalty(item, domain) > 0
        ),
        "top20_false_positive_count": sum(
            1 for item in top20 if _false_positive_penalty(item, domain) > 0
        ),
        "aspect_coverage_distribution": {
            "record_count": len(aspect_records),
            "zero_count": sum(1 for score in aspect_scores if score == 0),
            "nonzero_count": sum(1 for score in aspect_scores if score > 0),
            "average": (
                round(sum(aspect_scores) / len(aspect_scores), 4)
                if aspect_scores
                else 0.0
            ),
            "max": round(max(aspect_scores), 4) if aspect_scores else 0.0,
        },
    }


def _role_adjustment(record: PaperRoleRecord | None, domain: str = "") -> float:
    if record is None:
        return 0.0
    if domain == "ferroelectric_polarization":
        role_values = {
            "theory_origin": 0.10,
            "direct_probe_method": 0.12,
            "surface_probe_method": 0.12,
            "experimental_proof": 0.10,
            "interface_screening": 0.12,
            "material_case": 0.08,
            "device_application": 0.05,
            "application_bridge": 0.05,
            "review_background": 0.04,
            "limitation_or_challenge": 0.04,
        }
        value = max((role_values.get(role, 0.0) for role in record.roles), default=0.0)
        return round(value * 0.4, 4) if record.overbroad_role_warning else value
    if record.overbroad_role_warning:
        return 0.0
    generic_values = {
        "theory_mechanism": 0.012,
        "characterization_method": 0.012,
        "in_situ_operando": 0.014,
        "ex_situ_post_mortem": 0.010,
        "material_case": 0.008,
        "application_performance": 0.008,
        "failure_limitation": 0.010,
        "controversy_debate": 0.010,
        "review_background": 0.003,
        "peripheral_background": -0.004,
        "out_of_scope": -0.02,
    }
    if any(role in generic_values for role in record.roles):
        return round(max(generic_values.get(role, 0.0) for role in record.roles), 4)
    if record.primary_role == "review_background":
        return 0.003
    if record.primary_role in {"material_case", "limitation_or_challenge"}:
        return 0.008
    return 0.015


def _lane_adjustment(
    paper: Paper,
    query_provenance: dict[str, Any],
    domain: str = "",
) -> float:
    lane = _paper_lane(paper, query_provenance)
    if lane == "seed":
        return 0.03
    if lane.startswith("query_family"):
        if domain == "ferroelectric_polarization":
            if any(
                marker in lane
                for marker in [
                    "direct_probe_methods",
                    "interface_screening",
                    "theory_origin",
                ]
            ):
                return 0.04
            return 0.025
        return 0.012
    return 0.0


def _seed_or_title_mention_boost(paper: Paper, seed_hints: list[SeedHint]) -> float:
    if is_user_seed_paper(paper):
        return 0.05
    paper_title = _normalize_for_match(paper.title)
    for hint in seed_hints:
        hint_title = _normalize_for_match(hint.title or "")
        if hint_title and hint_title in paper_title:
            return 0.035
    return 0.0


def _false_positive_penalty(item: RankedPaper, domain: str = "") -> float:
    text = " ".join(
        [
            item.paper.title,
            item.paper.abstract,
            item.paper.venue,
            " ".join(item.paper.fields_of_study),
        ]
    ).lower()
    if domain == "ferroelectric_polarization":
        if any(
            term in text
            for term in [
                "drug screening",
                "clinical screening",
                "cognitive screening",
                "cell surface polarization",
                "generic solvent screening",
                "cosmo solvent",
            ]
        ):
            return 0.20
        if "surface plasmon polariton" in text and "ferroelectric" not in text:
            return 0.10
        if "thin film deposition" in text and "ferroelectric" not in text:
            return 0.10
    assessment = item.domain_assessment
    if assessment and assessment.domain_decision == "out_of_scope":
        return 0.05
    if assessment and assessment.domain_decision == "borderline":
        return 0.015
    return 0.0


def _semantic_scholar_zero_result_warning(query_provenance: dict[str, Any]) -> bool:
    """Return True when Semantic Scholar was queried but all tracked counts are zero."""

    queries = query_provenance.get("final_semantic_scholar_queries", [])
    if not queries:
        return False
    records = [
        record
        for record in query_provenance.get("records", [])
        if isinstance(record, dict) and record.get("provider") == "semantic_scholar"
    ]
    if not records:
        return False
    return all(int(record.get("paper_count") or 0) == 0 for record in records)


def _paper_lane(paper: Paper, query_provenance: dict[str, Any]) -> str:
    if is_user_seed_paper(paper):
        return "seed"
    source = str(paper.raw.get("matched_query_source") or "")
    family = str(paper.raw.get("matched_query_family") or "")
    if source == "query_family":
        return f"query_family:{family or 'unknown'}"
    if source:
        return source
    if paper.source_stage:
        return paper.source_stage
    return "keyword"


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_for_match(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def summarize_research_tensions(records: list[ResearchTension]) -> dict[str, Any]:
    """Summarize controversy and boundary-condition records."""

    return {
        "record_count": len(records),
        "tension_keys": [record.tension_key for record in records],
        "average_confidence": (
            sum(record.confidence for record in records) / len(records)
            if records
            else 0.0
        ),
    }


def classify_evidence_functions_for_records(
    evidence_records: list[EvidenceRecord],
    papers: list[Paper],
) -> list[EvidenceRecord]:
    """Attach research-argument function labels to evidence records."""

    papers_by_id = {paper.paper_id: paper for paper in papers}
    classified: list[EvidenceRecord] = []
    for record in evidence_records:
        paper = papers_by_id.get(record.paper_id)
        abstract = paper.abstract if paper else ""
        title = record.title or (paper.title if paper else "")
        evidence_text = record.evidence_sentence or record.claim
        evidence_function = classify_evidence_function(
            evidence_text,
            title=title,
            abstract=abstract,
        )
        classified.append(replace(record, evidence_function=evidence_function))
    return classified


def evidence_function_records(evidence_records: list[EvidenceRecord]) -> list[dict[str, Any]]:
    """Return a compact JSON payload for evidence-function audit."""

    return [
        {
            "paper_id": record.paper_id,
            "title": record.title,
            "claim": record.claim,
            "evidence_sentence": record.evidence_sentence,
            "evidence_function": evidence_function_value(record),
        }
        for record in evidence_records
    ]


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
    seed_exempt = 0
    examples: list[dict[str, Any]] = []
    for paper in papers:
        if is_user_seed_paper(paper):
            kept.append(paper)
            seed_exempt += 1
            continue
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
        "seed_exempt_count": seed_exempt,
        "excluded_examples": examples,
    }


def empty_import_result() -> ImportResult:
    """Return an empty import result for runs without external library files."""

    return ImportResult(papers=[], raw_count=0, detected_format="none")


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
    input_file: str | None = None,
    input_format: str = "auto",
    pilot_search: bool = False,
    pilot_max_per_query: int = 5,
    auto_repair_queries: bool = False,
    skip_pilot_search: bool = False,
    seed_papers: list[str] | None = None,
    seed_file: str | None = None,
    enable_snowballing: bool = False,
    snowball_top_n: int = 3,
    use_query_families: bool | None = True,
    intent_repair: bool = True,
    legacy_query_planning: bool = False,
    query_family_provider_cap: int = 18,
    disable_query_repair: bool = False,
    disable_domain_guardrail: bool = False,
    disable_intent_centrality: bool = False,
    disable_group_coverage_ranking: bool = False,
    ablation_config_name: str = "",
    ablation_cli_flags: list[str] | None = None,
    llm_client: GenericLLMClient | None = None,
    enable_llm_intent_enhancer: bool = False,
    llm_intent_provider: LLMIntentProvider | None = None,
    enable_llm_query_critic: bool = False,
    llm_query_critic_provider: LLMQueryPlanCriticProvider | None = None,
    apply_llm_query_critic_repairs: bool = False,
    retriever_agent: RetrieverAgent | None = None,
    snowball_agent: CitationSnowballAgent | None = None,
    progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> PipelineResult:
    """Run the full MVP pipeline and write output artifacts."""

    out = ensure_dir(output_dir)
    ensure_dir("data/cache")
    effective_intent_repair = bool(intent_repair and not legacy_query_planning)
    effective_use_query_families = (
        bool(use_query_families)
        if use_query_families is not None
        else not legacy_query_planning
    )
    if legacy_query_planning:
        effective_use_query_families = False
    ablation_config = build_ablation_config(
        ablation_config_name=ablation_config_name,
        cli_flags_used=ablation_cli_flags,
        legacy_query_planning=legacy_query_planning,
        use_query_families=effective_use_query_families,
        intent_repair=effective_intent_repair,
        disable_query_repair=disable_query_repair,
        disable_domain_guardrail=disable_domain_guardrail,
        disable_intent_centrality=disable_intent_centrality,
        disable_group_coverage_ranking=disable_group_coverage_ranking,
        enable_llm_intent_enhancer=enable_llm_intent_enhancer,
        enable_llm_query_critic=enable_llm_query_critic,
        apply_llm_query_critic_repairs=apply_llm_query_critic_repairs,
    )
    ablation_requested = bool(
        ablation_config_name
        or ablation_config["disabled_modules"]
        or enable_llm_intent_enhancer
        or enable_llm_query_critic
        or apply_llm_query_critic_repairs
    )
    if ablation_requested:
        write_json(out / "ablation_config.json", ablation_config)
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
            "input_file": input_file or "",
            "input_format": input_format,
            "pilot_search": pilot_search,
            "pilot_max_per_query": pilot_max_per_query,
            "auto_repair_queries": auto_repair_queries,
            "enable_snowballing": enable_snowballing,
            "snowball_top_n": snowball_top_n,
            "seed_paper_count": len(seed_papers or []),
            "seed_file": seed_file or "",
            "intent_repair": effective_intent_repair,
            "legacy_query_planning": legacy_query_planning,
            "use_query_families": effective_use_query_families,
            "query_family_provider_cap": query_family_provider_cap,
            "ablation_config_name": ablation_config["ablation_config_name"]
            if ablation_requested
            else "",
            "disabled_modules": ablation_config["disabled_modules"]
            if ablation_requested
            else [],
        },
    )

    config = PipelineConfig(
        providers=providers,
        max_per_query=max_per_query,
        from_year=from_year,
        output_dir=str(out),
        use_cache=use_cache,
        llm_backend=llm_backend,
        use_query_families=effective_use_query_families,
        intent_repair=effective_intent_repair,
        legacy_query_planning=legacy_query_planning,
        query_family_provider_cap=query_family_provider_cap,
    )

    seed_extraction_warning = ""
    try:
        seed_hints = SeedExtractionAgent().extract(question)
    except Exception as exc:
        seed_hints = []
        seed_extraction_warning = str(exc)[:240]
        if progress_callback:
            progress_callback(
                "seed_extraction",
                "Seed hint extraction failed; continuing without seed hints",
                {"warning": seed_extraction_warning},
            )
    write_json(out / "seed_hints.json", seed_hints)
    if progress_callback:
        progress_callback(
            "seed_extraction",
            "Seed hint extraction skipped after failure"
            if seed_extraction_warning
            else "Extracted explicit seed-paper hints from the question",
            {
                "seed_hint_count": len(seed_hints),
                "artifact_only": True,
                "warning": seed_extraction_warning,
            },
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
    expert_research_intent: ExpertResearchIntent | None = None
    intent_repair_warning = ""
    if effective_intent_repair:
        try:
            expert_research_intent = NoviceIntentInterpreter().repair(
                question,
                seed_hints=seed_hints,
                llm_client=active_llm_client,
                use_llm=True,
            )
            search_brief = _apply_expert_intent_to_search_brief(
                search_brief,
                expert_research_intent,
            )
            write_json(out / "expert_research_intent.json", expert_research_intent)
        except Exception as exc:
            intent_repair_warning = str(exc)[:240]
            expert_research_intent = None
            if progress_callback:
                progress_callback(
                    "intent_repair",
                    "Novice intent repair failed; continuing with legacy planning input",
                    {"warning": intent_repair_warning},
                )
    else:
        intent_repair_warning = "disabled_by_legacy_mode" if legacy_query_planning else "disabled"
    if progress_callback:
        intent_llm_metadata = (
            expert_research_intent.llm_metadata if expert_research_intent else {}
        )
        progress_callback(
            "intent_repair",
            "Repaired novice research intent"
            if expert_research_intent
            else "Novice intent repair skipped",
            {
                "enabled": effective_intent_repair,
                "executed": expert_research_intent is not None,
                "domain": intent_llm_metadata.get("domain_pack_domain", ""),
                "user_is_novice": expert_research_intent.user_is_novice
                if expert_research_intent
                else False,
                "downweighted_user_terms": expert_research_intent.ignored_or_downweighted_terms
                if expert_research_intent
                else [],
                "llm_metadata": expert_research_intent.llm_metadata
                if expert_research_intent
                else {},
                "llm_used": intent_llm_metadata.get("llm_used", False),
                "fallback_used": intent_llm_metadata.get("fallback_used", False),
                "invalid_json_count": intent_llm_metadata.get("invalid_json_count", 0),
                "schema_validation_errors": intent_llm_metadata.get(
                    "schema_validation_errors",
                    [],
                ),
                "llm_confidence": intent_llm_metadata.get("llm_confidence", 0.0),
                "fallback_reason": intent_llm_metadata.get("fallback_reason", ""),
                "warning": intent_repair_warning,
            },
        )
    question_refinement = QuestionRefinementAgent().refine(question, search_brief)
    ambiguity_analysis = AmbiguityDetectorAgent().analyze(question)
    search_contract = SearchContractAgent(
        mode=planner_mode,
        llm_client=active_llm_client,
    ).build(
        expert_research_intent.expert_rewritten_question
        if expert_research_intent
        else question,
        search_brief=search_brief,
        ambiguity_analysis=ambiguity_analysis,
        expert_intent=expert_research_intent,
    )
    llm_intent_enhancement = AdvisoryLLMIntentFrameEnhancer().enhance(
        original_question=question,
        rule_based_intent_frame=search_contract.generic_intent_frame,
        structured_concepts=expert_research_intent.structured_concepts
        if expert_research_intent
        else [],
        selected_domain=search_contract.domain_profile.domain_name,
        domain_candidates=search_contract.domain_profile.candidate_domains,
        seed_papers=seed_hints,
        llm_client=active_llm_client,
    )
    search_contract = AdvisoryLLMIntentFrameEnhancer().apply_to_contract(
        search_contract,
        llm_intent_enhancement,
    )
    write_json(out / "llm_intent_enhancement.json", llm_intent_enhancement)
    llm_intent_frame_trace: dict[str, Any] | None = None
    if enable_llm_intent_enhancer:
        search_contract_before_llm = search_contract
        write_json(out / "search_contract_before_llm.json", search_contract_before_llm)
        resolved_intent_provider = llm_intent_provider
        if resolved_intent_provider is None and active_llm_client is not None:
            resolved_intent_provider = GenericLLMIntentProvider(active_llm_client)
        optional_intent_result = OptionalLLMIntentFrameEnhancer(
            enabled=True,
            provider=resolved_intent_provider.provider_name
            if resolved_intent_provider is not None
            else active_llm_client.provider_name
            if active_llm_client is not None
            else llm_backend,
            model=resolved_intent_provider.model_name
            if resolved_intent_provider is not None
            else getattr(active_llm_client, "model", ""),
            llm_provider=resolved_intent_provider,
        ).enhance(
            search_contract.generic_intent_frame,
            input_question=question,
        )
        write_intent_enhancement_artifacts(
            out,
            intent_before_llm=search_contract.generic_intent_frame,
            raw=optional_intent_result.raw,
            verified_suggestion=optional_intent_result.verified_suggestion,
            trace=optional_intent_result.trace,
            verification_result=optional_intent_result.verification_result,
        )
        search_contract = apply_verified_suggestions_to_search_contract(
            search_contract,
            optional_intent_result,
        )
        write_json(out / "search_contract_after_llm.json", search_contract)
        llm_intent_frame_trace = {
            "trace": optional_intent_result.trace,
            "verification_result": optional_intent_result.verification_result,
        }
    write_json(out / "search_brief.json", search_brief)
    write_json(out / "question_refinement.json", question_refinement)
    write_json(out / "ambiguity_analysis.json", ambiguity_analysis)
    write_json(out / "search_contract.json", search_contract)
    concept_mapping_question = (
        f"{question} {expert_research_intent.expert_rewritten_question}"
        if expert_research_intent
        else question
    )
    concept_map, query_family_plan, research_lens_trace = build_research_lens_artifacts(
        question=concept_mapping_question,
        search_brief=search_brief,
        search_contract=search_contract,
        output_dir=out,
        seed_hints=seed_hints,
        progress_callback=progress_callback,
    )

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
            search_contract=search_contract,
            expert_research_intent=expert_research_intent,
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
                required_aspects=search_contract.required_aspects,
                must_terms=search_contract.must_include_concepts,
                exclude_terms=search_contract.must_exclude_concepts,
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
                structured_override.required_aspects = search_contract.required_aspects
            structured_override.must_terms = _unique_strings(
                [
                    *structured_override.must_terms,
                    *search_contract.must_include_concepts,
                ]
            )
            structured_override.exclude_terms = _unique_strings(
                [
                    *structured_override.exclude_terms,
                    *search_contract.must_exclude_concepts,
                ]
            )
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
            search_contract=search_contract,
            expert_research_intent=expert_research_intent,
        )

    query_plan = query_plan_payload["query_plan"]
    query_plan_payload["search_brief"] = search_brief
    query_plan_payload["search_contract"] = search_contract
    query_plan_payload["ambiguity_analysis"] = ambiguity_analysis
    query_plan_payload["question_refinement"] = question_refinement
    queries = query_plan_payload["queries"]
    planner_metadata = query_plan_payload["planner_metadata"]
    planning_question = str(query_plan_payload.get("planning_question") or question)
    queries_by_provider = _queries_by_provider(providers, query_plan, queries)
    retriever = retriever_agent or RetrieverAgent(config=config)
    query_pilot_diagnostics: dict[str, Any] = {
        "enabled": False,
        "reason": "pilot_search_not_requested",
        "results": [],
        "summary": {},
    }
    query_repair_suggestions: dict[str, Any] = {
        "enabled": False,
        "applied": False,
        "suggestions": [],
        "repaired_query_plan": None,
    }
    if pilot_search and not skip_pilot_search:
        if progress_callback:
            progress_callback(
                "query_pilot",
                "Running low-volume pilot search before full retrieval",
                {
                    "pilot_max_per_query": pilot_max_per_query,
                    "providers": providers,
                },
            )
        pilot_workflow = run_query_pilot_workflow(
            query_plan=query_plan,
            search_contract=search_contract,
            ambiguity_analysis=ambiguity_analysis,
            providers=providers,
            max_per_query=pilot_max_per_query,
            from_year=from_year,
            use_cache=use_cache,
            cache_dir=config.cache_dir,
            openalex_mode=openalex_mode,
            sort_preference=sort_preference,
            retriever_agent=retriever,
            progress_callback=progress_callback,
        )
        query_pilot_diagnostics = pilot_workflow["diagnostics"]
        query_repair_suggestions = pilot_workflow["repair_suggestions"]
        if auto_repair_queries and not disable_query_repair:
            repaired_plan = query_repair_suggestions.get("repaired_query_plan")
            if isinstance(repaired_plan, QueryPlan):
                query_plan = repaired_plan
            else:
                coerced = _coerce_query_plan(repaired_plan)
                if coerced is not None:
                    query_plan = coerced
            queries = _combined_queries(query_plan)
            queries_by_provider = _queries_by_provider(providers, query_plan, queries)
            query_repair_suggestions["applied"] = True
            query_plan_payload = _query_plan_payload(
                question,
                query_plan,
                planner_metadata,
                search_contract=search_contract,
                expert_research_intent=expert_research_intent,
            )
            query_plan_payload["search_brief"] = search_brief
            query_plan_payload["search_contract"] = search_contract
            query_plan_payload["ambiguity_analysis"] = ambiguity_analysis
            query_plan_payload["question_refinement"] = question_refinement
            planning_question = str(query_plan_payload.get("planning_question") or question)
        if progress_callback:
            progress_callback(
                "query_pilot",
                "Pilot search diagnostics ready",
                {
                    "summary": query_pilot_diagnostics.get("summary", {}),
                    "repair_suggestion_count": len(
                        query_repair_suggestions.get("suggestions", [])
                    ),
                    "auto_repair_applied": auto_repair_queries,
                    "query_repair_disabled": disable_query_repair,
                },
            )
    elif skip_pilot_search:
        query_pilot_diagnostics["reason"] = "pilot_search_skipped"
    queries_by_provider, query_provenance = build_query_provenance(
        providers=providers,
        queries_by_provider=queries_by_provider,
        query_family_plan=query_family_plan,
        use_query_families=effective_use_query_families,
        max_family_queries_per_provider=query_family_provider_cap,
        search_contract=search_contract,
        concept_map=concept_map,
    )
    raw_candidate_queries_before_repair = {
        provider: list(queries_by_provider.get(provider, []))
        for provider in providers
    }
    if disable_query_repair:
        query_repair_suggestions = {
            "enabled": False,
            "applied": False,
            "disabled_by_ablation": True,
            "reason": "query_repair_disabled_for_ablation",
            "suggestions": [],
            "repaired_query_plan": None,
            "trigger_reasons": [],
        }
    else:
        auto_query_repair = build_auto_query_repair_suggestions(
            query_plan=query_plan,
            query_provenance=query_provenance,
        )
        if (
            auto_query_repair.get("trigger_reasons")
            and not query_repair_suggestions.get("enabled")
        ):
            query_repair_suggestions = auto_query_repair
    query_plan.openalex_queries = list(query_provenance.get("final_openalex_queries", []))
    query_plan.semantic_scholar_queries = list(
        query_provenance.get("final_semantic_scholar_queries", [])
    )
    queries = _unique_strings(
        [
            query
            for provider in providers
            for query in queries_by_provider.get(provider, [])
        ]
    )
    query_plan.query_families_applied = bool(query_provenance.get("applied"))
    final_queries_after_repair = {
        provider: list(queries_by_provider.get(provider, []))
        for provider in providers
    }
    query_repair_stage_status = build_query_repair_stage_status(
        repair_enabled=not disable_query_repair,
        disabled_by_ablation=disable_query_repair,
        raw_candidate_queries_by_provider=raw_candidate_queries_before_repair,
        final_queries_by_provider=final_queries_after_repair,
        query_provenance=query_provenance,
        query_repair_suggestions=query_repair_suggestions,
    )
    query_plan_payload["query_plan"] = query_plan
    query_plan_payload["queries"] = queries
    query_plan_payload["queries_by_provider"] = {
        provider: list(queries_by_provider.get(provider, []))
        for provider in providers
    }
    query_plan_payload["query_families_applied"] = query_plan.query_families_applied
    query_plan_payload["selected_domain"] = search_contract.domain_profile.domain_name
    query_plan_payload["candidate_domains"] = search_contract.domain_profile.candidate_domains
    query_plan_payload["query_family_applied"] = query_plan.query_families_applied
    query_plan_payload["active_research_lenses"] = [
        lens.name for lens in (concept_map.lenses if concept_map else [])
    ]
    query_plan_payload["provider_queries_by_family"] = query_provenance.get(
        "provider_queries_by_family",
        {},
    )
    query_plan_payload["old_planner_queries"] = query_provenance.get(
        "old_planner_queries",
        {},
    )
    query_plan_payload["query_family_queries"] = query_provenance.get(
        "query_family_queries",
        [],
    )
    query_plan_payload["final_provider_queries"] = queries_by_provider
    query_plan_payload["final_openalex_queries"] = query_plan.openalex_queries
    query_plan_payload["final_semantic_scholar_queries"] = query_plan.semantic_scholar_queries
    query_plan_payload["dropped_queries"] = query_provenance.get("dropped_queries", [])
    query_plan_payload["generic_fallback_used"] = bool(
        research_lens_trace.get("generic_fallback_used")
    )
    query_plan_payload["domain_pack_enhancement_used"] = bool(
        research_lens_trace.get("domain_pack_enhancement_used")
    )
    query_plan_payload["removed_single_acronym_queries"] = query_provenance.get(
        "removed_single_acronym_queries",
        [],
    )
    query_plan_payload["repaired_queries"] = query_repair_suggestions.get(
        "added_queries",
        [],
    )
    query_plan_payload["query_repair_actions"] = query_repair_suggestions.get(
        "query_actions",
        [],
    )
    llm_query_critic = LLMQueryPlanCritic().critique(
        enhanced_intent_frame=llm_intent_enhancement,
        search_contract=search_contract,
        candidate_query_families=query_family_plan,
        final_provider_queries=queries_by_provider,
        llm_client=active_llm_client,
    )
    llm_query_critic = sync_query_critic_with_repair(
        llm_query_critic,
        query_repair_suggestions,
    )
    write_json(out / "llm_query_critic.json", llm_query_critic)
    query_plan_after_llm_critic: dict[str, Any] = {}
    if enable_llm_query_critic:
        query_plan_critic_input_payload = {
            "user_question": question,
            "structured_intent": expert_research_intent,
            "deterministic_intent": expert_research_intent,
            "search_contract_summary": search_contract,
            "query_plan": query_plan_payload,
            "planned_queries": query_plan_payload.get("queries", []),
            "query_provenance": query_provenance,
        }
        optional_query_critic = OptionalLLMQueryPlanCritic(
            enabled=True,
            provider=(
                getattr(llm_query_critic_provider, "provider_name", "")
                or llm_backend
            ),
            model=getattr(llm_query_critic_provider, "model_name", ""),
            llm_provider=llm_query_critic_provider,
        )
        optional_query_critic_result = optional_query_critic.critique(
            query_plan_payload,
            input_question=question,
            input_payload=query_plan_critic_input_payload,
        )
        query_repair_after_llm_critic = empty_query_critic_repair_result(
            optional_query_critic_result.query_plan_after_llm_critic,
            apply_enabled=apply_llm_query_critic_repairs,
        )
        query_critic_trace = optional_query_critic_result.trace
        query_plan_after_llm_critic = optional_query_critic_result.query_plan_after_llm_critic
        if (
            apply_llm_query_critic_repairs
            and optional_query_critic_result.verified_critique is not None
            and not optional_query_critic_result.trace.fallback_used
        ):
            query_repair_after_llm_critic = apply_verified_query_critic_issues_to_query_plan(
                optional_query_critic_result.query_plan_before_llm_critic,
                optional_query_critic_result.verified_critique,
                user_question=question,
                search_contract=search_contract,
                structured_intent=expert_research_intent,
                query_provenance=query_provenance,
            )
            query_plan_after_llm_critic = query_repair_after_llm_critic.query_plan_after_llm_critic
            if query_repair_after_llm_critic.applied_issue_count:
                query_plan_payload = query_plan_after_llm_critic
                query_plan = query_plan_payload["query_plan"]
                queries = query_plan_payload["queries"]
                queries_by_provider = query_plan_payload["final_provider_queries"]
                query_plan_payload["query_repair_after_llm_critic_applied"] = True
                query_plan_payload["llm_query_critic_repair_provenance"] = (
                    query_repair_after_llm_critic.provenance_updates
                )
                final_queries_after_repair, query_repair_stage_status = (
                    synchronize_final_query_artifacts_after_llm_critic_repair(
                        providers=providers,
                        query_plan_payload=query_plan_payload,
                        query_plan=query_plan,
                        queries_by_provider=queries_by_provider,
                        raw_candidate_queries_before_repair=raw_candidate_queries_before_repair,
                        query_provenance=query_provenance,
                        final_queries_after_repair=final_queries_after_repair,
                        query_repair_stage_status=query_repair_stage_status,
                        query_repair_suggestions=query_repair_suggestions,
                        query_repair_after_llm_critic=query_repair_after_llm_critic,
                        disable_query_repair=disable_query_repair,
                        apply_llm_query_critic_repairs=apply_llm_query_critic_repairs,
                    )
                )
                query_critic_trace = replace(
                    query_critic_trace,
                    applied_issue_count=query_repair_after_llm_critic.applied_issue_count,
                    rejected_for_application_count=query_repair_after_llm_critic.rejected_for_application_count,
                    query_added_count=query_repair_after_llm_critic.query_added_count,
                    query_dropped_count=query_repair_after_llm_critic.query_dropped_count,
                    query_modified_count=query_repair_after_llm_critic.query_modified_count,
                    reason_if_no_change="llm_query_critic_repairs_applied",
                )
            else:
                query_critic_trace = replace(
                    query_critic_trace,
                    rejected_for_application_count=query_repair_after_llm_critic.rejected_for_application_count,
                    reason_if_no_change="llm_query_critic_no_safe_repairs_applied",
                )
        query_repair_stage_status = update_query_repair_stage_status_with_llm_critic(
            query_repair_stage_status=query_repair_stage_status,
            query_critic_trace=query_critic_trace,
            query_repair_after_llm_critic=query_repair_after_llm_critic,
            apply_llm_query_critic_repairs=apply_llm_query_critic_repairs,
        )
        write_query_plan_critic_artifacts(
            out,
            query_plan_before_llm_critic=optional_query_critic_result.query_plan_before_llm_critic,
            raw=optional_query_critic_result.raw,
            verified_critique=optional_query_critic_result.verified_critique,
            trace=query_critic_trace,
            query_plan_after_llm_critic=query_plan_after_llm_critic,
            query_repair_after_llm_critic=query_repair_after_llm_critic,
        )
    write_json(out / "planned_queries.json", query_plan_payload)
    write_json(out / "query_provenance.json", query_provenance)
    write_json(out / "query_pilot_diagnostics.json", query_pilot_diagnostics)
    write_json(out / "query_repair_suggestions.json", query_repair_suggestions)
    write_json(
        out / "raw_candidate_queries_before_repair.json",
        {
            "queries_by_provider": raw_candidate_queries_before_repair,
            "queries": flattened_provider_queries(raw_candidate_queries_before_repair),
        },
    )
    write_json(
        out / "final_queries_after_repair.json",
        {
            "queries_by_provider": final_queries_after_repair,
            "queries": flattened_provider_queries(final_queries_after_repair),
        },
    )
    write_json(out / "query_repair_stage_status.json", query_repair_stage_status)
    if progress_callback:
        progress_callback(
            "planning",
            "Query plan ready",
            {
                "query_count": len(queries),
                "retrieval_query_count": sum(
                    len(provider_queries)
                    for provider_queries in queries_by_provider.values()
                ),
                "use_query_families": effective_use_query_families,
                "query_family_query_count": query_provenance.get(
                    "family_query_count",
                    0,
                ),
                "query_family_cap_per_provider": query_family_provider_cap,
                "planning_question": planning_question,
                "translation_mode": planner_metadata.get("translation_mode", "none"),
                "queries_by_provider": {
                    provider: len(provider_queries)
                    for provider, provider_queries in queries_by_provider.items()
                },
            },
        )

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
                "enable_snowballing": enable_snowballing,
                "snowball_top_n": snowball_top_n,
            },
        )
    user_seed_papers = [
        *parse_seed_values(seed_papers),
        *parse_seed_file(seed_file),
    ]
    seed_exact_papers, resolved_seed_papers, seed_resolution_report = resolve_seed_inputs(
        user_seed_papers,
        clients=retriever.clients,
    )
    write_json(out / "seed_resolution_report.json", seed_resolution_report)
    seed_expansion_report = empty_seed_expansion_report(
        seed_input_count=len(user_seed_papers),
        enabled=enable_snowballing,
    )
    seed_expansion_report["seed_resolved_count"] = seed_resolution_report.get(
        "seed_resolved_count",
        0,
    )
    seed_expansion_report["seed_unresolved_count"] = seed_resolution_report.get(
        "seed_unresolved_count",
        0,
    )
    seed_expansion_report["provider_errors"] = list(
        seed_resolution_report.get("provider_errors", [])
    )
    if progress_callback:
        progress_callback(
            "seed_resolution",
            "Resolved or retained user-provided seed papers",
            {
                "seed_input_count": seed_resolution_report["seed_input_count"],
                "seed_resolved_count": seed_resolution_report["seed_resolved_count"],
                "seed_unresolved_count": seed_resolution_report["seed_unresolved_count"],
                "seed_exact_paper_count": len(seed_exact_papers),
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
        openalex_mode=openalex_mode,
    )
    provider_status = build_provider_status(
        providers=providers,
        queries_by_provider=queries_by_provider,
        raw_by_provider=raw_by_provider,
        retrieval_counts=retrieval_counts,
    )
    write_json(out / "provider_status.json", provider_status)
    if seed_exact_papers:
        raw_papers = [*seed_exact_papers, *raw_papers]
        retrieval_counts["seed_exact"] = len(seed_exact_papers)
    annotate_papers_with_query_provenance(raw_papers, query_provenance)
    query_provenance = enrich_query_provenance_with_results(
        query_provenance,
        raw_by_provider,
    )
    write_json(out / "query_provenance.json", query_provenance)
    import_result = empty_import_result()
    if input_file:
        if progress_callback:
            progress_callback(
                "retrieval",
                "Importing external literature library",
                {"input_file": input_file, "input_format": input_format},
            )
        import_result = import_papers_from_file(input_file, input_format)
        raw_papers.extend(import_result.papers)
        retrieval_counts["imported"] = len(import_result.papers)
        write_csv(
            out / "imported_papers.csv",
            [paper_to_row(paper) for paper in import_result.papers],
            PAPER_FIELDS,
        )
        write_json(out / "import_diagnostics.json", import_result.diagnostics())
        if progress_callback:
            progress_callback(
                "retrieval",
                "External literature library imported",
                import_result.diagnostics(),
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
    evidence_records = classify_evidence_functions_for_records(
        evidence_records,
        merged_papers,
    )
    write_json(out / "evidence_functions.json", evidence_function_records(evidence_records))
    if progress_callback:
        missing_abstract_count = sum(1 for paper in merged_papers if not paper.abstract)
        function_counts: dict[str, int] = {}
        for record in evidence_records:
            function = evidence_function_value(record)
            function_counts[function] = function_counts.get(function, 0) + 1
        progress_callback(
            "extraction",
            "Evidence extraction finished",
            {
                "evidence_record_count": len(evidence_records),
                "missing_abstract_count": missing_abstract_count,
                "evidence_function_counts": function_counts,
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

    domain_guardrail = DomainGuardrailAgent()
    if progress_callback:
        progress_callback(
            "domain_guardrail",
            "Bypassing domain guardrail for ablation"
            if disable_domain_guardrail
            else "Assessing paper domain fit against the Search Contract",
            {
                "paper_count": len(merged_papers),
                "domain": search_contract.domain_profile.domain_name
                if search_contract
                else "",
                "disabled_by_ablation": disable_domain_guardrail,
            },
        )
    if disable_domain_guardrail:
        domain_assessments = neutral_domain_assessments_for_ablation(merged_papers)
    else:
        domain_assessments = domain_guardrail.assess_many(
            merged_papers,
            search_contract,
            query_plan=query_plan,
        )
    if disable_group_coverage_ranking:
        domain_assessments = neutralize_group_coverage_for_ablation(domain_assessments)
    if progress_callback:
        domain_counts: dict[str, int] = {}
        for assessment in domain_assessments:
            domain_counts[assessment.domain_decision] = (
                domain_counts.get(assessment.domain_decision, 0) + 1
            )
        progress_callback(
            "domain_guardrail",
            "Domain guardrail assessment finished",
            {"domain_decision_counts": domain_counts},
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
        domain_assessments=domain_assessments,
        use_intent_centrality=not disable_intent_centrality,
        use_group_coverage_ranking=not disable_group_coverage_ranking,
    )
    if progress_callback:
        progress_callback(
            "ranking",
            "Initial ranking finished",
            {"ranked_paper_count": len(ranked_before_feedback)},
        )

    citation_expansion_papers: list[Paper] = []
    retrieval_paths: list[RetrievalPath] = []
    if enable_snowballing:
        if progress_callback:
            progress_callback(
                "snowballing",
                "Starting seed-paper citation expansion",
                {
                    "provided_seed_count": len(user_seed_papers),
                    "snowball_top_n": snowball_top_n,
                },
            )
        active_snowball_agent = snowball_agent or CitationSnowballAgent(
            semantic_scholar_client=retriever.clients.get("semantic_scholar"),
            enabled=True,
        )
        citation_expansion_papers, retrieval_paths, snowball_seed_papers = (
            active_snowball_agent.expand(
                existing_papers=merged_papers,
                ranked_papers=ranked_before_feedback,
                seed_papers=resolved_seed_papers or user_seed_papers or None,
                top_n=snowball_top_n,
            )
        )
        if snowball_seed_papers:
            resolved_seed_papers = snowball_seed_papers
        agent_expansion_report = getattr(
            active_snowball_agent,
            "last_expansion_report",
            None,
        )
        if isinstance(agent_expansion_report, dict):
            seed_expansion_report = {
                **seed_expansion_report,
                **agent_expansion_report,
                "seed_input_count": len(user_seed_papers),
                "seed_resolved_count": seed_resolution_report.get(
                    "seed_resolved_count",
                    0,
                ),
                "seed_unresolved_count": seed_resolution_report.get(
                    "seed_unresolved_count",
                    0,
                ),
                "provider_errors": [
                    *list(seed_resolution_report.get("provider_errors", [])),
                    *list(agent_expansion_report.get("provider_errors", [])),
                ],
            }
        raw_paper_count_before_year_filter += len(citation_expansion_papers)
        retrieval_counts["citation_snowball"] = len(citation_expansion_papers)
        if citation_expansion_papers:
            citation_expansion_papers, expansion_year_filter = filter_papers_by_from_year(
                citation_expansion_papers,
                from_year,
            )
            kept_expanded_ids = {paper.paper_id for paper in citation_expansion_papers}
            retrieval_paths = [
                path for path in retrieval_paths if path.paper_id in kept_expanded_ids
            ]
            year_filter_stats["citation_expansion"] = expansion_year_filter
            retrieval_counts["citation_snowball"] = len(citation_expansion_papers)
            merged_papers, expansion_duplicate_count = deduplicate_with_stats(
                [*merged_papers, *citation_expansion_papers]
            )
            duplicate_count += expansion_duplicate_count
            evidence_records = extractor.extract_many(merged_papers, planning_question)
            evidence_records = classify_evidence_functions_for_records(
                evidence_records,
                merged_papers,
            )
            write_json(
                out / "evidence_functions.json",
                evidence_function_records(evidence_records),
            )
            verification_results = verifier.verify_many(merged_papers, evidence_records)
            aspect_coverage_records = aspect_agent.classify_many(
                merged_papers,
                evidence_records,
                required_aspects,
            )
            if disable_domain_guardrail:
                domain_assessments = neutral_domain_assessments_for_ablation(merged_papers)
            else:
                domain_assessments = domain_guardrail.assess_many(
                    merged_papers,
                    search_contract,
                    query_plan=query_plan,
                )
            if disable_group_coverage_ranking:
                domain_assessments = neutralize_group_coverage_for_ablation(
                    domain_assessments
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
                domain_assessments=domain_assessments,
                use_intent_centrality=not disable_intent_centrality,
                use_group_coverage_ranking=not disable_group_coverage_ranking,
            )
        if progress_callback:
            progress_callback(
                "snowballing",
                "Seed-paper citation expansion finished",
                {
                    "seed_count": len(resolved_seed_papers),
                    "expanded_paper_count": len(citation_expansion_papers),
                    "retrieval_path_count": len(retrieval_paths),
                },
            )
    seed_expansion_report["expanded_paper_count"] = len(citation_expansion_papers)
    seed_expansion_report["retrieval_path_count"] = len(retrieval_paths)
    seed_expansion_report["references_retrieved"] = max(
        int(seed_expansion_report.get("references_retrieved") or 0),
        sum(
            1
            for path in retrieval_paths
            if path.source_stage in {"reference", "seed_reference"}
        ),
    )
    seed_expansion_report["citations_retrieved"] = max(
        int(seed_expansion_report.get("citations_retrieved") or 0),
        sum(
            1
            for path in retrieval_paths
            if path.source_stage in {"citation", "seed_citation"}
        ),
    )
    seed_expansion_report["recommendations_retrieved"] = max(
        int(seed_expansion_report.get("recommendations_retrieved") or 0),
        sum(
            1
            for path in retrieval_paths
            if path.source_stage in {"recommendation", "seed_recommendation"}
        ),
    )
    write_json(out / "seed_expansion_report.json", seed_expansion_report)

    paper_role_records = PaperRoleClassifier().classify_many(
        merged_papers,
        query_provenance=query_provenance,
        search_contract=search_contract,
    )
    write_json(out / "paper_roles.json", paper_role_records)
    paper_role_summary = summarize_paper_roles(paper_role_records)
    ranked_before_feedback, ranking_diagnostics = apply_research_engine_ranking_adjustments(
        ranked_before_feedback,
        paper_role_records,
        query_provenance,
        seed_hints,
        search_contract=search_contract,
        aspect_coverage_records=aspect_coverage_records,
    )
    write_json(out / "ranking_diagnostics.json", ranking_diagnostics)
    if progress_callback:
        progress_callback(
            "paper_roles",
            "Classified papers into research roles",
            {
                "paper_count": len(paper_role_records),
                "primary_role_counts": paper_role_summary["primary_role_counts"],
                "artifact": "paper_roles.json",
                "ranking_diagnostics": "ranking_diagnostics.json",
            },
        )

    research_tension_warning = ""
    research_tension_domain = (
        search_contract.domain_profile.domain_name if search_contract else ""
    )
    try:
        research_tensions = ControversyAndBoundaryAgent().analyze(
            merged_papers,
            domain=research_tension_domain,
            query_provenance=query_provenance,
        )
    except Exception as exc:
        research_tension_warning = str(exc)[:240]
        research_tensions = []
    write_json(out / "research_tensions.json", research_tensions)
    research_tension_summary = summarize_research_tensions(research_tensions)
    if progress_callback:
        progress_callback(
            "controversy_boundary",
            "Identified research tensions and boundary conditions",
            {
                "domain": research_tension_domain,
                "tension_count": len(research_tensions),
                "artifact": "research_tensions.json",
                "ranking_unchanged": True,
                "warning": research_tension_warning,
            },
        )

    screening_decision_agent = ScreeningDecisionAgent()
    before_screening_decisions, ranked_before_feedback = (
        screening_decision_agent.decide_many(
            ranked_before_feedback,
            aspect_coverage_records,
        )
    )
    if progress_callback:
        progress_callback(
            "screening_decision",
            "Initial include/maybe/exclude decisions ready",
            summarize_screening_decisions(before_screening_decisions),
        )

    feedback_agent = HumanFeedbackAgent()
    preference_agent = PreferenceLearningAgent()
    ranked_after_feedback: list[RankedPaper] | None = None
    ranked_final = ranked_before_feedback
    screening_decisions = before_screening_decisions
    preference_learning = preference_agent.learn(
        ranked_before_feedback,
        {},
        search_contract,
    )
    feedback_query_refinement = preference_agent.query_refinement_payload(
        preference_learning
    )
    feedback_applied = False
    feedback_records: dict[str, FeedbackRecord] = {}
    if feedback_path:
        if progress_callback:
            progress_callback(
                "feedback",
                "Applying human feedback and reranking",
                {"feedback_path": feedback_path},
        )
        feedback_records = feedback_agent.read_feedback(feedback_path)
        preference_learning = preference_agent.learn(
            ranked_before_feedback,
            feedback_records,
            search_contract,
        )
        feedback_query_refinement = preference_agent.query_refinement_payload(
            preference_learning
        )
        ranked_after_feedback = feedback_agent.apply(
            ranked_before_feedback,
            feedback_records,
            scoring_weights=active_scoring_weights,
            preference_scores=preference_learning.preference_scores
            if preference_learning.enabled
            else None,
        )
        screening_decisions, ranked_after_feedback = (
            screening_decision_agent.decide_many(
                ranked_after_feedback,
                aspect_coverage_records,
            )
        )
        ranked_final = ranked_after_feedback
        feedback_applied = bool(feedback_records)
        if progress_callback:
            progress_callback(
                "feedback",
                "Feedback reranking finished",
                {"feedback_record_count": len(feedback_records)},
            )

    feedback_text = "\n".join(
        [
            f"{record.paper_id}: {record.label}; {record.note}"
            for record in feedback_records.values()
        ]
    )
    llm_feedback_interpretation = LLMFeedbackInterpreter().interpret(
        feedback_text=feedback_text,
        accepted_papers=[
            item.paper
            for item in ranked_final
            if item.screening_decision and item.screening_decision.decision == "include"
        ],
        rejected_papers=[
            item.paper
            for item in ranked_final
            if item.screening_decision and item.screening_decision.decision == "exclude"
        ],
        current_intent_frame=search_contract.generic_intent_frame,
        ranking_diagnostics=ranking_diagnostics,
        llm_client=active_llm_client,
    )
    write_json(out / "llm_feedback_interpretation.json", llm_feedback_interpretation)

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
        screening_decisions=screening_decisions,
    )

    if progress_callback:
        progress_callback(
            "evaluation",
            "Computing evaluation metrics",
            {"gold_labels_path": gold_labels_path or ""},
        )
    retrieval_context = build_retrieval_context(
        provider_status=provider_status,
        merged_paper_count=len(merged_papers),
        ranked_paper_count=len(ranked_final),
        max_per_query=max_per_query,
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
    metrics["provider_status"] = provider_status
    metrics["retrieval_status"] = retrieval_context["retrieval_status"]
    metrics["provider_summary"] = retrieval_context["provider_summary"]
    metrics["ranked_papers_based_on_real_retrieval"] = retrieval_context[
        "ranked_papers_based_on_real_retrieval"
    ]
    metrics["retrieval_performed"] = bool(
        retrieval_context["ranked_papers_based_on_real_retrieval"]
        or retrieval_context["retrieval_status"] in {"success", "partial_success"}
    )
    metrics["retrieval_context"] = retrieval_context
    final_query_artifact_consistent = provider_query_artifacts_consistent(
        [
            provider_queries_from_artifact(query_plan_payload),
            final_queries_after_repair,
            provider_queries_from_artifact(query_provenance),
        ]
    )
    llm_repair_provenance_count = int(
        query_repair_stage_status.get("llm_query_critic_repair_provenance_count", 0)
        or len(query_provenance.get("llm_query_critic_repairs", []) or [])
    )
    llm_query_critic_repair_artifact_consistent = (
        provider_query_artifacts_consistent(
            [
                provider_queries_from_artifact(query_plan_payload),
                final_queries_after_repair,
                provider_queries_from_artifact(query_provenance),
                provider_queries_from_artifact(query_plan_after_llm_critic),
            ]
        )
        and (
            not bool(query_repair_stage_status.get("llm_query_critic_repair_applied"))
            or llm_repair_provenance_count > 0
        )
    )
    metrics["final_query_artifact_consistent"] = final_query_artifact_consistent
    metrics["llm_query_critic_repair_artifact_consistent"] = (
        llm_query_critic_repair_artifact_consistent
    )
    metrics["llm_direct_paper_decision_mutation_count"] = 0
    metrics["llm_direct_evidence_validation_mutation_count"] = 0
    metrics["llm_direct_ranking_mutation_count"] = 0
    metrics["paper_decision_mutation_count"] = 0
    metrics["llm_query_critic_repair_enabled"] = bool(
        query_repair_stage_status.get("llm_query_critic_repair_enabled")
    )
    metrics["llm_query_critic_repair_applied"] = bool(
        query_repair_stage_status.get("llm_query_critic_repair_applied")
    )
    metrics["llm_query_critic_verified_issue_count"] = int(
        query_repair_stage_status.get("llm_query_critic_verified_issue_count", 0)
        or 0
    )
    metrics["llm_query_critic_applied_issue_count"] = int(
        query_repair_stage_status.get("llm_query_critic_applied_issue_count", 0)
        or 0
    )
    metrics["llm_query_critic_reason_if_no_change"] = str(
        query_repair_stage_status.get("llm_query_critic_reason_if_no_change", "")
        or ""
    )
    metrics["year_filter"] = year_filter_stats
    metrics["imported_library"] = import_result.diagnostics()
    metrics["scoring_weights"] = active_scoring_weights
    metrics["domain_guardrails"] = build_domain_guardrail_summary(
        domain_assessments,
        ranked_final,
    )
    metrics["query_pilot"] = {
        "enabled": bool(query_pilot_diagnostics.get("enabled")),
        "summary": query_pilot_diagnostics.get("summary", {}),
        "result_count": len(query_pilot_diagnostics.get("results", [])),
    }
    metrics["query_repair"] = {
        "enabled": bool(query_repair_suggestions.get("enabled")),
        "applied": bool(query_repair_suggestions.get("applied")),
        "deterministic_query_repair_enabled": bool(
            query_repair_stage_status.get("deterministic_query_repair_enabled")
        ),
        "deterministic_query_repair_applied": bool(
            query_repair_stage_status.get("deterministic_query_repair_applied")
        ),
        "llm_query_critic_repair_enabled": bool(
            query_repair_stage_status.get("llm_query_critic_repair_enabled")
        ),
        "llm_query_critic_repair_applied": bool(
            query_repair_stage_status.get("llm_query_critic_repair_applied")
        ),
        "llm_query_critic_verified_issue_count": int(
            query_repair_stage_status.get("llm_query_critic_verified_issue_count", 0)
            or 0
        ),
        "llm_query_critic_applied_issue_count": int(
            query_repair_stage_status.get("llm_query_critic_applied_issue_count", 0)
            or 0
        ),
        "llm_query_critic_reason_if_no_change": str(
            query_repair_stage_status.get("llm_query_critic_reason_if_no_change", "")
            or ""
        ),
        "final_query_artifact_consistent": final_query_artifact_consistent,
        "llm_query_critic_repair_artifact_consistent": (
            llm_query_critic_repair_artifact_consistent
        ),
        "llm_query_critic_repaired_query_count": int(
            query_repair_stage_status.get("llm_query_critic_repaired_query_count", 0)
            or 0
        ),
        "query_added_count": int(query_repair_stage_status.get("query_added_count", 0) or 0),
        "query_dropped_count": int(query_repair_stage_status.get("query_dropped_count", 0) or 0),
        "query_modified_count": int(
            query_repair_stage_status.get("query_modified_count", 0) or 0
        ),
        "repair_grounded_term_count": len(
            query_repair_stage_status.get("llm_query_critic_applied_terms", []) or []
        ),
        "repair_rejected_term_count": len(
            query_repair_stage_status.get("llm_query_critic_rejected_terms", []) or []
        ),
        "llm_query_critic_applied_terms": query_repair_stage_status.get(
            "llm_query_critic_applied_terms",
            [],
        ),
        "llm_query_critic_rejected_terms": query_repair_stage_status.get(
            "llm_query_critic_rejected_terms",
            [],
        ),
        "llm_query_critic_original_query_example": query_repair_stage_status.get(
            "llm_query_critic_original_query_example",
            "",
        ),
        "llm_query_critic_repaired_query_example": query_repair_stage_status.get(
            "llm_query_critic_repaired_query_example",
            "",
        ),
        "suggestion_count": len(query_repair_suggestions.get("suggestions", [])),
        "stage_status": query_repair_stage_status,
    }
    metrics["seed_paper_expansion"] = {
        "enabled": enable_snowballing,
        "seed_count": len(resolved_seed_papers),
        "seed_input_count": len(user_seed_papers),
        "seed_resolved_count": seed_resolution_report.get("seed_resolved_count", 0),
        "seed_unresolved_count": seed_resolution_report.get("seed_unresolved_count", 0),
        "expanded_paper_count": len(citation_expansion_papers),
        "retrieval_path_count": len(retrieval_paths),
        "references_retrieved": seed_expansion_report.get("references_retrieved", 0),
        "citations_retrieved": seed_expansion_report.get("citations_retrieved", 0),
        "recommendations_retrieved": seed_expansion_report.get(
            "recommendations_retrieved",
            0,
        ),
        "provider_errors": seed_expansion_report.get("provider_errors", []),
        "source_stage_counts": {
            stage: sum(1 for path in retrieval_paths if path.source_stage == stage)
            for stage in [
                "seed_reference",
                "seed_citation",
                "seed_recommendation",
                "reference",
                "citation",
                "recommendation",
            ]
        },
    }
    metrics["screening_decisions"] = summarize_screening_decisions(
        screening_decisions
    )
    reading_priority_policy = summarize_reading_priority_policy(ranked_final)
    metrics.update(reading_priority_policy)
    metrics["reading_priority_policy"] = reading_priority_policy
    metrics["paper_roles"] = paper_role_summary
    metrics["ranking_diagnostics"] = ranking_diagnostics
    metrics["research_tensions"] = research_tension_summary
    metrics["preference_learning"] = preference_learning_metrics(preference_learning)
    metrics["query_controls"] = {
        "strictness": strictness,
        "openalex_mode": openalex_mode,
        "sort_preference": sort_preference,
        "ranking_profile": ranking_profile,
        "intent_repair": effective_intent_repair,
        "legacy_query_planning": legacy_query_planning,
        "use_query_families": effective_use_query_families,
        "query_family_provider_cap": query_family_provider_cap,
        "disable_query_repair": disable_query_repair,
        "disable_domain_guardrail": disable_domain_guardrail,
        "disable_intent_centrality": disable_intent_centrality,
        "disable_group_coverage_ranking": disable_group_coverage_ranking,
    }
    if ablation_requested:
        metrics["ablation_config"] = ablation_config
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
        search_contract=search_contract,
        ambiguity_analysis=ambiguity_analysis,
        question_refinement=question_refinement,
        planner_metadata=planner_metadata,
        retrieval_counts=retrieval_counts,
        raw_paper_count=raw_paper_count_before_year_filter,
        merged_papers=merged_papers,
        duplicate_count=duplicate_count,
        evidence_records=evidence_records,
        verification_results=verification_results,
        aspect_coverage_records=aspect_coverage_records,
        domain_assessments=domain_assessments,
        screening_decisions=screening_decisions,
        preference_learning=preference_learning,
        feedback_query_refinement=feedback_query_refinement,
        ranked_papers=ranked_final,
        scoring_weights=active_scoring_weights,
        result_groups=result_groups,
        year_filter_stats=year_filter_stats,
        import_diagnostics=import_result.diagnostics(),
    )
    if ablation_requested:
        trace["ablation_config"] = ablation_config
    trace["concept_mapper"] = {
        **research_lens_trace.get("concept_mapper", {}),
        "domain": research_lens_trace.get("domain", ""),
        "reason": research_lens_trace.get("reason", ""),
        "generic_fallback_used": research_lens_trace.get("generic_fallback_used", False),
        "domain_pack_enhancement_used": research_lens_trace.get(
            "domain_pack_enhancement_used",
            False,
        ),
    }
    trace["query_family_planner"] = {
        **research_lens_trace.get("query_family_planner", {}),
        "domain": research_lens_trace.get("domain", ""),
        "reason": research_lens_trace.get("reason", ""),
    }
    intent_llm_metadata = (
        expert_research_intent.llm_metadata if expert_research_intent else {}
    )
    trace["intent_repair"] = {
        "enabled": effective_intent_repair,
        "executed": expert_research_intent is not None,
        "artifact": "expert_research_intent.json"
        if expert_research_intent
        else "",
        "user_is_novice": expert_research_intent.user_is_novice
        if expert_research_intent
        else False,
        "expert_rewritten_question": expert_research_intent.expert_rewritten_question
        if expert_research_intent
        else "",
        "downweighted_user_terms": expert_research_intent.ignored_or_downweighted_terms
        if expert_research_intent
        else [],
        "llm_metadata": expert_research_intent.llm_metadata
        if expert_research_intent
        else {},
        "llm_used": intent_llm_metadata.get("llm_used", False),
        "fallback_used": intent_llm_metadata.get("fallback_used", False),
        "invalid_json_count": intent_llm_metadata.get("invalid_json_count", 0),
        "schema_validation_errors": intent_llm_metadata.get(
            "schema_validation_errors",
            [],
        ),
        "llm_confidence": intent_llm_metadata.get("llm_confidence", 0.0),
        "fallback_reason": intent_llm_metadata.get("fallback_reason", ""),
        "assumptions": expert_research_intent.assumptions
        if expert_research_intent
        else [],
        "warning": intent_repair_warning,
        "legacy_query_planning": legacy_query_planning,
    }
    trace["generic_research_intent_frame"] = search_contract.generic_intent_frame
    trace["domain_activation"] = {
        "selected_domain": search_contract.domain_profile.domain_name,
        "candidate_domains": search_contract.domain_profile.candidate_domains,
        "activation_evidence": search_contract.domain_profile.activation_evidence,
        "negative_evidence": search_contract.domain_profile.negative_evidence,
        "confidence": search_contract.domain_profile.confidence,
        "fallback_reason": search_contract.domain_profile.fallback_reason,
    }
    trace["concept_validation_events"] = search_contract.concept_validation_events
    trace["fallback_reasons"] = {
        "domain_router": search_contract.domain_profile.fallback_reason,
        "intent_repair": intent_llm_metadata.get("fallback_reason", ""),
        "concept_mapper": research_lens_trace.get("reason", ""),
        "query_family": query_provenance.get("reason", ""),
    }
    trace["query_family_retrieval"] = {
        "enabled": query_provenance.get("enabled", False),
        "applied": query_provenance.get("applied", False),
        "reason": query_provenance.get("reason", ""),
        "domain": query_provenance.get("domain", ""),
        "generic_fallback_used": query_provenance.get("generic_fallback_used", False),
        "domain_pack_enhancement_used": query_provenance.get(
            "domain_pack_enhancement_used",
            False,
        ),
        "removed_single_acronym_queries": query_provenance.get(
            "removed_single_acronym_queries",
            [],
        ),
        "provider_query_counts": query_provenance.get("provider_query_counts", {}),
        "old_planner_query_count": query_provenance.get("old_planner_query_count", 0),
        "family_candidate_query_count": query_provenance.get(
            "family_candidate_query_count",
            0,
        ),
        "family_query_count": query_provenance.get("family_query_count", 0),
        "duplicate_family_query_count": query_provenance.get(
            "duplicate_family_query_count",
            0,
        ),
    }
    trace["seed_extraction"] = {
        "executed": True,
        "seed_hint_count": len(seed_hints),
        "artifact": "seed_hints.json",
        "artifact_only": True,
        "warning": seed_extraction_warning,
        "decision": "Seed hints are not converted into seed-paper expansion unless the existing seed-paper mode is explicitly requested.",
    }
    trace["paper_role_classifier"] = {
        "executed": True,
        "paper_role_count": len(paper_role_records),
        "artifact": "paper_roles.json",
        "ranking_diagnostics_artifact": "ranking_diagnostics.json",
        "primary_role_counts": paper_role_summary["primary_role_counts"],
    }
    trace["ranking_diagnostics"] = {
        "artifact": "ranking_diagnostics.json",
        "false_positive_top20_count": ranking_diagnostics.get(
            "false_positive_top20_count",
            0,
        ),
        "gold_paper_ranks": ranking_diagnostics.get("gold_paper_ranks", {}),
    }
    trace["controversy_boundary"] = {
        "executed": not research_tension_warning,
        "skipped": bool(research_tension_warning),
        "domain": research_tension_domain,
        "tension_count": len(research_tensions),
        "tension_keys": research_tension_summary["tension_keys"],
        "artifact": "research_tensions.json",
        "ranking_unchanged": True,
        "warning": research_tension_warning,
    }
    trace["query_pilot"] = query_pilot_diagnostics
    trace["query_repair"] = {
        **query_repair_suggestions,
        "trigger_reasons": query_repair_suggestions.get("trigger_reasons", []),
        "stage_status": query_repair_stage_status,
    }
    trace["llm_intent_enhancement"] = {
        "artifact": "llm_intent_enhancement.json",
        "llm_used": llm_intent_enhancement.get("llm_used", False),
        "fallback_used": llm_intent_enhancement.get("fallback_used", False),
        "schema_valid": llm_intent_enhancement.get("schema_valid", False),
        "accepted_suggestions": llm_intent_enhancement.get("accepted_suggestions", []),
        "rejected_suggestions": llm_intent_enhancement.get("rejected_suggestions", []),
    }
    trace["llm_query_critic"] = {
        "artifact": "llm_query_critic.json",
        "llm_used": llm_query_critic.get("llm_used", False),
        "fallback_used": llm_query_critic.get("fallback_used", False),
        "schema_valid": llm_query_critic.get("schema_valid", False),
        "accepted_suggestions": llm_query_critic.get("accepted_suggestions", []),
        "rejected_suggestions": llm_query_critic.get("rejected_suggestions", []),
    }
    trace["llm_feedback_interpretation"] = {
        "artifact": "llm_feedback_interpretation.json",
        "llm_used": llm_feedback_interpretation.get("llm_used", False),
        "fallback_used": llm_feedback_interpretation.get("fallback_used", False),
        "schema_valid": llm_feedback_interpretation.get("schema_valid", False),
        "accepted_suggestions": llm_feedback_interpretation.get("accepted_suggestions", []),
        "rejected_suggestions": llm_feedback_interpretation.get("rejected_suggestions", []),
    }
    trace["seed_paper_expansion"] = {
        "enabled": enable_snowballing,
        "seed_papers": resolved_seed_papers,
        "seed_resolution_report": seed_resolution_report,
        "seed_expansion_report": seed_expansion_report,
        "retrieval_paths": retrieval_paths,
        "expanded_paper_count": len(citation_expansion_papers),
        "decision": "Expanded from user-provided or high-confidence seed papers when enabled.",
    }
    write_json(out / "agent_trace.json", trace)
    write_json(out / "result_groups.json", result_groups)
    write_json(out / "prisma_like_flow.json", prisma_like_flow)
    write_json(out / "domain_assessments.json", domain_assessments)
    write_json(out / "paper_roles.json", paper_role_records)
    write_json(out / "research_tensions.json", research_tensions)
    write_preference_learning_outputs(
        out,
        preference_learning,
        feedback_query_refinement,
    )
    retrieval_diagnostics = build_retrieval_diagnostics(
        question=question,
        query_plan=query_plan,
        queries_by_provider=queries_by_provider,
        raw_by_provider=raw_by_provider,
        merged_papers=merged_papers,
        duplicate_count=duplicate_count,
        ranked_papers=ranked_final,
        year_filter_stats=year_filter_stats,
        import_diagnostics=import_result.diagnostics(),
    )
    retrieval_diagnostics["provider_status"] = provider_status
    write_json(out / "retrieval_diagnostics.json", retrieval_diagnostics)
    generate_paper_cards(
        out / "paper_cards.md",
        ranked_final,
        aspect_coverage_records,
    )
    reading_path_diagnostics = generate_reading_path(
        out / "reading_path.md",
        ranked_final,
        result_groups,
    )
    metrics.update(reading_path_diagnostics)
    metrics["reading_path_diagnostics"] = reading_path_diagnostics
    decision_artifacts = write_decision_artifacts(
        out,
        ranked_final,
        aspect_coverage_records,
        search_contract=search_contract,
        query_plan=query_plan,
        query_pilot_diagnostics=query_pilot_diagnostics,
        prisma_like_flow=prisma_like_flow,
        retrieval_context=retrieval_context,
    )
    metrics["decision_artifacts"] = {
        "method_comparison_rows": len(decision_artifacts["method_comparison_matrix"]),
        "research_gap_rows": len(decision_artifacts["research_gap_matrix"]),
        "suggested_next_search_count": len(decision_artifacts["suggested_next_searches"]),
    }
    exploration_quality = compute_exploration_quality(
        concept_map=concept_map,
        query_families=query_family_plan,
        paper_roles=paper_role_records,
        evidence_functions=evidence_records,
        gap_matrix=decision_artifacts["research_gap_matrix"],
        research_tensions=research_tensions,
        seed_hints=seed_hints,
        query_provenance=query_provenance,
        aspect_coverage_records=aspect_coverage_records,
        ranking_diagnostics=ranking_diagnostics,
    )
    save_exploration_quality(out / "exploration_quality.json", exploration_quality)
    user_report_markdown, llm_report_generation_trace = LLMUserReportAdapter().generate(
        ranked_papers=ranked_final,
        domain_assessments=domain_assessments,
        ranking_diagnostics=ranking_diagnostics,
        paper_roles=paper_role_records,
        provider_status=provider_status,
        exploration_quality=exploration_quality,
        search_contract=search_contract,
        research_gap_matrix=decision_artifacts["research_gap_matrix"],
        suggested_next_searches=decision_artifacts["suggested_next_searches"],
        llm_client=active_llm_client,
    )
    (out / "user_report.md").write_text(user_report_markdown, encoding="utf-8")
    write_json(out / "llm_report_generation_trace.json", llm_report_generation_trace)
    metrics["llm_enhancement"] = {
        "intent": {
            "llm_used": bool(llm_intent_enhancement.get("llm_used")),
            "fallback_used": bool(llm_intent_enhancement.get("fallback_used")),
            "schema_valid": bool(llm_intent_enhancement.get("schema_valid")),
        },
        "query_critic": {
            "llm_used": bool(llm_query_critic.get("llm_used")),
            "fallback_used": bool(llm_query_critic.get("fallback_used")),
            "schema_valid": bool(llm_query_critic.get("schema_valid")),
        },
        "feedback": {
            "llm_used": bool(llm_feedback_interpretation.get("llm_used")),
            "fallback_used": bool(llm_feedback_interpretation.get("fallback_used")),
            "schema_valid": bool(llm_feedback_interpretation.get("schema_valid")),
        },
        "user_report": {
            "llm_used": bool(llm_report_generation_trace.get("llm_used")),
            "fallback_used": bool(llm_report_generation_trace.get("fallback_used")),
            "schema_valid": bool(llm_report_generation_trace.get("schema_valid")),
        },
    }
    save_evaluation(out / "evaluation.json", metrics)

    write_pipeline_csvs(
        out,
        merged_papers,
        evidence_records,
        verification_results,
        aspect_coverage_records,
        screening_decisions,
        resolved_seed_papers,
        citation_expansion_papers,
        retrieval_paths,
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
        search_contract=search_contract,
        ambiguity_analysis=ambiguity_analysis,
        question_refinement=question_refinement,
        query_plan=query_plan,
        aspect_coverage_records=aspect_coverage_records,
        domain_assessments=domain_assessments,
        query_pilot_diagnostics=query_pilot_diagnostics,
        query_repair_suggestions=query_repair_suggestions,
        result_groups=result_groups,
        reading_path_path=out / "reading_path.md",
        paper_cards_path=out / "paper_cards.md",
        prisma_like_flow=prisma_like_flow,
        screening_decisions=screening_decisions,
        method_comparison_matrix=decision_artifacts["method_comparison_matrix"],
        research_gap_matrix=decision_artifacts["research_gap_matrix"],
        suggested_next_searches=decision_artifacts["suggested_next_searches"],
        preference_learning=preference_learning,
        feedback_query_refinement=feedback_query_refinement,
        seed_papers=resolved_seed_papers,
        retrieval_paths=retrieval_paths,
        citation_expansion_papers=citation_expansion_papers,
        research_tensions=research_tensions,
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
        search_contract=search_contract,
        ambiguity_analysis=ambiguity_analysis,
        domain_assessments=domain_assessments,
        screening_decisions=screening_decisions,
        paper_role_records=paper_role_records,
        research_tensions=research_tensions,
        seed_papers=resolved_seed_papers,
        seed_hints=seed_hints,
        retrieval_paths=retrieval_paths,
        citation_expansion_papers=citation_expansion_papers,
        preference_learning=preference_learning,
        feedback_query_refinement=feedback_query_refinement,
        query_pilot_diagnostics=query_pilot_diagnostics,
        query_repair_suggestions=query_repair_suggestions,
        question_refinement=question_refinement,
        aspect_coverage_records=aspect_coverage_records,
        result_groups=result_groups,
        concept_map=concept_map,
        query_family_plan=query_family_plan,
        query_provenance=query_provenance,
        expert_research_intent=expert_research_intent,
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
        "--input-file",
        default=None,
        help="Optional external literature library export: BibTeX, RIS, or CSV.",
    )
    run.add_argument(
        "--input-format",
        choices=["auto", "bibtex", "bib", "ris", "csv"],
        default="auto",
        help="Format for --input-file. Auto-detects from extension by default.",
    )
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
        "--enable-llm-intent-enhancer",
        action="store_true",
        help="Explicit opt-in: run optional LLM intent-frame enhancer before final SearchContract artifacts.",
    )
    run.add_argument(
        "--llm-intent-provider",
        choices=["none", "fake"],
        default="none",
        help="Provider for opt-in LLMIntentFrameEnhancer diagnostics. Defaults to none.",
    )
    run.add_argument(
        "--fake-llm-intent-mode",
        choices=[
            "valid",
            "malformed",
            "overbroad",
            "unsupported_domain_expansion",
            "decision_fields",
        ],
        default="valid",
        help="Fake LLMIntentFrameEnhancer response mode for offline tests.",
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
    run.add_argument(
        "--pilot-search",
        action="store_true",
        help="Run a small pilot retrieval before full retrieval.",
    )
    run.add_argument(
        "--pilot-max-per-query",
        type=int,
        default=5,
        help="Maximum pilot papers per provider query.",
    )
    run.add_argument(
        "--auto-repair-queries",
        action="store_true",
        help="Apply repaired queries from pilot diagnostics before full retrieval.",
    )
    run.add_argument(
        "--skip-pilot-search",
        action="store_true",
        help="Explicitly skip pilot search even if pilot settings are present.",
    )
    run.add_argument(
        "--seed-paper",
        action="append",
        default=[],
        help="Seed paper DOI, Semantic Scholar paper ID, OpenAlex ID, or title. Repeat for multiple seeds.",
    )
    run.add_argument(
        "--seed-file",
        default=None,
        help="CSV seed file with columns: seed_id,seed_type,title,doi,note.",
    )
    run.add_argument(
        "--enable-snowballing",
        action="store_true",
        help="Expand candidates through Semantic Scholar references, citations, and recommendations.",
    )
    run.add_argument(
        "--snowball-top-n",
        type=int,
        default=3,
        help="Maximum seed papers and linked papers per snowballing stage.",
    )
    run.add_argument(
        "--use-query-families",
        action="store_true",
        default=None,
        help="Include optional QueryFamilyPlan queries in retrieval.",
    )
    run.add_argument(
        "--skip-query-families",
        dest="use_query_families",
        action="store_false",
        help="Do not include QueryFamilyPlan queries in retrieval.",
    )
    run.add_argument(
        "--legacy-query-planning",
        action="store_true",
        help="Disable novice intent repair and QueryFamily retrieval for old planner behavior.",
    )
    run.add_argument(
        "--disable-intent-repair",
        dest="intent_repair",
        action="store_false",
        default=True,
        help="Disable novice-aware research intent repair.",
    )
    run.add_argument(
        "--disable-query-repair",
        action="store_true",
        help="Debug/evaluation only: disable query-repair suggestions and auto-repair.",
    )
    run.add_argument(
        "--disable-domain-guardrail",
        action="store_true",
        help="Debug/evaluation only: bypass domain guardrail demotion.",
    )
    run.add_argument(
        "--disable-intent-centrality",
        action="store_true",
        help="Debug/evaluation only: remove intent-centrality blending from ranking.",
    )
    run.add_argument(
        "--disable-group-coverage-ranking",
        action="store_true",
        help="Debug/evaluation only: neutralize required-group coverage ranking limits.",
    )
    run.add_argument(
        "--ablation-config-name",
        default="",
        help="Optional pilot ablation config name to record in ablation_config.json.",
    )
    run.add_argument(
        "--query-family-provider-cap",
        type=int,
        default=18,
        help="Maximum QueryFamily queries added per provider when enabled.",
    )
    run.add_argument(
        "--enable-llm-query-critic",
        action="store_true",
        help=(
            "Opt-in diagnostic only: run LLMQueryPlanCritic after query planning "
            "and write critique artifacts without changing final provider queries."
        ),
    )
    run.add_argument(
        "--apply-llm-query-critic-repairs",
        action="store_true",
        help=(
            "Opt-in evaluation/debug only: allow deterministic rules to apply "
            "verified LLMQueryPlanCritic issues as limited query repairs."
        ),
    )
    run.add_argument(
        "--llm-query-critic-provider",
        choices=["none", "fake"],
        default="none",
        help="Provider for opt-in LLMQueryPlanCritic diagnostics. Defaults to none.",
    )
    run.add_argument(
        "--fake-llm-query-critic-mode",
        choices=[
            "valid",
            "malformed_json",
            "forbidden_decision_fields",
            "unsupported_suggestion",
            "case_aware_weak_query",
            "valid_single_acronym_query",
            "valid_oer_single_acronym_query",
            "valid_mof_short_query",
            "no_issue",
        ],
        default="valid",
        help="Fake LLMQueryPlanCritic response mode for offline tests.",
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
            ablation_flags_used = []
            if args.use_query_families is False:
                ablation_flags_used.append("--skip-query-families")
            if args.legacy_query_planning:
                ablation_flags_used.append("--legacy-query-planning")
            if not args.intent_repair:
                ablation_flags_used.append("--disable-intent-repair")
            if args.disable_query_repair:
                ablation_flags_used.append("--disable-query-repair")
            if args.disable_domain_guardrail:
                ablation_flags_used.append("--disable-domain-guardrail")
            if args.disable_intent_centrality:
                ablation_flags_used.append("--disable-intent-centrality")
            if args.disable_group_coverage_ranking:
                ablation_flags_used.append("--disable-group-coverage-ranking")
            if args.enable_llm_intent_enhancer:
                ablation_flags_used.append("--enable-llm-intent-enhancer")
            if args.enable_llm_query_critic:
                ablation_flags_used.append("--enable-llm-query-critic")
            if args.apply_llm_query_critic_repairs:
                ablation_flags_used.append("--apply-llm-query-critic-repairs")
            llm_intent_provider = None
            if args.llm_intent_provider == "fake":
                ablation_flags_used.extend(
                    [
                        "--llm-intent-provider",
                        "fake",
                        "--fake-llm-intent-mode",
                        args.fake_llm_intent_mode,
                    ]
                )
                llm_intent_provider = FakeLLMIntentProvider(
                    response_mode=args.fake_llm_intent_mode
                )
            llm_query_critic_provider = None
            if args.llm_query_critic_provider == "fake":
                ablation_flags_used.extend(
                    [
                        "--llm-query-critic-provider",
                        "fake",
                        "--fake-llm-query-critic-mode",
                        args.fake_llm_query_critic_mode,
                    ]
                )
                llm_query_critic_provider = FakeLLMQueryPlanCriticProvider(
                    response_mode=args.fake_llm_query_critic_mode
                )
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
                enable_llm_intent_enhancer=args.enable_llm_intent_enhancer,
                llm_intent_provider=llm_intent_provider,
                enable_llm_query_critic=args.enable_llm_query_critic,
                llm_query_critic_provider=llm_query_critic_provider,
                apply_llm_query_critic_repairs=args.apply_llm_query_critic_repairs,
                planner_mode=args.planner_mode,
                extractor_mode=args.extractor_mode,
                verifier_mode=args.verifier_mode,
                scoring_weights=scoring_weights,
                strictness=args.strictness,
                openalex_mode=args.openalex_mode,
                sort_preference=args.sort_preference,
                ranking_profile=args.ranking_profile,
                input_file=args.input_file,
                input_format=args.input_format,
                pilot_search=args.pilot_search,
                pilot_max_per_query=args.pilot_max_per_query,
                auto_repair_queries=args.auto_repair_queries,
                skip_pilot_search=args.skip_pilot_search,
                seed_papers=args.seed_paper,
                seed_file=args.seed_file,
                enable_snowballing=args.enable_snowballing,
                snowball_top_n=args.snowball_top_n,
                use_query_families=args.use_query_families,
                intent_repair=args.intent_repair,
                legacy_query_planning=args.legacy_query_planning,
                query_family_provider_cap=args.query_family_provider_cap,
                disable_query_repair=args.disable_query_repair,
                disable_domain_guardrail=args.disable_domain_guardrail,
                disable_intent_centrality=args.disable_intent_centrality,
                disable_group_coverage_ranking=args.disable_group_coverage_ranking,
                ablation_config_name=args.ablation_config_name,
                ablation_cli_flags=ablation_flags_used,
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
