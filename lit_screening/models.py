"""Structured data models used by the literature-screening pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    """Structured, provider-aware search plan for a research question."""

    original_question: str = ""
    detected_language: str = "en"
    translated_question: str = ""
    core_terms: list[str] = field(default_factory=list)
    must_terms: list[str] = field(default_factory=list)
    optional_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    openalex_queries: list[str] = field(default_factory=list)
    semantic_scholar_queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Paper:
    """Normalized metadata for one scholarly paper."""

    paper_id: str
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    source_provider: str = ""
    provider_ids: dict[str, str] = field(default_factory=dict)
    citation_count: int = 0
    api_relevance_score: float = 0.0
    openalex_relevance_score: float = 0.0
    semantic_scholar_rank: int = 0
    publication_date: str = ""
    publication_types: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)
    influential_citation_count: int = 0
    reference_count: int = 0
    open_access_pdf_url: str = ""
    tldr: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class EvidenceRecord:
    """A claim-level evidence snippet extracted from a title or abstract."""

    paper_id: str
    title: str
    claim: str
    evidence_sentence: str
    relevance_reason: str
    limitation: str = ""
    keyword_overlap: float = 0.0
    extraction_mode: str = "rule"
    llm_used: bool = False
    invalid_llm_output: bool = False
    llm_error_type: str = ""


@dataclass
class VerificationResult:
    """Result of checking whether extracted evidence is grounded in the abstract."""

    paper_id: str
    supported: bool
    confidence: float
    error_type: str
    rationale: str
    support_level: str = "unverified"
    span_match_type: str = "none"
    span_match_confidence: float = 0.0
    matched_text: str = ""
    verification_mode: str = "rule"
    llm_used: bool = False
    invalid_llm_output: bool = False
    llm_error_type: str = ""


@dataclass
class ScoreBreakdown:
    """Transparent multi-objective score for a ranked paper."""

    relevance_score: float
    evidence_score: float
    recency_score: float
    quality_score: float
    diversity_score: float
    human_feedback_adjustment: float
    final_score: float


@dataclass
class FeedbackRecord:
    """Human review signal applied to a paper ranking."""

    paper_id: str
    label: str
    adjustment: float
    note: str = ""


@dataclass
class RankedPaper:
    """A paper plus its evidence, verification, and score."""

    rank: int
    paper: Paper
    evidence: EvidenceRecord
    verification: VerificationResult
    scores: ScoreBreakdown
    feedback: FeedbackRecord | None = None


@dataclass
class PipelineResult:
    """Paths and summary values produced by a pipeline run."""

    output_dir: str
    planned_queries: list[str]
    retrieval_counts: dict[str, int]
    merged_paper_count: int
    duplicate_count: int
    report_path: str
    evaluation_path: str
    ranked_papers_path: str
    question: str = ""
    raw_paper_count: int = 0
    merged_papers: list[Paper] = field(default_factory=list)
    evidence_records: list[EvidenceRecord] = field(default_factory=list)
    verification_results: list[VerificationResult] = field(default_factory=list)
    ranked_before_feedback: list[RankedPaper] = field(default_factory=list)
    ranked_after_feedback: list[RankedPaper] | None = None
    ranked_final: list[RankedPaper] = field(default_factory=list)
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    agent_trace: dict[str, Any] = field(default_factory=dict)
    scoring_weights: dict[str, float] = field(default_factory=dict)
    planning_question: str = ""
    translated_question: str = ""
    query_plan: QueryPlan | None = None
