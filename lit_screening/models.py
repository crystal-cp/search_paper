"""Structured data models used by the literature-screening pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class SearchBrief:
    """Intent-level interpretation of the user's literature search need."""

    original_question: str
    refined_question: str
    search_intent: str
    user_goal: str
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    required_aspects: list[str] = field(default_factory=list)
    preferred_paper_types: list[str] = field(default_factory=list)
    time_window: str = ""
    success_definition: str = ""


@dataclass
class DomainProfile:
    """Domain boundaries used to keep retrieval aligned with user intent."""

    domain_name: str
    positive_domains: list[str] = field(default_factory=list)
    negative_domains: list[str] = field(default_factory=list)
    required_concepts: list[str] = field(default_factory=list)
    forbidden_concepts: list[str] = field(default_factory=list)
    preferred_venues: list[str] = field(default_factory=list)
    excluded_venues: list[str] = field(default_factory=list)
    field_of_study_whitelist: list[str] = field(default_factory=list)
    field_of_study_blacklist: list[str] = field(default_factory=list)
    terminology_map: dict[str, list[str]] = field(default_factory=dict)
    candidate_domains: list[dict[str, Any]] = field(default_factory=list)
    activation_evidence: dict[str, list[str]] = field(default_factory=dict)
    negative_evidence: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.0
    fallback_reason: str = ""


@dataclass
class DomainConcept:
    """One concept entry from a lightweight domain pack."""

    synonyms: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)


@dataclass
class DomainPack:
    """Externalized domain knowledge for future query-planning extensions."""

    domain_name: str
    activation: dict[str, Any] = field(default_factory=dict)
    domain_anchors: list[str] = field(default_factory=list)
    concepts: dict[str, DomainConcept] = field(default_factory=dict)
    mechanisms: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)
    false_positive_terms: list[str] = field(default_factory=list)
    preferred_venues: list[str] = field(default_factory=list)
    field_of_study_whitelist: list[str] = field(default_factory=list)
    field_of_study_blacklist: list[str] = field(default_factory=list)
    constraint_groups: list[dict[str, Any]] = field(default_factory=list)
    aspect_groups: dict[str, list[str]] = field(default_factory=dict)
    query_expansions: dict[str, list[str]] = field(default_factory=dict)
    query_templates: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainActivationResult:
    """Auditable result from pack-driven domain routing."""

    selected_domain: str
    candidate_domains: list[dict[str, Any]] = field(default_factory=list)
    activation_evidence: dict[str, list[str]] = field(default_factory=dict)
    negative_evidence: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.0
    fallback_reason: str = ""
    domain_pack_enhancement_used: bool = False


@dataclass
class GenericResearchIntentFrame:
    """Domain-agnostic structure inferred from a noisy research question."""

    research_object: list[str] = field(default_factory=list)
    domain_context: list[str] = field(default_factory=list)
    target_process_or_property: list[str] = field(default_factory=list)
    relation_or_effect: list[str] = field(default_factory=list)
    mechanism_need: bool = False
    method_need: bool = False
    in_situ_or_operando_need: bool = False
    ex_situ_need: bool = False
    material_case_need: bool = False
    application_or_performance_need: bool = False
    failure_or_limitation_need: bool = False
    controversy_need: bool = False
    review_background_need: bool = False
    core_terms: list[str] = field(default_factory=list)
    method_terms: list[str] = field(default_factory=list)
    mechanism_terms: list[str] = field(default_factory=list)
    material_or_case_terms: list[str] = field(default_factory=list)
    application_or_metric_terms: list[str] = field(default_factory=list)
    failure_or_limitation_terms: list[str] = field(default_factory=list)
    controversy_terms: list[str] = field(default_factory=list)
    downweighted_terms: list[str] = field(default_factory=list)
    term_sources: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class IntentConcept:
    """One concept activated while repairing a novice research intent."""

    term: str
    category: str
    source: str
    confidence: float
    activation_reason: str
    query_role: str
    should_use_in_provider_query: bool


@dataclass
class ExpertResearchIntent:
    """Expert-level repair of a novice or underspecified research question."""

    original_question: str
    user_is_novice: bool
    inferred_goal: str
    expert_rewritten_question: str
    structured_concepts: list[IntentConcept] = field(default_factory=list)
    target_objects: list[str] = field(default_factory=list)
    target_properties: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)
    likely_user_misconceptions: list[str] = field(default_factory=list)
    ignored_or_downweighted_terms: list[str] = field(default_factory=list)
    must_not_overinterpret: list[str] = field(default_factory=list)
    ambiguity_points: list[str] = field(default_factory=list)
    possible_interpretations: list[str] = field(default_factory=list)
    selected_interpretation: str = ""
    selected_interpretation_reason: str = ""
    needs_user_confirmation: list[str] = field(default_factory=list)
    unsafe_or_overbroad_assumptions: list[str] = field(default_factory=list)
    llm_metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)


@dataclass
class ResearchLens:
    """A researcher-style viewpoint for exploring a central question."""

    name: str
    role: str
    question: str
    core_concepts: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)
    seed_paper_hints: list[str] = field(default_factory=list)
    expected_evidence_types: list[str] = field(default_factory=list)
    exclusion_risks: list[str] = field(default_factory=list)


@dataclass
class QueryFamily:
    """Provider queries generated for one research lens and purpose."""

    name: str
    purpose: str
    lens_name: str
    queries_by_provider: dict[str, list[str]] = field(default_factory=dict)
    expected_paper_roles: list[str] = field(default_factory=list)
    expected_evidence_types: list[str] = field(default_factory=list)
    exclusion_terms: list[str] = field(default_factory=list)
    stop_condition: str | None = None
    priority: int = 50
    budget: int = 3
    linked_seed_titles: list[str] = field(default_factory=list)
    seed_hint_confidence: float = 0.0


@dataclass
class ResearchLensPlan:
    """A collection of research lenses for a domain and central question."""

    domain: str
    central_question: str
    lenses: list[ResearchLens] = field(default_factory=list)


@dataclass
class QueryFamilyPlan:
    """A collection of query families for a domain and central question."""

    domain: str
    central_question: str
    families: list[QueryFamily] = field(default_factory=list)


@dataclass
class SearchConstraintGroup:
    """A group of search constraints derived from structured intent concepts."""

    group_name: str
    operator: str
    terms: list[str] = field(default_factory=list)
    source: str = ""
    required: bool = False


@dataclass
class SearchContract:
    """Explicit retrieval contract derived from a user question and search brief."""

    original_question: str
    refined_question: str
    user_goal: str
    search_intent: str
    domain_profile: DomainProfile
    must_include_concepts: list[str] = field(default_factory=list)
    must_exclude_concepts: list[str] = field(default_factory=list)
    optional_concepts: list[str] = field(default_factory=list)
    uncertain_concepts: list[str] = field(default_factory=list)
    dropped_downweighted_terms: list[str] = field(default_factory=list)
    constraint_groups: list[SearchConstraintGroup] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    required_aspects: list[str] = field(default_factory=list)
    preferred_paper_types: list[str] = field(default_factory=list)
    time_window: str = ""
    success_definition: str = ""
    generic_intent_frame: GenericResearchIntentFrame | None = None
    concept_validation_events: list[dict[str, Any]] = field(default_factory=list)


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
    required_aspects: list[str] = field(default_factory=list)
    openalex_queries: list[str] = field(default_factory=list)
    semantic_scholar_queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    expert_rewritten_question: str = ""
    intent_assumptions: list[str] = field(default_factory=list)
    downweighted_user_terms: list[str] = field(default_factory=list)
    query_families_applied: bool = False


@dataclass
class SeedPaper:
    """User-provided or auto-selected seed paper for citation expansion."""

    seed_id: str
    seed_type: str = "title"
    title: str = ""
    doi: str = ""
    note: str = ""


@dataclass
class SeedHint:
    """A bibliographic seed mention extracted from the user's question."""

    title: str | None = None
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None
    raw_mention: str = ""
    confidence: float = 0.0
    extraction_reason: str = ""


@dataclass
class RetrievalPath:
    """Trace how a paper entered the candidate set."""

    paper_id: str
    source_stage: str
    seed_paper_id: str
    seed_title: str
    reason: str
    seed_relation: str = ""
    seed_confidence: float = 0.0


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
    retrieval_provider: str = ""
    retrieval_stage: str = ""
    retrieval_query: str = ""
    source_stage: str = ""
    seed_paper_id: str = ""
    seed_title: str = ""
    seed_reason: str = ""
    seed_relation: str = ""
    seed_confidence: float = 0.0
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


SEED_EXACT_SOURCE_STAGES = {"seed_exact", "manual_seed"}
SEED_DERIVED_SOURCE_STAGES = {
    *SEED_EXACT_SOURCE_STAGES,
    "seed_reference",
    "seed_citation",
    "seed_recommendation",
}


def source_stage_values(source_stage: str) -> list[str]:
    """Return normalized semicolon-delimited source stages."""

    return [
        value.strip()
        for value in str(source_stage or "").split(";")
        if value.strip()
    ]


def is_user_seed_paper(paper: Paper) -> bool:
    """Return True for user-provided exact or manual seed records."""

    stages = set(source_stage_values(paper.source_stage))
    return bool(stages & SEED_EXACT_SOURCE_STAGES) or paper.seed_relation == "self"


def is_seed_derived_paper(paper: Paper) -> bool:
    """Return True for records that entered through seed exact/expansion paths."""

    stages = set(source_stage_values(paper.source_stage))
    return bool(stages & SEED_DERIVED_SOURCE_STAGES) or bool(paper.seed_title)


class EvidenceFunction(str, Enum):
    """Research-argument function of an extracted evidence sentence."""

    DEFINES_CONCEPT = "defines_concept"
    PREDICTS_EFFECT = "predicts_effect"
    REPORTS_EXPERIMENT = "reports_experiment"
    DIRECTLY_IMAGES_SIGNAL = "directly_images_signal"
    MEASURES_SPIN_POLARIZATION = "measures_spin_polarization"
    REPORTS_SURFACE_PROBE = "reports_surface_probe"
    CONNECTS_TO_APPLICATION = "connects_to_application"
    REPORTS_LIMITATION = "reports_limitation"
    REVIEW_BACKGROUND = "review_background"
    UNKNOWN = "unknown"


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
    evidence_function: EvidenceFunction = EvidenceFunction.UNKNOWN
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
class DomainAssessment:
    """Domain guardrail decision for one paper."""

    paper_id: str
    domain_match_score: float
    domain_decision: str
    off_topic_reason: str
    positive_domain_matches: list[str] = field(default_factory=list)
    negative_domain_matches: list[str] = field(default_factory=list)
    missing_required_concepts: list[str] = field(default_factory=list)
    forbidden_concepts_found: list[str] = field(default_factory=list)


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
    aspect_coverage_score: float = 0.0
    domain_penalty_multiplier: float = 1.0
    pre_domain_final_score: float = 0.0
    preference_score: float = 0.0
    preference_adjustment: float = 0.0
    role_adjustment: float = 0.0
    lane_adjustment: float = 0.0
    seed_or_title_mention_boost: float = 0.0
    false_positive_penalty: float = 0.0


@dataclass
class PreferenceLearningResult:
    """Learned relevance preferences from human feedback."""

    enabled: bool
    model_type: str = "none"
    labeled_paper_count: int = 0
    include_count: int = 0
    exclude_count: int = 0
    preference_scores: dict[str, float] = field(default_factory=dict)
    positive_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    suggested_must_terms: list[str] = field(default_factory=list)
    suggested_optional_terms: list[str] = field(default_factory=list)
    suggested_exclude_terms: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class AspectCoverageRecord:
    """Required-aspect coverage for one screened paper."""

    paper_id: str
    title: str
    covered_aspects: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    aspect_coverage_score: float = 0.0


@dataclass
class ScreeningDecision:
    """Human-readable screening recommendation for one ranked paper."""

    paper_id: str
    decision: str
    decision_confidence: float
    primary_reason: str
    exclusion_reasons: list[str] = field(default_factory=list)
    required_aspects_covered: list[str] = field(default_factory=list)
    required_aspects_missing: list[str] = field(default_factory=list)
    domain_match_score: float = 0.0
    domain_decision: str = ""
    reading_priority: str = ""
    suggested_action: str = ""


@dataclass
class PaperRoleRecord:
    """Research-role labels assigned to one paper for sensemaking artifacts."""

    paper_id: str
    title: str
    roles: list[str] = field(default_factory=list)
    primary_role: str = ""
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    linked_lenses: list[str] = field(default_factory=list)
    linked_query_families: list[str] = field(default_factory=list)
    content_roles: list[str] = field(default_factory=list)
    retrieval_lanes: list[str] = field(default_factory=list)
    overbroad_role_warning: str = ""


@dataclass
class ResearchTension:
    """A controversy, limitation, or boundary condition found across papers."""

    tension_key: str
    tension_label: str
    description: str
    supporting_paper_ids: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    why_it_matters: str = ""
    suggested_next_searches: list[str] = field(default_factory=list)
    confidence: float = 0.0


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
    domain_assessment: DomainAssessment | None = None
    screening_decision: ScreeningDecision | None = None


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
    search_brief: SearchBrief | None = None
    search_contract: SearchContract | None = None
    ambiguity_analysis: list[dict[str, Any]] = field(default_factory=list)
    domain_assessments: list[DomainAssessment] = field(default_factory=list)
    screening_decisions: list[ScreeningDecision] = field(default_factory=list)
    paper_role_records: list[PaperRoleRecord] = field(default_factory=list)
    research_tensions: list[ResearchTension] = field(default_factory=list)
    seed_papers: list[SeedPaper] = field(default_factory=list)
    seed_hints: list[SeedHint] = field(default_factory=list)
    retrieval_paths: list[RetrievalPath] = field(default_factory=list)
    citation_expansion_papers: list[Paper] = field(default_factory=list)
    preference_learning: PreferenceLearningResult | None = None
    feedback_query_refinement: dict[str, Any] = field(default_factory=dict)
    query_pilot_diagnostics: dict[str, Any] = field(default_factory=dict)
    query_repair_suggestions: dict[str, Any] = field(default_factory=dict)
    question_refinement: dict[str, Any] = field(default_factory=dict)
    aspect_coverage_records: list[AspectCoverageRecord] = field(default_factory=list)
    result_groups: dict[str, Any] = field(default_factory=dict)
    concept_map: ResearchLensPlan | None = None
    query_family_plan: QueryFamilyPlan | None = None
    query_provenance: dict[str, Any] = field(default_factory=dict)
    expert_research_intent: ExpertResearchIntent | None = None
