from lit_screening.agents.ambiguity_detector import AmbiguityDetectorAgent
from lit_screening.agents.extractor import ExtractorAgent
from lit_screening.agents.planner import (
    PlannerAgent,
    build_openalex_queries,
    build_semantic_scholar_queries,
)
from lit_screening.agents.research_intent import ResearchIntentAgent
from lit_screening.agents.search_contract import SearchContractAgent
from lit_screening.agents.verifier import VerifierAgent
from lit_screening.models import (
    DomainProfile,
    EvidenceRecord,
    Paper,
    QueryPlan,
    SearchContract,
)


def test_extractor_does_not_hallucinate_when_abstract_is_missing():
    paper = Paper(paper_id="p1", title="Title only")

    evidence = ExtractorAgent().extract(paper, "How can LLM agents verify evidence?")

    assert evidence.claim == ""
    assert evidence.evidence_sentence == ""
    assert "Missing abstract" in evidence.limitation


def test_verifier_flags_missing_evidence():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="This abstract discusses evidence verification.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Paper",
        claim="",
        evidence_sentence="",
        relevance_reason="test",
    )

    result = VerifierAgent().verify(paper, evidence)

    assert result.supported is False
    assert result.error_type == "missing_evidence"


def test_planner_does_not_inject_unrelated_llm_terms():
    queries = PlannerAgent().plan("the significance of surface magnetization")
    joined = " ".join(queries).lower()

    assert "surface magnetization" in joined
    assert "llm" not in joined
    assert "human-in-the-loop" not in joined
    assert "multi-agent" not in joined


def test_structured_planner_does_not_inject_llm_terms_for_materials_question():
    plan = PlannerAgent().plan_structured("the significance of surface magnetization")
    joined = " ".join(
        [
            *plan.core_terms,
            *plan.must_terms,
            *plan.optional_terms,
            *plan.openalex_queries,
            *plan.semantic_scholar_queries,
        ]
    ).lower()

    assert "surface magnetization" in joined
    assert "llm" not in joined
    assert "multi-agent" not in joined
    assert "human feedback" not in joined


def test_research_intent_defaults_plain_science_question_to_overview():
    brief = ResearchIntentAgent().analyze("surface magnetization")

    assert brief.search_intent == "overview"
    assert "systematic review" not in brief.inclusion_criteria
    assert "screening criteria" not in brief.inclusion_criteria


def test_planner_does_not_use_full_question_as_must_term():
    question = "the significance of surface magnetization"
    plan = PlannerAgent().plan_structured(question)

    assert question not in plan.must_terms
    assert "systematic review" not in plan.must_terms
    assert "surface magnetization" in plan.openalex_queries


def test_structured_planner_includes_llm_terms_for_llm_agent_question():
    plan = PlannerAgent().plan_structured(
        "How can human feedback improve multi-agent LLM literature screening?"
    )
    joined = " ".join([*plan.core_terms, *plan.openalex_queries]).lower()

    assert "human feedback" in joined
    assert "llm" in joined
    assert "literature screening" in joined


def test_ambiguity_detector_distinguishes_literature_screening():
    analysis = AmbiguityDetectorAgent().analyze(
        "How can LLM agents improve literature screening?"
    )
    screening = next(record for record in analysis if record["term"] == "screening")

    assert screening["selected_meaning"] == "literature screening"
    assert "literature screening" in screening["recommended_must_terms"]
    assert "patient screening" in screening["recommended_exclude_terms"]
    assert "drug screening" in screening["recommended_exclude_terms"]


def test_ambiguity_detector_distinguishes_llm_agent_from_biological_agent():
    analysis = AmbiguityDetectorAgent().analyze(
        "How should an LLM agent verify evidence?"
    )
    agent = next(record for record in analysis if record["term"] == "agent")

    assert agent["selected_meaning"] == "software/LLM agent"
    assert "LLM agent" in agent["recommended_must_terms"]
    assert "biological agent" in agent["recommended_exclude_terms"]
    assert "chemical agent" in agent["recommended_exclude_terms"]


def test_search_contract_materials_question_does_not_inject_llm_terms():
    question = "the significance of surface magnetization in antiferromagnets"
    brief = ResearchIntentAgent().analyze(question)
    contract = SearchContractAgent().build(
        question,
        search_brief=brief,
        ambiguity_analysis=AmbiguityDetectorAgent().analyze(question),
    )
    joined_must = " ".join(contract.must_include_concepts).lower()

    assert contract.domain_profile.domain_name == "materials_magnetism"
    assert "surface magnetization" in joined_must
    assert "llm" not in joined_must
    assert "large language model" in contract.must_exclude_concepts


def test_search_contract_ai_literature_screening_excludes_clinical_meanings():
    question = "How can LLM agents improve scientific literature screening?"
    brief = ResearchIntentAgent().analyze(question)
    ambiguity = AmbiguityDetectorAgent().analyze(question)
    contract = SearchContractAgent().build(
        question,
        search_brief=brief,
        ambiguity_analysis=ambiguity,
    )

    assert contract.domain_profile.domain_name == "ai_literature_screening"
    assert "literature screening" in contract.must_include_concepts
    assert "patient screening" in contract.must_exclude_concepts
    assert "drug screening" in contract.must_exclude_concepts


def test_planner_uses_search_contract_must_and_exclude_concepts():
    contract = SearchContract(
        original_question="ranking papers",
        refined_question="ranking papers for literature screening",
        user_goal="Find relevant papers.",
        search_intent="overview",
        domain_profile=DomainProfile(
            domain_name="ai_literature_screening",
            forbidden_concepts=["patient screening"],
        ),
        must_include_concepts=["literature screening", "relevance ranking"],
        must_exclude_concepts=["patient screening", "drug screening"],
        inclusion_criteria=["literature screening"],
        exclusion_criteria=["patient screening"],
        required_aspects=["ranking", "screening"],
        preferred_paper_types=["method paper"],
        time_window="no strict time window",
        success_definition="Relevant ranked papers.",
    )

    plan = PlannerAgent().plan_structured(
        "ranking papers",
        search_contract=contract,
    )
    joined_queries = " ".join([*plan.openalex_queries, *plan.semantic_scholar_queries])

    assert "literature screening" in plan.must_terms
    assert "relevance ranking" in plan.must_terms
    assert "patient screening" in plan.exclude_terms
    assert "drug screening" in plan.exclude_terms
    assert "ranking" in plan.required_aspects
    assert "patient screening" not in joined_queries


def test_openalex_query_builder_quotes_multi_word_core_terms():
    plan = QueryPlan(
        original_question="surface magnetization",
        core_terms=["surface magnetization"],
        must_terms=["surface magnetization"],
        optional_terms=["review"],
    )

    queries = build_openalex_queries(plan)

    assert any('"surface magnetization"' in query for query in queries)


def test_semantic_scholar_query_builder_keeps_excluded_terms_out_of_provider_query():
    plan = QueryPlan(
        original_question="surface magnetization",
        core_terms=["surface magnetization"],
        must_terms=["surface magnetization"],
        optional_terms=["review"],
        exclude_terms=["device noise"],
    )

    queries = build_semantic_scholar_queries(plan)

    assert any('"surface magnetization"' in query for query in queries)
    assert not any('+"surface magnetization"' in query for query in queries)
    assert not any('-"device noise"' in query for query in queries)


def test_planner_translates_common_chinese_terms_without_llm():
    planner = PlannerAgent()
    queries = planner.plan("表面磁化的重要性")
    joined = " ".join(queries).lower()

    assert "surface magnetization" in joined
    assert "importance" in joined
    assert "表面" not in joined
    assert planner.last_llm_metadata["question_language"] == "zh"
    assert planner.last_llm_metadata["translation_mode"] == "rule_glossary"
    assert planner.last_llm_metadata["planning_question"] == "surface magnetization importance"


def test_verifier_requires_strict_span_for_supported():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="Surface magnetization is important for boundary spin signals.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Paper",
        claim="Surface magnetization is important.",
        evidence_sentence="Surface magnetization is important for boundary spin signals.",
        relevance_reason="test",
    )

    result = VerifierAgent().verify(paper, evidence)

    assert result.supported is True
    assert result.support_level == "strict_support"
    assert result.span_match_type == "exact"


def test_verifier_downgrades_overlap_to_weak_support():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="Surface magnetization controls boundary spin signals in antiferromagnets.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Paper",
        claim="Surface magnetization controls spin signals.",
        evidence_sentence="Surface magnetization boundary spin antiferromagnets controls signals",
        relevance_reason="test",
    )

    result = VerifierAgent().verify(paper, evidence)

    assert result.supported is False
    assert result.support_level == "weak_support"
    assert result.error_type == "weak_support"


def test_verifier_marks_unmatched_llm_evidence_invalid():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="Surface magnetization controls boundary spin signals in antiferromagnets.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Paper",
        claim="Surface magnetization creates a giant device effect.",
        evidence_sentence="Surface magnetization creates a giant device effect.",
        relevance_reason="test",
        llm_used=True,
        extraction_mode="llm",
    )

    result = VerifierAgent().verify(paper, evidence)

    assert result.supported is False
    assert result.support_level == "llm_invalid_evidence"
    assert result.error_type == "llm_invalid_evidence"
