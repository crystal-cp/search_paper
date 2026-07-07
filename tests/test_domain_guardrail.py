import pytest

from lit_screening.agents.ambiguity_detector import AmbiguityDetectorAgent
from lit_screening.agents.domain_guardrail import DomainGuardrailAgent
from lit_screening.agents.research_intent import ResearchIntentAgent
from lit_screening.agents.search_contract import SearchContractAgent
from lit_screening.models import (
    DomainAssessment,
    EvidenceRecord,
    Paper,
    VerificationResult,
)
from lit_screening.scoring import compute_score_breakdown


def build_contract(question: str):
    brief = ResearchIntentAgent().analyze(question)
    ambiguity = AmbiguityDetectorAgent().analyze(question)
    return SearchContractAgent().build(
        question,
        search_brief=brief,
        ambiguity_analysis=ambiguity,
    )


def test_ai_literature_contract_marks_clinical_screening_out_of_scope():
    contract = build_contract(
        "How can LLM agents improve scientific literature screening?"
    )
    paper = Paper(
        paper_id="clinical-screening",
        title="Deep learning for patient screening and biomarker screening",
        abstract=(
            "This clinical study evaluates patient screening, drug screening, "
            "and biomarker screening workflows."
        ),
        fields_of_study=["Medicine"],
    )

    assessment = DomainGuardrailAgent().assess(paper, contract)

    assert assessment.domain_decision == "out_of_scope"
    assert "patient screening" in assessment.forbidden_concepts_found
    assert "drug screening" in assessment.forbidden_concepts_found
    assert assessment.domain_match_score < 0.3


def test_materials_magnetism_contract_marks_healthcare_llm_agents_out_of_scope():
    contract = build_contract(
        "the significance of surface magnetization in antiferromagnets"
    )
    paper = Paper(
        paper_id="healthcare-llm",
        title="Large language model agents for patient screening in healthcare",
        abstract=(
            "LLM agents help patient screening and clinical triage but do not "
            "study surface magnetization or antiferromagnetic materials."
        ),
        fields_of_study=["Computer Science", "Medicine"],
    )

    assessment = DomainGuardrailAgent().assess(paper, contract)

    assert assessment.domain_decision == "out_of_scope"
    assert "large language model" in assessment.forbidden_concepts_found
    assert "patient screening" in assessment.forbidden_concepts_found


def test_in_scope_paper_is_not_penalized():
    question = "How can LLM agents improve scientific literature screening and evidence verification?"
    contract = build_contract(question)
    paper = Paper(
        paper_id="ai-lit",
        title="Human-in-the-loop LLM agents for scientific literature screening and evidence verification",
        abstract=(
            "Human-in-the-loop LLM agents improve scientific literature screening "
            "and evidence verification by supporting claim extraction."
        ),
        year=2024,
        venue="ACL",
        citation_count=10,
        fields_of_study=["Computer Science", "Artificial Intelligence"],
    )
    evidence = EvidenceRecord(
        paper_id=paper.paper_id,
        title=paper.title,
        claim="LLM agents improve literature screening.",
        evidence_sentence=paper.abstract,
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id=paper.paper_id,
        supported=True,
        confidence=1.0,
        error_type="",
        rationale="test",
        support_level="strict_support",
    )
    assessment = DomainGuardrailAgent().assess(paper, contract)
    scores = compute_score_breakdown(
        paper,
        evidence,
        verification,
        question,
        domain_assessment=assessment,
    )

    assert assessment.domain_decision == "in_scope"
    assert scores.domain_penalty_multiplier == 1.0
    assert scores.final_score == scores.pre_domain_final_score


def test_out_of_scope_paper_receives_final_score_penalty():
    paper = Paper(
        paper_id="clinical",
        title="Patient screening with biomarkers",
        abstract="Drug screening and biomarker screening are evaluated.",
        year=2024,
        venue="Medical Journal",
        citation_count=10,
    )
    evidence = EvidenceRecord(
        paper_id=paper.paper_id,
        title=paper.title,
        claim="Patient screening is evaluated.",
        evidence_sentence=paper.abstract,
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id=paper.paper_id,
        supported=True,
        confidence=1.0,
        error_type="",
        rationale="test",
        support_level="strict_support",
    )
    assessment = DomainAssessment(
        paper_id=paper.paper_id,
        domain_match_score=0.1,
        domain_decision="out_of_scope",
        off_topic_reason="Forbidden clinical screening concept.",
        forbidden_concepts_found=["patient screening"],
    )

    scores = compute_score_breakdown(
        paper,
        evidence,
        verification,
        "LLM agents for scientific literature screening",
        domain_assessment=assessment,
    )

    assert scores.domain_penalty_multiplier == 0.3
    assert scores.final_score == pytest.approx(scores.pre_domain_final_score * 0.3)
