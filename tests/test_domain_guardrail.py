import pytest

from lit_screening.agents.ambiguity_detector import AmbiguityDetectorAgent
from lit_screening.agents.domain_guardrail import DomainGuardrailAgent
from lit_screening.agents.ranker import RankerAgent
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


def test_query_provenance_blacklist_terms_do_not_pollute_domain_judgment():
    contract = build_contract(
        "How can LLM agents improve scientific literature screening?"
    )
    paper = Paper(
        paper_id="clean-ai-paper",
        title="Human-in-the-loop LLM agents for scientific literature screening",
        abstract=(
            "This paper studies evidence verification for scientific literature "
            "screening with human feedback."
        ),
        venue="ACL",
        fields_of_study=["Computer Science"],
        retrieval_query="patient screening drug screening",
        raw={
            "matched_query": "patient screening drug screening",
            "seed_reason": "drug screening provenance noise",
        },
    )

    assessment = DomainGuardrailAgent().assess(paper, contract)

    assert assessment.domain_decision == "in_scope"
    assert "patient screening" not in assessment.forbidden_concepts_found
    assert "drug screening" not in assessment.forbidden_concepts_found


def test_field_blacklist_is_weak_when_content_has_strong_positive_evidence():
    contract = build_contract(
        "the significance of surface magnetization in antiferromagnets"
    )
    paper = Paper(
        paper_id="materials-with-cs-field",
        title="Surface magnetization in antiferromagnetic materials",
        abstract=(
            "Surface magnetization and antiferromagnetic materials provide "
            "strong positive evidence for the materials magnetism domain."
        ),
        fields_of_study=["Computer Science"],
    )

    assessment = DomainGuardrailAgent().assess(paper, contract)

    assert assessment.domain_decision != "out_of_scope"
    assert "Computer Science" in assessment.negative_domain_matches


def test_sei_lithium_context_required_for_top20():
    contract = build_contract(
        "我想了解锂离子电池中 SEI 界面的组成、结构、演化和失效机制。"
    )
    lithium = Paper(
        paper_id="li-sei",
        title="Solid electrolyte interphase evolution in lithium-ion batteries",
        abstract="The SEI composition and structure evolve on graphite anodes in lithium-ion batteries.",
    )
    sodium = Paper(
        paper_id="na-sei",
        title="Solid electrolyte interphase in sodium-ion batteries",
        abstract="This review studies SEI formation in sodium-ion battery anodes.",
    )

    li_assessment = DomainGuardrailAgent().assess(lithium, contract)
    sodium_assessment = DomainGuardrailAgent().assess(sodium, contract)

    assert li_assessment.domain_decision == "in_scope"
    assert li_assessment.intent_centrality_score > sodium_assessment.intent_centrality_score
    assert sodium_assessment.domain_decision != "in_scope"


def test_sei_sodium_zinc_potassium_downranked():
    contract = build_contract("我想了解锂离子电池中 SEI 界面的失效机制。")
    papers = [
        Paper(
            paper_id="li",
            title="SEI failure mechanism in lithium-ion batteries",
            abstract="Lithium-ion battery SEI failure is linked to electrolyte decomposition.",
            year=2021,
            citation_count=20,
        ),
        Paper(
            paper_id="zn",
            title="SEI stability in zinc-ion batteries",
            abstract="Zinc-ion battery SEI stability is reviewed for aqueous electrolytes.",
            year=2025,
            citation_count=200,
        ),
    ]
    ranked = _rank_for_contract(papers, contract)

    assert ranked[0].paper.paper_id == "li"
    assert ranked[1].domain_assessment.domain_decision != "in_scope"


def test_oer_requires_reaction_and_spin_groups_for_include():
    contract = build_contract(
        "我想了解过渡金属氧化物催化剂表面自旋态对析氧反应 OER 活性的影响。"
    )
    intersection = Paper(
        paper_id="intersection",
        title="Surface spin state controls oxygen evolution reaction activity in transition metal oxide catalysts",
        abstract="The OER activity of oxide catalysts is governed by surface spin state and electronic structure.",
    )
    oer_only = Paper(
        paper_id="oer-review",
        title="Advanced transition metal-based OER electrocatalysts",
        abstract="This review covers oxygen evolution reaction catalysts but not spin-state descriptors.",
    )
    spin_only = Paper(
        paper_id="spin-catalysis",
        title="Spin effects in chemisorption and catalysis",
        abstract="Spin polarization affects chemisorption but oxygen evolution reaction is not studied.",
    )

    good = DomainGuardrailAgent().assess(intersection, contract)
    oer = DomainGuardrailAgent().assess(oer_only, contract)
    spin = DomainGuardrailAgent().assess(spin_only, contract)

    assert good.domain_decision == "in_scope"
    assert all(good.required_group_matches.values())
    assert oer.domain_decision != "in_scope"
    assert spin.domain_decision != "in_scope"


def test_oer_intersection_papers_rank_above_broad_single_axis_reviews():
    contract = build_contract(
        "我想了解过渡金属氧化物催化剂表面自旋态对析氧反应 OER 活性的影响。"
    )
    papers = [
        Paper(
            paper_id="intersection",
            title="Engineering electrocatalytic activity through surface spin-state transition",
            abstract="A transition metal oxide catalyst improves oxygen evolution reaction activity through a surface spin-state transition and electronic structure tuning.",
            year=2020,
            citation_count=20,
        ),
        Paper(
            paper_id="broad",
            title="Advanced transition metal-based OER electrocatalysts",
            abstract="This broad OER review summarizes transition metal catalysts and performance trends.",
            year=2025,
            citation_count=500,
        ),
    ]
    ranked = _rank_for_contract(papers, contract)

    assert ranked[0].paper.paper_id == "intersection"
    assert ranked[0].scores.intent_centrality_score > ranked[1].scores.intent_centrality_score


def _rank_for_contract(papers: list[Paper], contract):
    evidence = [
        EvidenceRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            claim=paper.abstract,
            evidence_sentence=paper.abstract,
            relevance_reason="test",
        )
        for paper in papers
    ]
    verification = [
        VerificationResult(
            paper_id=paper.paper_id,
            supported=True,
            confidence=1.0,
            error_type="",
            rationale="test",
            support_level="strict_support",
        )
        for paper in papers
    ]
    assessments = DomainGuardrailAgent().assess_many(papers, contract)
    return RankerAgent().rank(
        papers,
        evidence,
        verification,
        contract.refined_question,
        domain_assessments=assessments,
    )
