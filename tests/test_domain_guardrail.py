import pytest

from lit_screening.agents.ambiguity_detector import AmbiguityDetectorAgent
from lit_screening.agents.domain_guardrail import DomainGuardrailAgent
from lit_screening.agents.intent_repair import NoviceIntentInterpreter
from lit_screening.agents.ranker import RankerAgent
from lit_screening.agents.research_intent import ResearchIntentAgent
from lit_screening.agents.screening_decision import ScreeningDecisionAgent, reading_priority
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


def build_contract_with_intent_repair(question: str):
    brief = ResearchIntentAgent().analyze(question)
    ambiguity = AmbiguityDetectorAgent().analyze(question)
    intent = NoviceIntentInterpreter().repair(question)
    return SearchContractAgent().build(
        question,
        search_brief=brief,
        ambiguity_analysis=ambiguity,
        expert_intent=intent,
    )


def _priority_domain(
    *,
    required_group_matches: dict[str, bool] | None = None,
    target_context_match: list[str] | None = None,
    negative_context_match: list[str] | None = None,
    peripheral_context_reason: str = "",
    intent_centrality_score: float = 0.9,
    required_group_coverage_score: float = 1.0,
    missing_required_group_count: int = 0,
    topic_focus_score: float = 0.9,
    domain_decision: str = "in_scope",
) -> DomainAssessment:
    return DomainAssessment(
        paper_id="priority-test",
        domain_match_score=0.95,
        domain_decision=domain_decision,
        off_topic_reason="test",
        required_group_matches=required_group_matches
        or {"target_chemistry_group": True, "core_object_group": True},
        target_context_match=(
            ["lithium metal battery"]
            if target_context_match is None
            else target_context_match
        ),
        negative_context_match=(
            [] if negative_context_match is None else negative_context_match
        ),
        peripheral_context_reason=peripheral_context_reason,
        intent_centrality_score=intent_centrality_score,
        required_group_coverage_score=required_group_coverage_score,
        missing_required_group_count=missing_required_group_count,
        topic_focus_score=topic_focus_score,
    )


def test_include_does_not_imply_must_read():
    domain = _priority_domain()

    assert (
        reading_priority(
            "include",
            0.82,
            "in_scope",
            domain=domain,
            rank=16,
        )
        == "read_later"
    )


def test_must_read_count_is_bounded_for_sei_lithium_case():
    domain = _priority_domain()
    priorities = [
        reading_priority("include", 0.9, "in_scope", domain=domain, rank=rank)
        for rank in range(1, 26)
    ]

    assert priorities.count("must_read") == 12
    assert len(priorities) > priorities.count("must_read")


def test_must_read_requires_target_context_match():
    domain = _priority_domain(target_context_match=[])

    assert (
        reading_priority("include", 0.9, "in_scope", domain=domain, rank=1)
        == "read_later"
    )


def test_negative_context_paper_never_must_read():
    domain = _priority_domain(negative_context_match=["sodium-ion"])

    assert (
        reading_priority("include", 0.95, "in_scope", domain=domain, rank=1)
        == "optional"
    )


def test_mixed_chemistry_review_optional_not_must_read():
    domain = _priority_domain(
        target_context_match=["lithium metal battery"],
        negative_context_match=["sodium", "potassium"],
        peripheral_context_reason="negative_or_beyond_target_chemistry_context",
    )

    assert (
        reading_priority("include", 0.95, "in_scope", domain=domain, rank=1)
        == "optional"
    )


def test_high_centrality_lithium_artificial_sei_can_be_must_read():
    domain = _priority_domain(
        target_context_match=["lithium metal battery", "lithium metal anode"],
        intent_centrality_score=0.95,
        required_group_coverage_score=1.0,
        topic_focus_score=0.92,
    )

    assert (
        reading_priority("include", 0.86, "in_scope", domain=domain, rank=3)
        == "must_read"
    )


def test_must_read_does_not_require_target_context_when_no_target_group():
    domain = _priority_domain(
        required_group_matches={
            "oer_reaction_group": True,
            "spin_electronic_group": True,
            "catalyst_material_group": True,
        },
        target_context_match=[],
        intent_centrality_score=1.0,
        required_group_coverage_score=1.0,
        topic_focus_score=0.95,
    )

    assert (
        reading_priority(
            "include",
            0.9,
            "in_scope",
            domain=domain,
            rank=1,
            support_level="strict_support",
        )
        == "must_read"
    )


def test_oer_full_coverage_in_scope_papers_can_be_must_read():
    domain = _priority_domain(
        required_group_matches={
            "oxygen_evolution_group": True,
            "spin_state_group": True,
            "electronic_structure_group": True,
        },
        target_context_match=[],
        intent_centrality_score=0.98,
        required_group_coverage_score=1.0,
        topic_focus_score=0.9,
    )

    assert (
        reading_priority(
            "include",
            0.88,
            "in_scope",
            domain=domain,
            rank=4,
            support_level="strict_support",
        )
        == "must_read"
    )


def test_oer_must_read_count_nonzero_and_bounded():
    domain = _priority_domain(
        required_group_matches={
            "oxygen_evolution_group": True,
            "spin_state_group": True,
            "electronic_structure_group": True,
        },
        target_context_match=[],
        intent_centrality_score=0.99,
        required_group_coverage_score=1.0,
        topic_focus_score=0.92,
    )
    priorities = [
        reading_priority(
            "include",
            0.9,
            "in_scope",
            domain=domain,
            rank=rank,
            support_level="strict_support",
        )
        for rank in range(1, 28)
    ]

    assert 8 <= priorities.count("must_read") <= 12
    assert priorities.count("must_read") < priorities.count("read_later")


def test_sei_must_read_still_requires_target_context():
    domain = _priority_domain(target_context_match=[])

    assert (
        reading_priority(
            "include",
            0.95,
            "in_scope",
            domain=domain,
            rank=1,
            support_level="strict_support",
        )
        == "read_later"
    )


def test_negative_context_never_must_read_when_target_context_required():
    domain = _priority_domain(
        target_context_match=["lithium metal battery"],
        negative_context_match=["sodium-ion battery"],
    )

    assert (
        reading_priority(
            "include",
            0.95,
            "in_scope",
            domain=domain,
            rank=1,
            support_level="strict_support",
        )
        == "optional"
    )


def test_chinese_lithium_battery_maps_to_lithium_target_context():
    question = "我想找锂电池里 SEI 是什么、怎么影响循环。"
    intent = NoviceIntentInterpreter().repair(question)
    contract = build_contract_with_intent_repair(question)
    concept_terms = {concept.term for concept in intent.structured_concepts}
    group_by_name = {group.group_name: group for group in contract.constraint_groups}

    assert "lithium battery" in concept_terms
    assert "lithium-ion battery" in concept_terms
    assert "target_chemistry_group" in group_by_name
    assert "lithium-ion battery" in group_by_name["target_chemistry_group"].terms


def test_lithium_metal_battery_maps_to_lithium_metal_context():
    question = "我想找锂金属电池里 SEI 怎么影响枝晶。"
    intent = NoviceIntentInterpreter().repair(question)
    contract = build_contract_with_intent_repair(question)
    concept_terms = {concept.term for concept in intent.structured_concepts}
    target_group = next(
        group for group in contract.constraint_groups if group.group_name == "target_chemistry_group"
    )

    assert "lithium metal battery" in concept_terms
    assert "lithium metal anode" in concept_terms
    assert "lithium metal battery" in target_group.terms
    assert "lithium metal anode" in target_group.terms


def test_artificial_sei_maps_to_engineered_sei_terms():
    intent = NoviceIntentInterpreter().repair("我想找锂电池里人工 SEI 的文章。")
    concept_terms = {concept.term for concept in intent.structured_concepts}

    assert "artificial SEI" in concept_terms
    assert "engineered SEI" in concept_terms
    assert "artificial solid electrolyte interphase" in concept_terms


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


def test_sei_non_lithium_battery_not_must_read():
    contract = build_contract(
        "我想了解锂离子电池中 SEI 界面的组成、结构、演化和失效机制。"
    )
    paper = Paper(
        paper_id="zn-sei",
        title="Solid Electrolyte Interface in Zn-Based Battery Systems",
        abstract="This review studies SEI formation and stability in zinc-based battery systems.",
        year=2025,
        citation_count=200,
    )
    ranked = _rank_for_contract([paper], contract)
    decisions, _updated = ScreeningDecisionAgent().decide_many(ranked)
    decision = decisions[0]

    assert ranked[0].domain_assessment.domain_decision != "in_scope"
    assert ranked[0].scores.intent_centrality_score <= 0.45
    assert decision.decision != "include"
    assert decision.reading_priority != "must_read"
    assert "missing_required_group" in decision.exclusion_reasons


def test_sei_sodium_potassium_mixed_review_not_must_read_for_lithium_query():
    contract = build_contract_with_intent_repair(
        "我想找锂电池里 SEI 是什么、怎么影响锂金属电池循环和枝晶，尤其人工 SEI 的文章。"
    )
    paper = Paper(
        paper_id="mixed-li-na-k",
        title=(
            "Review of Emerging Concepts in SEI Analysis and Artificial SEI Membranes "
            "for Lithium, Sodium, and Potassium Metal Battery Anodes"
        ),
        abstract=(
            "This review compares lithium, sodium, and potassium metal battery anodes "
            "and discusses artificial SEI membranes."
        ),
        year=2025,
        citation_count=500,
    )
    ranked = _rank_for_contract([paper], contract)
    decisions, _updated = ScreeningDecisionAgent().decide_many(ranked)
    assessment = ranked[0].domain_assessment

    assert assessment.domain_decision != "in_scope"
    assert assessment.negative_context_match
    assert assessment.peripheral_context_reason == "negative_or_beyond_target_chemistry_context"
    assert decisions[0].decision != "include"
    assert decisions[0].reading_priority != "must_read"


def test_sei_sodium_ion_paper_not_include_for_lithium_query():
    contract = build_contract_with_intent_repair(
        "我想找锂电池里 SEI 是什么、怎么影响锂金属电池循环和枝晶，尤其人工 SEI 的文章。"
    )
    paper = Paper(
        paper_id="sodium-sei",
        title="Solubility of the Solid Electrolyte Interphase (SEI) in Sodium Ion Batteries",
        abstract="This paper studies SEI solubility in sodium ion batteries and hard carbon anodes.",
        year=2024,
        citation_count=300,
    )
    ranked = _rank_for_contract([paper], contract)
    decisions, _updated = ScreeningDecisionAgent().decide_many(ranked)

    assert ranked[0].domain_assessment.negative_context_match
    assert ranked[0].domain_assessment.domain_decision != "in_scope"
    assert decisions[0].decision != "include"
    assert decisions[0].reading_priority != "must_read"


def test_mixed_li_na_k_review_marked_peripheral_background():
    contract = build_contract_with_intent_repair(
        "我想找锂电池里 SEI 是什么、怎么影响锂金属电池循环和枝晶，尤其人工 SEI 的文章。"
    )
    paper = Paper(
        paper_id="mixed-review",
        title="Artificial SEI membranes for lithium, sodium, and potassium metal battery anodes",
        abstract="The review compares Li, Na, and K metal batteries rather than centering only lithium.",
    )
    assessment = DomainGuardrailAgent().assess(paper, contract)

    assert assessment.target_context_match
    assert assessment.negative_context_match
    assert assessment.domain_decision == "borderline"
    assert assessment.peripheral_context_reason == "negative_or_beyond_target_chemistry_context"
    assert assessment.topic_focus_score <= 0.55


def test_sei_zinc_sodium_potassium_downranked():
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
        Paper(
            paper_id="na",
            title="SEI formation in sodium-ion batteries",
            abstract="Sodium-ion battery SEI formation is reviewed for hard carbon anodes.",
            year=2024,
            citation_count=150,
        ),
        Paper(
            paper_id="k",
            title="SEI evolution in potassium-ion batteries",
            abstract="Potassium-ion battery SEI evolution is studied during cycling.",
            year=2024,
            citation_count=120,
        ),
    ]
    ranked = _rank_for_contract(papers, contract)
    by_id = {item.paper.paper_id: item for item in ranked}

    assert ranked[0].paper.paper_id == "li"
    for paper_id in {"zn", "na", "k"}:
        assert by_id[paper_id].domain_assessment.domain_decision != "in_scope"
        assert by_id[paper_id].scores.final_score < by_id["li"].scores.final_score
        assert by_id[paper_id].domain_assessment.missing_required_group_count > 0


def test_ablation_sei_shorter_query_forbidden_must_read_reduced():
    contract = build_contract_with_intent_repair(
        "我想找锂电池里 SEI 是什么、怎么影响锂金属电池循环和枝晶，尤其人工 SEI 的文章。"
    )
    papers = [
        Paper(
            paper_id="focused-li-artificial",
            title="Artificial SEI for stable lithium metal battery anodes",
            abstract=(
                "Artificial solid electrolyte interphase layers suppress lithium dendrites "
                "and improve lithium metal battery cycling."
            ),
            year=2022,
            citation_count=100,
        ),
        Paper(
            paper_id="mixed-li-na-k",
            title="Review of SEI membranes for Lithium, Sodium, and Potassium Metal Battery Anodes",
            abstract="This review compares lithium, sodium, and potassium metal batteries.",
            year=2025,
            citation_count=500,
        ),
        Paper(
            paper_id="sodium-sei",
            title="Solubility of the Solid Electrolyte Interphase in Sodium Ion Batteries",
            abstract="Sodium-ion battery SEI solubility is measured for hard carbon anodes.",
            year=2024,
            citation_count=300,
        ),
        Paper(
            paper_id="zinc-sei",
            title="Interphase chemistry in aqueous zinc-ion batteries",
            abstract="AZIB and zinc-ion battery interphases are reviewed.",
            year=2024,
            citation_count=250,
        ),
    ]
    ranked = _rank_for_contract(papers, contract)
    decisions, _updated = ScreeningDecisionAgent().decide_many(ranked)
    decision_by_id = {decision.paper_id: decision for decision in decisions}

    assert ranked[0].paper.paper_id == "focused-li-artificial"
    forbidden_ids = {"mixed-li-na-k", "sodium-sei", "zinc-sei"}
    assert all(
        decision_by_id[paper_id].reading_priority != "must_read"
        for paper_id in forbidden_ids
    )
    assert all(
        decision_by_id[paper_id].decision != "include"
        for paper_id in {"sodium-sei", "zinc-sei"}
    )


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


def test_oer_missing_required_group_cannot_include():
    contract = build_contract(
        "我想了解过渡金属氧化物催化剂表面自旋态对析氧反应 OER 活性的影响。"
    )
    papers = [
        Paper(
            paper_id="oer-only",
            title="Advanced transition metal oxide OER electrocatalysts",
            abstract="This paper studies oxygen evolution reaction activity in transition metal oxide catalysts.",
            year=2025,
            citation_count=500,
        ),
        Paper(
            paper_id="spin-only",
            title="Surface spin state in oxide catalysts",
            abstract="Surface spin state and electronic structure are measured in oxide catalysts.",
            year=2025,
            citation_count=500,
        ),
    ]
    ranked = _rank_for_contract(papers, contract)
    decisions, _updated = ScreeningDecisionAgent().decide_many(ranked)

    for item, decision in zip(ranked, decisions):
        assert item.domain_assessment.missing_required_group_count > 0
        assert decision.decision != "include"
        assert decision.reading_priority != "must_read"
        assert "missing_required_group" in decision.exclusion_reasons


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


def test_intent_centrality_dominates_recency_quality():
    contract = build_contract(
        "我想了解过渡金属氧化物催化剂表面自旋态对析氧反应 OER 活性的影响。"
    )
    papers = [
        Paper(
            paper_id="intersection",
            title="Surface spin state controls oxygen evolution reaction activity in oxide catalysts",
            abstract="Transition metal oxide catalysts improve OER activity through surface spin state and electronic structure tuning.",
            year=2019,
            citation_count=10,
        ),
        Paper(
            paper_id="broad",
            title="Highly cited recent transition metal oxide OER catalyst review",
            abstract="This review summarizes oxygen evolution reaction activity and transition metal oxide electrocatalyst performance.",
            year=2026,
            citation_count=1000,
        ),
    ]
    ranked = _rank_for_contract(papers, contract)
    by_id = {item.paper.paper_id: item for item in ranked}

    assert ranked[0].paper.paper_id == "intersection"
    assert by_id["intersection"].scores.intent_centrality_score > by_id["broad"].scores.intent_centrality_score
    assert by_id["intersection"].scores.final_score > by_id["broad"].scores.final_score


def test_sei_beyond_lithium_not_must_read():
    contract = build_contract(
        "我想了解锂离子电池中 SEI 界面的组成、结构、演化和失效机制。"
    )
    paper = Paper(
        paper_id="beyond-li",
        title="Beyond Lithium-Ion Batteries",
        abstract=(
            "This review discusses beyond lithium-ion batteries, AZIBs, and zinc battery "
            "interfaces. SEI formation is mentioned as one interfacial chemistry topic."
        ),
        year=2026,
        citation_count=1000,
    )
    ranked = _rank_for_contract([paper], contract)
    decisions, _updated = ScreeningDecisionAgent().decide_many(ranked)
    assessment = ranked[0].domain_assessment

    assert assessment.domain_decision != "in_scope"
    assert assessment.negative_context_match
    assert assessment.peripheral_context_reason
    assert decisions[0].decision != "include"
    assert decisions[0].reading_priority != "must_read"


def test_sei_non_target_chemistry_downranked():
    contract = build_contract("我想了解锂离子电池中 SEI 界面的失效机制。")
    papers = [
        Paper(
            paper_id="focused-li",
            title="SEI failure mechanism in graphite anodes for lithium-ion batteries",
            abstract="Lithium-ion battery SEI composition, structure, and evolution govern graphite anode degradation.",
            year=2020,
            citation_count=20,
        ),
        Paper(
            paper_id="azib",
            title="Stable interfacial chemistry for aqueous zinc-ion batteries",
            abstract="AZIBs and zinc-based batteries form SEI-like interphases in aqueous electrolytes.",
            year=2026,
            citation_count=800,
        ),
    ]
    ranked = _rank_for_contract(papers, contract)
    by_id = {item.paper.paper_id: item for item in ranked}

    assert ranked[0].paper.paper_id == "focused-li"
    assert by_id["azib"].domain_assessment.negative_context_match
    assert by_id["azib"].domain_assessment.intent_centrality_score < by_id["focused-li"].domain_assessment.intent_centrality_score
    assert by_id["azib"].scores.final_score < by_id["focused-li"].scores.final_score


def test_sei_broad_battery_overview_not_top_core():
    contract = build_contract(
        "我想了解锂离子电池中 SEI 界面的组成、结构、演化和失效机制。"
    )
    papers = [
        Paper(
            paper_id="focused-sei",
            title="Solid electrolyte interphase composition and evolution in lithium-ion batteries",
            abstract="The lithium-ion battery SEI composition, structure, and evolution are characterized on graphite anodes.",
            year=2019,
            citation_count=15,
        ),
        Paper(
            paper_id="broad-overview",
            title="Lithium-Ion Batteries",
            abstract="This broad overview of lithium-ion batteries briefly mentions SEI composition and interfacial degradation.",
            year=2026,
            citation_count=1000,
        ),
    ]
    ranked = _rank_for_contract(papers, contract)
    decisions, _updated = ScreeningDecisionAgent().decide_many(ranked)
    by_id = {item.paper.paper_id: item for item in ranked}
    decision_by_id = {decision.paper_id: decision for decision in decisions}

    assert ranked[0].paper.paper_id == "focused-sei"
    assert by_id["broad-overview"].domain_assessment.topic_focus_score < by_id["focused-sei"].domain_assessment.topic_focus_score
    assert decision_by_id["broad-overview"].reading_priority != "must_read"


def test_oer_spin_state_synonyms_count_as_spin_group():
    contract = build_contract(
        "我想了解过渡金属氧化物催化剂表面自旋态对析氧反应 OER 活性的影响。"
    )
    paper = Paper(
        paper_id="coooh-spin-transition",
        title="An enhanced oxygen evolution reaction on 2D CoOOH via strain engineering: an insightful view from spin-state transition",
        abstract="The CoOOH electrocatalyst improves OER activity through spin-state transition and electronic structure changes.",
    )
    assessment = DomainGuardrailAgent().assess(paper, contract)

    assert assessment.missing_required_group_count == 0
    assert assessment.required_group_coverage_score == pytest.approx(1.0)
    assert assessment.domain_decision == "in_scope"


def test_oer_covalency_electronic_structure_count_as_spin_electronic_group():
    contract = build_contract(
        "我想了解过渡金属氧化物催化剂表面自旋态对析氧反应 OER 活性的影响。"
    )
    paper = Paper(
        paper_id="lacoo3-covalency",
        title="Tailoring the Co 3d-O 2p Covalency in LaCoO3 by Fe Substitution To Promote Oxygen Evolution Reaction",
        abstract="Metal-oxygen covalency and electronic structure tune the perovskite oxide catalyst for OER.",
    )
    assessment = DomainGuardrailAgent().assess(paper, contract)

    assert assessment.missing_required_group_count == 0
    assert assessment.required_group_coverage_score == pytest.approx(1.0)
    assert assessment.domain_decision == "in_scope"


def test_oer_intersection_papers_missing_required_group_zero():
    contract = build_contract(
        "我想了解过渡金属氧化物催化剂表面自旋态对析氧反应 OER 活性的影响。"
    )
    papers = [
        Paper(
            paper_id="low-spin",
            title="Low-Spin Fe3+ Evoked by Multiple Defects for Efficient Oxygen Evolution",
            abstract="Low-spin Fe3+ electronic structure improves the oxide electrocatalyst OER activity.",
        ),
        Paper(
            paper_id="spin-density",
            title="Tuning the Spin Density of Cobalt Single-Atom Catalysts for Efficient Oxygen Evolution",
            abstract="Spin density modulation in cobalt single-atom catalysts improves the oxygen evolution reaction.",
        ),
        Paper(
            paper_id="perovskite-cobaltite",
            title="Engineering electrocatalytic activity in nanosized perovskite cobaltite through surface spin-state transition",
            abstract="Surface spin-state transition in perovskite cobaltite electrocatalysts promotes OER activity.",
        ),
    ]

    for paper in papers:
        assessment = DomainGuardrailAgent().assess(paper, contract)
        assert assessment.missing_required_group_count == 0
        assert assessment.required_group_coverage_score == pytest.approx(1.0)
        assert assessment.domain_decision == "in_scope"


def test_oer_broad_spin_review_background_not_top_core():
    contract = build_contract(
        "我想了解过渡金属氧化物催化剂表面自旋态对析氧反应 OER 活性的影响。"
    )
    papers = [
        Paper(
            paper_id="focused-oer-spin",
            title="Surface spin-state transition promotes oxygen evolution reaction in perovskite cobaltite electrocatalysts",
            abstract="A perovskite cobaltite oxide catalyst shows enhanced OER through surface spin-state transition.",
            year=2020,
            citation_count=20,
        ),
        Paper(
            paper_id="broad-spin-review",
            title="Spin Effects in Chemisorption and Catalysis",
            abstract="This broad review covers spin polarization, chemisorption, and catalysis without studying oxygen evolution reaction catalysts.",
            year=2026,
            citation_count=1000,
        ),
    ]
    ranked = _rank_for_contract(papers, contract)
    by_id = {item.paper.paper_id: item for item in ranked}

    assert ranked[0].paper.paper_id == "focused-oer-spin"
    assert by_id["broad-spin-review"].domain_assessment.missing_required_group_count > 0
    assert by_id["broad-spin-review"].scores.final_score < by_id["focused-oer-spin"].scores.final_score


def test_topic_focus_affects_intent_centrality():
    contract = build_contract(
        "我想了解锂离子电池中 SEI 界面的组成、结构、演化和失效机制。"
    )
    focused = Paper(
        paper_id="focused",
        title="SEI composition and structural evolution in lithium-ion batteries",
        abstract="The lithium-ion battery SEI composition, structure, evolution, and failure are directly characterized.",
    )
    broad = Paper(
        paper_id="broad",
        title="Lithium-Ion Batteries",
        abstract="This overview briefly discusses lithium-ion battery SEI composition, structure, evolution, and failure.",
    )
    focused_assessment = DomainGuardrailAgent().assess(focused, contract)
    broad_assessment = DomainGuardrailAgent().assess(broad, contract)

    assert focused_assessment.required_group_coverage_score == pytest.approx(1.0)
    assert broad_assessment.required_group_coverage_score == pytest.approx(1.0)
    assert focused_assessment.topic_focus_score > broad_assessment.topic_focus_score
    assert focused_assessment.intent_centrality_score > broad_assessment.intent_centrality_score


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
