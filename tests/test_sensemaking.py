from lit_screening.agents.aspect_classifier import AspectCoverageAgent
from lit_screening.agents.planner import PlannerAgent
from lit_screening.agents.research_intent import ResearchIntentAgent
from lit_screening.models import (
    AspectCoverageRecord,
    DomainAssessment,
    EvidenceRecord,
    Paper,
    RankedPaper,
    SearchBrief,
    ScreeningDecision,
    VerificationResult,
)
from lit_screening.agents.llm_enhancement import render_user_report_markdown, verified_report_input
from lit_screening.reading_path import generate_reading_path
from lit_screening.result_groups import group_ranked_papers
from lit_screening.scoring import compute_final_score
from lit_screening.screening_flow import build_prisma_like_flow


def test_research_intent_agent_detects_common_modes():
    agent = ResearchIntentAgent()

    assert agent.analyze("latest frontier in surface magnetization").search_intent == "frontier"
    assert agent.analyze("implementation method for evidence screening systems").search_intent == "implementation"
    assert agent.analyze("research gap for a PhD proposal on surface magnetization").search_intent == "proposal"


def test_planner_uses_search_brief_inclusion_and_exclusion_terms():
    brief = SearchBrief(
        original_question="surface magnetization",
        refined_question="surface magnetization in antiferromagnetic materials",
        search_intent="overview",
        user_goal="background",
        inclusion_criteria=["antiferromagnetic materials"],
        exclusion_criteria=["battery"],
        required_aspects=["surface magnetization", "spin signals"],
        preferred_paper_types=["review"],
    )

    plan = PlannerAgent().plan_structured("surface magnetization", search_brief=brief)
    joined_queries = " ".join(plan.openalex_queries + plan.semantic_scholar_queries).lower()

    assert "antiferromagnetic materials" in " ".join(plan.must_terms).lower()
    assert "battery" in plan.exclude_terms
    assert "spin signals" in plan.required_aspects
    assert "battery" not in joined_queries


def test_aspect_coverage_detects_covered_and_missing_aspects():
    paper = Paper(
        paper_id="p1",
        title="Surface magnetization in antiferromagnets",
        abstract="Surface magnetization creates boundary spin signals in antiferromagnets.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title=paper.title,
        claim="Surface magnetization creates boundary spin signals.",
        evidence_sentence="Surface magnetization creates boundary spin signals in antiferromagnets.",
        relevance_reason="test",
    )

    record = AspectCoverageAgent().classify(
        paper,
        evidence,
        ["surface magnetization", "spin signals", "device fabrication"],
    )

    assert "surface magnetization" in record.covered_aspects
    assert "device fabrication" in record.missing_aspects
    assert 0 < record.aspect_coverage_score < 1


def test_result_grouping_uses_aspect_coverage_and_grounding():
    ranked = _ranked_paper(
        paper_id="p1",
        title="Surface magnetization in antiferromagnetic materials",
        final_score=0.8,
        support_level="strict_support",
    )
    aspect = AspectCoverageRecord(
        paper_id="p1",
        title=ranked.paper.title,
        covered_aspects=["surface magnetization"],
        missing_aspects=[],
        aspect_coverage_score=1.0,
    )

    groups = group_ranked_papers([ranked], [aspect], SearchBrief(
        original_question="surface magnetization",
        refined_question="surface magnetization",
        search_intent="overview",
        user_goal="background",
    ))

    assert groups["must_read"][0]["paper_id"] == "p1"


def test_reading_path_is_generated(tmp_path):
    ranked = _ranked_paper("p1", "Review of surface magnetization", 0.7)
    groups = {
        "background_or_survey": [
            {"rank": 1, "title": ranked.paper.title, "paper_id": ranked.paper.paper_id}
        ],
        "must_read": [],
        "implementation_relevant": [],
        "recent_frontier": [],
        "evaluation_relevant": [],
        "peripheral": [],
    }
    path = tmp_path / "reading_path.md"

    generate_reading_path(path, [ranked], groups)

    assert "Recommended Reading Path" in path.read_text()


def test_reading_path_excludes_out_of_scope_papers(tmp_path):
    excluded = _ranked_paper(
        "ocean",
        "Ocean Seismic Network Pilot Experiment",
        0.9,
        decision="exclude",
        reading_priority="exclude",
        domain_decision="out_of_scope",
    )
    path = tmp_path / "reading_path.md"

    diagnostics = generate_reading_path(
        path,
        [excluded],
        {"evaluation_relevant": [_group_row(excluded)]},
    )

    text = path.read_text()
    assert "Ocean Seismic Network Pilot Experiment" not in text
    assert diagnostics["reading_path_out_of_scope_count"] == 0


def test_reading_path_excludes_decision_exclude(tmp_path):
    excluded = _ranked_paper(
        "exclude-decision",
        "Excluded SEI paper",
        0.9,
        decision="exclude",
        reading_priority="read_later",
        domain_decision="in_scope",
    )
    path = tmp_path / "reading_path.md"

    diagnostics = generate_reading_path(
        path,
        [excluded],
        {"recent_frontier": [_group_row(excluded)]},
    )

    assert "Excluded SEI paper" not in path.read_text()
    assert diagnostics["reading_path_exclude_count"] == 0


def test_reading_path_excludes_reading_priority_exclude(tmp_path):
    excluded = _ranked_paper(
        "exclude-priority",
        "Priority excluded paper",
        0.9,
        decision="maybe",
        reading_priority="exclude",
        domain_decision="in_scope",
    )
    path = tmp_path / "reading_path.md"

    generate_reading_path(
        path,
        [excluded],
        {"background_or_survey": [_group_row(excluded)]},
    )

    assert "Priority excluded paper" not in path.read_text()


def test_reading_path_deduplicates_papers_across_sections(tmp_path):
    paper = _ranked_paper("p1", "Artificial SEI for lithium metal batteries", 0.9)
    path = tmp_path / "reading_path.md"
    groups = {
        "implementation_relevant": [_group_row(paper)],
        "evaluation_relevant": [_group_row(paper)],
        "recent_frontier": [_group_row(paper)],
    }

    diagnostics = generate_reading_path(path, [paper], groups)
    text = path.read_text()

    assert text.count("Artificial SEI for lithium metal batteries") == 1
    assert diagnostics["reading_path_duplicate_count"] == 0


def test_user_report_read_first_excludes_out_of_scope():
    good = _ranked_paper("good", "Artificial SEI for lithium metal batteries", 0.9)
    ocean = _ranked_paper(
        "ocean",
        "Ocean Seismic Network Pilot Experiment",
        0.95,
        decision="exclude",
        reading_priority="exclude",
        domain_decision="out_of_scope",
    )
    artifact_input = verified_report_input(
        [ocean, good],
        [ocean.domain_assessment, good.domain_assessment],
        {},
        [],
        {},
        {},
    )

    report = render_user_report_markdown(artifact_input)

    assert "Artificial SEI for lithium metal batteries" in report
    assert "Ocean Seismic Network Pilot Experiment" not in report


def test_reading_path_uses_priority_filtered_candidates(tmp_path):
    must = _ranked_paper("must", "Core lithium artificial SEI", 0.9)
    later = _ranked_paper(
        "later",
        "Read later lithium SEI characterization",
        0.7,
        reading_priority="read_later",
    )
    optional = _ranked_paper(
        "optional",
        "Optional mixed chemistry SEI review",
        0.6,
        decision="maybe",
        reading_priority="optional",
        domain_decision="borderline",
    )
    path = tmp_path / "reading_path.md"

    generate_reading_path(
        path,
        [must, later, optional],
        {
            "must_read": [_group_row(must)],
            "evaluation_relevant": [_group_row(later), _group_row(optional)],
            "peripheral": [_group_row(optional)],
        },
    )
    text = path.read_text()

    assert "Core lithium artificial SEI" in text
    assert "Read later lithium SEI characterization" in text
    assert "Optional mixed chemistry SEI review" in text


def test_ocean_seismic_pilot_not_in_sei_reading_path(tmp_path):
    lithium = _ranked_paper("li", "Artificial SEI for lithium metal batteries", 0.9)
    ocean = _ranked_paper(
        "ocean",
        "Ocean Seismic Network Pilot Experiment",
        0.99,
        decision="exclude",
        reading_priority="exclude",
        domain_decision="out_of_scope",
        peripheral_context_reason="missing_target_lithium_context",
    )
    path = tmp_path / "reading_path.md"

    diagnostics = generate_reading_path(
        path,
        [ocean, lithium],
        {
            "evaluation_relevant": [_group_row(ocean)],
            "must_read": [_group_row(lithium)],
        },
    )

    text = path.read_text()
    assert "Ocean Seismic Network Pilot Experiment" not in text
    assert "Artificial SEI for lithium metal batteries" in text
    assert diagnostics["reading_path_exclude_count"] == 0
    assert diagnostics["reading_path_out_of_scope_count"] == 0


def test_reading_path_filters_still_exclude_out_of_scope(tmp_path):
    ocean = _ranked_paper(
        "ocean",
        "Ocean Seismic Network Pilot Experiment",
        0.99,
        decision="exclude",
        reading_priority="exclude",
        domain_decision="out_of_scope",
    )
    path = tmp_path / "reading_path.md"

    diagnostics = generate_reading_path(
        path,
        [ocean],
        {"must_read": [_group_row(ocean)], "evaluation_relevant": [_group_row(ocean)]},
    )

    assert "Ocean Seismic Network Pilot Experiment" not in path.read_text()
    assert diagnostics["reading_path_exclude_count"] == 0
    assert diagnostics["reading_path_out_of_scope_count"] == 0


def test_user_report_read_first_uses_must_read_when_available():
    read_later = _ranked_paper(
        "later",
        "Read later OER catalyst overview",
        0.8,
        reading_priority="read_later",
    )
    must_read = _ranked_paper(
        "must",
        "Spin-state control of oxygen evolution reaction activity",
        0.92,
        reading_priority="must_read",
    )
    read_later.rank = 1
    must_read.rank = 2
    artifact_input = verified_report_input(
        [read_later, must_read],
        [read_later.domain_assessment, must_read.domain_assessment],
        {},
        [],
        {},
        {},
    )

    report = render_user_report_markdown(artifact_input)

    assert report.index("Spin-state control of oxygen evolution reaction activity") < report.index(
        "Read later OER catalyst overview"
    )


def test_prisma_like_flow_counts_records_and_reasons():
    paper = Paper(paper_id="p1", title="No abstract")
    verification = VerificationResult(
        paper_id="p1",
        supported=False,
        confidence=0.0,
        error_type="missing_abstract",
        rationale="missing",
        support_level="missing_abstract",
    )

    flow = build_prisma_like_flow(
        retrieval_counts={"openalex": 1, "semantic_scholar": 0},
        raw_paper_count=1,
        merged_papers=[paper],
        duplicate_count=0,
        verification_results=[verification],
        ranked_papers=[],
        screening_decisions=[
            ScreeningDecision(
                paper_id="p1",
                decision="exclude",
                decision_confidence=0.8,
                primary_reason="missing_abstract",
                exclusion_reasons=["missing_abstract"],
                domain_match_score=0.0,
                domain_decision="unknown",
                reading_priority="exclude",
                suggested_action="exclude",
            )
        ],
    )

    assert flow["records_screened"] == 1
    assert flow["records_with_missing_abstracts"] == 1
    assert flow["records_excluded"] == 1
    assert flow["common_exclusion_reasons"]["missing_abstract"] == 1


def _ranked_paper(
    paper_id: str,
    title: str,
    final_score: float,
    support_level: str = "strict_support",
    decision: str = "include",
    reading_priority: str = "must_read",
    domain_decision: str = "in_scope",
    peripheral_context_reason: str = "",
) -> RankedPaper:
    paper = Paper(
        paper_id=paper_id,
        title=title,
        abstract="Surface magnetization creates boundary spin signals.",
        year=2024,
    )
    evidence = EvidenceRecord(
        paper_id=paper_id,
        title=title,
        claim="Surface magnetization creates boundary spin signals.",
        evidence_sentence="Surface magnetization creates boundary spin signals.",
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id=paper_id,
        supported=support_level == "strict_support",
        confidence=1.0 if support_level == "strict_support" else 0.3,
        error_type="",
        rationale="test",
        support_level=support_level,
    )
    scores = compute_final_score(
        relevance_score=final_score,
        evidence_score=final_score,
        recency_score=final_score,
        quality_score=final_score,
        diversity_score=final_score,
    )
    scores.final_score = final_score
    domain = DomainAssessment(
        paper_id=paper_id,
        domain_match_score=0.95 if domain_decision == "in_scope" else 0.3,
        domain_decision=domain_decision,
        off_topic_reason="test",
        target_context_match=["lithium metal battery"]
        if domain_decision != "out_of_scope"
        else [],
        peripheral_context_reason=peripheral_context_reason,
        intent_centrality_score=0.9,
        required_group_coverage_score=1.0,
        topic_focus_score=0.9,
    )
    screening = ScreeningDecision(
        paper_id=paper_id,
        decision=decision,
        decision_confidence=0.9,
        primary_reason="test",
        exclusion_reasons=[],
        domain_match_score=domain.domain_match_score,
        domain_decision=domain_decision,
        reading_priority=reading_priority,
        suggested_action=decision,
    )
    return RankedPaper(
        1,
        paper,
        evidence,
        verification,
        scores,
        domain_assessment=domain,
        screening_decision=screening,
    )


def _group_row(item: RankedPaper) -> dict:
    return {
        "rank": item.rank,
        "title": item.paper.title,
        "paper_id": item.paper.paper_id,
    }
