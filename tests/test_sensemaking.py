from pathlib import Path

from lit_screening.agents.aspect_classifier import AspectCoverageAgent
from lit_screening.agents.planner import PlannerAgent
from lit_screening.agents.research_intent import ResearchIntentAgent
from lit_screening.models import (
    AspectCoverageRecord,
    EvidenceRecord,
    Paper,
    RankedPaper,
    SearchBrief,
    ScreeningDecision,
    VerificationResult,
)
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
    assert "battery" in joined_queries


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
    return RankedPaper(1, paper, evidence, verification, scores)
