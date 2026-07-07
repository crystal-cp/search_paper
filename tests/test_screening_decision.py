import pytest

from lit_screening.agents.screening_decision import ScreeningDecisionAgent
from lit_screening.models import (
    AspectCoverageRecord,
    DomainAssessment,
    EvidenceRecord,
    Paper,
    RankedPaper,
    VerificationResult,
)
from lit_screening.scoring import compute_final_score


def test_include_decision_for_high_scoring_in_domain_supported_paper():
    ranked = _ranked_paper(
        final_score=0.82,
        support_level="strict_support",
        domain_decision="in_scope",
        domain_match_score=0.95,
    )
    aspect = AspectCoverageRecord(
        paper_id=ranked.paper.paper_id,
        title=ranked.paper.title,
        covered_aspects=["literature screening", "evidence verification"],
        missing_aspects=[],
        aspect_coverage_score=1.0,
    )

    decisions, updated = ScreeningDecisionAgent().decide_many([ranked], [aspect])

    assert decisions[0].decision == "include"
    assert decisions[0].reading_priority in {"must_read", "read_later"}
    assert decisions[0].suggested_action == "include"
    assert updated[0].screening_decision == decisions[0]


def test_maybe_decision_for_borderline_paper():
    ranked = _ranked_paper(
        final_score=0.56,
        support_level="strict_support",
        domain_decision="borderline",
        domain_match_score=0.52,
    )

    decision = ScreeningDecisionAgent().decide(ranked)

    assert decision.decision == "maybe"
    assert decision.reading_priority == "optional"
    assert decision.suggested_action == "uncertain"


def test_exclude_decision_for_out_of_scope_paper():
    ranked = _ranked_paper(
        final_score=0.61,
        support_level="strict_support",
        domain_decision="out_of_scope",
        domain_match_score=0.12,
        forbidden_concepts_found=["patient screening"],
    )

    decision = ScreeningDecisionAgent().decide(ranked)

    assert decision.decision == "exclude"
    assert decision.primary_reason == "off_topic_domain"
    assert "off_topic_domain" in decision.exclusion_reasons
    assert "ambiguous_screening_meaning" in decision.exclusion_reasons


@pytest.mark.parametrize(
    ("final_score", "expected_decision"),
    [
        (0.60, "maybe"),
        (0.20, "exclude"),
    ],
)
def test_missing_abstract_can_be_maybe_or_exclude_depending_on_score(
    final_score,
    expected_decision,
):
    ranked = _ranked_paper(
        final_score=final_score,
        support_level="missing_abstract",
        domain_decision="in_scope",
        domain_match_score=0.8,
        abstract="",
    )

    decision = ScreeningDecisionAgent().decide(ranked)

    assert decision.decision == expected_decision
    assert "missing_abstract" in decision.exclusion_reasons


def _ranked_paper(
    final_score: float,
    support_level: str,
    domain_decision: str,
    domain_match_score: float,
    abstract: str = (
        "Human-in-the-loop LLM agents improve scientific literature screening "
        "and evidence verification."
    ),
    forbidden_concepts_found: list[str] | None = None,
) -> RankedPaper:
    paper = Paper(
        paper_id="paper-1",
        title="Human-in-the-loop LLM agents for literature screening",
        abstract=abstract,
        year=2024,
        venue="Demo Journal",
        citation_count=12,
    )
    evidence = EvidenceRecord(
        paper_id=paper.paper_id,
        title=paper.title,
        claim="LLM agents improve literature screening.",
        evidence_sentence=abstract,
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id=paper.paper_id,
        supported=support_level == "strict_support",
        confidence=1.0 if support_level == "strict_support" else 0.2,
        error_type=support_level if support_level != "strict_support" else "",
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
    assessment = DomainAssessment(
        paper_id=paper.paper_id,
        domain_match_score=domain_match_score,
        domain_decision=domain_decision,
        off_topic_reason="test" if domain_decision != "in_scope" else "",
        forbidden_concepts_found=forbidden_concepts_found or [],
    )
    return RankedPaper(
        rank=1,
        paper=paper,
        evidence=evidence,
        verification=verification,
        scores=scores,
        domain_assessment=assessment,
    )
