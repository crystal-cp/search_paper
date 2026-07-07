from lit_screening.agents.human_feedback import HumanFeedbackAgent
from lit_screening.agents.preference_learning import PreferenceLearningAgent
from lit_screening.models import (
    DomainAssessment,
    EvidenceRecord,
    FeedbackRecord,
    Paper,
    RankedPaper,
    VerificationResult,
)
from lit_screening.scoring import compute_final_score


def test_preference_learning_produces_positive_and_negative_terms():
    ranked = [
        _ranked(
            "p1",
            "Human feedback for literature screening",
            "Human feedback improves evidence verification in literature screening.",
        ),
        _ranked(
            "p2",
            "Span-grounded evidence verification",
            "Evidence verification checks claim support in abstracts.",
        ),
        _ranked(
            "p3",
            "Drug screening with biomarkers",
            "Patient screening and biomarker screening are clinical workflows.",
        ),
        _ranked(
            "p4",
            "Clinical patient screening",
            "Drug screening prioritizes biomarkers in medicine.",
        ),
    ]
    feedback = {
        "p1": FeedbackRecord("p1", "include", 0.1),
        "p2": FeedbackRecord("p2", "include", 0.1),
        "p3": FeedbackRecord("p3", "exclude", -0.2),
        "p4": FeedbackRecord("p4", "exclude", -0.2),
    }

    result = PreferenceLearningAgent().learn(ranked, feedback)

    assert result.enabled is True
    assert result.preference_scores["p1"] > result.preference_scores["p3"]
    assert any(term in " ".join(result.positive_terms) for term in ["feedback", "evidence", "literature"])
    assert any(term in " ".join(result.negative_terms) for term in ["patient", "drug", "clinical"])
    assert result.suggested_must_terms or result.suggested_optional_terms
    assert result.suggested_exclude_terms


def test_preference_score_is_added_only_when_feedback_model_exists():
    ranked = [_ranked("p1", "Relevant", "Relevant evidence verification paper.")]
    feedback = {"p1": FeedbackRecord("p1", "include", 0.1)}

    no_preference = HumanFeedbackAgent().apply(ranked, feedback)[0]
    with_preference = HumanFeedbackAgent().apply(
        ranked,
        feedback,
        preference_scores={"p1": 1.0},
    )[0]

    assert no_preference.scores.preference_adjustment == 0.0
    assert with_preference.scores.preference_score == 1.0
    assert with_preference.scores.preference_adjustment > 0
    assert with_preference.scores.final_score > no_preference.scores.final_score


def test_out_of_scope_paper_stays_penalized_with_high_preference_score():
    scores = compute_final_score(
        relevance_score=1.0,
        evidence_score=1.0,
        recency_score=1.0,
        quality_score=1.0,
        diversity_score=1.0,
        preference_score=1.0,
        domain_penalty_multiplier=0.3,
    )

    assert scores.preference_adjustment > 0
    assert scores.domain_penalty_multiplier == 0.3
    assert scores.final_score == scores.pre_domain_final_score * 0.3
    assert scores.final_score < scores.pre_domain_final_score


def _ranked(paper_id: str, title: str, abstract: str) -> RankedPaper:
    paper = Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=2024,
        venue="Demo Journal",
        citation_count=4,
    )
    evidence = EvidenceRecord(
        paper_id=paper_id,
        title=title,
        claim=abstract,
        evidence_sentence=abstract,
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id=paper_id,
        supported=True,
        confidence=1.0,
        error_type="",
        rationale="test",
        support_level="strict_support",
    )
    scores = compute_final_score(0.5, 0.5, 0.5, 0.5, 0.5)
    assessment = DomainAssessment(
        paper_id=paper_id,
        domain_match_score=0.9,
        domain_decision="in_scope",
        off_topic_reason="",
    )
    return RankedPaper(
        rank=1,
        paper=paper,
        evidence=evidence,
        verification=verification,
        scores=scores,
        domain_assessment=assessment,
    )
