from lit_screening.agents.human_feedback import HumanFeedbackAgent
from lit_screening.models import EvidenceRecord, FeedbackRecord, Paper, RankedPaper, VerificationResult
from lit_screening.scoring import compute_final_score


def test_scoring_formula():
    scores = compute_final_score(
        relevance_score=0.8,
        evidence_score=0.6,
        recency_score=0.4,
        quality_score=0.2,
        diversity_score=1.0,
        human_feedback_adjustment=0.1,
    )

    assert scores.final_score == 0.40 * 0.8 + 0.25 * 0.6 + 0.15 * 0.4 + 0.15 * 0.2 + 0.05 * 1.0 + 0.1


def test_feedback_adjustment():
    paper = Paper(paper_id="p1", title="Relevant paper")
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Relevant paper",
        claim="A claim.",
        evidence_sentence="A claim.",
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id="p1",
        supported=True,
        confidence=1.0,
        error_type="",
        rationale="test",
    )
    scores = compute_final_score(0.5, 0.5, 0.5, 0.5, 0.5)
    ranked = RankedPaper(1, paper, evidence, verification, scores)
    feedback = {"p1": FeedbackRecord("p1", "include", 0.2, "important")}

    adjusted = HumanFeedbackAgent().apply([ranked], feedback)

    assert adjusted[0].scores.human_feedback_adjustment == 0.2
    assert adjusted[0].scores.final_score == scores.final_score + 0.2
    assert adjusted[0].feedback.note == "important"
