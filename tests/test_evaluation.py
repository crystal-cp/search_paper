from lit_screening.evaluation import (
    average_precision,
    compute_evaluation,
    ndcg_at_k,
    recall_at_k,
)
from lit_screening.models import (
    EvidenceRecord,
    Paper,
    RankedPaper,
    ScoreBreakdown,
    VerificationResult,
)


def ranked(paper_id: str, rank: int) -> RankedPaper:
    paper = Paper(paper_id=paper_id, title=f"Paper {paper_id}", abstract="Evidence sentence.")
    evidence = EvidenceRecord(
        paper_id=paper_id,
        title=paper.title,
        claim="Evidence sentence.",
        evidence_sentence="Evidence sentence.",
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id=paper_id,
        supported=True,
        confidence=1.0,
        error_type="",
        rationale="exact",
        support_level="strict_support",
        span_match_type="exact",
        span_match_confidence=1.0,
        matched_text="Evidence sentence.",
    )
    scores = ScoreBreakdown(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
    return RankedPaper(rank, paper, evidence, verification, scores)


def test_ranking_metrics_with_gold_labels():
    ranked_papers = [ranked("p1", 1), ranked("p2", 2), ranked("p3", 3)]
    gold = {"p1": "include", "p2": "exclude", "p3": "include"}

    assert ndcg_at_k(ranked_papers, gold, 3) is not None
    assert average_precision(ranked_papers, gold) == (1 / 1 + 2 / 3) / 2
    assert recall_at_k(ranked_papers, gold, 2) == 0.5


def test_compute_evaluation_adds_grounding_and_feedback_delta():
    before = [ranked("p1", 1), ranked("p2", 2)]
    after = [ranked("p2", 1), ranked("p1", 2)]
    papers = [item.paper for item in before]
    evidence = [item.evidence for item in before]
    verification = [item.verification for item in before]

    metrics = compute_evaluation(
        retrieval_counts={"fake": 2},
        original_paper_count=2,
        merged_papers=papers,
        evidence_records=evidence,
        verification_results=verification,
        ranked_before_feedback=before,
        ranked_after_feedback=after,
    )

    assert metrics["grounding_accuracy"] == 1.0
    assert metrics["strict_support_rate"] == 1.0
    assert metrics["weak_support_rate"] == 0.0
    assert metrics["feedback_before_after_ranking_delta"]["moved_count"] == 2
    assert metrics["feedback_before_after_ranking_delta"]["mean_abs_rank_delta"] == 1.0
