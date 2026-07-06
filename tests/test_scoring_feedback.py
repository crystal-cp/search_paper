from lit_screening.agents.human_feedback import HumanFeedbackAgent
from lit_screening.models import EvidenceRecord, FeedbackRecord, Paper, QueryPlan, RankedPaper, VerificationResult
from lit_screening.reranking import hybrid_relevance_score
from lit_screening.scoring import compute_final_score, score_evidence


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


def test_scoring_formula_accepts_custom_weights():
    scores = compute_final_score(
        relevance_score=1.0,
        evidence_score=0.0,
        recency_score=0.0,
        quality_score=0.0,
        diversity_score=0.0,
        weights={
            "relevance": 0.8,
            "evidence": 0.1,
            "recency": 0.05,
            "quality": 0.03,
            "diversity": 0.02,
        },
    )

    assert scores.final_score == 0.8


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


def test_tfidf_reranker_scores_obviously_relevant_abstract_higher():
    plan = QueryPlan(
        original_question="surface magnetization in antiferromagnetic materials",
        core_terms=["surface magnetization", "antiferromagnetic materials"],
        must_terms=["surface magnetization"],
    )
    relevant = Paper(
        paper_id="p1",
        title="Surface magnetization in antiferromagnetic materials",
        abstract="Surface magnetization controls boundary spin signals in antiferromagnetic materials.",
    )
    irrelevant = Paper(
        paper_id="p2",
        title="Battery electrolyte degradation",
        abstract="This paper studies electrolyte aging and ion transport in batteries.",
    )
    relevant_evidence = EvidenceRecord(
        paper_id="p1",
        title=relevant.title,
        claim=relevant.abstract,
        evidence_sentence=relevant.abstract,
        relevance_reason="test",
    )
    irrelevant_evidence = EvidenceRecord(
        paper_id="p2",
        title=irrelevant.title,
        claim=irrelevant.abstract,
        evidence_sentence=irrelevant.abstract,
        relevance_reason="test",
    )

    assert hybrid_relevance_score(relevant, relevant_evidence, plan) > hybrid_relevance_score(
        irrelevant,
        irrelevant_evidence,
        plan,
    )


def test_evidence_score_is_lower_when_grounded_evidence_is_not_question_relevant():
    verification = VerificationResult(
        paper_id="p1",
        supported=True,
        confidence=1.0,
        error_type="",
        rationale="grounded",
    )
    relevant = EvidenceRecord(
        paper_id="p1",
        title="Relevant",
        claim="Surface magnetization controls spin signals.",
        evidence_sentence="Surface magnetization controls spin signals.",
        relevance_reason="test",
    )
    unrelated = EvidenceRecord(
        paper_id="p1",
        title="Unrelated",
        claim="The dataset contains calibrated microscopy images.",
        evidence_sentence="The dataset contains calibrated microscopy images.",
        relevance_reason="test",
    )
    question = "surface magnetization spin signals"

    assert score_evidence(verification, unrelated, question) < score_evidence(
        verification,
        relevant,
        question,
    )
