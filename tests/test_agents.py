from lit_screening.agents.extractor import ExtractorAgent
from lit_screening.agents.planner import PlannerAgent
from lit_screening.agents.verifier import VerifierAgent
from lit_screening.models import EvidenceRecord, Paper


def test_extractor_does_not_hallucinate_when_abstract_is_missing():
    paper = Paper(paper_id="p1", title="Title only")

    evidence = ExtractorAgent().extract(paper, "How can LLM agents verify evidence?")

    assert evidence.claim == ""
    assert evidence.evidence_sentence == ""
    assert "Missing abstract" in evidence.limitation


def test_verifier_flags_missing_evidence():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="This abstract discusses evidence verification.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Paper",
        claim="",
        evidence_sentence="",
        relevance_reason="test",
    )

    result = VerifierAgent().verify(paper, evidence)

    assert result.supported is False
    assert result.error_type == "missing_evidence"


def test_planner_does_not_inject_unrelated_llm_terms():
    queries = PlannerAgent().plan("the significance of surface magnetization")
    joined = " ".join(queries).lower()

    assert "surface magnetization" in joined
    assert "llm" not in joined
    assert "human-in-the-loop" not in joined
    assert "multi-agent" not in joined


def test_planner_translates_common_chinese_terms_without_llm():
    planner = PlannerAgent()
    queries = planner.plan("表面磁化的重要性")
    joined = " ".join(queries).lower()

    assert "surface magnetization" in joined
    assert "importance" in joined
    assert "表面" not in joined
    assert planner.last_llm_metadata["question_language"] == "zh"
    assert planner.last_llm_metadata["translation_mode"] == "rule_glossary"
    assert planner.last_llm_metadata["planning_question"] == "surface magnetization importance"


def test_verifier_requires_strict_span_for_supported():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="Surface magnetization is important for boundary spin signals.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Paper",
        claim="Surface magnetization is important.",
        evidence_sentence="Surface magnetization is important for boundary spin signals.",
        relevance_reason="test",
    )

    result = VerifierAgent().verify(paper, evidence)

    assert result.supported is True
    assert result.support_level == "strict_support"
    assert result.span_match_type == "exact"


def test_verifier_downgrades_overlap_to_weak_support():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="Surface magnetization controls boundary spin signals in antiferromagnets.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Paper",
        claim="Surface magnetization controls spin signals.",
        evidence_sentence="Surface magnetization boundary spin antiferromagnets controls signals",
        relevance_reason="test",
    )

    result = VerifierAgent().verify(paper, evidence)

    assert result.supported is False
    assert result.support_level == "weak_support"
    assert result.error_type == "weak_support"


def test_verifier_marks_unmatched_llm_evidence_invalid():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="Surface magnetization controls boundary spin signals in antiferromagnets.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Paper",
        claim="Surface magnetization creates a giant device effect.",
        evidence_sentence="Surface magnetization creates a giant device effect.",
        relevance_reason="test",
        llm_used=True,
        extraction_mode="llm",
    )

    result = VerifierAgent().verify(paper, evidence)

    assert result.supported is False
    assert result.support_level == "llm_invalid_evidence"
    assert result.error_type == "llm_invalid_evidence"
