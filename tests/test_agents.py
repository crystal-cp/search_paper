from lit_screening.agents.extractor import ExtractorAgent
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
