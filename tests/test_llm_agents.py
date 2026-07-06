from lit_screening.agents.extractor import ExtractorAgent
from lit_screening.agents.planner import PlannerAgent
from lit_screening.agents.verifier import VerifierAgent
from lit_screening.llm_client import LLMJSONResult
from lit_screening.models import EvidenceRecord, Paper


class FakeLLMClient:
    provider_name = "fake"
    is_available = True

    def __init__(self, result):
        self.result = result

    def chat_json(self, system_prompt, user_prompt):
        return self.result


def test_planner_uses_mocked_llm_queries():
    client = FakeLLMClient(
        LLMJSONResult(
            data={
                "translated_question": "How can human feedback improve LLM literature screening?",
                "queries": [
                    "human feedback LLM agents literature screening",
                    "claim extraction evidence verification systematic review",
                    "multi agent LLM scientific abstracts",
                    "human in the loop ranking literature review",
                ]
            }
        )
    )

    queries = PlannerAgent(mode="llm", llm_client=client).plan(
        "How can human feedback improve LLM literature screening?"
    )

    assert queries[0] == "How can human feedback improve LLM literature screening?"
    assert "human feedback LLM agents literature screening" in queries


def test_planner_uses_mocked_llm_translation_for_chinese_question():
    client = FakeLLMClient(
        LLMJSONResult(
            data={
                "translated_question": "the significance of surface magnetization",
                "queries": [
                    "surface magnetization significance review",
                    "surface magnetization mechanism",
                    "surface magnetization experimental study",
                    "surface magnetization theoretical study",
                ],
            }
        )
    )

    planner = PlannerAgent(mode="llm", llm_client=client)
    queries = planner.plan("表面磁化的重要性")

    assert queries[0] == "the significance of surface magnetization"
    assert "surface magnetization mechanism" in queries
    assert planner.last_llm_metadata["question_language"] == "zh"
    assert planner.last_llm_metadata["translation_mode"] == "llm"
    assert planner.last_llm_metadata["translated_question"] == (
        "the significance of surface magnetization"
    )
    assert "表面" not in " ".join(queries)


def test_extractor_uses_mocked_llm_evidence():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="Human feedback improves screening by correcting evidence errors.",
    )
    client = FakeLLMClient(
        LLMJSONResult(
            data={
                "claim": "Human feedback improves screening.",
                "evidence_sentence": "Human feedback improves screening by correcting evidence errors.",
                "relevance_reason": "Directly answers the question.",
                "limitation": "",
            }
        )
    )

    evidence = ExtractorAgent(mode="llm", llm_client=client).extract(
        paper,
        "How can human feedback improve screening?",
    )

    assert evidence.extraction_mode == "llm"
    assert evidence.llm_used is True
    assert evidence.invalid_llm_output is False
    assert evidence.claim == "Human feedback improves screening."


def test_verifier_invalid_llm_output_falls_back_and_records_flag():
    paper = Paper(
        paper_id="p1",
        title="Paper",
        abstract="Human feedback improves screening by correcting evidence errors.",
    )
    evidence = EvidenceRecord(
        paper_id="p1",
        title="Paper",
        claim="Human feedback improves screening.",
        evidence_sentence="Human feedback improves screening by correcting evidence errors.",
        relevance_reason="test",
    )
    client = FakeLLMClient(
        LLMJSONResult(data={}, invalid_llm_output=True, error_type="invalid_json")
    )

    result = VerifierAgent(mode="llm", llm_client=client).verify(paper, evidence)

    assert result.supported is True
    assert result.llm_used is True
    assert result.invalid_llm_output is True
    assert result.llm_error_type == "invalid_json"
