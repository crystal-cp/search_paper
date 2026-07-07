from lit_screening.agents.intent_repair import NoviceIntentInterpreter
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.llm_client import LLMJSONResult
from lit_screening.pipeline import plan_screening_queries, run_pipeline
from lit_screening.retrieval.base import RetrievalResult


FERROELECTRIC_QUESTION = "我想调研铁电薄膜表面极化为什么重要，以及怎么探测"


class MockIntentLLM:
    provider_name = "mock"
    is_available = True

    def __init__(self, result):
        self.result = result
        self.calls = []

    def chat_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.result


class EmptyOpenAlexClient:
    provider_name = "openalex"

    def search(self, *args, **kwargs):
        return RetrievalResult(raw={"results": []}, papers=[])


def valid_payload():
    return {
        "original_question": FERROELECTRIC_QUESTION,
        "user_is_novice": True,
        "expert_rewritten_question": (
            "Find papers explaining surface polarization in ferroelectric thin films "
            "and how it can be probed experimentally."
        ),
        "inferred_goal": "Map ferroelectric surface-polarization concepts and probes.",
        "structured_concepts": [
            {
                "term": "ferroelectric polarization",
                "category": "property",
                "source": "user_text",
                "confidence": 0.92,
                "activation_reason": "The user asks about ferroelectric polarization.",
                "query_role": "must",
                "should_use_in_provider_query": True,
            },
            {
                "term": "surface polarization",
                "category": "property",
                "source": "user_text",
                "confidence": 0.9,
                "activation_reason": "The user asks about surface polarization.",
                "query_role": "must",
                "should_use_in_provider_query": True,
            },
            {
                "term": "PFM",
                "category": "method",
                "source": "domain_pack",
                "confidence": 0.74,
                "activation_reason": "Probe wording suggests a polarization microscopy route.",
                "query_role": "optional",
                "should_use_in_provider_query": True,
            },
        ],
        "likely_user_misconceptions": [
            "Importance is a motivation, not a hard search term.",
        ],
        "downweighted_user_terms": ["importance"],
        "assumptions": ["PFM is a plausible but optional probe route."],
        "needs_user_confirmation": ["Whether the user wants theory, experiments, or both."],
        "confidence": 0.84,
    }


def test_llm_mode_accepts_valid_json_and_stores_structured_concepts():
    llm = MockIntentLLM(LLMJSONResult(data=valid_payload()))

    intent = NoviceIntentInterpreter().repair(
        FERROELECTRIC_QUESTION,
        llm_client=llm,
    )

    assert intent.llm_metadata["llm_attempted"] is True
    assert intent.llm_metadata["llm_used"] is True
    assert intent.llm_metadata["fallback_used"] is False
    assert "surface polarization" in intent.expert_rewritten_question
    assert {concept.term for concept in intent.structured_concepts} >= {
        "ferroelectric polarization",
        "surface polarization",
        "PFM",
    }
    assert intent.needs_user_confirmation


def test_llm_mode_invalid_json_falls_back_to_rules():
    llm = MockIntentLLM(
        LLMJSONResult(
            data={},
            invalid_llm_output=True,
            error_type="invalid_json",
        )
    )

    intent = NoviceIntentInterpreter().repair(
        FERROELECTRIC_QUESTION,
        llm_client=llm,
    )

    assert intent.llm_metadata["llm_attempted"] is True
    assert intent.llm_metadata["llm_used"] is False
    assert intent.llm_metadata["fallback_used"] is True
    assert intent.llm_metadata["invalid_json_count"] == 1
    assert "invalid_json" in intent.llm_metadata["fallback_reason"]
    assert "ferroelectric polarization" in {c.term for c in intent.structured_concepts}


def test_llm_mode_missing_schema_fields_falls_back_with_errors():
    payload = {
        "original_question": FERROELECTRIC_QUESTION,
        "user_is_novice": True,
        "confidence": 0.8,
    }
    llm = MockIntentLLM(LLMJSONResult(data=payload))

    intent = NoviceIntentInterpreter().repair(
        FERROELECTRIC_QUESTION,
        llm_client=llm,
    )

    assert intent.llm_metadata["fallback_used"] is True
    assert intent.llm_metadata["schema_validation_errors"]
    assert any(
        "missing_field:structured_concepts" in error
        for error in intent.llm_metadata["schema_validation_errors"]
    )


def test_llm_overinjected_spaldin_concepts_are_downgraded_and_not_queried():
    payload = valid_payload()
    payload["structured_concepts"] = [
        *payload["structured_concepts"],
        {
            "term": "boundary magnetization",
            "category": "object",
            "source": "llm_inferred",
            "confidence": 0.9,
            "activation_reason": "Over-broad guess from another domain.",
            "query_role": "must",
            "should_use_in_provider_query": True,
        },
        {
            "term": "Cr2O3",
            "category": "material",
            "source": "llm_inferred",
            "confidence": 0.88,
            "activation_reason": "Over-broad guess from another domain.",
            "query_role": "must",
            "should_use_in_provider_query": True,
        },
    ]
    llm = MockIntentLLM(LLMJSONResult(data=payload))

    intent = NoviceIntentInterpreter().repair(
        FERROELECTRIC_QUESTION,
        llm_client=llm,
    )
    by_term = {concept.term: concept for concept in intent.structured_concepts}
    assert by_term["boundary magnetization"].query_role == "uncertain"
    assert by_term["Cr2O3"].query_role == "uncertain"
    assert by_term["boundary magnetization"].should_use_in_provider_query is False
    assert by_term["Cr2O3"].should_use_in_provider_query is False
    assert intent.llm_metadata["domain_validation_events"]

    plan = plan_screening_queries(
        FERROELECTRIC_QUESTION,
        llm_client=MockIntentLLM(LLMJSONResult(data=payload)),
    )["query_plan"]
    query_text = " ".join(
        [*plan.openalex_queries, *plan.semantic_scholar_queries]
    ).lower()
    assert "boundary magnetization" not in query_text
    assert "cr2o3" not in query_text


def test_pipeline_report_includes_llm_assisted_intent_repair(tmp_path):
    output_dir = tmp_path / "outputs"
    result = run_pipeline(
        FERROELECTRIC_QUESTION,
        providers=["openalex"],
        max_per_query=0,
        output_dir=str(output_dir),
        llm_client=MockIntentLLM(LLMJSONResult(data=valid_payload())),
        retriever_agent=RetrieverAgent(clients={"openalex": EmptyOpenAlexClient()}),
    )

    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "LLM-assisted intent repair" in report
    assert "LLM used: True" in report
    assert "Fallback used: False" in report
    assert "Needs user confirmation" in report
    assert result.agent_trace["intent_repair"]["llm_metadata"]["llm_used"] is True
