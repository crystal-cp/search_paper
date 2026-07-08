import json

from lit_screening.agents.ambiguity_detector import AmbiguityDetectorAgent
from lit_screening.agents.llm_enhancement import (
    LLMFeedbackInterpreter,
    LLMIntentFrameEnhancer,
    LLMQueryPlanCritic,
    LLMUserReportAdapter,
)
from lit_screening.agents.research_intent import ResearchIntentAgent
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.agents.search_contract import SearchContractAgent
from lit_screening.llm_client import LLMJSONResult
from lit_screening.models import (
    EvidenceRecord,
    Paper,
    RankedPaper,
    VerificationResult,
)
from lit_screening.pipeline import plan_screening_queries, run_pipeline
from lit_screening.retrieval.base import RetrievalResult
from lit_screening.scoring import compute_final_score


class MockLLM:
    provider_name = "mock"
    is_available = True

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def chat_json(self, system_prompt, user_prompt):
        self.prompts.append((system_prompt, user_prompt))
        payload = self.payloads.pop(0) if self.payloads else {}
        if isinstance(payload, LLMJSONResult):
            return payload
        return LLMJSONResult(data=payload)


class EmptyFakeClient:
    provider_name = "fake"

    def search(self, *args, **kwargs):
        return RetrievalResult(raw={"data": []}, papers=[])


def valid_intent_payload():
    return {
        "interpreted_user_goal": "Find lithium-ion SEI characterization papers.",
        "expert_rewritten_question": "solid electrolyte interphase lithium-ion battery characterization",
        "user_level": "novice",
        "research_intent_frame": {
            "core_object_terms": ["solid electrolyte interphase"],
            "domain_context_terms": ["lithium-ion battery"],
            "process_or_property_terms": ["composition"],
            "method_scope_terms": ["in situ characterization", "ex situ characterization"],
            "mechanism_terms": ["failure mechanism"],
            "case_or_system_terms": ["graphite anode"],
            "application_or_metric_terms": [],
            "failure_or_limitation_terms": ["failure mechanism"],
            "controversy_terms": ["controversy"],
            "review_background_terms": ["theoretical background"],
        },
        "concept_groups": [
            {
                "group_name": "explicit_sei",
                "group_role": "required",
                "operator_within_group": "OR",
                "terms": ["SEI", "solid electrolyte interphase"],
                "source": "user_text",
                "confidence": 0.95,
                "activation_reason": "explicit user term",
            },
            {
                "group_name": "unsafe_inferred_method",
                "group_role": "required",
                "operator_within_group": "OR",
                "terms": ["cryo-TEM"],
                "source": "llm_inferred",
                "confidence": 0.9,
                "activation_reason": "model guess",
            },
        ],
        "assumptions": ["Lithium-ion context is intended."],
        "ambiguities": [],
        "clarification": {
            "needed": False,
            "question": "",
            "reason": "Assumption-first.",
            "blocking": False,
        },
    }


def build_contract(question):
    return SearchContractAgent().build(
        question,
        search_brief=ResearchIntentAgent().analyze(question),
        ambiguity_analysis=AmbiguityDetectorAgent().analyze(question),
    )


def test_llm_intent_enhancer_schema_validation():
    llm = MockLLM([valid_intent_payload()])
    contract = build_contract("我想了解锂离子电池中 SEI 的原位/非原位表征。")

    payload = LLMIntentFrameEnhancer().enhance(
        original_question=contract.original_question,
        rule_based_intent_frame=contract.generic_intent_frame,
        structured_concepts=[],
        selected_domain=contract.domain_profile.domain_name,
        domain_candidates=[],
        seed_papers=[],
        llm_client=llm,
    )

    assert payload["llm_used"] is True
    assert payload["schema_valid"] is True
    assert payload["accepted_suggestions"]
    assert payload["rejected_suggestions"]


def test_llm_disabled_rule_mode_still_passes(tmp_path):
    run_pipeline(
        question="How can LLM systems improve systematic review screening?",
        providers=["fake"],
        max_per_query=0,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=RetrieverAgent(clients={"fake": EmptyFakeClient()}),
        llm_backend="none",
    )

    output_dir = tmp_path / "outputs"
    assert json.loads((output_dir / "llm_intent_enhancement.json").read_text())["llm_used"] is False
    assert json.loads((output_dir / "llm_query_critic.json").read_text())["fallback_used"] is True
    assert (output_dir / "llm_feedback_interpretation.json").exists()
    assert (output_dir / "llm_report_generation_trace.json").exists()
    assert (output_dir / "user_report.md").exists()


def test_llm_inferred_terms_not_required_without_validation():
    payload = LLMIntentFrameEnhancer().enhance(
        original_question="SEI in lithium-ion batteries",
        rule_based_intent_frame=build_contract("SEI in lithium-ion batteries").generic_intent_frame,
        structured_concepts=[],
        selected_domain="general_science",
        domain_candidates=[],
        seed_papers=[],
        llm_client=MockLLM([valid_intent_payload()]),
    )

    rejected_names = {item["group_name"] for item in payload["rejected_suggestions"]}
    assert "unsafe_inferred_method" in rejected_names


def test_llm_detects_in_situ_ex_situ_from_chinese():
    contract = build_contract("我想了解 SEI 的原位和非原位表征方法。")
    payload = LLMIntentFrameEnhancer().enhance(
        original_question=contract.original_question,
        rule_based_intent_frame=contract.generic_intent_frame,
        llm_client=MockLLM([valid_intent_payload()]),
    )
    updated = LLMIntentFrameEnhancer().apply_to_contract(contract, payload)

    assert updated.generic_intent_frame.in_situ_or_operando_need is True
    assert updated.generic_intent_frame.ex_situ_need is True
    assert "in situ characterization" in updated.generic_intent_frame.method_scope
    assert "ex situ characterization" in updated.generic_intent_frame.method_scope


def test_llm_query_critic_flags_missing_aspects():
    contract = build_contract("我想了解 SEI 的原位和非原位表征方法。")
    payload = LLMQueryPlanCritic().critique(
        enhanced_intent_frame={},
        search_contract=contract,
        candidate_query_families=None,
        final_provider_queries={"openalex": ["SEI lithium-ion battery review"]},
    )

    assert "missing_in_situ_or_operando_query" in payload["missing_user_aspects"]
    assert "missing_ex_situ_query" in payload["missing_user_aspects"]


def test_llm_query_critic_flags_cross_domain_pollution():
    contract = build_contract("LLM 用于系统综述文献筛选")
    payload = LLMQueryPlanCritic().critique(
        enhanced_intent_frame={},
        search_contract=contract,
        candidate_query_families=None,
        final_provider_queries={
            "openalex": [
                "LLM systematic review screening representative materials experimental characterization"
            ]
        },
    )

    assert "ai_screening_query_contains_material_template_terms" in payload["cross_domain_pollution"]


def test_llm_feedback_interpreter_downranks_non_target_battery_context():
    payload = LLMFeedbackInterpreter().interpret("这些钠电池不是我要的")

    assert any("sodium" in item for item in payload["downrank"])


def test_llm_user_report_does_not_invent_evidence():
    ranked = _ranked_paper()
    markdown, trace = LLMUserReportAdapter().generate(
        ranked_papers=[ranked],
        domain_assessments=[],
        ranking_diagnostics={},
        paper_roles=[],
        provider_status={},
        exploration_quality={},
        llm_client=MockLLM([{"markdown": "Invented Paper Title\nUnsupported claim."}]),
    )

    assert trace["fallback_used"] is True
    assert "Known grounded title" in markdown
    assert "Invented Paper Title" not in markdown


def test_llm_enabled_does_not_reduce_query_quality_against_rule_baseline(tmp_path):
    question = "LLM 用于系统综述文献筛选的人机协作，关注人工反馈、证据验证、召回率和准确率。"
    baseline = plan_screening_queries(question)
    run_pipeline(
        question=question,
        providers=["fake"],
        max_per_query=0,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=RetrieverAgent(clients={"fake": EmptyFakeClient()}),
        llm_client=MockLLM([valid_intent_payload()]),
        intent_repair=False,
    )
    planned = json.loads((tmp_path / "outputs" / "planned_queries.json").read_text())
    queries = "\n".join(planned["queries"]).lower()

    assert len(planned["queries"]) >= len(baseline["queries"])
    assert "pfm" not in queries
    assert "cryo-tem" not in queries


def _ranked_paper():
    paper = Paper(
        paper_id="p1",
        title="Known grounded title",
        abstract="Known grounded evidence sentence.",
    )
    evidence = EvidenceRecord(
        paper_id=paper.paper_id,
        title=paper.title,
        claim="Known claim",
        evidence_sentence="Known grounded evidence sentence.",
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id=paper.paper_id,
        supported=True,
        confidence=1.0,
        error_type="",
        rationale="test",
        support_level="strict_support",
    )
    return RankedPaper(
        rank=1,
        paper=paper,
        evidence=evidence,
        verification=verification,
        scores=compute_final_score(0.8, 0.8, 0.8, 0.8, 0.8),
    )
