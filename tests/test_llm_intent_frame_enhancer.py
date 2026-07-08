import json

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.llm.intent_frame_enhancer import (
    FORBIDDEN_DECISION_FIELDS,
    FakeLLMIntentProvider,
    LLMIntentFrameEnhancer,
    LLMIntentFrameSuggestion,
    assert_llm_intent_frame_schema_is_non_decisive,
    build_intent_frame_prompt,
    find_forbidden_decision_fields,
    llm_intent_frame_suggestion_fields,
    parse_llm_intent_frame_json,
    verify_intent_frame_suggestions,
    write_intent_enhancement_artifacts,
)
from lit_screening.models import Paper
from lit_screening.pipeline import run_pipeline
from lit_screening.retrieval.base import RetrievalResult


class ExplodingLLMClient:
    def chat_json(self, *args, **kwargs):
        raise AssertionError("disabled LLMIntentFrameEnhancer must not call LLM")


class PipelineFakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        paper = Paper(
            paper_id="fake-paper-1",
            title="Artificial SEI improves lithium metal battery cycling",
            abstract=(
                "Artificial SEI layers can improve lithium metal battery cycling "
                "and suppress dendrite growth."
            ),
            authors=["A. Researcher"],
            year=2024,
            venue="Demo Journal",
            source_provider="fake",
        )
        return RetrievalResult(raw={"query": query}, papers=[paper])


def fake_retriever():
    return RetrieverAgent(clients={"fake": PipelineFakeClient()})


def test_llm_intent_enhancer_disabled_noop():
    deterministic_intent = {"topic": "SEI", "target_context": ["lithium battery"]}

    result = LLMIntentFrameEnhancer(enabled=False).enhance(
        deterministic_intent,
        input_question="我想找锂电池 SEI 文章",
        llm_client=ExplodingLLMClient(),
    )

    assert result.raw.enabled is False
    assert result.raw.raw_text == ""
    assert result.raw.parsed_json is None
    assert result.raw.malformed_output is False
    assert result.raw.error == ""
    assert result.verified_suggestion is None


def test_llm_intent_enhancer_disabled_does_not_change_intent():
    deterministic_intent = {"required_groups": ["sei", "lithium"]}

    result = LLMIntentFrameEnhancer(enabled=False).enhance(deterministic_intent)

    assert result.intent_before_llm is deterministic_intent
    assert result.intent_after_llm is deterministic_intent
    assert result.intent_after_llm == deterministic_intent


def test_llm_intent_enhancer_trace_disabled():
    result = LLMIntentFrameEnhancer(enabled=False, provider="mock", model="mock-model").enhance(
        {"topic": "OER spin state"},
        input_question="OER spin state papers",
    )

    assert result.trace.llm_enabled is False
    assert result.trace.llm_called is False
    assert result.trace.fallback_used is False
    assert result.trace.malformed_output is False
    assert result.trace.verified_candidate_count == 0
    assert result.trace.applied_suggestion_count == 0
    assert result.trace.accepted_suggestion_count == 0
    assert result.trace.rejected_suggestion_count == 0
    assert result.trace.unsupported_suggestion_count == 0
    assert result.trace.reason_if_no_change == "llm_intent_enhancer_disabled"
    assert result.raw.provider == "mock"
    assert result.raw.model == "mock-model"
    assert result.raw.input_question == "OER spin state papers"


def test_llm_intent_frame_schema_contains_no_decision_fields():
    fields = llm_intent_frame_suggestion_fields()

    assert not (fields & FORBIDDEN_DECISION_FIELDS)
    assert "intent_summary" in fields
    assert "target_context_candidates" in fields
    assert "negative_context_candidates" in fields
    assert_llm_intent_frame_schema_is_non_decisive()


def test_write_intent_enhancement_artifacts_reserved_noop(tmp_path):
    deterministic_intent = {"topic": "MOF CO2 capture"}
    result = LLMIntentFrameEnhancer(enabled=False).enhance(deterministic_intent)

    artifacts = write_intent_enhancement_artifacts(
        tmp_path,
        intent_before_llm=result.intent_before_llm,
        raw=result.raw,
        verified_suggestion=result.verified_suggestion,
        trace=result.trace,
    )

    assert set(artifacts) == {
        "intent_frame_before_llm",
        "llm_intent_frame_raw",
        "llm_intent_frame_verified",
        "llm_intent_enhancement_trace",
    }
    assert json.loads((tmp_path / "intent_frame_before_llm.json").read_text()) == deterministic_intent
    assert json.loads((tmp_path / "llm_intent_frame_raw.json").read_text())["enabled"] is False
    assert json.loads((tmp_path / "llm_intent_frame_verified.json").read_text()) is None
    assert (
        json.loads((tmp_path / "llm_intent_enhancement_trace.json").read_text())["reason_if_no_change"]
        == "llm_intent_enhancer_disabled"
    )


def test_fake_llm_provider_returns_valid_intent_frame():
    provider = FakeLLMIntentProvider(response_mode="valid")

    raw_text = provider.generate_intent_frame(
        "我想找锂电池 SEI 和人工 SEI 文章",
        {"topic": "SEI"},
    )
    payload = json.loads(raw_text)

    assert provider.provider_name == "fake_llm_intent_provider"
    assert provider.model_name == "fake-intent-frame-model"
    assert provider.call_count == 1
    assert "JSON object only" in provider.last_prompt
    assert "include, exclude, must_read" in provider.last_prompt
    assert payload["intent_summary"]
    assert "normalized_research_intent" in payload
    assert payload["confidence"]["overall"] == "high"


def test_llm_intent_parser_accepts_valid_json():
    provider = FakeLLMIntentProvider(response_mode="valid")

    raw, suggestion, trace = parse_llm_intent_frame_json(
        provider.generate_intent_frame("SEI in lithium batteries"),
        input_question="SEI in lithium batteries",
        provider=provider.provider_name,
        model=provider.model_name,
    )

    assert raw.parsed_json is not None
    assert raw.malformed_output is False
    assert raw.error == ""
    assert suggestion is not None
    assert "solid electrolyte interphase" in suggestion.topic
    assert "lithium battery" in suggestion.target_context_candidates
    assert suggestion.confidence == 0.9
    assert trace.llm_called is True
    assert trace.fallback_used is False
    assert trace.verified_candidate_count == 0
    assert trace.applied_suggestion_count == 0
    assert trace.accepted_suggestion_count == 0
    assert trace.reason_if_no_change == "llm_intent_suggestion_parsed_pending_verification"


def test_llm_intent_parser_fallback_on_malformed_json():
    deterministic_intent = {"topic": "SEI"}
    provider = FakeLLMIntentProvider(response_mode="malformed")

    result = LLMIntentFrameEnhancer(enabled=True, llm_provider=provider).enhance(
        deterministic_intent,
        input_question="SEI in lithium batteries",
    )

    assert result.intent_after_llm is deterministic_intent
    assert result.raw.malformed_output is True
    assert result.raw.error == "malformed_json"
    assert result.trace.fallback_used is True
    assert result.trace.malformed_output is True
    assert result.trace.reason_if_no_change == "malformed_llm_intent_frame_json"


def test_llm_intent_parser_rejects_decision_fields():
    deterministic_intent = {"topic": "OER spin state"}
    provider = FakeLLMIntentProvider(response_mode="decision_fields")

    result = LLMIntentFrameEnhancer(enabled=True, llm_provider=provider).enhance(
        deterministic_intent,
        input_question="OER spin state",
    )

    assert result.intent_after_llm is deterministic_intent
    assert result.verified_suggestion is None
    assert result.raw.parsed_json is not None
    assert result.raw.error == "llm_output_contains_forbidden_decision_fields"
    assert result.trace.fallback_used is True
    assert result.trace.verified_candidate_count == 0
    assert result.trace.applied_suggestion_count == 0
    assert result.trace.rejected_suggestion_count == 1
    assert result.trace.reason_if_no_change == "llm_output_contains_forbidden_decision_fields"


def test_llm_intent_enhancer_does_not_emit_include_exclude():
    provider = FakeLLMIntentProvider(response_mode="valid")

    result = LLMIntentFrameEnhancer(enabled=True, llm_provider=provider).enhance(
        {"topic": "SEI"},
        input_question="SEI in lithium batteries",
    )
    suggestion_data = result.verified_suggestion.__dataclass_fields__

    assert result.verified_suggestion is not None
    assert "include" not in suggestion_data
    assert "exclude" not in suggestion_data
    assert result.trace.fallback_used is False


def test_llm_intent_enhancer_no_real_api_required():
    provider = FakeLLMIntentProvider(response_mode="valid")

    result = LLMIntentFrameEnhancer(enabled=True, llm_provider=provider).enhance(
        {"topic": "SEI"},
        input_question="SEI in lithium batteries",
    )

    assert provider.call_count == 1
    assert result.raw.provider == "fake_llm_intent_provider"
    assert result.raw.model == "fake-intent-frame-model"
    assert result.trace.llm_called is True
    assert result.trace.fallback_used is False
    assert result.trace.verified_candidate_count >= result.trace.applied_suggestion_count
    assert result.trace.applied_suggestion_count > 0
    assert result.trace.accepted_suggestion_count > 0


def test_llm_intent_parser_fallback_on_unsupported_domain_expansion():
    provider = FakeLLMIntentProvider(response_mode="unsupported_domain_expansion")

    result = LLMIntentFrameEnhancer(enabled=True, llm_provider=provider).enhance(
        {"topic": "SEI"},
        input_question="SEI in lithium batteries",
    )

    assert result.verified_suggestion is None
    assert result.trace.fallback_used is True
    assert result.trace.verified_candidate_count > 0
    assert result.trace.applied_suggestion_count == 0
    assert result.trace.accepted_suggestion_count == 0
    assert result.trace.unsupported_suggestion_count >= 1
    assert result.trace.reason_if_no_change == "llm_output_contains_unsupported_domain_expansion"
    assert result.verification_result is not None
    assert "llm_warning_unsupported_domain_expansion" in result.verification_result["verification_warnings"]


def test_llm_intent_parser_fallback_on_overbroad_json():
    provider = FakeLLMIntentProvider(response_mode="overbroad")

    result = LLMIntentFrameEnhancer(enabled=True, llm_provider=provider).enhance(
        {"topic": "AI literature screening"},
        input_question="AI literature screening",
    )

    assert result.verified_suggestion is None
    assert result.trace.fallback_used is True
    assert result.trace.unsupported_suggestion_count >= 1
    assert result.trace.reason_if_no_change == "no_verified_llm_intent_suggestions"


def test_prompt_builder_forbids_decision_fields():
    prompt = build_intent_frame_prompt("OER spin state", {"topic": "OER"})

    assert "JSON object only" in prompt
    assert "include, exclude, must_read" in prompt
    assert "domain_decision" in prompt
    assert "Question: OER spin state" in prompt


def test_llm_intent_enhancer_disabled_by_default_matches_deterministic(tmp_path):
    question = "我想找锂电池里 SEI 是什么、尤其人工 SEI 的文章。"
    first_dir = tmp_path / "default"
    second_dir = tmp_path / "explicit_disabled"

    run_pipeline(
        question=question,
        providers=["fake"],
        max_per_query=0,
        output_dir=str(first_dir),
        retriever_agent=fake_retriever(),
        planned_queries_override=["SEI lithium battery artificial SEI"],
    )
    run_pipeline(
        question=question,
        providers=["fake"],
        max_per_query=0,
        output_dir=str(second_dir),
        retriever_agent=fake_retriever(),
        planned_queries_override=["SEI lithium battery artificial SEI"],
        enable_llm_intent_enhancer=False,
    )

    first_contract = json.loads((first_dir / "search_contract.json").read_text())
    second_contract = json.loads((second_dir / "search_contract.json").read_text())
    assert first_contract == second_contract
    assert not (first_dir / "llm_intent_frame_raw.json").exists()
    assert not (second_dir / "llm_intent_frame_raw.json").exists()


def test_llm_intent_enhancer_writes_auditable_artifacts(tmp_path):
    out = tmp_path / "outputs"

    run_pipeline(
        question="我想找锂电池里 SEI 是什么、尤其人工 SEI 的文章。",
        providers=["fake"],
        max_per_query=0,
        output_dir=str(out),
        retriever_agent=fake_retriever(),
        planned_queries_override=["SEI lithium battery artificial SEI"],
        enable_llm_intent_enhancer=True,
        llm_intent_provider=FakeLLMIntentProvider(response_mode="valid"),
    )

    for filename in [
        "intent_frame_before_llm.json",
        "llm_intent_frame_raw.json",
        "llm_intent_frame_verified.json",
        "search_contract_before_llm.json",
        "search_contract_after_llm.json",
        "llm_intent_enhancement_trace.json",
    ]:
        assert (out / filename).exists()
    trace = json.loads((out / "llm_intent_enhancement_trace.json").read_text())
    assert trace["llm_enabled"] is True
    assert trace["llm_called"] is True
    assert trace["fallback_used"] is False
    assert trace["verified_candidate_count"] >= trace["applied_suggestion_count"]
    assert trace["applied_suggestion_count"] > 0
    assert trace["accepted_suggestion_count"] > 0


def test_llm_trace_distinguishes_verified_candidates_from_applied_suggestions():
    provider = FakeLLMIntentProvider(response_mode="unsupported_domain_expansion")

    result = LLMIntentFrameEnhancer(enabled=True, llm_provider=provider).enhance(
        {"topic": "SEI", "target_context": ["lithium battery"]},
        input_question="我想找锂电池 SEI 文章。",
    )

    assert result.trace.fallback_used is True
    assert result.trace.verified_candidate_count > 0
    assert result.trace.applied_suggestion_count == 0
    assert result.trace.accepted_suggestion_count == 0
    assert result.trace.rejected_suggestion_count > 0
    assert result.trace.unsupported_suggestion_count > 0


def test_fallback_trace_has_zero_applied_suggestions():
    provider = FakeLLMIntentProvider(response_mode="unsupported_domain_expansion")

    result = LLMIntentFrameEnhancer(enabled=True, llm_provider=provider).enhance(
        {"topic": "SEI"},
        input_question="SEI in lithium batteries",
    )

    assert result.trace.fallback_used is True
    assert result.trace.applied_suggestion_count == 0
    assert result.trace.accepted_suggestion_count == 0
    assert result.trace.reason_if_no_change == "llm_output_contains_unsupported_domain_expansion"


def test_search_contract_records_llm_verified_provenance(tmp_path):
    out = tmp_path / "outputs"

    run_pipeline(
        question="我想找锂电池里 SEI 是什么、怎么影响锂金属电池循环和枝晶，尤其人工 SEI 的文章。",
        providers=["fake"],
        max_per_query=0,
        output_dir=str(out),
        retriever_agent=fake_retriever(),
        planned_queries_override=["SEI lithium battery artificial SEI dendrite"],
        enable_llm_intent_enhancer=True,
        llm_intent_provider=FakeLLMIntentProvider(response_mode="valid"),
    )

    contract = json.loads((out / "search_contract.json").read_text())
    sources = {
        group.get("source")
        for group in contract.get("constraint_groups", [])
    }
    assert "llm_suggested_rule_verified" in sources
    assert contract["llm_intent_provenance"]["source"] == "llm_suggested_rule_verified"
    assert contract["llm_verified_suggestions"]


def test_valid_fake_llm_applies_verified_suggestions_to_contract(tmp_path):
    out = tmp_path / "outputs"

    run_pipeline(
        question="我想找锂电池里 SEI 是什么、怎么影响锂金属电池循环和枝晶，尤其人工 SEI 的文章。",
        providers=["fake"],
        max_per_query=0,
        output_dir=str(out),
        retriever_agent=fake_retriever(),
        planned_queries_override=["SEI lithium battery artificial SEI dendrite"],
        enable_llm_intent_enhancer=True,
        llm_intent_provider=FakeLLMIntentProvider(response_mode="valid"),
    )

    trace = json.loads((out / "llm_intent_enhancement_trace.json").read_text())
    contract = json.loads((out / "search_contract_after_llm.json").read_text())
    sources = {
        group.get("source")
        for group in contract.get("constraint_groups", [])
    }

    assert trace["llm_enabled"] is True
    assert trace["llm_called"] is True
    assert trace["fallback_used"] is False
    assert trace["verified_candidate_count"] > 0
    assert trace["applied_suggestion_count"] > 0
    assert trace["accepted_suggestion_count"] == trace["applied_suggestion_count"]
    assert contract["llm_intent_provenance"]["source"] == "llm_suggested_rule_verified"
    assert contract["llm_verified_suggestions"]
    assert "llm_suggested_rule_verified" in sources
    assert not find_forbidden_decision_fields(contract["llm_verified_suggestions"])


def test_rejected_llm_suggestions_do_not_enter_search_contract(tmp_path):
    out = tmp_path / "outputs"

    run_pipeline(
        question="我想找锂电池 SEI 文章。",
        providers=["fake"],
        max_per_query=0,
        output_dir=str(out),
        retriever_agent=fake_retriever(),
        planned_queries_override=["SEI lithium battery"],
        enable_llm_intent_enhancer=True,
        llm_intent_provider=FakeLLMIntentProvider(response_mode="unsupported_domain_expansion"),
    )

    contract = json.loads((out / "search_contract.json").read_text())
    group_text = json.dumps(contract.get("constraint_groups", []), ensure_ascii=False).lower()
    assert "quantum transport" not in group_text
    assert "cryo-em" not in group_text
    assert contract["llm_rejected_suggestions"]
    assert contract["llm_intent_provenance"]["source"] == "deterministic"


def test_unsupported_fake_llm_records_rejections_without_applying_candidates(tmp_path):
    out = tmp_path / "outputs"

    run_pipeline(
        question="我想找锂电池 SEI 文章。",
        providers=["fake"],
        max_per_query=0,
        output_dir=str(out),
        retriever_agent=fake_retriever(),
        planned_queries_override=["SEI lithium battery"],
        enable_llm_intent_enhancer=True,
        llm_intent_provider=FakeLLMIntentProvider(response_mode="unsupported_domain_expansion"),
    )

    trace = json.loads((out / "llm_intent_enhancement_trace.json").read_text())
    contract = json.loads((out / "search_contract_after_llm.json").read_text())
    group_text = json.dumps(contract.get("constraint_groups", []), ensure_ascii=False).lower()
    rejected_text = json.dumps(contract.get("llm_rejected_suggestions", []), ensure_ascii=False).lower()

    assert trace["fallback_used"] is True
    assert trace["verified_candidate_count"] > 0
    assert trace["applied_suggestion_count"] == 0
    assert trace["accepted_suggestion_count"] == 0
    assert trace["rejected_suggestion_count"] > 0
    assert trace["unsupported_suggestion_count"] > 0
    assert contract["llm_verified_suggestions"] == {}
    assert "quantum transport" not in group_text
    assert "cryo-em" not in group_text
    assert "quantum transport" in rejected_text
    assert "cryo-em" in rejected_text


def test_llm_intent_artifacts_positive_and_reject_paths_are_written(tmp_path):
    positive = tmp_path / "positive"
    reject = tmp_path / "reject"
    required_files = {
        "intent_frame_before_llm.json",
        "llm_intent_frame_raw.json",
        "llm_intent_frame_verified.json",
        "search_contract_before_llm.json",
        "search_contract_after_llm.json",
        "llm_intent_enhancement_trace.json",
    }

    for out, mode in [
        (positive, "valid"),
        (reject, "unsupported_domain_expansion"),
    ]:
        run_pipeline(
            question="我想找锂电池里 SEI 是什么、怎么影响锂金属电池循环和枝晶，尤其人工 SEI 的文章。",
            providers=["fake"],
            max_per_query=0,
            output_dir=str(out),
            retriever_agent=fake_retriever(),
            planned_queries_override=["SEI lithium battery artificial SEI dendrite"],
            enable_llm_intent_enhancer=True,
            llm_intent_provider=FakeLLMIntentProvider(response_mode=mode),
        )
        assert required_files <= {path.name for path in out.iterdir()}

    positive_trace = json.loads((positive / "llm_intent_enhancement_trace.json").read_text())
    reject_trace = json.loads((reject / "llm_intent_enhancement_trace.json").read_text())
    positive_contract = json.loads((positive / "search_contract_after_llm.json").read_text())
    reject_contract = json.loads((reject / "search_contract_after_llm.json").read_text())

    assert positive_trace["fallback_used"] is False
    assert positive_contract["llm_verified_suggestions"]
    assert reject_trace["fallback_used"] is True
    assert reject_contract["llm_verified_suggestions"] == {}


def test_llm_provider_unavailable_falls_back_without_crash(tmp_path):
    out = tmp_path / "outputs"

    run_pipeline(
        question="OER spin state papers",
        providers=["fake"],
        max_per_query=0,
        output_dir=str(out),
        retriever_agent=fake_retriever(),
        planned_queries_override=['OER "spin state"'],
        enable_llm_intent_enhancer=True,
        llm_backend="none",
    )

    trace = json.loads((out / "llm_intent_enhancement_trace.json").read_text())
    raw = json.loads((out / "llm_intent_frame_raw.json").read_text())
    assert trace["llm_enabled"] is True
    assert trace["llm_called"] is False
    assert trace["fallback_used"] is True
    assert trace["reason_if_no_change"] == "llm_provider_unavailable"
    assert raw["error"] == "llm_provider_unavailable"


def test_llm_intent_enhancer_does_not_change_paper_decision_fields_directly(tmp_path):
    out = tmp_path / "outputs"

    run_pipeline(
        question="我想找锂电池里 SEI 是什么、尤其人工 SEI 的文章。",
        providers=["fake"],
        max_per_query=1,
        output_dir=str(out),
        retriever_agent=fake_retriever(),
        planned_queries_override=["SEI lithium battery artificial SEI"],
        enable_llm_intent_enhancer=True,
        llm_intent_provider=FakeLLMIntentProvider(response_mode="decision_fields"),
    )

    verified = json.loads((out / "llm_intent_frame_verified.json").read_text())
    ranked_header = (out / "ranked_papers.csv").read_text().splitlines()[0].split(",")
    assert verified is None or verified["verified_suggestion"] is None
    assert "decision" in ranked_header
    assert "reading_priority" in ranked_header
    contract = json.loads((out / "search_contract.json").read_text())
    contract_text = json.dumps(contract, ensure_ascii=False).lower()
    assert '"include"' not in contract_text
    assert "paper a" not in contract_text


def test_chinese_sei_question_accepts_lithium_context_when_grounded():
    question = "我想找锂电池里 SEI 是什么、怎么影响锂金属电池循环和枝晶，尤其人工 SEI 的文章。"
    suggestion = LLMIntentFrameSuggestion(
        topic=["solid electrolyte interphase"],
        target_context_candidates=[
            "lithium battery",
            "lithium metal battery",
            "lithium metal anode",
        ],
        mechanism_need=["SEI formation", "dendrite suppression", "cycling stability"],
        evidence_from_user_question=["锂电池", "锂金属电池", "SEI", "枝晶", "人工 SEI"],
    )

    verification = verify_intent_frame_suggestions(question, {}, suggestion)

    accepted = verification["accepted_suggestions"]
    assert {"value": "lithium battery", "reason": "lithium_battery_context_normalization"} in accepted["target_context_candidates"]
    assert {"value": "lithium metal battery", "reason": "lithium_metal_context_normalization"} in accepted["target_context_candidates"]
    assert {"value": "lithium metal anode", "reason": "lithium_metal_context_normalization"} in accepted["target_context_candidates"]
    assert any(item["value"] == "dendrite suppression" for item in accepted["mechanism_need"])
    assert any(item["value"] == "cycling stability" for item in accepted["mechanism_need"])


def test_artificial_sei_alias_can_be_accepted_when_grounded():
    question = "我想找锂电池里 SEI 是什么，尤其人工 SEI 的文章。"
    suggestion = LLMIntentFrameSuggestion(
        material_or_domain_terms=[
            "artificial SEI",
            "artificial solid electrolyte interphase",
        ],
        evidence_from_user_question=["人工 SEI"],
    )

    verification = verify_intent_frame_suggestions(question, {}, suggestion)

    accepted = verification["accepted_suggestions"]["material_or_domain_terms"]
    assert any(item["value"] == "artificial SEI" for item in accepted)
    assert any(item["value"] == "artificial solid electrolyte interphase" for item in accepted)


def test_lithium_metal_context_accepts_lmb_alias_when_grounded():
    question = "我想找锂金属电池循环和枝晶相关的 SEI 文章。"
    suggestion = LLMIntentFrameSuggestion(
        abbreviation_or_alias_candidates=["LMB", "Li metal"],
        evidence_from_user_question=["锂金属电池"],
    )

    verification = verify_intent_frame_suggestions(question, {}, suggestion)

    accepted = verification["accepted_suggestions"]["abbreviation_or_alias_candidates"]
    assert any(item["value"] == "LMB" for item in accepted)
    assert any(item["value"] == "Li metal" for item in accepted)


def test_li_abbreviation_requires_battery_context():
    unsupported = verify_intent_frame_suggestions(
        "Li diffusion",
        {},
        LLMIntentFrameSuggestion(abbreviation_or_alias_candidates=["Li"]),
    )
    supported = verify_intent_frame_suggestions(
        "Li SEI dendrite cycling in battery anodes",
        {},
        LLMIntentFrameSuggestion(abbreviation_or_alias_candidates=["Li"]),
    )

    assert unsupported["rejected_suggestions"][0]["reason"] == "isolated_abbreviation_without_context"
    assert supported["accepted_suggestions"]["abbreviation_or_alias_candidates"][0]["reason"] == "li_abbreviation_supported_by_battery_context"


def test_rejects_unsupported_domain_expansion():
    question = "我想找锂电池 SEI 文章。"
    suggestion = LLMIntentFrameSuggestion(
        method_need=["cryo-EM"],
        material_or_domain_terms=["quantum transport"],
        warnings=["unsupported_domain_expansion"],
        evidence_from_user_question=["锂电池", "SEI"],
    )

    verification = verify_intent_frame_suggestions(question, {}, suggestion)

    rejected_values = {item["value"] for item in verification["rejected_suggestions"]}
    assert "cryo-EM" in rejected_values
    assert "quantum transport" in rejected_values
    assert "llm_warning_unsupported_domain_expansion" in verification["verification_warnings"]


def test_rejects_unrelated_material_terms():
    verification = verify_intent_frame_suggestions(
        "I need papers on AI literature screening.",
        {"topic": "AI literature screening"},
        LLMIntentFrameSuggestion(material_or_domain_terms=["perovskite oxide"]),
    )

    assert verification["rejected_suggestions"][0]["reason"] == "unrelated_or_unsupported_material_domain_term"


def test_negative_context_suggestions_require_explicit_target_context():
    without_target = verify_intent_frame_suggestions(
        "I want SEI papers.",
        {"topic": "SEI"},
        LLMIntentFrameSuggestion(negative_context_candidates=["sodium-ion battery"]),
    )
    with_target = verify_intent_frame_suggestions(
        "我想找锂电池 SEI 文章。",
        {"target_context": ["lithium battery"]},
        LLMIntentFrameSuggestion(negative_context_candidates=["sodium-ion battery", "potassium-ion battery", "zinc-ion battery"]),
    )

    assert without_target["rejected_suggestions"][0]["reason"] == "negative_context_requires_explicit_target_context"
    accepted_values = {
        item["value"]
        for item in with_target["accepted_suggestions"]["negative_context_candidates"]
    }
    assert {"sodium-ion battery", "potassium-ion battery", "zinc-ion battery"} <= accepted_values
    assert all(
        item["reason"] == "explicit_target_context_non_target_contrast"
        for item in with_target["accepted_suggestions"]["negative_context_candidates"]
    )


def test_verifier_records_accepted_and_rejected_suggestions():
    question = "我想找锂电池里 SEI 是什么、尤其人工 SEI 的文章。"
    suggestion = LLMIntentFrameSuggestion(
        target_context_candidates=["lithium battery"],
        material_or_domain_terms=["artificial SEI", "quantum transport"],
        evidence_from_user_question=["锂电池", "not in question"],
    )

    verification = verify_intent_frame_suggestions(question, {}, suggestion)

    assert verification["accepted_suggestions"]["target_context_candidates"]
    assert verification["accepted_suggestions"]["material_or_domain_terms"][0]["value"] == "artificial SEI"
    rejected = verification["rejected_suggestions"]
    assert any(item["value"] == "quantum transport" for item in rejected)
    assert any(item["value"] == "not in question" for item in rejected)
    assert verification["unsupported_suggestion_count"] == len(rejected)
