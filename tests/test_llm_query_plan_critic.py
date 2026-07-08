import json

import pytest

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.llm.query_plan_critic import (
    ALLOWED_QUERY_PLAN_ACTIONS,
    ALLOWED_QUERY_PLAN_ISSUE_TYPES,
    FakeLLMQueryPlanCriticProvider,
    LLMQueryPlanCritic,
    LLMQueryPlanCritique,
    LLMQueryPlanIssue,
    LLMQueryPlanVerifiedCritique,
    apply_verified_query_critic_issues_to_query_plan,
    build_query_plan_critic_prompt,
    parse_query_plan_critique,
    assert_llm_query_plan_issue_schema_is_non_decisive,
    is_single_acronym_query,
    llm_query_plan_issue_fields,
    verify_query_plan_critique,
    write_query_plan_critic_artifacts,
)
from lit_screening.models import Paper
from lit_screening.models import QueryPlan
from lit_screening.pipeline import build_ablation_config, run_pipeline
from lit_screening.retrieval.base import RetrievalResult


class ExplodingLLMClient:
    def chat_json(self, *args, **kwargs):
        raise AssertionError("disabled LLMQueryPlanCritic must not call LLM")


class QueryCriticPipelineFakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        paper = Paper(
            paper_id="paper-query-critic-1",
            title="Artificial SEI stabilizes lithium metal batteries",
            abstract=(
                "Artificial solid electrolyte interphase layers can stabilize "
                "lithium metal batteries by suppressing dendrite growth."
            ),
            authors=["A. Researcher"],
            year=2024,
            venue="Demo Journal",
            doi="10.1234/query-critic",
            url="https://example.test/query-critic",
            source_provider="fake",
            citation_count=5,
        )
        return RetrievalResult(raw={"query": query}, papers=[paper])


def sample_query_plan():
    return {
        "question": "MOF CO2 capture water stability",
        "planned_queries": [
            {"query_id": "q1", "query": 'MOF "CO2 adsorption" "water stability"'},
            {"query_id": "q2", "query": 'MOF "CO2 capture" "functional groups"'},
        ],
    }


def sample_input_payload():
    return {
        "user_question": "I need artificial SEI papers for lithium batteries.",
        "structured_intent": {"target_context": ["lithium battery"]},
        "search_contract_summary": {"required_aspects": ["artificial SEI"]},
        "query_plan": sample_query_plan(),
        "query_provenance": [{"query_id": "q1", "source": "query_family"}],
    }


def verifier_payload(user_question, queries, **extra):
    payload = {
        "user_question": user_question,
        "query_plan": {
            "planned_queries": [
                {"query_id": query_id, "query": query}
                for query_id, query in queries
            ]
        },
    }
    payload.update(extra)
    return payload


def critique_with(*issues):
    return LLMQueryPlanCritique(issues=list(issues), confidence=0.9)


def run_query_critic_pipeline(
    output_dir,
    *,
    question="I need artificial SEI papers for lithium batteries.",
    enable_llm_query_critic=False,
    llm_query_critic_provider=None,
    max_per_query=0,
    planned_queries_override=None,
    apply_llm_query_critic_repairs=False,
):
    return run_pipeline(
        question,
        providers=["fake"],
        max_per_query=max_per_query,
        output_dir=str(output_dir),
        use_query_families=False,
        planned_queries_override=planned_queries_override
        if planned_queries_override is not None
        else ["SEI lithium battery dendrite"],
        retriever_agent=RetrieverAgent(clients={"fake": QueryCriticPipelineFakeClient()}),
        enable_llm_query_critic=enable_llm_query_critic,
        llm_query_critic_provider=llm_query_critic_provider,
        apply_llm_query_critic_repairs=apply_llm_query_critic_repairs,
    )


def verified_issue(**overrides):
    payload = {
        "issue_type": "missing_aspect",
        "severity": "high",
        "affected_query_ids": ["q1"],
        "affected_aspect": "artificial SEI",
        "evidence": ["The affected query lacks artificial SEI."],
        "suggested_action": "add_anchor",
        "suggested_terms": ["artificial SEI"],
        "rationale": "The user explicitly asks for artificial SEI.",
        "verification_reason": "missing_aspect_verified_by_deterministic_query_plan_checks",
    }
    payload.update(overrides)
    return payload


def verified_critique(*issues):
    return LLMQueryPlanVerifiedCritique(
        verified_issues=list(issues),
        verified_issue_count=len(issues),
    )


def test_llm_query_plan_critic_disabled_noop():
    query_plan = sample_query_plan()

    result = LLMQueryPlanCritic(enabled=False).critique(
        query_plan,
        input_question="MOF CO2 capture water stability",
        llm_client=ExplodingLLMClient(),
    )

    assert result.raw.enabled is False
    assert result.raw.raw_text == ""
    assert result.raw.parsed_json is None
    assert result.raw.malformed_output is False
    assert result.raw.error == ""
    assert result.verified_critique is None
    assert result.query_plan_after_llm_critic is query_plan


def test_llm_query_plan_critic_disabled_does_not_change_query_plan():
    query_plan = sample_query_plan()

    result = LLMQueryPlanCritic(enabled=False).critique(query_plan)

    assert result.query_plan_before_llm_critic is query_plan
    assert result.query_plan_after_llm_critic is query_plan
    assert result.query_plan_after_llm_critic == query_plan


def test_llm_query_plan_critic_trace_disabled():
    result = LLMQueryPlanCritic(
        enabled=False,
        provider="mock",
        model="mock-model",
    ).critique(
        sample_query_plan(),
        input_question="thin film deposition method comparison",
    )

    assert result.trace.llm_query_critic_enabled is False
    assert result.trace.llm_called is False
    assert result.trace.fallback_used is False
    assert result.trace.malformed_output is False
    assert result.trace.verified_issue_count == 0
    assert result.trace.unsupported_issue_count == 0
    assert result.trace.applied_issue_count == 0
    assert result.trace.rejected_issue_count == 0
    assert result.trace.query_added_count == 0
    assert result.trace.query_dropped_count == 0
    assert result.trace.query_modified_count == 0
    assert result.trace.reason_if_no_change == "llm_query_critic_disabled"
    assert result.raw.provider == "mock"
    assert result.raw.model == "mock-model"
    assert result.raw.input_question == "thin film deposition method comparison"


def test_llm_query_plan_issue_schema_contains_no_decision_fields():
    fields = llm_query_plan_issue_fields()

    assert "issue_type" in fields
    assert "suggested_action" in fields
    assert "suggested_terms" in fields
    assert "include" not in fields
    assert "exclude" not in fields
    assert "must_read" not in fields
    assert "domain_decision" not in fields
    assert "final_score" not in fields
    assert "evidence_validity" not in fields
    assert "reading_priority" not in fields
    assert_llm_query_plan_issue_schema_is_non_decisive()


def test_llm_query_plan_critic_allowed_issue_types():
    expected = {
        "missing_aspect",
        "overbroad_query",
        "single_acronym_query",
        "weak_anchor",
        "duplicate_query",
        "cross_domain_injection",
        "missing_target_context",
        "missing_method_anchor",
        "missing_mechanism_anchor",
        "provider_query_too_short",
        "unsupported_suggestion",
        "no_issue",
    }

    assert ALLOWED_QUERY_PLAN_ISSUE_TYPES == expected
    assert LLMQueryPlanIssue(issue_type="weak_anchor").issue_type in expected


def test_llm_query_plan_critic_allowed_actions():
    expected = {
        "add_anchor",
        "drop_query",
        "merge_queries",
        "strengthen_query",
        "add_query_variant",
        "no_change",
    }

    assert ALLOWED_QUERY_PLAN_ACTIONS == expected
    assert LLMQueryPlanIssue(suggested_action="strengthen_query").suggested_action in expected


def test_llm_query_plan_issue_rejects_invalid_issue_type():
    with pytest.raises(ValueError, match="invalid_issue"):
        LLMQueryPlanIssue(issue_type="invalid_issue")


def test_llm_query_plan_issue_rejects_invalid_suggested_action():
    with pytest.raises(ValueError, match="rewrite_everything"):
        LLMQueryPlanIssue(suggested_action="rewrite_everything")


def test_llm_query_plan_issue_accepts_all_allowed_issue_types():
    for issue_type in ALLOWED_QUERY_PLAN_ISSUE_TYPES:
        issue = LLMQueryPlanIssue(issue_type=issue_type)
        assert issue.issue_type == issue_type


def test_llm_query_plan_issue_accepts_all_allowed_actions():
    for action in ALLOWED_QUERY_PLAN_ACTIONS:
        issue = LLMQueryPlanIssue(suggested_action=action)
        assert issue.suggested_action == action


def test_fake_llm_query_plan_critic_provider_returns_valid_json():
    provider = FakeLLMQueryPlanCriticProvider(response_mode="valid")

    raw_text = provider.critique_query_plan(sample_input_payload())
    payload = json.loads(raw_text)

    assert provider.provider_name == "fake_llm_query_plan_critic_provider"
    assert provider.model_name == "fake-query-plan-critic-model"
    assert provider.call_count == 1
    assert "JSON object only" in provider.last_prompt
    assert "Do not output final provider queries" in provider.last_prompt
    assert "include, exclude, must_read" in provider.last_prompt
    assert payload["issues"][0]["issue_type"] == "missing_aspect"
    assert payload["issues"][0]["suggested_action"] == "add_anchor"


def test_fake_llm_query_plan_critic_provider_returns_valid_single_acronym_issue():
    provider = FakeLLMQueryPlanCriticProvider(response_mode="valid_single_acronym_query")
    payload = {
        "user_question": "SEI",
        "query_plan": {
            "planned_queries": [
                {"query_id": "q1", "query": "SEI"},
            ]
        },
    }

    payload = json.loads(provider.critique_query_plan(payload))

    assert payload["issues"][0]["issue_type"] == "single_acronym_query"
    assert payload["issues"][0]["suggested_action"] == "strengthen_query"
    assert payload["issues"][0]["affected_query_ids"] == ["q1"]


def test_single_acronym_normalizer_accepts_repeated_case_variant():
    assert is_single_acronym_query("sei SEI")
    assert is_single_acronym_query("SEI SEI")


def test_single_acronym_normalizer_accepts_boolean_repeated_acronym():
    assert is_single_acronym_query("sei AND SEI")
    assert is_single_acronym_query('"SEI" AND SEI')


def test_single_acronym_normalizer_accepts_plus_or_quoted_acronym():
    assert is_single_acronym_query("+SEI")
    assert is_single_acronym_query('"SEI"')


def test_single_acronym_normalizer_rejects_multi_meaningful_token_query():
    assert not is_single_acronym_query("SEI lithium battery")
    assert not is_single_acronym_query("SEI characterization")
    assert not is_single_acronym_query("artificial SEI lithium battery")
    assert not is_single_acronym_query("SEI solid electrolyte interphase")
    assert not is_single_acronym_query("OER spin state catalyst")
    assert not is_single_acronym_query("MOF CO2 adsorption")


def test_fake_valid_single_acronym_uses_actual_query_text_in_evidence():
    provider = FakeLLMQueryPlanCriticProvider(response_mode="valid_single_acronym_query")
    payload = {
        "user_question": "SEI",
        "query_plan": {
            "planned_queries": [
                {"query_id": "q7", "query": "sei SEI"},
                {"query_id": "q8", "query": "SEI lithium battery"},
            ]
        },
    }

    response = json.loads(provider.critique_query_plan(payload))
    issue = response["issues"][0]

    assert issue["affected_query_ids"] == ["q7"]
    assert issue["affected_query_text"] == "sei SEI"
    assert "sei SEI" in issue["evidence"][0]


def test_llm_query_plan_critic_parser_accepts_valid_json():
    provider = FakeLLMQueryPlanCriticProvider(response_mode="valid")

    raw, critique, trace = parse_query_plan_critique(
        provider.critique_query_plan(sample_input_payload()),
        input_question="I need artificial SEI papers for lithium batteries.",
        provider=provider.provider_name,
        model=provider.model_name,
    )

    assert raw.parsed_json is not None
    assert raw.malformed_output is False
    assert raw.error == ""
    assert critique is not None
    assert critique.confidence == 0.9
    assert critique.issues[0].issue_type == "missing_aspect"
    assert critique.issues[0].suggested_action == "add_anchor"
    assert trace.llm_called is True
    assert trace.fallback_used is False
    assert trace.verified_issue_count == 0
    assert trace.applied_issue_count == 0
    assert trace.reason_if_no_change == "llm_query_critic_parsed_pending_verification"
    assert trace.query_added_count == 0
    assert trace.query_dropped_count == 0
    assert trace.query_modified_count == 0


def test_llm_query_plan_critic_fallback_on_malformed_json():
    query_plan = sample_query_plan()
    provider = FakeLLMQueryPlanCriticProvider(response_mode="malformed_json")

    result = LLMQueryPlanCritic(enabled=True, llm_provider=provider).critique(
        query_plan,
        input_question="I need artificial SEI papers for lithium batteries.",
    )

    assert result.query_plan_after_llm_critic is query_plan
    assert result.verified_critique is None
    assert result.raw.malformed_output is True
    assert result.trace.llm_called is True
    assert result.trace.fallback_used is True
    assert result.trace.malformed_output is True
    assert result.trace.verified_issue_count == 0
    assert result.trace.applied_issue_count == 0
    assert result.trace.query_added_count == 0
    assert result.trace.query_dropped_count == 0
    assert result.trace.query_modified_count == 0
    assert result.trace.reason_if_no_change == "llm_query_critic_malformed_output"


def test_llm_query_plan_critic_fallback_on_forbidden_decision_fields():
    query_plan = sample_query_plan()
    provider = FakeLLMQueryPlanCriticProvider(response_mode="forbidden_decision_fields")

    result = LLMQueryPlanCritic(enabled=True, llm_provider=provider).critique(
        query_plan,
        input_question="I need artificial SEI papers for lithium batteries.",
    )

    assert result.query_plan_after_llm_critic is query_plan
    assert result.verified_critique is None
    assert result.raw.error == "llm_query_critic_output_contains_forbidden_decision_fields"
    assert result.trace.llm_called is True
    assert result.trace.fallback_used is True
    assert result.trace.malformed_output is False
    assert result.trace.verified_issue_count == 0
    assert result.trace.applied_issue_count == 0
    assert (
        result.trace.reason_if_no_change
        == "llm_query_critic_output_contains_forbidden_decision_fields"
    )


def test_llm_query_plan_critic_fallback_on_invalid_issue_type():
    raw_text = json.dumps(
        {
            "issues": [
                {
                    "issue_type": "invented_issue",
                    "severity": "high",
                    "affected_query_ids": ["q1"],
                    "affected_aspect": "artificial SEI",
                    "evidence": ["Missing artificial SEI."],
                    "suggested_action": "add_anchor",
                    "suggested_terms": ["artificial SEI"],
                    "rationale": "Invalid issue type should fallback.",
                }
            ],
            "warnings": [],
            "confidence": "high",
            "evidence_from_query_plan": [],
        }
    )

    raw, critique, trace = parse_query_plan_critique(raw_text)

    assert critique is None
    assert raw.error == "llm_query_critic_invalid_schema"
    assert trace.fallback_used is True
    assert trace.malformed_output is False
    assert trace.verified_issue_count == 0
    assert trace.applied_issue_count == 0
    assert trace.reason_if_no_change == "llm_query_critic_invalid_schema"


def test_llm_query_plan_critic_fallback_on_invalid_suggested_action():
    raw_text = json.dumps(
        {
            "issues": [
                {
                    "issue_type": "missing_aspect",
                    "severity": "high",
                    "affected_query_ids": ["q1"],
                    "affected_aspect": "artificial SEI",
                    "evidence": ["Missing artificial SEI."],
                    "suggested_action": "rewrite_everything",
                    "suggested_terms": ["artificial SEI"],
                    "rationale": "Invalid suggested action should fallback.",
                }
            ],
            "warnings": [],
            "confidence": "high",
            "evidence_from_query_plan": [],
        }
    )

    raw, critique, trace = parse_query_plan_critique(raw_text)

    assert critique is None
    assert raw.error == "llm_query_critic_invalid_schema"
    assert trace.fallback_used is True
    assert trace.malformed_output is False
    assert trace.verified_issue_count == 0
    assert trace.applied_issue_count == 0
    assert trace.reason_if_no_change == "llm_query_critic_invalid_schema"


def test_llm_query_plan_critic_no_issue_valid_output():
    provider = FakeLLMQueryPlanCriticProvider(response_mode="no_issue")

    raw, critique, trace = parse_query_plan_critique(
        provider.critique_query_plan(sample_input_payload()),
    )

    assert raw.error == ""
    assert critique is not None
    assert critique.issues[0].issue_type == "no_issue"
    assert critique.issues[0].suggested_action == "no_change"
    assert trace.fallback_used is False
    assert trace.verified_issue_count == 0
    assert trace.applied_issue_count == 0
    assert trace.reason_if_no_change == "llm_query_critic_no_issues"


def test_llm_query_plan_critic_no_real_api_required():
    query_plan = sample_query_plan()
    provider = FakeLLMQueryPlanCriticProvider(response_mode="valid")

    result = LLMQueryPlanCritic(enabled=True, llm_provider=provider).critique(
        query_plan,
        input_question="I need artificial SEI papers for lithium batteries.",
        input_payload=sample_input_payload(),
    )

    assert provider.call_count == 1
    assert result.query_plan_after_llm_critic is query_plan
    assert result.raw.provider == "fake_llm_query_plan_critic_provider"
    assert result.raw.model == "fake-query-plan-critic-model"
    assert result.verified_critique is not None
    assert result.trace.llm_called is True
    assert result.trace.fallback_used is False
    assert result.trace.verified_issue_count == 1
    assert result.trace.applied_issue_count == 0
    assert result.trace.query_added_count == 0
    assert result.trace.query_dropped_count == 0
    assert result.trace.query_modified_count == 0
    assert result.trace.reason_if_no_change == "llm_query_critic_verified_but_not_applied"


def test_query_plan_critic_verifier_accepts_grounded_missing_aspect():
    payload = verifier_payload(
        "I need artificial SEI papers for lithium batteries.",
        [("q1", "SEI lithium battery dendrite")],
        search_contract_summary={"required_aspects": ["artificial SEI"]},
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="missing_aspect",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="artificial SEI",
            evidence=["No query contains artificial or engineered SEI."],
            suggested_action="add_anchor",
            suggested_terms=["artificial SEI", "engineered SEI"],
            rationale="The user explicitly asks for artificial SEI.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 1
    assert verified.rejected_issue_count == 0
    assert verified.verified_issues[0]["issue_type"] == "missing_aspect"


def test_query_plan_critic_verifier_rejects_unsupported_suggestion():
    payload = verifier_payload(
        "I need AI literature screening papers.",
        [("q1", "AI literature screening human feedback")],
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="unsupported_suggestion",
            severity="medium",
            affected_query_ids=["q1"],
            affected_aspect="quantum transport",
            evidence=["Quantum transport is not grounded in the question."],
            suggested_action="no_change",
            suggested_terms=["quantum transport"],
            rationale="Flag unsupported expansion only.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 0
    assert verified.rejected_issue_count == 1
    assert verified.rejected_issues[0]["reason"] == "unsupported_suggestion"


def test_query_plan_critic_verifier_accepts_single_acronym_query():
    payload = verifier_payload(
        "I need artificial SEI papers for lithium batteries.",
        [("q1", "SEI")],
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="single_acronym_query",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="SEI",
            evidence=["The affected query is only SEI."],
            suggested_action="strengthen_query",
            suggested_terms=["lithium battery"],
            rationale="Single-acronym queries are under-constrained.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 1
    assert verified.verified_issues[0]["issue_type"] == "single_acronym_query"


def test_query_plan_critic_verifier_rejects_acronym_inside_anchored_query():
    payload = verifier_payload(
        "I need artificial SEI papers for lithium batteries.",
        [("q1", "SEI lithium battery artificial SEI")],
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="single_acronym_query",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="SEI",
            evidence=["The query contains SEI."],
            suggested_action="strengthen_query",
            suggested_terms=["lithium battery"],
            rationale="This should not be accepted because the query is anchored.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 0
    assert verified.rejected_issues[0]["reason"] == "affected_query_not_single_acronym"


def test_query_plan_critic_verifier_accepts_overbroad_short_query():
    payload = verifier_payload(
        "I need MOF CO2 capture water stability papers.",
        [("q1", "MOF CO2")],
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="overbroad_query",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="MOF CO2 capture",
            evidence=["The affected query is only MOF CO2."],
            suggested_action="strengthen_query",
            suggested_terms=["water stability"],
            rationale="The query is too short for a multi-axis question.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 1
    assert verified.verified_issues[0]["issue_type"] == "overbroad_query"


def test_query_plan_critic_verifier_accepts_missing_target_context_when_contract_requires_it():
    payload = verifier_payload(
        "I need artificial SEI papers for lithium batteries.",
        [("q1", "artificial SEI dendrite")],
        search_contract_summary={
            "constraint_groups": [
                {"group_name": "target_context_group", "terms": ["lithium battery"]}
            ]
        },
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="missing_target_context",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="lithium battery",
            evidence=["The affected query lacks lithium battery target context."],
            suggested_action="add_anchor",
            suggested_terms=["lithium battery"],
            rationale="The contract requires lithium target context.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 1
    assert verified.verified_issues[0]["issue_type"] == "missing_target_context"


def test_query_plan_critic_verifier_rejects_missing_target_context_without_target_group():
    payload = verifier_payload(
        "I need SEI papers.",
        [("q1", "SEI dendrite")],
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="missing_target_context",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="lithium battery",
            evidence=["The affected query lacks lithium battery target context."],
            suggested_action="add_anchor",
            suggested_terms=["lithium battery"],
            rationale="No deterministic target context exists.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 0
    assert verified.rejected_issues[0]["reason"] == "missing_target_context_requires_target_group"


def test_query_plan_critic_verifier_accepts_duplicate_query():
    payload = verifier_payload(
        "I need MOF CO2 capture papers.",
        [("q1", "MOF CO2"), ("q2", '"MOF" "CO2"')],
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="duplicate_query",
            severity="medium",
            affected_query_ids=["q1", "q2"],
            affected_aspect="MOF CO2",
            evidence=["The affected queries normalize to the same terms."],
            suggested_action="merge_queries",
            suggested_terms=[],
            rationale="Duplicate queries do not add coverage.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 1
    assert verified.verified_issues[0]["issue_type"] == "duplicate_query"


def test_query_plan_critic_verifier_accepts_cross_domain_injection():
    payload = verifier_payload(
        "I need AI literature screening papers.",
        [("q1", "AI literature screening quantum transport")],
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="cross_domain_injection",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="quantum transport",
            evidence=["The affected query contains quantum transport."],
            suggested_action="drop_query",
            suggested_terms=["quantum transport"],
            rationale="Quantum transport is unrelated to AI literature screening.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 1
    assert verified.verified_issues[0]["issue_type"] == "cross_domain_injection"


def test_query_plan_critic_verifier_rejects_unrelated_suggested_terms():
    payload = verifier_payload(
        "I need artificial SEI papers for lithium batteries.",
        [("q1", "SEI lithium battery")],
        search_contract_summary={"required_aspects": ["artificial SEI"]},
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="missing_aspect",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="artificial SEI",
            evidence=["The affected query lacks artificial SEI."],
            suggested_action="add_anchor",
            suggested_terms=["quantum transport"],
            rationale="The suggested term introduces an unrelated field.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 0
    assert verified.rejected_issues[0]["reason"] == "suggested_terms_introduce_unsupported_domain"


def test_query_plan_critic_verifier_records_verified_and_rejected_counts():
    payload = verifier_payload(
        "I need artificial SEI papers for lithium batteries.",
        [("q1", "SEI lithium battery")],
        search_contract_summary={"required_aspects": ["artificial SEI"]},
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="missing_aspect",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="artificial SEI",
            evidence=["The affected query lacks artificial SEI."],
            suggested_action="add_anchor",
            suggested_terms=["artificial SEI"],
            rationale="Artificial SEI is grounded in the user question.",
        ),
        LLMQueryPlanIssue(
            issue_type="unsupported_suggestion",
            severity="medium",
            affected_query_ids=["q1"],
            affected_aspect="quantum transport",
            evidence=["Quantum transport is unsupported."],
            suggested_action="no_change",
            suggested_terms=["quantum transport"],
            rationale="Unsupported expansion should be rejected.",
        ),
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 1
    assert verified.rejected_issue_count == 1
    assert verified.unsupported_issue_count == 1
    assert verified.verification_warnings == ["some_llm_query_plan_issues_rejected"]


def mixed_rejection_counts_verified_critique():
    payload = verifier_payload(
        "I need SEI papers.",
        [("q1", "SEI dendrite")],
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="unsupported_suggestion",
            severity="medium",
            affected_query_ids=["q1"],
            affected_aspect="quantum transport",
            evidence=["Quantum transport is unsupported."],
            suggested_action="no_change",
            suggested_terms=["quantum transport"],
            rationale="Unsupported expansion should be rejected.",
        ),
        LLMQueryPlanIssue(
            issue_type="missing_target_context",
            severity="high",
            affected_query_ids=["q1"],
            affected_aspect="lithium battery",
            evidence=["The query lacks lithium battery target context."],
            suggested_action="add_anchor",
            suggested_terms=["lithium battery"],
            rationale="This should be rejected because no target group exists.",
        ),
    )
    return verify_query_plan_critique(payload, critique)


def test_query_plan_critic_unsupported_issue_count_counts_only_unsupported_reasons():
    verified = mixed_rejection_counts_verified_critique()

    assert verified.unsupported_issue_count == 1


def test_query_plan_critic_rejected_issue_count_counts_all_rejections():
    verified = mixed_rejection_counts_verified_critique()

    assert verified.rejected_issue_count == 2


def test_query_plan_critic_rejection_reason_counts_if_available():
    verified = mixed_rejection_counts_verified_critique()

    assert verified.rejection_reason_counts == {
        "unsupported_suggestion": 1,
        "missing_target_context_requires_target_group": 1,
    }


def test_query_plan_critic_verifier_does_not_modify_query_plan():
    query_plan = sample_query_plan()
    provider = FakeLLMQueryPlanCriticProvider(response_mode="valid")

    result = LLMQueryPlanCritic(enabled=True, llm_provider=provider).critique(
        query_plan,
        input_question="I need artificial SEI papers for lithium batteries.",
        input_payload=sample_input_payload(),
    )

    assert result.query_plan_before_llm_critic is query_plan
    assert result.query_plan_after_llm_critic is query_plan
    assert result.trace.verified_issue_count == 1
    assert result.trace.applied_issue_count == 0
    assert result.trace.query_added_count == 0
    assert result.trace.query_dropped_count == 0
    assert result.trace.query_modified_count == 0
    assert result.trace.reason_if_no_change == "llm_query_critic_verified_but_not_applied"


def test_write_query_plan_critic_artifacts_reserved_noop(tmp_path):
    query_plan = sample_query_plan()
    result = LLMQueryPlanCritic(enabled=False).critique(query_plan)

    artifacts = write_query_plan_critic_artifacts(
        tmp_path,
        query_plan_before_llm_critic=result.query_plan_before_llm_critic,
        raw=result.raw,
        verified_critique=result.verified_critique,
        trace=result.trace,
        query_plan_after_llm_critic=result.query_plan_after_llm_critic,
    )

    assert set(artifacts) == {
        "query_plan_before_llm_critic",
        "llm_query_critic_raw",
        "llm_query_critic_verified",
        "query_plan_after_llm_critic",
        "query_repair_after_llm_critic",
        "llm_query_critic_trace",
    }
    assert json.loads((tmp_path / "query_plan_before_llm_critic.json").read_text()) == query_plan
    assert json.loads((tmp_path / "llm_query_critic_raw.json").read_text())["enabled"] is False
    assert json.loads((tmp_path / "llm_query_critic_verified.json").read_text()) is None
    assert json.loads((tmp_path / "query_plan_after_llm_critic.json").read_text()) == query_plan
    assert json.loads((tmp_path / "query_repair_after_llm_critic.json").read_text())[
        "apply_enabled"
    ] is False
    assert (
        json.loads((tmp_path / "llm_query_critic_trace.json").read_text())["reason_if_no_change"]
        == "llm_query_critic_disabled"
    )


def test_llm_query_critic_disabled_by_default_pipeline_unchanged(tmp_path):
    default_dir = tmp_path / "default"
    disabled_dir = tmp_path / "disabled"

    run_query_critic_pipeline(default_dir)
    run_query_critic_pipeline(disabled_dir, enable_llm_query_critic=False)

    default_plan = json.loads((default_dir / "planned_queries.json").read_text())
    disabled_plan = json.loads((disabled_dir / "planned_queries.json").read_text())

    assert default_plan["final_provider_queries"] == disabled_plan["final_provider_queries"]
    assert not (default_dir / "llm_query_critic_trace.json").exists()
    assert not (disabled_dir / "llm_query_critic_trace.json").exists()


def test_llm_query_critic_enabled_writes_artifacts(tmp_path):
    output_dir = tmp_path / "enabled"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )

    for name in [
        "query_plan_before_llm_critic.json",
        "llm_query_critic_raw.json",
        "llm_query_critic_verified.json",
        "query_plan_after_llm_critic.json",
        "llm_query_critic_trace.json",
    ]:
        assert (output_dir / name).exists()


def test_llm_query_critic_artifacts_keep_query_plan_unchanged_in_phase1(tmp_path):
    output_dir = tmp_path / "unchanged"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )

    before = json.loads((output_dir / "query_plan_before_llm_critic.json").read_text())
    after = json.loads((output_dir / "query_plan_after_llm_critic.json").read_text())

    assert after == before


def test_llm_query_critic_trace_verified_but_not_applied(tmp_path):
    output_dir = tmp_path / "verified"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )

    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())
    assert trace["llm_query_critic_enabled"] is True
    assert trace["llm_called"] is True
    assert trace["fallback_used"] is False
    assert trace["verified_issue_count"] > 0
    assert trace["unsupported_issue_count"] == 0
    assert trace["applied_issue_count"] == 0
    assert trace["query_added_count"] == 0
    assert trace["query_dropped_count"] == 0
    assert trace["query_modified_count"] == 0
    assert trace["reason_if_no_change"] == "llm_query_critic_verified_but_not_applied"


def test_llm_query_critic_provider_unavailable_fallback_without_crash(tmp_path):
    output_dir = tmp_path / "provider-unavailable"

    run_query_critic_pipeline(output_dir, enable_llm_query_critic=True)

    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())
    raw = json.loads((output_dir / "llm_query_critic_raw.json").read_text())

    assert trace["llm_query_critic_enabled"] is True
    assert trace["llm_called"] is False
    assert trace["fallback_used"] is True
    assert trace["verified_issue_count"] == 0
    assert trace["unsupported_issue_count"] == 0
    assert trace["applied_issue_count"] == 0
    assert trace["reason_if_no_change"] == "llm_query_critic_provider_unavailable"
    assert raw["error"] == "llm_query_critic_provider_unavailable"


def test_llm_query_critic_fake_valid_pipeline_artifacts(tmp_path):
    output_dir = tmp_path / "fake-valid"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )

    verified = json.loads((output_dir / "llm_query_critic_verified.json").read_text())
    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())

    assert verified["verified_issue_count"] > 0
    assert verified["verified_issues"][0]["issue_type"] == "missing_aspect"
    assert trace["verified_issue_count"] == verified["verified_issue_count"]


def test_llm_query_critic_fake_reject_pipeline_artifacts(tmp_path):
    output_dir = tmp_path / "fake-reject"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(
            response_mode="unsupported_suggestion"
        ),
    )

    verified = json.loads((output_dir / "llm_query_critic_verified.json").read_text())
    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())

    assert verified["verified_issue_count"] == 0
    assert verified["rejected_issue_count"] > 0
    assert verified["unsupported_issue_count"] > 0
    assert trace["verified_issue_count"] == 0
    assert trace["rejected_issue_count"] > 0
    assert trace["unsupported_issue_count"] > 0
    assert trace["reason_if_no_change"] == "llm_query_critic_no_verified_issues"


def test_llm_query_critic_does_not_change_final_provider_queries(tmp_path):
    output_dir = tmp_path / "provider-queries"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )

    planned = json.loads((output_dir / "planned_queries.json").read_text())
    before = json.loads((output_dir / "query_plan_before_llm_critic.json").read_text())
    after = json.loads((output_dir / "query_plan_after_llm_critic.json").read_text())

    assert before["final_provider_queries"] == after["final_provider_queries"]
    assert planned["final_provider_queries"] == before["final_provider_queries"]


def test_llm_query_critic_does_not_change_paper_decision_fields(tmp_path):
    default_dir = tmp_path / "paper-default"
    enabled_dir = tmp_path / "paper-enabled"

    run_query_critic_pipeline(default_dir, max_per_query=1)
    run_query_critic_pipeline(
        enabled_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
        max_per_query=1,
    )

    default_ranked = (default_dir / "ranked_papers.csv").read_text()
    enabled_ranked = (enabled_dir / "ranked_papers.csv").read_text()

    assert enabled_ranked == default_ranked


def test_llm_query_critic_pipeline_fake_valid_artifact_trace_fields(tmp_path):
    output_dir = tmp_path / "fake-valid-trace-fields"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )

    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())
    assert trace["llm_query_critic_enabled"] is True
    assert trace["llm_called"] is True
    assert trace["fallback_used"] is False
    assert trace["verified_issue_count"] > 0
    assert trace["applied_issue_count"] == 0
    assert trace["query_added_count"] == 0
    assert trace["query_dropped_count"] == 0
    assert trace["query_modified_count"] == 0
    assert trace["reason_if_no_change"] == "llm_query_critic_verified_but_not_applied"


def test_llm_query_critic_pipeline_fake_reject_artifact_trace_fields(tmp_path):
    output_dir = tmp_path / "fake-reject-trace-fields"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(
            response_mode="unsupported_suggestion"
        ),
    )

    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())
    assert trace["llm_query_critic_enabled"] is True
    assert trace["llm_called"] is True
    assert trace["fallback_used"] is False
    assert trace["verified_issue_count"] == 0
    assert trace["rejected_issue_count"] > 0
    assert trace["unsupported_issue_count"] > 0
    assert trace["applied_issue_count"] == 0
    assert trace["query_added_count"] == 0
    assert trace["query_dropped_count"] == 0
    assert trace["query_modified_count"] == 0


def test_llm_query_critic_artifact_query_plan_before_after_equal(tmp_path):
    output_dir = tmp_path / "before-after-equal"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )

    before = json.loads((output_dir / "query_plan_before_llm_critic.json").read_text())
    after = json.loads((output_dir / "query_plan_after_llm_critic.json").read_text())
    assert after == before


def test_llm_query_critic_phase1_support_status_is_diagnostic():
    config = build_ablation_config(
        ablation_config_name="llm_query_critic_only",
        enable_llm_query_critic=True,
    )

    assert (
        config["support_status"]["llm_query_plan_critic"]
        == "diagnostic_artifacts_only"
    )


def test_llm_query_critic_pipeline_fake_positive_verified_artifact_trace_fields(tmp_path):
    output_dir = tmp_path / "fake-positive-verified"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
        planned_queries_override=["SEI lithium battery dendrite"],
    )

    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())
    assert trace["llm_query_critic_enabled"] is True
    assert trace["llm_called"] is True
    assert trace["fallback_used"] is False
    assert trace["verified_issue_count"] > 0
    assert trace["unsupported_issue_count"] == 0
    assert trace["applied_issue_count"] == 0
    assert trace["query_added_count"] == 0
    assert trace["query_dropped_count"] == 0
    assert trace["query_modified_count"] == 0
    assert trace["reason_if_no_change"] == "llm_query_critic_verified_but_not_applied"


def test_llm_query_critic_positive_verified_run_keeps_query_plan_unchanged(tmp_path):
    output_dir = tmp_path / "fake-positive-unchanged"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
        planned_queries_override=["SEI lithium battery dendrite"],
    )

    before = json.loads((output_dir / "query_plan_before_llm_critic.json").read_text())
    after = json.loads((output_dir / "query_plan_after_llm_critic.json").read_text())
    assert after == before


def test_llm_query_critic_positive_verified_run_has_verified_issue_count(tmp_path):
    output_dir = tmp_path / "fake-positive-count"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
        planned_queries_override=["SEI lithium battery dendrite"],
    )

    verified = json.loads((output_dir / "llm_query_critic_verified.json").read_text())
    assert verified["verified_issue_count"] > 0
    assert verified["verified_issues"][0]["issue_type"] == "missing_aspect"


def test_llm_query_critic_full_query_family_rejects_already_covered_missing_aspect(tmp_path):
    output_dir = tmp_path / "full-query-family-covered"

    run_pipeline(
        "I need artificial SEI papers for lithium batteries.",
        providers=["fake"],
        max_per_query=0,
        output_dir=str(output_dir),
        retriever_agent=RetrieverAgent(clients={"fake": QueryCriticPipelineFakeClient()}),
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )

    verified = json.loads((output_dir / "llm_query_critic_verified.json").read_text())
    assert verified["verified_issue_count"] == 0
    assert verified["rejected_issue_count"] > 0
    assert verified["rejected_issues"][0]["reason"] == "affected_aspect_already_covered"


def test_verifier_accepts_repeated_acronym_query_as_single_acronym():
    payload = verifier_payload(
        "SEI",
        [("q1", "sei SEI")],
    )
    critique = critique_with(
        LLMQueryPlanIssue(
            issue_type="single_acronym_query",
            severity="high",
            affected_query_ids=["q1"],
            affected_query_text="sei SEI",
            affected_aspect="SEI",
            evidence=["The affected provider query contains only repeated SEI tokens."],
            suggested_action="strengthen_query",
            suggested_terms=["solid electrolyte interphase", "lithium battery"],
            rationale="Repeated acronym-only query is too broad.",
        )
    )

    verified = verify_query_plan_critique(payload, critique)

    assert verified.verified_issue_count == 1
    assert verified.rejected_issue_count == 0


def test_apply_verified_repeated_acronym_strengthens_query():
    query_plan = {
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase", "lithium battery"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
    )

    assert result.applied_issue_count == 1
    assert result.query_modified_count == 1
    assert result.applied_issue_records[0]["original_query"] == "sei SEI"
    assert "solid electrolyte interphase" in result.applied_issue_records[0]["new_query"]


def test_no_apply_verified_repeated_acronym_keeps_query_plan_unchanged(tmp_path):
    output_dir = tmp_path / "repeated-acronym-no-apply"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(
            response_mode="valid_single_acronym_query"
        ),
        planned_queries_override=["sei SEI"],
    )

    before = json.loads((output_dir / "query_plan_before_llm_critic.json").read_text())
    after = json.loads((output_dir / "query_plan_after_llm_critic.json").read_text())
    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())

    assert before == after
    assert trace["verified_issue_count"] > 0
    assert trace["applied_issue_count"] == 0
    assert trace["query_modified_count"] == 0
    assert trace["reason_if_no_change"] == "llm_query_critic_verified_but_not_applied"


def test_llm_query_critic_repairs_disabled_by_default(tmp_path):
    output_dir = tmp_path / "repairs-disabled-default"

    run_query_critic_pipeline(output_dir)

    assert not (output_dir / "query_repair_after_llm_critic.json").exists()


def test_llm_query_critic_verified_but_not_applied_without_apply_flag(tmp_path):
    output_dir = tmp_path / "verified-not-applied"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )

    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())
    before = json.loads((output_dir / "query_plan_before_llm_critic.json").read_text())
    after = json.loads((output_dir / "query_plan_after_llm_critic.json").read_text())
    repair = json.loads((output_dir / "query_repair_after_llm_critic.json").read_text())

    assert before == after
    assert repair["apply_enabled"] is False
    assert trace["verified_issue_count"] > 0
    assert trace["applied_issue_count"] == 0
    assert trace["reason_if_no_change"] == "llm_query_critic_verified_but_not_applied"


def test_apply_verified_missing_aspect_adds_anchored_query():
    query_plan = {
        "queries": ["SEI lithium battery dendrite"],
        "final_provider_queries": {"fake": ["SEI lithium battery dendrite"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(verified_issue()),
        user_question="I need artificial SEI papers for lithium batteries.",
    )

    queries = result.query_plan_after_llm_critic["final_provider_queries"]["fake"]
    assert result.applied_issue_count == 1
    assert result.query_added_count == 1
    assert any("artificial SEI" in query for query in queries)


def test_apply_verified_single_acronym_strengthens_query():
    query_plan = {
        "queries": ["SEI"],
        "final_provider_queries": {"fake": ["SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase", "lithium battery"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
    )

    queries = result.query_plan_after_llm_critic["final_provider_queries"]["fake"]
    assert result.applied_issue_count == 1
    assert result.query_modified_count == 1
    assert queries == ['SEI "solid electrolyte interphase"']


def test_apply_repair_accepts_glossary_expansion_for_acronym():
    query_plan = {
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
        user_question="SEI",
    )

    record = result.applied_issue_records[0]
    assert result.applied_issue_count == 1
    assert record["applied_terms"] == ["solid electrolyte interphase"]
    assert record["term_grounding"][0]["grounding_source"] == "generic_glossary_expansion"


def test_apply_repair_rejects_ungrounded_target_context_anchor():
    query_plan = {
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase", "lithium battery"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
        user_question="SEI",
    )

    record = result.applied_issue_records[0]
    assert record["applied_terms"] == ["solid electrolyte interphase"]
    assert record["rejected_terms"][0]["term"] == "lithium battery"
    assert record["rejected_terms"][0]["grounding_reason"] == "ungrounded_target_context_anchor"


def test_apply_repair_records_rejected_terms_with_reason():
    query_plan = {
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase", "lithium battery"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
        user_question="SEI",
    )

    rejected = result.applied_issue_records[0]["rejected_terms"]
    assert rejected == [
        {
            "term": "lithium battery",
            "grounded": False,
            "grounding_source": "",
            "grounding_reason": "ungrounded_target_context_anchor",
            "normalized_term": "lithium battery",
        }
    ]


def test_apply_repair_does_not_add_lithium_battery_for_bare_sei_question():
    query_plan = {
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase", "lithium battery"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
        user_question="SEI",
    )

    new_query = result.applied_issue_records[0]["new_query"]
    assert "solid electrolyte interphase" in new_query
    assert "lithium battery" not in new_query


def test_apply_repair_allows_lithium_battery_when_user_question_mentions_lithium_battery():
    query_plan = {
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase", "lithium battery"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
        user_question="I need artificial SEI papers for lithium batteries.",
    )

    record = result.applied_issue_records[0]
    assert "lithium battery" in record["applied_terms"]
    assert any(
        item["term"] == "lithium battery" and item["grounding_source"] == "user_question"
        for item in record["term_grounding"]
    )


def test_apply_repair_allows_artificial_sei_when_user_question_mentions_artificial_sei():
    query_plan = {
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase", "artificial SEI"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
        user_question="I need artificial SEI papers for lithium batteries.",
    )

    record = result.applied_issue_records[0]
    assert "artificial SEI" in record["applied_terms"]
    assert any(
        item["term"] == "artificial SEI" and item["grounding_source"] == "user_question"
        for item in record["term_grounding"]
    )


def test_query_repair_after_llm_critic_records_applied_and_rejected_terms(tmp_path):
    output_dir = tmp_path / "term-grounding-artifact"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(
            response_mode="valid_single_acronym_query"
        ),
        question="SEI",
        planned_queries_override=["sei SEI"],
        apply_llm_query_critic_repairs=True,
    )

    repair = json.loads((output_dir / "query_repair_after_llm_critic.json").read_text())
    record = repair["applied_issue_records"][0]
    assert record["applied_terms"] == ["solid electrolyte interphase"]
    assert record["rejected_terms"][0]["term"] == "lithium battery"


def test_applied_repair_provenance_includes_grounding_source():
    query_plan = {
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
        user_question="SEI",
    )

    grounding = result.applied_issue_records[0]["term_grounding"][0]
    assert grounding["grounded"] is True
    assert grounding["grounding_source"] == "generic_glossary_expansion"


def test_apply_verified_duplicate_query_drops_later_duplicate():
    query_plan = {
        "queries": ["SEI lithium battery", "SEI lithium battery"],
        "final_provider_queries": {"fake": ["SEI lithium battery", "SEI lithium battery"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="duplicate_query",
                affected_query_ids=["q1", "q2"],
                affected_aspect="SEI lithium battery",
                suggested_action="drop_query",
                suggested_terms=[],
                verification_reason="duplicate_query_verified_by_deterministic_query_plan_checks",
            )
        ),
    )

    assert result.applied_issue_count == 1
    assert result.query_dropped_count == 1
    assert result.query_plan_after_llm_critic["final_provider_queries"]["fake"] == [
        "SEI lithium battery"
    ]


def test_apply_rejects_unsupported_suggestion_even_if_llm_proposed():
    query_plan = {
        "queries": ["SEI lithium battery"],
        "final_provider_queries": {"fake": ["SEI lithium battery"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="unsupported_suggestion",
                affected_aspect="quantum transport",
                suggested_action="no_change",
                suggested_terms=["quantum transport"],
                verification_reason="unsupported_suggestion",
            )
        ),
    )

    assert result.applied_issue_count == 0
    assert result.rejected_for_application_count == 1
    assert result.rejected_for_application_records[0]["rule_reason"] == "unsupported_suggestion"


def test_apply_rejects_unverified_issue():
    query_plan = {
        "queries": ["SEI lithium battery"],
        "final_provider_queries": {"fake": ["SEI lithium battery"]},
    }
    issue = verified_issue()
    issue.pop("verification_reason")

    result = apply_verified_query_critic_issues_to_query_plan(query_plan, [issue])

    assert result.applied_issue_count == 0
    assert result.rejected_for_application_records[0]["rule_reason"] == "issue_not_verified"


def test_apply_does_not_create_duplicate_queries():
    query_plan = {
        "queries": ["SEI lithium battery", 'SEI lithium battery "artificial SEI"'],
        "final_provider_queries": {
            "fake": ["SEI lithium battery", 'SEI lithium battery "artificial SEI"']
        },
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(verified_issue()),
        user_question="I need artificial SEI papers for lithium batteries.",
    )

    assert result.applied_issue_count == 0
    assert result.rejected_for_application_records[0]["rule_reason"] in {
        "affected_aspect_already_covered",
        "would_create_duplicate_query",
    }


def test_apply_does_not_drop_all_queries():
    query_plan = {
        "queries": ["SEI lithium battery", "SEI lithium battery"],
        "final_provider_queries": {"fake": ["SEI lithium battery", "SEI lithium battery"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="duplicate_query",
                affected_query_ids=["q1", "q2"],
                suggested_action="drop_query",
                suggested_terms=[],
                verification_reason="duplicate_query_verified_by_deterministic_query_plan_checks",
            )
        ),
        min_query_count=2,
    )

    assert result.applied_issue_count == 0
    assert result.rejected_for_application_records[0]["rule_reason"] == "would_drop_all_or_too_many_queries"


def test_applied_query_change_records_provenance():
    query_plan = {
        "queries": ["SEI lithium battery dendrite"],
        "final_provider_queries": {"fake": ["SEI lithium battery dendrite"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(verified_issue()),
        user_question="I need artificial SEI papers for lithium batteries.",
    )

    record = result.applied_issue_records[0]
    assert record["source"] == "llm_query_critic_suggested_rule_applied"
    assert record["applied"] is True
    assert record["llm_issue_id"] == 0
    assert record["verifier_reason"]


def test_query_repair_after_llm_critic_artifact_written(tmp_path):
    output_dir = tmp_path / "repair-artifact"

    run_query_critic_pipeline(
        output_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
        apply_llm_query_critic_repairs=True,
    )

    repair = json.loads((output_dir / "query_repair_after_llm_critic.json").read_text())
    trace = json.loads((output_dir / "llm_query_critic_trace.json").read_text())
    assert repair["apply_enabled"] is True
    assert repair["applied_issue_count"] > 0
    assert trace["applied_issue_count"] == repair["applied_issue_count"]


def test_apply_flag_required_for_query_mutation(tmp_path):
    critique_only_dir = tmp_path / "critique-only"
    apply_dir = tmp_path / "apply"

    run_query_critic_pipeline(
        critique_only_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
    )
    run_query_critic_pipeline(
        apply_dir,
        enable_llm_query_critic=True,
        llm_query_critic_provider=FakeLLMQueryPlanCriticProvider(response_mode="valid"),
        apply_llm_query_critic_repairs=True,
    )

    critique_before = json.loads(
        (critique_only_dir / "query_plan_before_llm_critic.json").read_text()
    )
    critique_after = json.loads(
        (critique_only_dir / "query_plan_after_llm_critic.json").read_text()
    )
    apply_before = json.loads((apply_dir / "query_plan_before_llm_critic.json").read_text())
    apply_after = json.loads((apply_dir / "query_plan_after_llm_critic.json").read_text())

    assert critique_after == critique_before
    assert apply_after != apply_before
    assert apply_after["final_provider_queries"] != apply_before["final_provider_queries"]


def test_llm_query_critic_fake_provider_does_not_populate_openalex_queries():
    nested = QueryPlan(
        core_terms=["SEI"],
        openalex_queries=[],
        semantic_scholar_queries=[],
    )
    query_plan = {
        "query_plan": nested,
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
    )

    after = result.query_plan_after_llm_critic
    assert after["final_provider_queries"]["fake"][0] == 'sei SEI "solid electrolyte interphase"'
    assert after["final_openalex_queries"] == []
    assert after["query_plan"].openalex_queries == []


def test_llm_query_critic_fake_provider_does_not_populate_semantic_scholar_queries():
    nested = QueryPlan(
        core_terms=["SEI"],
        openalex_queries=[],
        semantic_scholar_queries=[],
    )
    query_plan = {
        "query_plan": nested,
        "queries": ["sei SEI"],
        "final_provider_queries": {"fake": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
    )

    after = result.query_plan_after_llm_critic
    assert after["final_semantic_scholar_queries"] == []
    assert after["query_plan"].semantic_scholar_queries == []


def test_llm_query_critic_openalex_provider_updates_openalex_queries_when_present():
    nested = QueryPlan(
        core_terms=["SEI"],
        openalex_queries=["sei SEI"],
        semantic_scholar_queries=[],
    )
    query_plan = {
        "query_plan": nested,
        "queries": ["sei SEI"],
        "final_provider_queries": {"openalex": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
    )

    after = result.query_plan_after_llm_critic
    assert after["final_openalex_queries"] == ['sei SEI "solid electrolyte interphase"']
    assert after["query_plan"].openalex_queries == ['sei SEI "solid electrolyte interphase"']
    assert after["query_plan"].semantic_scholar_queries == []


def test_llm_query_critic_semantic_scholar_provider_updates_semantic_queries_when_present():
    nested = QueryPlan(
        core_terms=["SEI"],
        openalex_queries=[],
        semantic_scholar_queries=["sei SEI"],
    )
    query_plan = {
        "query_plan": nested,
        "queries": ["sei SEI"],
        "final_provider_queries": {"semantic_scholar": ["sei SEI"]},
    }

    result = apply_verified_query_critic_issues_to_query_plan(
        query_plan,
        verified_critique(
            verified_issue(
                issue_type="single_acronym_query",
                affected_query_ids=["q1"],
                affected_query_text="sei SEI",
                affected_aspect="SEI",
                suggested_action="strengthen_query",
                suggested_terms=["solid electrolyte interphase"],
                verification_reason="single_acronym_query_verified_by_deterministic_query_plan_checks",
            )
        ),
    )

    after = result.query_plan_after_llm_critic
    assert after["final_semantic_scholar_queries"] == ['sei SEI "solid electrolyte interphase"']
    assert after["query_plan"].semantic_scholar_queries == ['sei SEI "solid electrolyte interphase"']
    assert after["query_plan"].openalex_queries == []
