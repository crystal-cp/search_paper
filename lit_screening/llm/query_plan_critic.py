"""Optional LLM query-plan critic.

By default, LLMQueryPlanCritic is an auditable critique layer only: it parses,
verifies, and writes query-plan critique artifacts without changing provider
queries. If and only if the explicit ``--apply-llm-query-critic-repairs`` flag is
enabled, a deterministic rule applier may apply verified issues as limited query
repairs with provenance. The LLM itself still cannot directly generate final
provider queries or decide paper-level fields such as include/exclude, must_read,
domain_decision, final_score, reading_priority, or evidence validity. The v9
deterministic baseline remains unaffected by default.
"""

from __future__ import annotations

import json
import copy
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from lit_screening.llm.intent_frame_enhancer import (
    FORBIDDEN_DECISION_FIELDS,
    find_forbidden_decision_fields,
)


ALLOWED_QUERY_PLAN_ISSUE_TYPES = {
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

ALLOWED_QUERY_PLAN_ACTIONS = {
    "add_anchor",
    "drop_query",
    "merge_queries",
    "strengthen_query",
    "add_query_variant",
    "no_change",
}

CONFIDENCE_TO_FLOAT = {
    "low": 0.33,
    "medium": 0.66,
    "high": 0.9,
}

VERIFIABLE_QUERY_PLAN_ISSUE_TYPES = {
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
}

UNSUPPORTED_DOMAIN_TERMS = {
    "cryo-em",
    "cryo em",
    "quantum transport",
    "perovskite oxide",
    "unrelated material family",
}

UNSUPPORTED_REJECTION_REASONS = {
    "unsupported_suggestion",
    "suggested_terms_introduce_unsupported_domain",
    "affected_aspect_not_grounded",
    "cross_domain_injection_not_found",
    "issue_type_not_verifiable",
    "unsupported_issue_type",
}

GENERIC_GLOSSARY_EXPANSIONS = {
    "sei": {"solid electrolyte interphase"},
    "oer": {"oxygen evolution reaction"},
    "mof": {"metal organic framework", "metal-organic framework"},
}

TARGET_CONTEXT_ANCHORS = {
    "lithium battery",
    "lithium-ion battery",
    "lithium ion battery",
    "lithium metal battery",
    "sodium-ion battery",
    "sodium ion battery",
    "zinc-ion battery",
    "zinc ion battery",
    "oer catalyst",
    "mof co2 capture",
}


class LLMQueryPlanCriticProvider(Protocol):
    """Minimal provider interface for optional query-plan critique."""

    provider_name: str
    model_name: str

    def critique_query_plan(self, input_payload: dict[str, Any]) -> str:
        """Return raw text that should contain one JSON critique object."""


class FakeLLMQueryPlanCriticProvider:
    """Deterministic fake provider for offline query-plan critic tests."""

    provider_name = "fake_llm_query_plan_critic_provider"
    model_name = "fake-query-plan-critic-model"

    def __init__(self, response_mode: str = "valid", raw_text: str | None = None) -> None:
        self.response_mode = response_mode
        self.raw_text = raw_text
        self.call_count = 0
        self.last_input_payload: dict[str, Any] | None = None
        self.last_prompt = ""

    def critique_query_plan(self, input_payload: dict[str, Any]) -> str:
        self.call_count += 1
        self.last_input_payload = input_payload
        self.last_prompt = build_query_plan_critic_prompt(input_payload)
        if self.raw_text is not None:
            return self.raw_text
        if self.response_mode == "malformed_json":
            return "{not valid json"
        if self.response_mode == "forbidden_decision_fields":
            return json.dumps(
                {
                    "issues": [
                        {
                            "issue_type": "missing_aspect",
                            "severity": "high",
                            "affected_query_ids": ["q1"],
                            "affected_aspect": "artificial SEI",
                            "evidence": ["The query misses artificial SEI."],
                            "suggested_action": "add_anchor",
                            "suggested_terms": ["artificial SEI"],
                            "rationale": "Unsafe payload also contains a decision.",
                            "include": ["paper A"],
                        }
                    ],
                    "warnings": [],
                    "confidence": "high",
                    "evidence_from_query_plan": ["No planned query contains artificial SEI."],
                }
            )
        if self.response_mode == "unsupported_suggestion":
            return json.dumps(
                {
                    "issues": [
                        {
                            "issue_type": "unsupported_suggestion",
                            "severity": "medium",
                            "affected_query_ids": ["q2"],
                            "affected_aspect": "quantum transport",
                            "evidence": ["The suggested aspect is not grounded in the user question."],
                            "suggested_action": "no_change",
                            "suggested_terms": ["quantum transport"],
                            "rationale": "The critique flags, but does not apply, unsupported expansion.",
                        }
                    ],
                    "warnings": ["unsupported_suggestion_detected"],
                    "confidence": "medium",
                    "evidence_from_query_plan": ["No user aspect mentions quantum transport."],
                }
            )
        if self.response_mode in {
            "case_aware_weak_query",
            "valid_single_acronym_query",
            "valid_oer_single_acronym_query",
            "valid_mof_short_query",
        }:
            return json.dumps(
                build_case_aware_fake_query_critic_payload(
                    input_payload,
                    response_mode=self.response_mode,
                ),
                ensure_ascii=False,
            )
        if self.response_mode == "no_issue":
            return json.dumps(
                {
                    "issues": [
                        {
                            "issue_type": "no_issue",
                            "severity": "info",
                            "affected_query_ids": [],
                            "affected_aspect": "",
                            "evidence": ["Query plan already covers stated anchors."],
                            "suggested_action": "no_change",
                            "suggested_terms": [],
                            "rationale": "No critique issue found.",
                        }
                    ],
                    "warnings": [],
                    "confidence": "high",
                    "evidence_from_query_plan": ["All planned queries include core anchors."],
                }
            )
        return json.dumps(
            {
                "issues": [
                    {
                        "issue_type": "missing_aspect",
                        "severity": "high",
                        "affected_query_ids": ["q1"],
                        "affected_aspect": "artificial SEI",
                        "evidence": (
                            "User asks for artificial SEI but no query contains "
                            "artificial or engineered SEI terms."
                        ),
                        "suggested_action": "add_anchor",
                        "suggested_terms": [
                            "artificial SEI",
                            "engineered SEI",
                            "artificial solid electrolyte interphase",
                        ],
                        "rationale": (
                            "Adds missing user-specified aspect without changing "
                            "paper-level decisions."
                        ),
                    }
                ],
                "warnings": [],
                "confidence": "high",
                "evidence_from_query_plan": ["No planned query contains artificial SEI."],
            }
        )


def build_case_aware_fake_query_critic_payload(
    input_payload: dict[str, Any],
    *,
    response_mode: str = "case_aware_weak_query",
) -> dict[str, Any]:
    """Return a fake critique grounded in the actual planned queries."""

    query_records = extract_query_records(input_payload)
    preferred_profile = {
        "valid_single_acronym_query": "sei",
        "valid_oer_single_acronym_query": "oer",
        "valid_mof_short_query": "mof_co2",
    }.get(response_mode, "")
    issue_record = _first_fake_weak_query_record(
        query_records,
        preferred_profile=preferred_profile,
    )
    if issue_record is None:
        return _fake_no_issue_payload(query_records)
    return _fake_weak_query_payload(issue_record)


def _first_fake_weak_query_record(
    query_records: list[dict[str, str]],
    *,
    preferred_profile: str = "",
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for record in query_records:
        profile = _fake_weak_query_profile(record["query"])
        if profile is None:
            continue
        matches.append({**record, **profile})
    if preferred_profile:
        for match in matches:
            if match["profile_id"] == preferred_profile:
                return match
        return None
    return matches[0] if matches else None


def _fake_weak_query_profile(query: str) -> dict[str, Any] | None:
    normalized = normalize_text(query)
    if is_single_acronym_query(query):
        token = _single_acronym_token(query)
        if token == "sei":
            return {
                "profile_id": "sei",
                "issue_type": "single_acronym_query",
                "severity": "high",
                "affected_aspect": "SEI",
                "suggested_terms": [
                    "solid electrolyte interphase",
                    "lithium battery",
                ],
                "rationale": (
                    "A single-acronym query is too broad for an SEI screening task; "
                    "only grounded terms can be applied by deterministic rules."
                ),
            }
        if token == "oer":
            return {
                "profile_id": "oer",
                "issue_type": "single_acronym_query",
                "severity": "high",
                "affected_aspect": "OER",
                "suggested_terms": [
                    "oxygen evolution reaction",
                    "spin state",
                ],
                "rationale": (
                    "A single-acronym query is too broad for an OER screening task; "
                    "the glossary expansion is grounded by the acronym."
                ),
            }
        if token == "mof":
            return {
                "profile_id": "mof",
                "issue_type": "single_acronym_query",
                "severity": "high",
                "affected_aspect": "MOF",
                "suggested_terms": [
                    "metal-organic framework",
                    "CO2 capture",
                ],
                "rationale": (
                    "A single-acronym query is too broad for a MOF screening task; "
                    "the framework expansion is grounded by the acronym."
                ),
            }
    if normalized in {"mof co2", "co2 mof", "mof capture"}:
        return {
            "profile_id": "mof_co2",
            "issue_type": "provider_query_too_short",
            "severity": "medium",
            "affected_aspect": "MOF CO2",
            "suggested_terms": [
                "metal-organic framework",
                "CO2 capture",
            ],
            "rationale": (
                "The short MOF/CO2 provider query lacks an explicit framework "
                "expansion; deterministic grounding decides which terms apply."
            ),
        }
    return None


def _single_acronym_token(query: str) -> str:
    tokens = acronym_only_meaningful_tokens(query)
    return tokens[0] if tokens else ""


def _fake_weak_query_payload(issue_record: dict[str, Any]) -> dict[str, Any]:
    query_id = str(issue_record["query_id"])
    query_text = str(issue_record["query"])
    issue_type = str(issue_record["issue_type"])
    if issue_type == "single_acronym_query":
        evidence_text = (
            "The affected provider query contains only repeated acronym tokens: "
            f"{query_text}."
        )
    else:
        evidence_text = (
            "The affected provider query is too short to carry the intended "
            f"terminology expansion: {query_text}."
        )
    return {
        "issues": [
            {
                "issue_type": issue_type,
                "severity": str(issue_record["severity"]),
                "affected_query_ids": [query_id],
                "affected_query_text": query_text,
                "affected_aspect": str(issue_record["affected_aspect"]),
                "evidence": [evidence_text],
                "suggested_action": "strengthen_query",
                "suggested_terms": list(issue_record["suggested_terms"]),
                "rationale": str(issue_record["rationale"]),
            }
        ],
        "warnings": [],
        "confidence": "high",
        "evidence_from_query_plan": [
            f"Planned query {query_id} is weak: {query_text}."
        ],
    }


def _fake_no_issue_payload(query_records: list[dict[str, str]]) -> dict[str, Any]:
    examples = [record["query"] for record in query_records[:3]]
    return {
        "issues": [
            {
                "issue_type": "no_issue",
                "severity": "info",
                "affected_query_ids": [],
                "affected_aspect": "",
                "evidence": [
                    "No acronym-only or short weak provider query was found."
                ],
                "suggested_action": "no_change",
                "suggested_terms": [],
                "rationale": "No verified repair opportunity in the current query plan.",
            }
        ],
        "warnings": [],
        "confidence": "high",
        "evidence_from_query_plan": examples
        or ["No planned provider queries were available."],
    }


@dataclass(frozen=True)
class LLMQueryPlanCritiqueRaw:
    """Raw optional LLM query-plan critique audit record."""

    enabled: bool = False
    provider: str = ""
    model: str = ""
    input_question: str = ""
    raw_text: str = ""
    parsed_json: dict[str, Any] | None = None
    malformed_output: bool = False
    error: str = ""


@dataclass(frozen=True)
class LLMQueryPlanIssue:
    """Advisory query-plan issue with no screening or provider-query authority."""

    issue_type: str = "no_issue"
    severity: str = "info"
    affected_query_ids: list[str] = field(default_factory=list)
    affected_query_text: str = ""
    affected_aspect: str = ""
    evidence: list[str] = field(default_factory=list)
    suggested_action: str = "no_change"
    suggested_terms: list[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.issue_type not in ALLOWED_QUERY_PLAN_ISSUE_TYPES:
            raise ValueError(
                "Invalid LLMQueryPlanIssue.issue_type: "
                f"{self.issue_type!r}"
            )
        if self.suggested_action not in ALLOWED_QUERY_PLAN_ACTIONS:
            raise ValueError(
                "Invalid LLMQueryPlanIssue.suggested_action: "
                f"{self.suggested_action!r}"
            )


@dataclass(frozen=True)
class LLMQueryPlanCritique:
    """Structured advisory critique of a query plan."""

    issues: list[LLMQueryPlanIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_from_query_plan: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMQueryPlanVerifiedCritique:
    """Deterministically verified query-plan critique issues."""

    verified_issues: list[dict[str, Any]] = field(default_factory=list)
    rejected_issues: list[dict[str, Any]] = field(default_factory=list)
    verification_warnings: list[str] = field(default_factory=list)
    verified_issue_count: int = 0
    rejected_issue_count: int = 0
    unsupported_issue_count: int = 0
    rejection_reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMQueryPlanCriticTrace:
    """Trace for optional LLM query-plan critique."""

    llm_query_critic_enabled: bool = False
    llm_called: bool = False
    fallback_used: bool = False
    malformed_output: bool = False
    verified_issue_count: int = 0
    unsupported_issue_count: int = 0
    applied_issue_count: int = 0
    rejected_issue_count: int = 0
    rejected_for_application_count: int = 0
    query_added_count: int = 0
    query_dropped_count: int = 0
    query_modified_count: int = 0
    reason_if_no_change: str = ""


@dataclass(frozen=True)
class LLMQueryPlanCriticResult:
    """No-op critique result returned by phase 1 scaffolding."""

    query_plan_before_llm_critic: Any
    query_plan_after_llm_critic: Any
    raw: LLMQueryPlanCritiqueRaw
    verified_critique: LLMQueryPlanCritique | LLMQueryPlanVerifiedCritique | None
    trace: LLMQueryPlanCriticTrace


@dataclass(frozen=True)
class LLMQueryCriticRepairResult:
    """Deterministic application result for verified query-plan critique issues."""

    apply_enabled: bool = False
    query_plan_after_llm_critic: Any = None
    applied_issue_records: list[dict[str, Any]] = field(default_factory=list)
    rejected_for_application_records: list[dict[str, Any]] = field(default_factory=list)
    provenance_updates: list[dict[str, Any]] = field(default_factory=list)
    query_added_count: int = 0
    query_dropped_count: int = 0
    query_modified_count: int = 0
    applied_issue_count: int = 0
    rejected_for_application_count: int = 0


class LLMQueryPlanCritic:
    """Optional LLM query-plan critic.

    Phase 1 intentionally supports only disabled/no-op behavior. The class
    accepts a future-compatible API shape, but it does not call LLMs or mutate
    query plans while disabled.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        provider: str = "",
        model: str = "",
        llm_provider: LLMQueryPlanCriticProvider | None = None,
    ) -> None:
        self.enabled = enabled
        self.provider = provider
        self.model = model
        self.llm_provider = llm_provider

    def critique(
        self,
        query_plan: Any,
        *,
        input_question: str = "",
        llm_client: Any | None = None,
        input_payload: dict[str, Any] | None = None,
        llm_provider: LLMQueryPlanCriticProvider | None = None,
    ) -> LLMQueryPlanCriticResult:
        """Return a no-op critique result without changing the query plan."""

        if not self.enabled:
            raw = LLMQueryPlanCritiqueRaw(
                enabled=False,
                provider=self.provider,
                model=self.model,
                input_question=input_question,
            )
            trace = LLMQueryPlanCriticTrace(
                llm_query_critic_enabled=False,
                llm_called=False,
                fallback_used=False,
                malformed_output=False,
                verified_issue_count=0,
                unsupported_issue_count=0,
                applied_issue_count=0,
                rejected_issue_count=0,
                rejected_for_application_count=0,
                query_added_count=0,
                query_dropped_count=0,
                query_modified_count=0,
                reason_if_no_change="llm_query_critic_disabled",
            )
            return LLMQueryPlanCriticResult(
                query_plan_before_llm_critic=query_plan,
                query_plan_after_llm_critic=query_plan,
                raw=raw,
                verified_critique=None,
                trace=trace,
            )

        provider = llm_provider or self.llm_provider
        if provider is not None:
            payload = dict(input_payload or {})
            payload.setdefault("user_question", input_question)
            payload.setdefault("query_plan", query_plan)
            raw_text = provider.critique_query_plan(payload)
            raw, critique, trace = parse_query_plan_critique(
                raw_text,
                input_question=input_question,
                provider=getattr(provider, "provider_name", self.provider),
                model=getattr(provider, "model_name", self.model),
            )
            verified_critique = None
            if critique is not None and not trace.fallback_used:
                verified_critique = verify_query_plan_critique(payload, critique)
                trace = trace_from_verified_query_plan_critique(verified_critique)
            return LLMQueryPlanCriticResult(
                query_plan_before_llm_critic=query_plan,
                query_plan_after_llm_critic=query_plan,
                raw=raw,
                verified_critique=verified_critique if not trace.fallback_used else None,
                trace=trace,
            )

        raw = LLMQueryPlanCritiqueRaw(
            enabled=True,
            provider=self.provider,
            model=self.model,
            input_question=input_question,
            error="llm_query_critic_provider_unavailable",
        )
        trace = LLMQueryPlanCriticTrace(
            llm_query_critic_enabled=True,
            llm_called=False,
            fallback_used=True,
            malformed_output=False,
            verified_issue_count=0,
            unsupported_issue_count=0,
            applied_issue_count=0,
            rejected_issue_count=0,
            rejected_for_application_count=0,
            query_added_count=0,
            query_dropped_count=0,
            query_modified_count=0,
            reason_if_no_change="llm_query_critic_provider_unavailable",
        )
        return LLMQueryPlanCriticResult(
            query_plan_before_llm_critic=query_plan,
            query_plan_after_llm_critic=query_plan,
            raw=raw,
            verified_critique=None,
            trace=trace,
        )


def build_query_plan_critic_prompt(input_payload: dict[str, Any]) -> str:
    """Build a short JSON-only prompt for query-plan critique."""

    return (
        "Return one JSON object only. Critique the QueryFamily plan and planned "
        "queries only. Do not output final provider queries. Do not output "
        "include, exclude, must_read, out_of_scope, final_score, domain_decision, "
        "evidence_validity, or reading_priority. Do not make paper-level "
        "decisions. Allowed issue_type values: "
        f"{sorted(ALLOWED_QUERY_PLAN_ISSUE_TYPES)}. Allowed suggested_action "
        f"values: {sorted(ALLOWED_QUERY_PLAN_ACTIONS)}. Input payload: "
        f"{json.dumps(_to_plain_data(input_payload), ensure_ascii=False)}"
    )


def parse_query_plan_critique(
    raw_text: str,
    *,
    input_question: str = "",
    provider: str = "",
    model: str = "",
) -> tuple[LLMQueryPlanCritiqueRaw, LLMQueryPlanCritique | None, LLMQueryPlanCriticTrace]:
    """Parse and safety-check a raw LLM query-plan critique response."""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        raw = LLMQueryPlanCritiqueRaw(
            enabled=True,
            provider=provider,
            model=model,
            input_question=input_question,
            raw_text=raw_text,
            parsed_json=None,
            malformed_output=True,
            error="malformed_json",
        )
        trace = _fallback_trace(
            malformed_output=True,
            reason="llm_query_critic_malformed_output",
        )
        return raw, None, trace

    if not isinstance(parsed, dict):
        raw = LLMQueryPlanCritiqueRaw(
            enabled=True,
            provider=provider,
            model=model,
            input_question=input_question,
            raw_text=raw_text,
            parsed_json=None,
            malformed_output=True,
            error="json_not_object",
        )
        trace = _fallback_trace(
            malformed_output=True,
            reason="llm_query_critic_malformed_output",
        )
        return raw, None, trace

    forbidden = sorted(find_forbidden_decision_fields(parsed))
    if forbidden:
        raw = LLMQueryPlanCritiqueRaw(
            enabled=True,
            provider=provider,
            model=model,
            input_question=input_question,
            raw_text=raw_text,
            parsed_json=parsed,
            malformed_output=False,
            error="llm_query_critic_output_contains_forbidden_decision_fields",
        )
        trace = _fallback_trace(
            malformed_output=False,
            reason="llm_query_critic_output_contains_forbidden_decision_fields",
        )
        return raw, None, trace

    try:
        critique = query_plan_critique_from_parsed_json(parsed)
    except (TypeError, ValueError, KeyError):
        raw = LLMQueryPlanCritiqueRaw(
            enabled=True,
            provider=provider,
            model=model,
            input_question=input_question,
            raw_text=raw_text,
            parsed_json=parsed,
            malformed_output=False,
            error="llm_query_critic_invalid_schema",
        )
        trace = _fallback_trace(
            malformed_output=False,
            reason="llm_query_critic_invalid_schema",
        )
        return raw, None, trace

    parsed_issue_count = sum(1 for issue in critique.issues if issue.issue_type != "no_issue")
    reason = (
        "llm_query_critic_no_issues"
        if parsed_issue_count == 0
        else "llm_query_critic_parsed_pending_verification"
    )
    raw = LLMQueryPlanCritiqueRaw(
        enabled=True,
        provider=provider,
        model=model,
        input_question=input_question,
        raw_text=raw_text,
        parsed_json=parsed,
        malformed_output=False,
        error="",
    )
    trace = LLMQueryPlanCriticTrace(
        llm_query_critic_enabled=True,
        llm_called=True,
        fallback_used=False,
        malformed_output=False,
        verified_issue_count=0,
        unsupported_issue_count=0,
        applied_issue_count=0,
        rejected_issue_count=0,
        rejected_for_application_count=0,
        query_added_count=0,
        query_dropped_count=0,
        query_modified_count=0,
        reason_if_no_change=reason,
    )
    return raw, critique, trace


def verify_query_plan_critique(
    input_payload: dict[str, Any],
    critique: LLMQueryPlanCritique,
) -> LLMQueryPlanVerifiedCritique:
    """Deterministically verify critique issues without modifying queries."""

    context = build_query_plan_verifier_context(input_payload)
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warnings: list[str] = []

    for issue in critique.issues:
        if issue.issue_type == "no_issue":
            continue
        reject_reason = rejection_reason_for_query_plan_issue(issue, context)
        issue_payload = _issue_to_dict(issue)
        if reject_reason:
            rejected.append(
                {
                    "issue_type": issue.issue_type,
                    "suggested_action": issue.suggested_action,
                    "affected_query_text": issue.affected_query_text,
                    "affected_aspect": issue.affected_aspect,
                    "reason": reject_reason,
                }
            )
        else:
            verified.append(
                {
                    **issue_payload,
                    "verification_reason": verified_reason_for_query_plan_issue(issue),
                }
            )

    reason_counts = count_rejection_reasons(rejected)
    unsupported_count = sum(
        count
        for reason, count in reason_counts.items()
        if reason in UNSUPPORTED_REJECTION_REASONS
    )
    if unsupported_count:
        warnings.append("some_llm_query_plan_issues_rejected")
    return LLMQueryPlanVerifiedCritique(
        verified_issues=verified,
        rejected_issues=rejected,
        verification_warnings=warnings,
        verified_issue_count=len(verified),
        rejected_issue_count=len(rejected),
        unsupported_issue_count=unsupported_count,
        rejection_reason_counts=reason_counts,
    )


class LLMQueryPlanCritiqueVerifier:
    """Small wrapper class for deterministic query-plan critique verification."""

    def verify(
        self,
        input_payload: dict[str, Any],
        critique: LLMQueryPlanCritique,
    ) -> LLMQueryPlanVerifiedCritique:
        return verify_query_plan_critique(input_payload, critique)


def trace_from_verified_query_plan_critique(
    verified_critique: LLMQueryPlanVerifiedCritique,
) -> LLMQueryPlanCriticTrace:
    if verified_critique.verified_issue_count > 0:
        reason = "llm_query_critic_verified_but_not_applied"
    elif verified_critique.rejected_issue_count > 0:
        reason = "llm_query_critic_no_verified_issues"
    else:
        reason = "llm_query_critic_no_issues"
    return LLMQueryPlanCriticTrace(
        llm_query_critic_enabled=True,
        llm_called=True,
        fallback_used=False,
        malformed_output=False,
        verified_issue_count=verified_critique.verified_issue_count,
        unsupported_issue_count=verified_critique.unsupported_issue_count,
        applied_issue_count=0,
        rejected_issue_count=verified_critique.rejected_issue_count,
        rejected_for_application_count=0,
        query_added_count=0,
        query_dropped_count=0,
        query_modified_count=0,
        reason_if_no_change=reason,
    )


def apply_verified_query_critic_issues_to_query_plan(
    query_plan_before_llm_critic: Any,
    verified_issues: LLMQueryPlanVerifiedCritique | dict[str, Any] | list[dict[str, Any]] | None,
    *,
    user_question: str = "",
    search_contract: Any | None = None,
    structured_intent: Any | None = None,
    query_provenance: dict[str, Any] | None = None,
    min_query_count: int = 1,
) -> LLMQueryCriticRepairResult:
    """Apply a small deterministic subset of verified LLM query critiques.

    The LLM critique never writes provider queries directly. This rule applier
    only accepts verifier-approved issue records and performs tightly bounded
    query repairs with explicit provenance.
    """

    query_plan_after = copy.deepcopy(query_plan_before_llm_critic)
    issues = _verified_issue_records(verified_issues)
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    provenance_updates: list[dict[str, Any]] = []
    counts = {"added": 0, "dropped": 0, "modified": 0}

    for issue_index, issue in enumerate(issues):
        reject_reason = _application_reject_reason(
            issue,
            query_plan_after,
            user_question=user_question,
            search_contract=search_contract,
            structured_intent=structured_intent,
            min_query_count=min_query_count,
        )
        if reject_reason:
            rejected.append(_application_rejection_record(issue, issue_index, reject_reason))
            continue

        issue_type = str(issue.get("issue_type") or "")
        suggested_action = str(issue.get("suggested_action") or "")
        if issue_type == "missing_aspect" and suggested_action == "add_anchor":
            record = _apply_missing_aspect_add_anchor(
                query_plan_after,
                issue,
                issue_index,
                user_question=user_question,
                search_contract=search_contract,
                structured_intent=structured_intent,
            )
        elif issue_type in {"weak_anchor", "provider_query_too_short", "single_acronym_query"} and suggested_action == "strengthen_query":
            record = _apply_strengthen_query(
                query_plan_after,
                issue,
                issue_index,
                user_question=user_question,
                search_contract=search_contract,
                structured_intent=structured_intent,
            )
        elif issue_type == "duplicate_query" and suggested_action == "drop_query":
            record = _apply_drop_duplicate_query(query_plan_after, issue, issue_index, min_query_count)
        else:
            rejected.append(
                _application_rejection_record(
                    issue,
                    issue_index,
                    "verified_issue_type_not_supported_for_application",
                )
            )
            continue

        if not record.get("applied"):
            rejected.append(_application_rejection_record(issue, issue_index, record["rule_reason"]))
            continue
        applied.append(record)
        provenance_updates.append(record)
        counts["added"] += int(record.get("query_added_count", 0))
        counts["dropped"] += int(record.get("query_dropped_count", 0))
        counts["modified"] += int(record.get("query_modified_count", 0))

    if isinstance(query_provenance, dict) and provenance_updates:
        query_provenance.setdefault("llm_query_critic_repairs", []).extend(provenance_updates)

    return LLMQueryCriticRepairResult(
        apply_enabled=True,
        query_plan_after_llm_critic=query_plan_after,
        applied_issue_records=applied,
        rejected_for_application_records=rejected,
        provenance_updates=provenance_updates,
        query_added_count=counts["added"],
        query_dropped_count=counts["dropped"],
        query_modified_count=counts["modified"],
        applied_issue_count=len(applied),
        rejected_for_application_count=len(rejected),
    )


def empty_query_critic_repair_result(
    query_plan_after_llm_critic: Any,
    *,
    apply_enabled: bool = False,
) -> LLMQueryCriticRepairResult:
    return LLMQueryCriticRepairResult(
        apply_enabled=apply_enabled,
        query_plan_after_llm_critic=query_plan_after_llm_critic,
    )


def _verified_issue_records(
    verified_issues: LLMQueryPlanVerifiedCritique | dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if verified_issues is None:
        return []
    if isinstance(verified_issues, LLMQueryPlanVerifiedCritique):
        return [dict(item) for item in verified_issues.verified_issues]
    if isinstance(verified_issues, dict):
        values = verified_issues.get("verified_issues", [])
        return [dict(item) for item in values if isinstance(item, dict)]
    if isinstance(verified_issues, list):
        return [dict(item) for item in verified_issues if isinstance(item, dict)]
    return []


def _application_reject_reason(
    issue: dict[str, Any],
    query_plan: Any,
    *,
    user_question: str,
    search_contract: Any | None,
    structured_intent: Any | None,
    min_query_count: int,
) -> str:
    if find_forbidden_decision_fields(issue):
        return "forbidden_decision_field"
    if str(issue.get("issue_type") or "") == "unsupported_suggestion":
        return "unsupported_suggestion"
    if not issue.get("verification_reason"):
        return "issue_not_verified"
    if suggested_terms_introduce_unsupported_domain(
        _issue_from_dict(issue),
        build_query_plan_verifier_context(
            {
                "query_plan": query_plan,
                "search_contract_summary": search_contract,
                "structured_intent": structured_intent,
            }
        ),
    ):
        return "suggested_terms_introduce_unsupported_domain"
    grounding = _ground_suggested_terms(
        issue,
        query_plan,
        user_question=user_question,
        search_contract=search_contract,
        structured_intent=structured_intent,
    )
    issue_type = str(issue.get("issue_type") or "")
    suggested_action = str(issue.get("suggested_action") or "")
    if issue_type == "missing_aspect" and suggested_action == "add_anchor":
        context = build_query_plan_verifier_context({"query_plan": query_plan})
        if aspect_covered_by_queries(_issue_from_dict(issue), context):
            return "affected_aspect_already_covered"
        if not grounding["applied_terms"]:
            return "no_grounded_suggested_terms"
        new_query = _anchored_query_variant(
            _first_affected_or_first_query(query_plan, issue),
            issue,
            grounding["applied_terms"],
        )
        if not new_query:
            return "no_safe_query_variant_generated"
        if _query_exists(query_plan, new_query):
            return "would_create_duplicate_query"
        return ""
    if issue_type in {"weak_anchor", "provider_query_too_short", "single_acronym_query"} and suggested_action == "strengthen_query":
        original = _first_affected_or_first_query(query_plan, issue)
        if not original:
            return "affected_query_not_found"
        context = build_query_plan_verifier_context({"query_plan": query_plan})
        if issue_type == "single_acronym_query" and not is_single_acronym_query(original):
            return "affected_query_not_single_acronym"
        if issue_type != "single_acronym_query" and not is_overbroad_query(original, context):
            return "affected_query_not_overbroad"
        if not grounding["applied_terms"]:
            return "no_grounded_suggested_terms"
        new_query = _strengthened_query_variant(original, issue, grounding["applied_terms"])
        if not new_query or normalize_query_for_duplicate(new_query) == normalize_query_for_duplicate(original):
            return "no_safe_query_variant_generated"
        if _query_exists(query_plan, new_query):
            return "would_create_duplicate_query"
        return ""
    if issue_type == "duplicate_query" and suggested_action == "drop_query":
        drop_query = _later_duplicate_query(query_plan, issue)
        if not drop_query:
            return "affected_queries_not_duplicate"
        if _total_provider_query_count(query_plan) <= min_query_count:
            return "would_drop_all_or_too_many_queries"
        return ""
    return "verified_issue_type_not_supported_for_application"


def _apply_missing_aspect_add_anchor(
    query_plan: Any,
    issue: dict[str, Any],
    issue_index: int,
    *,
    user_question: str,
    search_contract: Any | None,
    structured_intent: Any | None,
) -> dict[str, Any]:
    original = _first_affected_or_first_query(query_plan, issue)
    grounding = _ground_suggested_terms(
        issue,
        query_plan,
        user_question=user_question,
        search_contract=search_contract,
        structured_intent=structured_intent,
    )
    new_query = _anchored_query_variant(original, issue, grounding["applied_terms"])
    if not new_query or not _append_provider_query(query_plan, new_query):
        return {"applied": False, "rule_reason": "no_safe_query_variant_generated"}
    return _application_record(
        issue,
        issue_index,
        rule_reason="missing_aspect_add_anchor_rule_applied",
        original_query=original,
        added_query=new_query,
        query_added_count=1,
        applied_terms=grounding["applied_terms"],
        rejected_terms=grounding["rejected_terms"],
        term_grounding=grounding["term_grounding"],
    )


def _apply_strengthen_query(
    query_plan: Any,
    issue: dict[str, Any],
    issue_index: int,
    *,
    user_question: str,
    search_contract: Any | None,
    structured_intent: Any | None,
) -> dict[str, Any]:
    original = _first_affected_or_first_query(query_plan, issue)
    grounding = _ground_suggested_terms(
        issue,
        query_plan,
        user_question=user_question,
        search_contract=search_contract,
        structured_intent=structured_intent,
    )
    new_query = _strengthened_query_variant(original, issue, grounding["applied_terms"])
    if not original or not new_query or not _replace_provider_query(query_plan, original, new_query):
        return {"applied": False, "rule_reason": "affected_query_not_found"}
    return _application_record(
        issue,
        issue_index,
        rule_reason="strengthen_query_rule_applied",
        original_query=original,
        new_query=new_query,
        query_modified_count=1,
        applied_terms=grounding["applied_terms"],
        rejected_terms=grounding["rejected_terms"],
        term_grounding=grounding["term_grounding"],
    )


def _apply_drop_duplicate_query(
    query_plan: Any,
    issue: dict[str, Any],
    issue_index: int,
    min_query_count: int,
) -> dict[str, Any]:
    drop_query = _later_duplicate_query(query_plan, issue)
    if not drop_query or _total_provider_query_count(query_plan) <= min_query_count:
        return {"applied": False, "rule_reason": "would_drop_all_or_too_many_queries"}
    if not _remove_provider_query_once(query_plan, drop_query):
        return {"applied": False, "rule_reason": "affected_query_not_found"}
    return _application_record(
        issue,
        issue_index,
        rule_reason="duplicate_query_drop_later_rule_applied",
        original_query=drop_query,
        query_dropped_count=1,
    )


def _application_record(
    issue: dict[str, Any],
    issue_index: int,
    *,
    rule_reason: str,
    original_query: str = "",
    new_query: str = "",
    added_query: str = "",
    query_added_count: int = 0,
    query_dropped_count: int = 0,
    query_modified_count: int = 0,
    applied_terms: list[str] | None = None,
    rejected_terms: list[dict[str, Any]] | None = None,
    term_grounding: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "source": "llm_query_critic_suggested_rule_applied",
        "issue_type": str(issue.get("issue_type") or ""),
        "suggested_action": str(issue.get("suggested_action") or ""),
        "affected_query_text": str(issue.get("affected_query_text") or ""),
        "affected_aspect": str(issue.get("affected_aspect") or ""),
        "original_query": original_query,
        "new_query": new_query,
        "added_query": added_query,
        "applied": True,
        "rule_reason": rule_reason,
        "llm_issue_id": issue_index,
        "verifier_reason": str(issue.get("verification_reason") or ""),
        "applied_terms": applied_terms or [],
        "rejected_terms": rejected_terms or [],
        "term_grounding": term_grounding or [],
        "query_added_count": query_added_count,
        "query_dropped_count": query_dropped_count,
        "query_modified_count": query_modified_count,
    }


def _application_rejection_record(
    issue: dict[str, Any],
    issue_index: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "source": "llm_query_critic_suggested_rule_rejected_for_application",
        "issue_type": str(issue.get("issue_type") or ""),
        "suggested_action": str(issue.get("suggested_action") or ""),
        "affected_query_text": str(issue.get("affected_query_text") or ""),
        "affected_aspect": str(issue.get("affected_aspect") or ""),
        "applied": False,
        "rule_reason": reason,
        "llm_issue_id": issue_index,
        "verifier_reason": str(issue.get("verification_reason") or ""),
    }


def _issue_from_dict(issue: dict[str, Any]) -> LLMQueryPlanIssue:
    return LLMQueryPlanIssue(
        issue_type=str(issue.get("issue_type") or "unsupported_suggestion"),
        severity=str(issue.get("severity") or "info"),
        affected_query_ids=_string_list(issue.get("affected_query_ids")),
        affected_query_text=str(issue.get("affected_query_text") or ""),
        affected_aspect=str(issue.get("affected_aspect") or ""),
        evidence=_string_list(issue.get("evidence")),
        suggested_action=str(issue.get("suggested_action") or "no_change"),
        suggested_terms=_string_list(issue.get("suggested_terms")),
        rationale=str(issue.get("rationale") or ""),
    )


def _first_affected_or_first_query(query_plan: Any, issue: dict[str, Any]) -> str:
    records = extract_query_records({"query_plan": query_plan})
    by_id = {record["query_id"]: record["query"] for record in records}
    affected_ids = _string_list(issue.get("affected_query_ids"))
    for query_id in affected_ids:
        if query_id in by_id:
            return by_id[query_id]
    if affected_ids:
        return ""
    affected_query_text = str(issue.get("affected_query_text") or "")
    if affected_query_text:
        expected = normalize_query_for_duplicate(affected_query_text)
        for record in records:
            if normalize_query_for_duplicate(record["query"]) == expected:
                return record["query"]
        return ""
    return records[0]["query"] if records else ""


def _anchored_query_variant(
    original_query: str,
    issue: dict[str, Any],
    applied_terms: list[str] | None = None,
) -> str:
    anchors = applied_terms if applied_terms is not None else _safe_suggested_terms(issue)
    if not anchors:
        anchors = [str(issue.get("affected_aspect") or "").strip()]
    return _compose_query(original_query, anchors[:1])


def _strengthened_query_variant(
    original_query: str,
    issue: dict[str, Any],
    applied_terms: list[str] | None = None,
) -> str:
    terms = applied_terms if applied_terms is not None else _safe_suggested_terms(issue)
    return _compose_query(original_query, terms[:2])


def _safe_suggested_terms(issue: dict[str, Any]) -> list[str]:
    terms = []
    for term in _string_list(issue.get("suggested_terms")):
        normalized = normalize_text(term)
        if not normalized or normalized in UNSUPPORTED_DOMAIN_TERMS:
            continue
        terms.append(term.strip())
    return terms


def _ground_suggested_terms(
    issue: dict[str, Any],
    query_plan: Any,
    *,
    user_question: str,
    search_contract: Any | None,
    structured_intent: Any | None,
) -> dict[str, Any]:
    context = _term_grounding_context(
        query_plan,
        user_question=user_question,
        search_contract=search_contract,
        structured_intent=structured_intent,
    )
    applied_terms: list[str] = []
    rejected_terms: list[dict[str, Any]] = []
    term_grounding: list[dict[str, Any]] = []
    for term in _safe_suggested_terms(issue):
        grounding = _grounding_for_term(term, context)
        term_grounding.append(grounding)
        if grounding["grounded"]:
            applied_terms.append(term)
        else:
            rejected_terms.append(grounding)
    return {
        "applied_terms": applied_terms,
        "rejected_terms": rejected_terms,
        "term_grounding": term_grounding,
    }


def _term_grounding_context(
    query_plan: Any,
    *,
    user_question: str,
    search_contract: Any | None,
    structured_intent: Any | None,
) -> dict[str, str]:
    contract = _to_plain_data(search_contract)
    intent = _to_plain_data(structured_intent)
    existing_queries = " ".join(
        record["query"] for record in extract_query_records({"query_plan": query_plan})
    )
    return {
        "user_question": str(user_question or ""),
        "search_contract.must_include_concepts": _values_for_named_keys(
            contract,
            {"must_include_concepts", "must_terms", "required_aspects"},
        ),
        "search_contract.constraint_groups": _constraint_group_text(contract),
        "deterministic_intent": " ".join(_flatten_strings(intent)),
        "verified_llm_intent_frame": _values_for_named_keys(
            contract,
            {"llm_verified_suggestions", "verified_llm_intent_frame"},
        ),
        "existing_query_plan_terms": existing_queries,
    }


def _grounding_for_term(term: str, context: dict[str, str]) -> dict[str, Any]:
    normalized = normalize_text(term)
    target_anchor = _is_target_context_anchor(term)
    ordered_sources = [
        "user_question",
        "search_contract.must_include_concepts",
        "search_contract.constraint_groups",
        "deterministic_intent",
        "verified_llm_intent_frame",
        "existing_query_plan_terms",
    ]
    for source in ordered_sources:
        if _term_in_text(term, context.get(source, "")):
            return {
                "term": term,
                "grounded": True,
                "grounding_source": source,
                "grounding_reason": "term_present_in_grounding_context",
            }
    glossary_source = _generic_glossary_grounding_source(term, context)
    if glossary_source and not target_anchor:
        return {
            "term": term,
            "grounded": True,
            "grounding_source": "generic_glossary_expansion",
            "grounding_reason": glossary_source,
        }
    return {
        "term": term,
        "grounded": False,
        "grounding_source": "",
        "grounding_reason": (
            "ungrounded_target_context_anchor"
            if target_anchor
            else "ungrounded_suggested_term"
        ),
        "normalized_term": normalized,
    }


def _generic_glossary_grounding_source(term: str, context: dict[str, str]) -> str:
    normalized = normalize_text(term)
    for acronym, expansions in GENERIC_GLOSSARY_EXPANSIONS.items():
        if normalized not in expansions:
            continue
        combined_context = " ".join(context.values())
        if _term_in_text(acronym, combined_context):
            return f"{acronym}_expansion_grounded_in_context"
    return ""


def _is_target_context_anchor(term: str) -> bool:
    return any(_term_in_text(anchor, term) for anchor in TARGET_CONTEXT_ANCHORS)


def _term_in_text(term: str, text: str) -> bool:
    term_tokens = _match_tokens(term)
    text_tokens = _match_tokens(text)
    if not term_tokens or not text_tokens or len(term_tokens) > len(text_tokens):
        return False
    for index in range(0, len(text_tokens) - len(term_tokens) + 1):
        if text_tokens[index : index + len(term_tokens)] == term_tokens:
            return True
    return False


def _match_tokens(value: str) -> list[str]:
    return [_singularize_match_token(token) for token in re.findall(r"[a-z0-9]+", normalize_text(value))]


def _singularize_match_token(token: str) -> str:
    if token == "batteries":
        return "battery"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _values_for_named_keys(payload: Any, keys: set[str]) -> str:
    if isinstance(payload, dict):
        values: list[str] = []
        for key, value in payload.items():
            if str(key) in keys:
                values.extend(_flatten_strings(value))
            values.append(_values_for_named_keys(value, keys))
        return " ".join(value for value in values if value)
    if isinstance(payload, list):
        return " ".join(_values_for_named_keys(item, keys) for item in payload)
    return ""


def _constraint_group_text(payload: Any) -> str:
    if isinstance(payload, dict):
        values: list[str] = []
        for key, value in payload.items():
            if str(key) == "constraint_groups":
                values.extend(_flatten_strings(value))
            else:
                values.append(_constraint_group_text(value))
        return " ".join(value for value in values if value)
    if isinstance(payload, list):
        return " ".join(_constraint_group_text(item) for item in payload)
    return ""


def _compose_query(original_query: str, anchors: list[str]) -> str:
    pieces = [str(original_query or "").strip()]
    normalized = normalize_text(original_query)
    for anchor in anchors:
        clean = str(anchor or "").strip()
        if not clean or normalize_text(clean) in normalized:
            continue
        if " " in clean and not (clean.startswith('"') and clean.endswith('"')):
            clean = f'"{clean}"'
        pieces.append(clean)
    query = " ".join(piece for piece in pieces if piece).strip()
    return query if len(query_tokens(query)) > 1 else ""


def _query_exists(query_plan: Any, query: str) -> bool:
    normalized = normalize_query_for_duplicate(query)
    return any(
        normalize_query_for_duplicate(record["query"]) == normalized
        for record in extract_query_records({"query_plan": query_plan})
    )


def _provider_query_map(query_plan: Any) -> dict[str, list[str]]:
    if isinstance(query_plan, dict):
        existing = query_plan.get("final_provider_queries") or query_plan.get("queries_by_provider")
        if isinstance(existing, dict) and existing:
            return {
                str(provider): [str(query) for query in queries]
                for provider, queries in existing.items()
                if isinstance(queries, list)
            }
        queries = [record["query"] for record in extract_query_records({"query_plan": query_plan})]
        return {"generic": queries}
    return {"generic": [record["query"] for record in extract_query_records({"query_plan": query_plan})]}


def _write_provider_query_map(query_plan: Any, provider_map: dict[str, list[str]]) -> None:
    flat_queries = _unique_preserve_order(
        query for queries in provider_map.values() for query in queries
    )
    if isinstance(query_plan, dict):
        query_plan["queries"] = flat_queries
        query_plan["final_provider_queries"] = {
            provider: list(queries) for provider, queries in provider_map.items()
        }
        if "queries_by_provider" in query_plan:
            query_plan["queries_by_provider"] = {
                provider: list(queries) for provider, queries in provider_map.items()
            }
        query_plan["final_openalex_queries"] = list(provider_map.get("openalex", []))
        query_plan["final_semantic_scholar_queries"] = list(
            provider_map.get("semantic_scholar", [])
        )
        nested_plan = query_plan.get("query_plan")
        if nested_plan is not None:
            _set_nested_query_plan_queries(nested_plan, provider_map, flat_queries)


def _set_nested_query_plan_queries(
    nested_plan: Any,
    provider_map: dict[str, list[str]],
    flat_queries: list[str],
) -> None:
    if "openalex" in provider_map and hasattr(nested_plan, "openalex_queries"):
        setattr(nested_plan, "openalex_queries", list(provider_map.get("openalex", [])))
    if "semantic_scholar" in provider_map and hasattr(nested_plan, "semantic_scholar_queries"):
        setattr(
            nested_plan,
            "semantic_scholar_queries",
            list(provider_map.get("semantic_scholar", [])),
        )


def _append_provider_query(query_plan: Any, new_query: str) -> bool:
    provider_map = _provider_query_map(query_plan)
    if not provider_map:
        provider_map = {"generic": []}
    changed = False
    for provider, queries in provider_map.items():
        if not any(
            normalize_query_for_duplicate(query) == normalize_query_for_duplicate(new_query)
            for query in queries
        ):
            provider_map[provider] = [*queries, new_query]
            changed = True
    if changed:
        _write_provider_query_map(query_plan, provider_map)
    return changed


def _replace_provider_query(query_plan: Any, original_query: str, new_query: str) -> bool:
    provider_map = _provider_query_map(query_plan)
    changed = False
    for provider, queries in provider_map.items():
        updated: list[str] = []
        replaced = False
        for query in queries:
            if not replaced and normalize_query_for_duplicate(query) == normalize_query_for_duplicate(original_query):
                updated.append(new_query)
                replaced = True
                changed = True
            else:
                updated.append(query)
        provider_map[provider] = _unique_preserve_order(updated)
    if changed:
        _write_provider_query_map(query_plan, provider_map)
    return changed


def _remove_provider_query_once(query_plan: Any, query_to_remove: str) -> bool:
    provider_map = _provider_query_map(query_plan)
    changed = False
    for provider, queries in provider_map.items():
        updated: list[str] = []
        removed = False
        for query in queries:
            if not removed and normalize_query_for_duplicate(query) == normalize_query_for_duplicate(query_to_remove):
                removed = True
                changed = True
                continue
            updated.append(query)
        provider_map[provider] = updated
    if changed:
        _write_provider_query_map(query_plan, provider_map)
    return changed


def _later_duplicate_query(query_plan: Any, issue: dict[str, Any]) -> str:
    records = extract_query_records({"query_plan": query_plan})
    affected_ids = set(_string_list(issue.get("affected_query_ids")))
    affected = [record for record in records if not affected_ids or record["query_id"] in affected_ids]
    seen: set[str] = set()
    for record in affected:
        normalized = normalize_query_for_duplicate(record["query"])
        if normalized in seen:
            return record["query"]
        seen.add(normalized)
    return ""


def _total_provider_query_count(query_plan: Any) -> int:
    return len(extract_query_records({"query_plan": query_plan}))


def _unique_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = normalize_query_for_duplicate(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def query_plan_critique_from_parsed_json(parsed: dict[str, Any]) -> LLMQueryPlanCritique:
    issues_payload = parsed.get("issues")
    if not isinstance(issues_payload, list):
        raise ValueError("issues must be a list")
    issues: list[LLMQueryPlanIssue] = []
    for item in issues_payload:
        if not isinstance(item, dict):
            raise ValueError("issue must be an object")
        issues.append(
            LLMQueryPlanIssue(
                issue_type=str(item.get("issue_type") or ""),
                severity=str(item.get("severity") or "info"),
                affected_query_ids=_string_list(item.get("affected_query_ids")),
                affected_query_text=str(item.get("affected_query_text") or ""),
                affected_aspect=str(item.get("affected_aspect") or ""),
                evidence=_string_list(item.get("evidence")),
                suggested_action=str(item.get("suggested_action") or ""),
                suggested_terms=_string_list(item.get("suggested_terms")),
                rationale=str(item.get("rationale") or ""),
            )
        )
    return LLMQueryPlanCritique(
        issues=issues,
        warnings=_string_list(parsed.get("warnings")),
        confidence=_confidence_to_float(parsed.get("confidence")),
        evidence_from_query_plan=_string_list(parsed.get("evidence_from_query_plan")),
    )


def write_query_plan_critic_artifacts(
    output_dir: str | Path,
    *,
    query_plan_before_llm_critic: Any,
    raw: LLMQueryPlanCritiqueRaw,
    verified_critique: LLMQueryPlanCritique | LLMQueryPlanVerifiedCritique | None,
    trace: LLMQueryPlanCriticTrace,
    query_plan_after_llm_critic: Any | None = None,
    query_repair_after_llm_critic: LLMQueryCriticRepairResult | dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write reserved LLM query-plan critic artifacts.

    This helper is called only for explicit opt-in LLMQueryPlanCritic runs. It
    writes diagnostic artifacts for review and does not modify final provider
    queries. The v9 deterministic baseline remains unaffected by default.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    after_plan = (
        query_plan_before_llm_critic
        if query_plan_after_llm_critic is None
        else query_plan_after_llm_critic
    )
    artifacts = {
        "query_plan_before_llm_critic": output / "query_plan_before_llm_critic.json",
        "llm_query_critic_raw": output / "llm_query_critic_raw.json",
        "llm_query_critic_verified": output / "llm_query_critic_verified.json",
        "query_plan_after_llm_critic": output / "query_plan_after_llm_critic.json",
        "query_repair_after_llm_critic": output / "query_repair_after_llm_critic.json",
        "llm_query_critic_trace": output / "llm_query_critic_trace.json",
    }
    repair_record = (
        query_repair_after_llm_critic
        if query_repair_after_llm_critic is not None
        else empty_query_critic_repair_result(after_plan, apply_enabled=False)
    )
    _write_json(artifacts["query_plan_before_llm_critic"], query_plan_before_llm_critic)
    _write_json(artifacts["llm_query_critic_raw"], raw)
    _write_json(artifacts["llm_query_critic_verified"], verified_critique)
    _write_json(artifacts["query_plan_after_llm_critic"], after_plan)
    _write_json(artifacts["query_repair_after_llm_critic"], repair_record)
    _write_json(artifacts["llm_query_critic_trace"], trace)
    return artifacts


def llm_query_plan_issue_fields() -> set[str]:
    """Return issue schema fields for safety tests and documentation."""

    return set(LLMQueryPlanIssue.__dataclass_fields__)


def assert_llm_query_plan_issue_schema_is_non_decisive() -> None:
    """Raise if the issue schema accidentally gains decision fields."""

    overlap = llm_query_plan_issue_fields() & FORBIDDEN_DECISION_FIELDS
    if overlap:
        raise ValueError(
            "LLMQueryPlanIssue contains decision fields: "
            + ", ".join(sorted(overlap))
        )


def build_query_plan_verifier_context(input_payload: dict[str, Any]) -> dict[str, Any]:
    query_records = extract_query_records(input_payload)
    intent_text = " ".join(
        [
            str(input_payload.get("user_question") or ""),
            " ".join(_flatten_strings(input_payload.get("structured_intent"))),
            " ".join(_flatten_strings(input_payload.get("deterministic_intent"))),
            " ".join(_flatten_strings(input_payload.get("search_contract_summary"))),
        ]
    )
    target_terms = extract_target_context_terms(input_payload)
    required_aspects = extract_required_aspects(input_payload)
    all_query_text = " ".join(record["query"] for record in query_records)
    return {
        "intent_text": intent_text,
        "normalized_intent_text": normalize_text(intent_text),
        "query_records": query_records,
        "query_by_id": {record["query_id"]: record for record in query_records},
        "all_query_text": all_query_text,
        "normalized_all_query_text": normalize_text(all_query_text),
        "target_terms": target_terms,
        "required_aspects": required_aspects,
        "has_target_context_group": bool(target_terms),
    }


def rejection_reason_for_query_plan_issue(
    issue: LLMQueryPlanIssue,
    context: dict[str, Any],
) -> str:
    if find_forbidden_decision_fields(_issue_to_dict(issue)):
        return "forbidden_decision_field"
    if issue.issue_type == "unsupported_suggestion":
        return "unsupported_suggestion"
    if issue.issue_type not in VERIFIABLE_QUERY_PLAN_ISSUE_TYPES:
        return "issue_type_not_verifiable"
    if not issue.evidence or not issue.rationale.strip():
        return "missing_evidence_or_rationale"
    if not suggested_action_matches_issue_type(issue):
        return "suggested_action_mismatch"
    if suggested_terms_introduce_unsupported_domain(issue, context):
        return "suggested_terms_introduce_unsupported_domain"

    if issue.issue_type in {
        "missing_aspect",
        "missing_method_anchor",
        "missing_mechanism_anchor",
    }:
        if not issue_aspect_grounded(issue, context):
            return "affected_aspect_not_grounded"
        if aspect_covered_by_queries(issue, context):
            return "affected_aspect_already_covered"
        return ""
    if issue.issue_type == "single_acronym_query":
        records = affected_query_records(issue, context)
        if not records:
            return "affected_query_not_found"
        return "" if any(is_single_acronym_query(record["query"]) for record in records) else "affected_query_not_single_acronym"
    if issue.issue_type in {"overbroad_query", "provider_query_too_short"}:
        records = affected_query_records(issue, context)
        if not records:
            return "affected_query_not_found"
        return "" if any(is_overbroad_query(record["query"], context) for record in records) else "affected_query_not_overbroad"
    if issue.issue_type == "weak_anchor":
        records = affected_query_records(issue, context)
        if not records:
            return "affected_query_not_found"
        return "" if any(query_lacks_required_anchor(record["query"], context) for record in records) else "affected_query_has_required_anchors"
    if issue.issue_type == "missing_target_context":
        if not context["has_target_context_group"]:
            return "missing_target_context_requires_target_group"
        records = affected_query_records(issue, context)
        if not records:
            return "affected_query_not_found"
        return "" if any(query_lacks_target_context(record["query"], context) for record in records) else "affected_query_has_target_context"
    if issue.issue_type == "duplicate_query":
        records = affected_query_records(issue, context)
        if not records:
            return "affected_query_not_found"
        return "" if has_duplicate_queries(records) else "affected_queries_not_duplicate"
    if issue.issue_type == "cross_domain_injection":
        records = affected_query_records(issue, context)
        if not records:
            return "affected_query_not_found"
        return "" if any(query_has_cross_domain_injection(record["query"], context) for record in records) else "cross_domain_injection_not_found"
    return "unsupported_issue_type"


def verified_reason_for_query_plan_issue(issue: LLMQueryPlanIssue) -> str:
    return f"{issue.issue_type}_verified_by_deterministic_query_plan_checks"


def count_rejection_reasons(rejected: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rejected:
        reason = str(item.get("reason") or "")
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def suggested_action_matches_issue_type(issue: LLMQueryPlanIssue) -> bool:
    allowed = {
        "missing_aspect": {"add_anchor", "strengthen_query", "add_query_variant"},
        "missing_method_anchor": {"add_anchor", "strengthen_query", "add_query_variant"},
        "missing_mechanism_anchor": {"add_anchor", "strengthen_query", "add_query_variant"},
        "overbroad_query": {"strengthen_query", "drop_query", "add_anchor"},
        "single_acronym_query": {"strengthen_query", "drop_query", "add_anchor"},
        "weak_anchor": {"strengthen_query", "add_anchor"},
        "provider_query_too_short": {"strengthen_query", "add_anchor", "drop_query"},
        "missing_target_context": {"add_anchor", "strengthen_query"},
        "duplicate_query": {"merge_queries", "drop_query"},
        "cross_domain_injection": {"drop_query", "strengthen_query", "no_change"},
        "unsupported_suggestion": {"no_change"},
        "no_issue": {"no_change"},
    }
    return issue.suggested_action in allowed.get(issue.issue_type, set())


def issue_aspect_grounded(issue: LLMQueryPlanIssue, context: dict[str, Any]) -> bool:
    candidates = [issue.affected_aspect, *issue.suggested_terms]
    intent_text = context["normalized_intent_text"]
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if normalized and (normalized in intent_text or normalized in context["required_aspects"]):
            return True
        if any(term and term in intent_text for term in aspect_aliases(candidate)):
            return True
    return False


def aspect_covered_by_queries(issue: LLMQueryPlanIssue, context: dict[str, Any]) -> bool:
    candidates = [issue.affected_aspect, *issue.suggested_terms]
    query_text = context["normalized_all_query_text"]
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if normalized and normalized in query_text:
            return True
        aliases = aspect_aliases(candidate)
        if aliases and any(alias in query_text for alias in aliases):
            return True
    return False


def suggested_terms_introduce_unsupported_domain(
    issue: LLMQueryPlanIssue,
    context: dict[str, Any],
) -> bool:
    if issue.issue_type == "cross_domain_injection":
        return False
    intent_text = context["normalized_intent_text"]
    for term in issue.suggested_terms:
        normalized = normalize_text(term)
        if normalized in UNSUPPORTED_DOMAIN_TERMS and normalized not in intent_text:
            return True
    return False


def affected_query_records(
    issue: LLMQueryPlanIssue,
    context: dict[str, Any],
) -> list[dict[str, str]]:
    query_by_id = context["query_by_id"]
    if issue.affected_query_ids:
        return [
            query_by_id[query_id]
            for query_id in issue.affected_query_ids
            if query_id in query_by_id
        ]
    if issue.affected_query_text:
        expected = normalize_query_for_duplicate(issue.affected_query_text)
        return [
            record
            for record in context["query_records"]
            if normalize_query_for_duplicate(record["query"]) == expected
        ]
    return context["query_records"]


def is_single_acronym_query(query: str) -> bool:
    tokens = acronym_only_meaningful_tokens(query)
    if not tokens:
        return False
    unique_tokens = set(tokens)
    if len(unique_tokens) != 1:
        return False
    token = next(iter(unique_tokens))
    known_acronyms = {
        "sei",
        "oer",
        "mof",
        "lib",
        "lmb",
        "orr",
        "aem",
        "lom",
        "co2",
        "ai",
    }
    if token in known_acronyms:
        return True
    raw_tokens = acronym_only_raw_tokens(query)
    return any(raw.upper() == raw and 2 <= len(raw) <= 6 for raw in raw_tokens)


def acronym_only_meaningful_tokens(query: str) -> list[str]:
    return [
        token.lower()
        for token in acronym_only_raw_tokens(query)
        if token.lower() not in {"and", "or", "not"}
    ]


def acronym_only_raw_tokens(query: str) -> list[str]:
    cleaned = (
        str(query or "")
        .replace('"', " ")
        .replace("'", " ")
        .replace("+", " ")
    )
    return re.findall(r"[A-Za-z0-9]+", cleaned)


def is_overbroad_query(query: str, context: dict[str, Any]) -> bool:
    tokens = query_tokens(query)
    normalized = normalize_text(query)
    if len(tokens) <= 2:
        return True
    if is_single_acronym_query(query):
        return True
    return normalized in {"sei", "oer", "mof", "mof co2", "co2 mof", "mof capture"}


def query_lacks_required_anchor(query: str, context: dict[str, Any]) -> bool:
    normalized = normalize_text(query)
    anchors = [anchor for anchor in context["required_aspects"] if anchor]
    if not anchors:
        anchors = [term for term in normalize_text(context["intent_text"]).split() if len(term) > 3][:3]
    return bool(anchors) and not any(anchor in normalized for anchor in anchors)


def query_lacks_target_context(query: str, context: dict[str, Any]) -> bool:
    normalized = normalize_text(query)
    return bool(context["target_terms"]) and not any(
        term and term in normalized for term in context["target_terms"]
    )


def has_duplicate_queries(records: list[dict[str, str]]) -> bool:
    seen: set[str] = set()
    for record in records:
        normalized = normalize_query_for_duplicate(record["query"])
        if normalized in seen:
            return True
        seen.add(normalized)
    return False


def query_has_cross_domain_injection(query: str, context: dict[str, Any]) -> bool:
    normalized = normalize_text(query)
    intent_text = context["normalized_intent_text"]
    return any(term in normalized and term not in intent_text for term in UNSUPPORTED_DOMAIN_TERMS)


def extract_query_records(input_payload: dict[str, Any]) -> list[dict[str, str]]:
    query_plan = input_payload.get("query_plan") or input_payload.get("planned_queries") or {}
    if isinstance(query_plan, dict):
        candidates = (
            query_plan.get("planned_queries")
            or query_plan.get("queries")
            or query_plan.get("openalex_queries")
            or query_plan.get("final_openalex_queries")
            or query_plan.get("semantic_scholar_queries")
            or query_plan.get("final_semantic_scholar_queries")
            or []
        )
        if not candidates and isinstance(query_plan.get("final_provider_queries"), dict):
            candidates = [
                query
                for queries in query_plan.get("final_provider_queries", {}).values()
                if isinstance(queries, list)
                for query in queries
            ]
    else:
        candidates = query_plan
    records: list[dict[str, str]] = []
    if isinstance(candidates, list):
        for index, item in enumerate(candidates, start=1):
            if isinstance(item, dict):
                query = str(item.get("query") or item.get("text") or item.get("provider_query") or "")
                query_id = str(item.get("query_id") or item.get("id") or f"q{index}")
            else:
                query = str(item or "")
                query_id = f"q{index}"
            if query.strip():
                records.append({"query_id": query_id, "query": query})
    return records


def extract_target_context_terms(input_payload: dict[str, Any]) -> list[str]:
    values = []
    for key in ["structured_intent", "deterministic_intent", "search_contract_summary"]:
        payload = input_payload.get(key)
        if isinstance(payload, dict):
            values.extend(_flatten_target_terms(payload))
    return sorted({normalize_text(value) for value in values if normalize_text(value)})


def _flatten_target_terms(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in payload.items():
        normalized_key = normalize_text(key)
        if "target_context" in normalized_key or "target_chemistry" in normalized_key:
            values.extend(_flatten_strings(value))
        if key == "constraint_groups" and isinstance(value, list):
            for group in value:
                if isinstance(group, dict) and (
                    "target_context" in normalize_text(group.get("group_name", ""))
                    or "target_chemistry" in normalize_text(group.get("group_name", ""))
                ):
                    values.extend(_flatten_strings(group.get("terms")))
    return values


def extract_required_aspects(input_payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ["structured_intent", "deterministic_intent", "search_contract_summary"]:
        payload = input_payload.get(key)
        if isinstance(payload, dict):
            for aspect_key in ["required_aspects", "expected_aspects", "method_need", "mechanism_need"]:
                values.extend(_flatten_strings(payload.get(aspect_key)))
    return sorted({normalize_text(value) for value in values if normalize_text(value)})


def aspect_aliases(value: str) -> list[str]:
    normalized = normalize_text(value)
    if "artificial sei" in normalized or "engineered sei" in normalized:
        return [
            "artificial sei",
            "engineered sei",
            "artificial solid electrolyte interphase",
        ]
    if "lithium" in normalized:
        return ["lithium", "lithium battery", "lithium metal"]
    return [normalized] if normalized else []


def query_tokens(query: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(query or ""))


def normalize_query_for_duplicate(query: str) -> str:
    return " ".join(query_tokens(query.lower()))


def normalize_text(value: Any) -> str:
    return (
        str(value or "")
        .lower()
        .replace("：", ":")
        .replace("–", "-")
        .replace("—", "-")
        .replace('"', " ")
        .replace("'", " ")
        .strip()
    )


def _fallback_trace(*, malformed_output: bool, reason: str) -> LLMQueryPlanCriticTrace:
    return LLMQueryPlanCriticTrace(
        llm_query_critic_enabled=True,
        llm_called=True,
        fallback_used=True,
        malformed_output=malformed_output,
        verified_issue_count=0,
        unsupported_issue_count=0,
        applied_issue_count=0,
        rejected_issue_count=0,
        rejected_for_application_count=0,
        query_added_count=0,
        query_dropped_count=0,
        query_modified_count=0,
        reason_if_no_change=reason,
    )


def _confidence_to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        value = value.get("overall")
    return CONFIDENCE_TO_FLOAT.get(str(value or "").lower(), 0.0)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [f"{key}: {child}" for key, child in value.items() if str(key).strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        strings: list[str] = []
        for child in value.values():
            strings.extend(_flatten_strings(child))
        return strings
    if isinstance(value, (list, tuple, set)):
        strings = []
        for child in value:
            strings.extend(_flatten_strings(child))
        return strings
    return [str(value)] if str(value).strip() else []


def _issue_to_dict(issue: LLMQueryPlanIssue) -> dict[str, Any]:
    return _to_plain_data(issue)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_to_plain_data(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain_data(child) for key, child in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_plain_data(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value
