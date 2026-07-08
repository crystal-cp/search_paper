"""Optional LLM intent-frame enhancer.

This module defines the data structures, provider interface, fake provider,
parser, verifier, and provenance helpers for an opt-in LLM-assisted intent-frame
enhancement layer. It is only connected to the main pipeline behind an explicit
flag, remains disabled by default, and does not affect the v9 deterministic
baseline unless that flag is used. The LLM can propose intent-frame suggestions,
but it must not directly make paper-level decisions such as include/exclude,
must_read, domain_decision, final_score, reading_priority, or evidence validity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from lit_screening.llm_client import GenericLLMClient
from lit_screening.models import SearchConstraintGroup, SearchContract


FORBIDDEN_DECISION_FIELDS = {
    "include",
    "exclude",
    "must_read",
    "out_of_scope",
    "final_score",
    "domain_decision",
    "evidence_validity",
    "reading_priority",
}

CONFIDENCE_TO_FLOAT = {
    "low": 0.33,
    "medium": 0.66,
    "high": 0.9,
}

CONTRACT_APPLICABLE_SUGGESTION_FIELDS = {
    "target_context_candidates",
    "negative_context_candidates",
    "method_need",
    "mechanism_need",
    "application_need",
    "abbreviation_or_alias_candidates",
    "ambiguities",
}


class LLMIntentProvider(Protocol):
    """Minimal provider interface for optional intent-frame generation."""

    provider_name: str
    model_name: str

    def generate_intent_frame(
        self,
        question: str,
        deterministic_intent: Any | None = None,
    ) -> str:
        """Return raw text that should contain one JSON object."""


@dataclass(frozen=True)
class LLMIntentFrameRaw:
    """Raw optional LLM intent-frame response audit record."""

    enabled: bool = False
    provider: str = ""
    model: str = ""
    input_question: str = ""
    raw_text: str = ""
    parsed_json: dict[str, Any] | None = None
    malformed_output: bool = False
    error: str = ""


@dataclass(frozen=True)
class LLMIntentFrameSuggestion:
    """Advisory intent-frame suggestion with no screening-decision authority."""

    intent_summary: str = ""
    topic: list[str] = field(default_factory=list)
    target_context_candidates: list[str] = field(default_factory=list)
    negative_context_candidates: list[str] = field(default_factory=list)
    method_need: list[str] = field(default_factory=list)
    mechanism_need: list[str] = field(default_factory=list)
    application_need: list[str] = field(default_factory=list)
    material_or_domain_terms: list[str] = field(default_factory=list)
    abbreviation_or_alias_candidates: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_from_user_question: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMIntentEnhancementTrace:
    """Trace for optional LLM intent enhancement."""

    llm_enabled: bool = False
    llm_called: bool = False
    fallback_used: bool = False
    malformed_output: bool = False
    verified_candidate_count: int = 0
    applied_suggestion_count: int = 0
    accepted_suggestion_count: int = 0
    rejected_suggestion_count: int = 0
    unsupported_suggestion_count: int = 0
    reason_if_no_change: str = ""


@dataclass(frozen=True)
class LLMIntentEnhancementResult:
    """Enhancement result returned by the scaffolding enhancer."""

    intent_before_llm: Any
    intent_after_llm: Any
    raw: LLMIntentFrameRaw
    verified_suggestion: LLMIntentFrameSuggestion | None
    trace: LLMIntentEnhancementTrace
    verification_result: dict[str, Any] | None = None


class FakeLLMIntentProvider:
    """Deterministic fake provider for offline tests.

    ``response_mode`` supports ``valid``, ``malformed``, ``overbroad``,
    ``unsupported_domain_expansion``, and ``decision_fields``. No network or API
    key is used.
    """

    provider_name = "fake_llm_intent_provider"
    model_name = "fake-intent-frame-model"

    def __init__(self, response_mode: str = "valid", raw_text: str | None = None) -> None:
        self.response_mode = response_mode
        self.raw_text = raw_text
        self.call_count = 0
        self.last_prompt = ""
        self.last_question = ""
        self.last_deterministic_intent: Any | None = None

    def generate_intent_frame(
        self,
        question: str,
        deterministic_intent: Any | None = None,
    ) -> str:
        self.call_count += 1
        self.last_question = question
        self.last_deterministic_intent = deterministic_intent
        self.last_prompt = build_intent_frame_prompt(question, deterministic_intent)
        if self.raw_text is not None:
            return self.raw_text
        if self.response_mode == "malformed":
            return "{not valid json"
        if self.response_mode == "overbroad":
            return json.dumps(
                {
                    "intent_summary": "Find useful science papers.",
                    "normalized_research_intent": {
                        "topic": ["science"],
                        "target_context_candidates": [],
                        "negative_context_candidates": [],
                        "method_need": [],
                        "mechanism_need": [],
                        "application_need": [],
                        "material_or_domain_terms": [],
                        "abbreviation_or_alias_candidates": [],
                        "ambiguities": ["question is too broad"],
                    },
                    "confidence": {"overall": "low", "reasons": ["overbroad"]},
                    "evidence_from_user_question": [],
                    "warnings": ["overbroad_without_user_evidence"],
                }
            )
        if self.response_mode == "unsupported_domain_expansion":
            return json.dumps(
                {
                    "intent_summary": "Add cryo-EM and quantum transport context.",
                    "normalized_research_intent": {
                        "topic": ["solid electrolyte interphase"],
                        "target_context_candidates": ["lithium battery"],
                        "negative_context_candidates": [],
                        "method_need": ["cryo-EM"],
                        "mechanism_need": [],
                        "application_need": [],
                        "material_or_domain_terms": ["quantum transport"],
                        "abbreviation_or_alias_candidates": ["SEI: solid electrolyte interphase"],
                        "ambiguities": [],
                    },
                    "confidence": {"overall": "medium", "reasons": ["model extrapolated"]},
                    "evidence_from_user_question": ["SEI"],
                    "warnings": ["unsupported_domain_expansion"],
                }
            )
        if self.response_mode == "decision_fields":
            return json.dumps(
                {
                    "intent_summary": "Unsafe output with a decision field.",
                    "include": ["paper A"],
                    "normalized_research_intent": {
                        "topic": ["OER spin state"],
                        "target_context_candidates": [],
                        "negative_context_candidates": [],
                        "method_need": [],
                        "mechanism_need": ["spin state"],
                        "application_need": [],
                        "material_or_domain_terms": [],
                        "abbreviation_or_alias_candidates": [],
                        "ambiguities": [],
                    },
                    "confidence": {"overall": "high", "reasons": []},
                    "evidence_from_user_question": ["OER", "spin state"],
                    "warnings": [],
                }
            )
        return json.dumps(
            {
                "intent_summary": "Find SEI papers for lithium battery context.",
                "normalized_research_intent": {
                    "topic": ["solid electrolyte interphase"],
                    "target_context_candidates": ["lithium battery", "lithium metal battery"],
                    "negative_context_candidates": ["sodium-ion battery", "potassium-ion battery"],
                    "method_need": ["in situ characterization"],
                    "mechanism_need": ["dendrite suppression", "cycling degradation"],
                    "application_need": ["cycle life"],
                    "material_or_domain_terms": ["artificial SEI", "engineered interphase"],
                    "abbreviation_or_alias_candidates": ["SEI: solid electrolyte interphase"],
                    "ambiguities": [],
                },
                "confidence": {"overall": "high", "reasons": ["explicit user terms"]},
                "evidence_from_user_question": ["锂电池", "SEI", "人工 SEI"],
                "warnings": [],
            },
            ensure_ascii=False,
        )


class GenericLLMIntentProvider:
    """Adapter from the existing JSON chat client to the intent provider API."""

    def __init__(self, client: GenericLLMClient) -> None:
        self.client = client
        self.provider_name = client.provider_name
        self.model_name = getattr(client, "model", "")

    def generate_intent_frame(
        self,
        question: str,
        deterministic_intent: Any | None = None,
    ) -> str:
        prompt = build_intent_frame_prompt(question, deterministic_intent)
        result = self.client.chat_json(
            "Return one JSON object only. Intent framing suggestions only.",
            prompt,
        )
        if result.raw_text:
            return result.raw_text
        return json.dumps(result.data, ensure_ascii=False)


class LLMIntentFrameEnhancer:
    """Optional LLM intent-frame enhancer skeleton.

    Disabled mode remains a strict no-op. Enabled mode can call an injected
    provider interface, parse its JSON, and record a verified suggestion. It
    still does not apply the suggestion to the deterministic intent or the main
    pipeline.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        provider: str = "",
        model: str = "",
        llm_provider: LLMIntentProvider | None = None,
    ) -> None:
        self.enabled = enabled
        self.provider = provider
        self.model = model
        self.llm_provider = llm_provider

    def enhance(
        self,
        deterministic_intent: Any,
        *,
        input_question: str = "",
        llm_client: Any | None = None,
        llm_provider: LLMIntentProvider | None = None,
    ) -> LLMIntentEnhancementResult:
        """Return an enhancement audit result without changing intent.

        ``llm_client`` is accepted for backward compatibility with the previous
        scaffold API, but this module uses ``llm_provider``. No real API is
        called unless a future explicit integration passes a real provider.
        """

        if not self.enabled:
            raw = LLMIntentFrameRaw(
                enabled=False,
                provider=self.provider,
                model=self.model,
                input_question=input_question,
            )
            trace = LLMIntentEnhancementTrace(
                llm_enabled=False,
                llm_called=False,
                fallback_used=False,
                malformed_output=False,
                verified_candidate_count=0,
                applied_suggestion_count=0,
                accepted_suggestion_count=0,
                rejected_suggestion_count=0,
                unsupported_suggestion_count=0,
                reason_if_no_change="llm_intent_enhancer_disabled",
            )
            return LLMIntentEnhancementResult(
                intent_before_llm=deterministic_intent,
                intent_after_llm=deterministic_intent,
                raw=raw,
                verified_suggestion=None,
                trace=trace,
            )

        provider = llm_provider or self.llm_provider
        if provider is None:
            raw = LLMIntentFrameRaw(
                enabled=True,
                provider=self.provider,
                model=self.model,
                input_question=input_question,
                error="llm_provider_unavailable",
            )
            trace = LLMIntentEnhancementTrace(
                llm_enabled=True,
                llm_called=False,
                fallback_used=True,
                malformed_output=False,
                verified_candidate_count=0,
                applied_suggestion_count=0,
                accepted_suggestion_count=0,
                rejected_suggestion_count=0,
                unsupported_suggestion_count=0,
                reason_if_no_change="llm_provider_unavailable",
            )
            return LLMIntentEnhancementResult(
                intent_before_llm=deterministic_intent,
                intent_after_llm=deterministic_intent,
                raw=raw,
                verified_suggestion=None,
                trace=trace,
            )

        raw_text = provider.generate_intent_frame(input_question, deterministic_intent)
        raw, suggestion, parse_trace = parse_llm_intent_frame_json(
            raw_text,
            input_question=input_question,
            provider=getattr(provider, "provider_name", self.provider),
            model=getattr(provider, "model_name", self.model),
        )
        if suggestion is None:
            return LLMIntentEnhancementResult(
                intent_before_llm=deterministic_intent,
                intent_after_llm=deterministic_intent,
                raw=raw,
                verified_suggestion=None,
                trace=parse_trace,
                verification_result=None,
            )
        verification = verify_intent_frame_suggestions(
            input_question,
            deterministic_intent,
            suggestion,
        )
        verified_candidate_count = count_verified_candidates(verification)
        rejected_count = len(verification.get("rejected_suggestions", []) or [])
        unsupported_count = int(verification.get("unsupported_suggestion_count") or 0)
        verification_warnings = verification.get("verification_warnings", []) or []
        unsafe_warning = "llm_warning_unsupported_domain_expansion" in verification_warnings
        fallback_used = verified_candidate_count == 0 or unsafe_warning
        applied_suggestion_count = (
            0
            if fallback_used
            else count_contract_applicable_suggestions(verification)
        )
        reason_if_no_change = ""
        if unsafe_warning:
            reason_if_no_change = "llm_output_contains_unsupported_domain_expansion"
        elif verified_candidate_count == 0:
            reason_if_no_change = "no_verified_llm_intent_suggestions"
        elif applied_suggestion_count == 0:
            reason_if_no_change = "verified_suggestions_not_contract_applicable"
        trace = LLMIntentEnhancementTrace(
            llm_enabled=True,
            llm_called=True,
            fallback_used=fallback_used,
            malformed_output=False,
            verified_candidate_count=verified_candidate_count,
            applied_suggestion_count=applied_suggestion_count,
            accepted_suggestion_count=applied_suggestion_count,
            rejected_suggestion_count=rejected_count,
            unsupported_suggestion_count=unsupported_count,
            reason_if_no_change=reason_if_no_change,
        )
        return LLMIntentEnhancementResult(
            intent_before_llm=deterministic_intent,
            intent_after_llm=deterministic_intent,
            raw=raw,
            verified_suggestion=suggestion if verified_candidate_count and not unsafe_warning else None,
            trace=trace,
            verification_result=verification,
        )


def build_intent_frame_prompt(
    question: str,
    deterministic_intent: Any | None = None,
) -> str:
    """Build a short JSON-only prompt for intent-frame suggestions."""

    return (
        "Return one JSON object only for optional research-intent framing. "
        "Do not output include, exclude, must_read, out_of_scope, final_score, "
        "domain_decision, evidence_validity, or reading_priority. "
        "Schema: intent_summary; normalized_research_intent with topic, "
        "target_context_candidates, negative_context_candidates, method_need, "
        "mechanism_need, application_need, material_or_domain_terms, "
        "abbreviation_or_alias_candidates, ambiguities; confidence with overall "
        "low|medium|high and reasons; evidence_from_user_question; warnings. "
        f"Question: {question}\n"
        f"Deterministic intent: {json.dumps(_to_plain_data(deterministic_intent), ensure_ascii=False)}"
    )


def parse_llm_intent_frame_json(
    raw_text: str,
    *,
    input_question: str = "",
    provider: str = "",
    model: str = "",
) -> tuple[LLMIntentFrameRaw, LLMIntentFrameSuggestion | None, LLMIntentEnhancementTrace]:
    """Parse and safety-check a raw LLM intent-frame JSON response."""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        raw = LLMIntentFrameRaw(
            enabled=True,
            provider=provider,
            model=model,
            input_question=input_question,
            raw_text=raw_text,
            parsed_json=None,
            malformed_output=True,
            error="malformed_json",
        )
        trace = LLMIntentEnhancementTrace(
            llm_enabled=True,
            llm_called=True,
            fallback_used=True,
            malformed_output=True,
            verified_candidate_count=0,
            applied_suggestion_count=0,
            accepted_suggestion_count=0,
            reason_if_no_change="malformed_llm_intent_frame_json",
        )
        return raw, None, trace

    if not isinstance(parsed, dict):
        raw = LLMIntentFrameRaw(
            enabled=True,
            provider=provider,
            model=model,
            input_question=input_question,
            raw_text=raw_text,
            parsed_json=None,
            malformed_output=True,
            error="json_not_object",
        )
        trace = LLMIntentEnhancementTrace(
            llm_enabled=True,
            llm_called=True,
            fallback_used=True,
            malformed_output=True,
            verified_candidate_count=0,
            applied_suggestion_count=0,
            accepted_suggestion_count=0,
            reason_if_no_change="malformed_llm_intent_frame_json",
        )
        return raw, None, trace

    forbidden = sorted(find_forbidden_decision_fields(parsed))
    if forbidden:
        raw = LLMIntentFrameRaw(
            enabled=True,
            provider=provider,
            model=model,
            input_question=input_question,
            raw_text=raw_text,
            parsed_json=parsed,
            malformed_output=False,
            error="llm_output_contains_forbidden_decision_fields",
        )
        trace = LLMIntentEnhancementTrace(
            llm_enabled=True,
            llm_called=True,
            fallback_used=True,
            malformed_output=False,
            verified_candidate_count=0,
            applied_suggestion_count=0,
            accepted_suggestion_count=0,
            rejected_suggestion_count=1,
            unsupported_suggestion_count=0,
            reason_if_no_change="llm_output_contains_forbidden_decision_fields",
        )
        return raw, None, trace

    suggestion = suggestion_from_parsed_json(parsed)
    raw = LLMIntentFrameRaw(
        enabled=True,
        provider=provider,
        model=model,
        input_question=input_question,
        raw_text=raw_text,
        parsed_json=parsed,
        malformed_output=False,
        error="",
    )
    trace = LLMIntentEnhancementTrace(
        llm_enabled=True,
        llm_called=True,
        fallback_used=False,
        malformed_output=False,
        verified_candidate_count=0,
        applied_suggestion_count=0,
        accepted_suggestion_count=0,
        rejected_suggestion_count=0,
        unsupported_suggestion_count=0,
        reason_if_no_change="llm_intent_suggestion_parsed_pending_verification",
    )
    return raw, suggestion, trace


def find_forbidden_decision_fields(value: Any) -> set[str]:
    """Find forbidden screening/ranking decision fields recursively."""

    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FORBIDDEN_DECISION_FIELDS:
                found.add(str(key))
            found.update(find_forbidden_decision_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(find_forbidden_decision_fields(child))
    return found


def verify_intent_frame_suggestions(
    user_question: str,
    deterministic_intent: Any,
    llm_suggestion: LLMIntentFrameSuggestion | dict[str, Any],
) -> dict[str, Any]:
    """Deterministically verify intent-level LLM suggestions.

    This verifier only accepts suggestions grounded in the user question,
    strongly supported by deterministic intent, safe terminology normalization,
    or explicit non-target contrasts for a known target context. It never emits
    screening, ranking, evidence-validity, or domain-decision labels.
    """

    suggestion = _to_plain_data(llm_suggestion)
    accepted: dict[str, list[dict[str, str]]] = {}
    rejected: list[dict[str, str]] = []
    warnings: list[str] = []

    forbidden = sorted(find_forbidden_decision_fields(suggestion))
    for field in forbidden:
        rejected.append(
            {
                "field": field,
                "value": field,
                "reason": "forbidden_decision_field",
            }
        )
    if forbidden:
        return {
            "accepted_suggestions": accepted,
            "rejected_suggestions": rejected,
            "unsupported_suggestion_count": len(rejected),
            "verification_warnings": ["llm_output_contains_forbidden_decision_fields"],
        }

    if not isinstance(suggestion, dict):
        return {
            "accepted_suggestions": accepted,
            "rejected_suggestions": [
                {
                    "field": "llm_suggestion",
                    "value": str(type(llm_suggestion).__name__),
                    "reason": "invalid_suggestion_type",
                }
            ],
            "unsupported_suggestion_count": 1,
            "verification_warnings": ["invalid_suggestion_type"],
        }

    question_text = str(user_question or "")
    deterministic_text = " ".join(flatten_strings(deterministic_intent))
    context = build_verifier_context(question_text, deterministic_text)
    for warning in _string_list(suggestion.get("warnings")):
        if warning.lower() == "unsupported_domain_expansion":
            warnings.append("llm_warning_unsupported_domain_expansion")

    for field in [
        "topic",
        "target_context_candidates",
        "negative_context_candidates",
        "method_need",
        "mechanism_need",
        "application_need",
        "material_or_domain_terms",
        "abbreviation_or_alias_candidates",
        "ambiguities",
    ]:
        for value in _string_list(suggestion.get(field)):
            reason = accepted_reason_for_value(field, value, context)
            if reason:
                accepted.setdefault(field, []).append(
                    {"value": value, "reason": reason}
                )
            else:
                rejected.append(
                    {
                        "field": field,
                        "value": value,
                        "reason": rejected_reason_for_value(field, value, context),
                    }
                )

    summary = str(suggestion.get("intent_summary") or "").strip()
    if summary:
        reason = accepted_reason_for_value("intent_summary", summary, context)
        if reason:
            accepted.setdefault("intent_summary", []).append(
                {"value": summary, "reason": reason}
            )
        else:
            rejected.append(
                {
                    "field": "intent_summary",
                    "value": summary,
                    "reason": "broad_claim_not_grounded_in_user_question",
                }
            )

    evidence_values = _string_list(suggestion.get("evidence_from_user_question"))
    for evidence in evidence_values:
        if value_appears_in_text(evidence, question_text):
            accepted.setdefault("evidence_from_user_question", []).append(
                {"value": evidence, "reason": "literal_user_question_evidence"}
            )
        else:
            rejected.append(
                {
                    "field": "evidence_from_user_question",
                    "value": evidence,
                    "reason": "evidence_not_found_in_user_question",
                }
            )

    unsupported_count = len(rejected)
    return {
        "accepted_suggestions": accepted,
        "rejected_suggestions": rejected,
        "unsupported_suggestion_count": unsupported_count,
        "verification_warnings": sorted(set(warnings)),
    }


def apply_verified_suggestions_to_search_contract(
    search_contract: SearchContract,
    enhancement_result: LLMIntentEnhancementResult,
) -> SearchContract:
    """Add verified intent suggestions to SearchContract with provenance.

    This function never overwrites deterministic fields. Accepted LLM
    suggestions are appended as optional constraint groups with
    ``source=llm_suggested_rule_verified`` and are also recorded in provenance
    fields for audit. Rejected suggestions are retained only as provenance.
    """

    verification = enhancement_result.verification_result or {}
    accepted = (
        {}
        if enhancement_result.trace.fallback_used
        else verification.get("accepted_suggestions") or {}
    )
    rejected = verification.get("rejected_suggestions") or []
    groups = list(search_contract.constraint_groups)
    for field_name, suggestions in accepted.items():
        if field_name not in CONTRACT_APPLICABLE_SUGGESTION_FIELDS:
            continue
        terms = [
            str(item.get("value") or "")
            for item in suggestions
            if isinstance(item, dict) and str(item.get("value") or "").strip()
        ]
        if not terms:
            continue
        groups.append(
            SearchConstraintGroup(
                group_name=f"llm_{field_name}",
                operator="OR",
                terms=_unique_strings(terms),
                source="llm_suggested_rule_verified",
                required=False,
            )
        )

    provenance = {
        "source": "llm_suggested_rule_verified"
        if any(accepted.values())
        else "deterministic",
        "accepted_suggestions": accepted,
        "rejected_suggestions": [
            {**item, "source": "rejected_llm_suggestion"}
            for item in rejected
            if isinstance(item, dict)
        ],
        "trace": _to_plain_data(enhancement_result.trace),
    }
    validation_events = [
        *search_contract.concept_validation_events,
        *[
            {
                "source": "llm_suggested_rule_verified",
                "field": field_name,
                "value": item.get("value", ""),
                "reason": item.get("reason", ""),
            }
            for field_name, suggestions in accepted.items()
            for item in suggestions
            if isinstance(item, dict)
        ],
        *[
            {
                "source": "rejected_llm_suggestion",
                "field": item.get("field", ""),
                "value": item.get("value", ""),
                "reason": item.get("reason", ""),
            }
            for item in rejected
            if isinstance(item, dict)
        ],
    ]
    return replace(
        search_contract,
        constraint_groups=_unique_constraint_groups(groups),
        concept_validation_events=validation_events,
        llm_intent_provenance=provenance,
        llm_verified_suggestions=accepted,
        llm_rejected_suggestions=provenance["rejected_suggestions"],
    )


def count_verified_candidates(verification: dict[str, Any]) -> int:
    accepted = verification.get("accepted_suggestions") or {}
    if not isinstance(accepted, dict):
        return 0
    return sum(
        len(values)
        for values in accepted.values()
        if isinstance(values, list)
    )


def count_contract_applicable_suggestions(verification: dict[str, Any]) -> int:
    accepted = verification.get("accepted_suggestions") or {}
    if not isinstance(accepted, dict):
        return 0
    return sum(
        len(values)
        for field_name, values in accepted.items()
        if field_name in CONTRACT_APPLICABLE_SUGGESTION_FIELDS
        and isinstance(values, list)
    )


def build_verifier_context(user_question: str, deterministic_text: str) -> dict[str, Any]:
    combined = f"{user_question}\n{deterministic_text}"
    normalized_combined = normalize_for_match(combined)
    supported_aliases = chinese_and_contextual_aliases(combined)
    return {
        "user_question": user_question,
        "deterministic_text": deterministic_text,
        "combined": combined,
        "normalized_combined": normalized_combined,
        "supported_aliases": supported_aliases,
        "has_lithium_target": has_lithium_target_context(combined),
        "has_lithium_ion_context": has_lithium_ion_context(combined),
        "has_lithium_metal_context": has_lithium_metal_context(combined),
        "has_battery_support_context": has_battery_support_context(combined),
    }


def accepted_reason_for_value(field: str, value: str, context: dict[str, Any]) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    if field == "negative_context_candidates":
        if is_non_target_battery_context(cleaned) and context["has_lithium_target"]:
            return "explicit_target_context_non_target_contrast"
        if value_supported_by_question_or_intent(cleaned, context):
            return "grounded_negative_context_from_user_or_intent"
        return ""
    if field == "abbreviation_or_alias_candidates":
        return abbreviation_acceptance_reason(cleaned, context)
    if value_supported_by_question_or_intent(cleaned, context):
        return "grounded_in_user_question_or_deterministic_intent"
    alias_key = normalize_for_match(cleaned)
    if alias_key in context["supported_aliases"]:
        return context["supported_aliases"][alias_key]
    return ""


def rejected_reason_for_value(field: str, value: str, context: dict[str, Any]) -> str:
    normalized = normalize_for_match(value)
    if field == "negative_context_candidates":
        return "negative_context_requires_explicit_target_context"
    if field == "abbreviation_or_alias_candidates":
        if normalized in {"li", "li lithium battery", "li: lithium battery"}:
            return "isolated_abbreviation_without_context"
        return "isolated_abbreviation_without_supported_context"
    if field == "material_or_domain_terms":
        return "unrelated_or_unsupported_material_domain_term"
    if field in {"topic", "intent_summary"}:
        return "broad_claim_not_grounded_in_user_question"
    if "quantum transport" in normalized or "cryo" in normalized or "perovskite" in normalized:
        return "arbitrary_domain_expansion"
    return "unsupported_by_user_question_or_deterministic_intent"


def value_supported_by_question_or_intent(value: str, context: dict[str, Any]) -> bool:
    normalized = normalize_for_match(value)
    if not normalized:
        return False
    if normalized in context["normalized_combined"]:
        return True
    compact = normalized.replace("-", " ")
    return compact in context["normalized_combined"].replace("-", " ")


def abbreviation_acceptance_reason(value: str, context: dict[str, Any]) -> str:
    normalized = normalize_for_match(value)
    if normalized.startswith("lib") or normalized.startswith("li ion"):
        if context["has_lithium_ion_context"]:
            return "lithium_ion_battery_alias_supported_by_context"
        return ""
    if normalized.startswith("lmb") or normalized.startswith("li metal"):
        if context["has_lithium_metal_context"]:
            return "lithium_metal_battery_alias_supported_by_context"
        return ""
    if normalized in {"li", "li lithium battery", "li: lithium battery"}:
        if context["has_battery_support_context"]:
            return "li_abbreviation_supported_by_battery_context"
        return ""
    return accepted_reason_for_value("generic_alias", value, context)


def chinese_and_contextual_aliases(text: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if any(marker in text for marker in ["锂电池", "锂离子电池"]) or any(
        marker in normalize_for_match(text)
        for marker in [
            "lithium battery",
            "lithium batteries",
            "lithium ion battery",
            "lithium-ion battery",
        ]
    ):
        for term in [
            "lithium battery",
            "lithium batteries",
            "lithium-ion battery",
            "lithium ion battery",
            "lib",
            "li-ion",
        ]:
            aliases[normalize_for_match(term)] = "lithium_battery_context_normalization"
    if "锂金属电池" in text or "lithium metal" in normalize_for_match(text):
        for term in ["lithium metal battery", "lithium metal anode", "lmb", "li metal"]:
            aliases[normalize_for_match(term)] = "lithium_metal_context_normalization"
    if "人工 sei" in normalize_for_match(text) or "artificial sei" in normalize_for_match(text):
        for term in [
            "artificial SEI",
            "engineered SEI",
            "artificial solid electrolyte interphase",
        ]:
            aliases[normalize_for_match(term)] = "artificial_sei_alias_grounded"
    if "sei" in normalize_for_match(text):
        for term in ["solid electrolyte interphase", "SEI formation"]:
            aliases[normalize_for_match(term)] = "sei_term_normalization"
    if "枝晶" in text or "dendrite" in normalize_for_match(text):
        for term in ["dendrite suppression", "lithium dendrite", "dendrite"]:
            aliases[normalize_for_match(term)] = "dendrite_mechanism_grounded"
    if "循环" in text or "cycling" in normalize_for_match(text) or "cycle" in normalize_for_match(text):
        for term in ["cycling stability", "cycle life", "cycling degradation"]:
            aliases[normalize_for_match(term)] = "cycling_term_grounded"
    return aliases


def has_lithium_target_context(text: str) -> bool:
    normalized = normalize_for_match(text)
    return any(
        marker in text or marker in normalized
        for marker in [
            "锂电池",
            "锂离子电池",
            "锂金属电池",
            "lithium battery",
            "lithium batteries",
            "lithium-ion battery",
            "lithium ion battery",
            "lithium metal battery",
        ]
    )


def has_lithium_ion_context(text: str) -> bool:
    normalized = normalize_for_match(text)
    return any(
        marker in text or marker in normalized
        for marker in [
            "锂电池",
            "锂离子电池",
            "lithium-ion",
            "lithium ion",
            "lithium battery",
            "lithium batteries",
        ]
    )


def has_lithium_metal_context(text: str) -> bool:
    normalized = normalize_for_match(text)
    return "锂金属电池" in text or "lithium metal" in normalized


def has_battery_support_context(text: str) -> bool:
    normalized = normalize_for_match(text)
    return any(
        marker in text or marker in normalized
        for marker in [
            "电池",
            "负极",
            "电解质",
            "枝晶",
            "循环",
            "battery",
            "anode",
            "electrolyte",
            "sei",
            "dendrite",
            "cycling",
        ]
    )


def is_non_target_battery_context(value: str) -> bool:
    normalized = normalize_for_match(value)
    return any(
        marker in normalized
        for marker in [
            "sodium-ion battery",
            "sodium ion battery",
            "potassium-ion battery",
            "potassium ion battery",
            "zinc-ion battery",
            "zinc ion battery",
        ]
    )


def value_appears_in_text(value: str, text: str) -> bool:
    if not value:
        return False
    return value in text or normalize_for_match(value) in normalize_for_match(text)


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for child in value.values():
            strings.extend(flatten_strings(child))
        return strings
    if isinstance(value, (list, tuple, set)):
        strings = []
        for child in value:
            strings.extend(flatten_strings(child))
        return strings
    if value is None:
        return []
    return [str(value)]


def normalize_for_match(value: str) -> str:
    return (
        str(value or "")
        .lower()
        .replace("：", ":")
        .replace("–", "-")
        .replace("—", "-")
        .strip()
    )


def suggestion_from_parsed_json(parsed: dict[str, Any]) -> LLMIntentFrameSuggestion:
    normalized = parsed.get("normalized_research_intent")
    normalized = normalized if isinstance(normalized, dict) else {}
    confidence = parsed.get("confidence")
    confidence_label = ""
    if isinstance(confidence, dict):
        confidence_label = str(confidence.get("overall") or "").lower()
    elif isinstance(confidence, str):
        confidence_label = confidence.lower()
    return LLMIntentFrameSuggestion(
        intent_summary=str(parsed.get("intent_summary") or ""),
        topic=_string_list(normalized.get("topic")),
        target_context_candidates=_string_list(normalized.get("target_context_candidates")),
        negative_context_candidates=_string_list(normalized.get("negative_context_candidates")),
        method_need=_string_list(normalized.get("method_need")),
        mechanism_need=_string_list(normalized.get("mechanism_need")),
        application_need=_string_list(normalized.get("application_need")),
        material_or_domain_terms=_string_list(normalized.get("material_or_domain_terms")),
        abbreviation_or_alias_candidates=_string_list(
            normalized.get("abbreviation_or_alias_candidates")
        ),
        ambiguities=_string_list(normalized.get("ambiguities")),
        confidence=CONFIDENCE_TO_FLOAT.get(confidence_label, 0.0),
        evidence_from_user_question=_string_list(parsed.get("evidence_from_user_question")),
        warnings=_string_list(parsed.get("warnings")),
    )


def write_intent_enhancement_artifacts(
    output_dir: str | Path,
    *,
    intent_before_llm: Any,
    raw: LLMIntentFrameRaw,
    verified_suggestion: LLMIntentFrameSuggestion | None,
    trace: LLMIntentEnhancementTrace,
    verification_result: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write reserved LLM intent-enhancement artifacts.

    This helper is intentionally not called by the main pipeline in the current
    baseline. It exists so future opt-in integration can write a stable artifact
    set without inventing file names later.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "intent_frame_before_llm": output / "intent_frame_before_llm.json",
        "llm_intent_frame_raw": output / "llm_intent_frame_raw.json",
        "llm_intent_frame_verified": output / "llm_intent_frame_verified.json",
        "llm_intent_enhancement_trace": output / "llm_intent_enhancement_trace.json",
    }
    _write_json(artifacts["intent_frame_before_llm"], intent_before_llm)
    _write_json(artifacts["llm_intent_frame_raw"], raw)
    verified_payload: Any
    if verification_result is None and verified_suggestion is None:
        verified_payload = None
    else:
        verified_payload = {
            "verified_suggestion": verified_suggestion,
            "verification_result": verification_result or {},
        }
    _write_json(artifacts["llm_intent_frame_verified"], verified_payload)
    _write_json(artifacts["llm_intent_enhancement_trace"], trace)
    return artifacts


def llm_intent_frame_suggestion_fields() -> set[str]:
    """Return suggestion schema fields for safety tests and documentation."""

    return set(LLMIntentFrameSuggestion.__dataclass_fields__)


def assert_llm_intent_frame_schema_is_non_decisive() -> None:
    """Raise if the suggestion schema accidentally gains decision fields."""

    overlap = llm_intent_frame_suggestion_fields() & FORBIDDEN_DECISION_FIELDS
    if overlap:
        raise ValueError(
            "LLMIntentFrameSuggestion contains decision fields: "
            + ", ".join(sorted(overlap))
        )


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


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _unique_constraint_groups(
    groups: list[SearchConstraintGroup],
) -> list[SearchConstraintGroup]:
    seen: set[tuple[str, str, tuple[str, ...], str, bool]] = set()
    result: list[SearchConstraintGroup] = []
    for group in groups:
        key = (
            group.group_name,
            group.operator,
            tuple(term.lower() for term in group.terms),
            group.source,
            group.required,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(group)
    return result


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
