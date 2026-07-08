"""Controlled optional LLM enhancement layers.

LLM outputs in this module are advisory. They may improve user-intent wording,
query-plan diagnostics, feedback interpretation, and novice-facing summaries,
but they do not decide evidence validity, domain fit, ranking, or screening
decisions.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from lit_screening.llm_client import GenericLLMClient
from lit_screening.models import (
    ExpertResearchIntent,
    GenericResearchIntentFrame,
    PaperRoleRecord,
    RankedPaper,
    SearchContract,
    SeedHint,
)
from lit_screening.utils import to_plain_data


INTENT_FRAME_FIELDS = {
    "core_object_terms",
    "domain_context_terms",
    "process_or_property_terms",
    "method_scope_terms",
    "mechanism_terms",
    "case_or_system_terms",
    "application_or_metric_terms",
    "failure_or_limitation_terms",
    "controversy_terms",
    "review_background_terms",
}

ALLOWED_GROUP_ROLES = {
    "required",
    "strong_context",
    "optional_aspect",
    "negative",
    "downrank",
}


class LLMIntentFrameEnhancer:
    """Schema-validate and safely apply optional LLM intent-frame hints."""

    def enhance(
        self,
        original_question: str,
        rule_based_intent_frame: GenericResearchIntentFrame | None,
        structured_concepts: list[Any] | None = None,
        selected_domain: str = "",
        domain_candidates: list[dict[str, Any]] | None = None,
        seed_papers: list[SeedHint] | None = None,
        llm_client: GenericLLMClient | None = None,
    ) -> dict[str, Any]:
        fallback = self._fallback_payload(rule_based_intent_frame)
        if not llm_client or not llm_client.is_available:
            return fallback
        result = llm_client.chat_json(
            self._system_prompt(),
            json.dumps(
                {
                    "original_question": original_question,
                    "rule_based_intent_frame": to_plain_data(rule_based_intent_frame),
                    "structured_concepts": to_plain_data(structured_concepts or []),
                    "selected_domain": selected_domain,
                    "domain_candidates": domain_candidates or [],
                    "seed_papers": to_plain_data(seed_papers or []),
                },
                ensure_ascii=False,
            ),
        )
        if result.invalid_llm_output:
            payload = dict(fallback)
            payload.update(
                {
                    "llm_used": False,
                    "fallback_used": True,
                    "schema_valid": False,
                    "rejection_reasons": [result.error_type or "invalid_llm_output"],
                }
            )
            return payload
        payload = result.data
        validation_errors = validate_intent_enhancement_payload(payload)
        if validation_errors:
            fallback.update(
                {
                    "llm_used": False,
                    "fallback_used": True,
                    "schema_valid": False,
                    "rejection_reasons": validation_errors,
                }
            )
            return fallback
        safe_payload = normalize_intent_payload(payload)
        safe_payload["clarification"] = ClarificationPolicy().apply(
            safe_payload.get("clarification", {}),
            safe_payload.get("ambiguities", []),
        )
        accepted, rejected = evaluate_intent_suggestions(safe_payload)
        safe_payload.update(
            {
                "llm_used": True,
                "fallback_used": False,
                "schema_valid": True,
                "accepted_suggestions": accepted,
                "rejected_suggestions": rejected,
                "rejection_reasons": sorted(
                    {
                        reason
                        for item in rejected
                        for reason in item.get("rejection_reasons", [])
                    }
                ),
            }
        )
        return safe_payload

    def apply_to_contract(
        self,
        contract: SearchContract,
        enhancement: dict[str, Any],
    ) -> SearchContract:
        """Safely merge accepted LLM hints into non-decisive frame fields."""

        frame = contract.generic_intent_frame
        if frame is None or not enhancement.get("schema_valid"):
            return contract
        research_frame = enhancement.get("research_intent_frame", {})
        method_scope = _string_list(research_frame.get("method_scope_terms", []))
        if not method_scope:
            return contract
        updated_frame = replace(
            frame,
            method_scope=_unique([*frame.method_scope, *method_scope], 12),
            method_terms=_unique([*frame.method_terms, *method_scope], 16),
            in_situ_or_operando_need=frame.in_situ_or_operando_need
            or any(_contains_any(term, ["in situ", "operando", "real-time"]) for term in method_scope),
            ex_situ_need=frame.ex_situ_need
            or any(_contains_any(term, ["ex situ", "post mortem"]) for term in method_scope),
            controversy_need=frame.controversy_need
            or bool(_string_list(research_frame.get("controversy_terms", []))),
            case_or_system_need=frame.case_or_system_need
            or bool(_string_list(research_frame.get("case_or_system_terms", []))),
            material_case_need=frame.material_case_need
            or bool(_string_list(research_frame.get("case_or_system_terms", []))),
            failure_or_limitation_need=frame.failure_or_limitation_need
            or bool(_string_list(research_frame.get("failure_or_limitation_terms", []))),
            review_background_need=frame.review_background_need
            or bool(_string_list(research_frame.get("review_background_terms", []))),
            theory_background_need=frame.theory_background_need
            or bool(_string_list(research_frame.get("review_background_terms", []))),
        )
        return replace(contract, generic_intent_frame=updated_frame)

    def _fallback_payload(
        self,
        frame: GenericResearchIntentFrame | None,
    ) -> dict[str, Any]:
        return {
            "llm_used": False,
            "fallback_used": True,
            "schema_valid": True,
            "interpreted_user_goal": "",
            "expert_rewritten_question": "",
            "user_level": "unknown",
            "research_intent_frame": frame_to_llm_frame(frame),
            "concept_groups": [],
            "assumptions": [],
            "ambiguities": [],
            "clarification": {
                "needed": False,
                "question": "",
                "reason": "Assumption-first fallback; no blocking clarification.",
                "blocking": False,
            },
            "accepted_suggestions": [],
            "rejected_suggestions": [],
            "rejection_reasons": [],
        }

    def _system_prompt(self) -> str:
        return (
            "You enhance novice scientific search intent only. Return JSON. "
            "Do not decide paper inclusion, exclusion, domain fit, ranking, or evidence validity. "
            "LLM-inferred domain-specific terms must not be required unless source=user_text."
        )


class LLMQueryPlanCritic:
    """Advisory query-plan critic with schema validation and rule fallback."""

    def critique(
        self,
        enhanced_intent_frame: dict[str, Any],
        search_contract: SearchContract,
        candidate_query_families: Any,
        final_provider_queries: dict[str, list[str]],
        llm_client: GenericLLMClient | None = None,
    ) -> dict[str, Any]:
        rule_findings = rule_query_critic(
            enhanced_intent_frame,
            search_contract,
            final_provider_queries,
        )
        if not llm_client or not llm_client.is_available:
            return {
                **rule_findings,
                "llm_used": False,
                "fallback_used": True,
                "schema_valid": True,
            }
        result = llm_client.chat_json(
            "Return JSON query-plan critique. Suggestions only; do not rewrite queries.",
            json.dumps(
                {
                    "enhanced_intent_frame": enhanced_intent_frame,
                    "search_contract": to_plain_data(search_contract),
                    "candidate_query_families": to_plain_data(candidate_query_families),
                    "final_provider_queries": final_provider_queries,
                },
                ensure_ascii=False,
            ),
        )
        if result.invalid_llm_output:
            return {
                **rule_findings,
                "llm_used": False,
                "fallback_used": True,
                "schema_valid": False,
                "rejection_reasons": [result.error_type or "invalid_llm_output"],
            }
        errors = validate_query_critic_payload(result.data)
        if errors:
            return {
                **rule_findings,
                "llm_used": False,
                "fallback_used": True,
                "schema_valid": False,
                "rejection_reasons": errors,
            }
        payload = normalize_query_critic_payload(result.data)
        accepted, rejected = QueryQualityRule().evaluate(payload)
        return {
            **payload,
            "llm_used": True,
            "fallback_used": False,
            "schema_valid": True,
            "accepted_suggestions": accepted,
            "rejected_suggestions": rejected,
            "rejection_reasons": sorted(
                {
                    reason
                    for item in rejected
                    for reason in item.get("rejection_reasons", [])
                }
            ),
        }


class LLMFeedbackInterpreter:
    """Turn novice feedback text into advisory preference hints."""

    def interpret(
        self,
        feedback_text: str,
        accepted_papers: list[Any] | None = None,
        rejected_papers: list[Any] | None = None,
        current_intent_frame: GenericResearchIntentFrame | None = None,
        ranking_diagnostics: dict[str, Any] | None = None,
        llm_client: GenericLLMClient | None = None,
    ) -> dict[str, Any]:
        fallback = rule_feedback_interpretation(feedback_text)
        if not llm_client or not llm_client.is_available or not feedback_text.strip():
            return {
                **fallback,
                "llm_used": False,
                "fallback_used": True,
                "schema_valid": True,
                "accepted_suggestions": fallback.get("accepted_suggestions", []),
                "rejected_suggestions": [],
                "rejection_reasons": [],
            }
        result = llm_client.chat_json(
            "Return JSON feedback interpretation. Do not decide ranking or inclusion.",
            json.dumps(
                {
                    "user_feedback_text": feedback_text,
                    "accepted_papers": to_plain_data(accepted_papers or []),
                    "rejected_papers": to_plain_data(rejected_papers or []),
                    "current_intent_frame": to_plain_data(current_intent_frame),
                    "ranking_diagnostics": ranking_diagnostics or {},
                },
                ensure_ascii=False,
            ),
        )
        if result.invalid_llm_output:
            return {
                **fallback,
                "llm_used": False,
                "fallback_used": True,
                "schema_valid": False,
                "rejection_reasons": [result.error_type or "invalid_llm_output"],
            }
        errors = validate_feedback_payload(result.data)
        if errors:
            return {
                **fallback,
                "llm_used": False,
                "fallback_used": True,
                "schema_valid": False,
            "rejection_reasons": errors,
            }
        payload = normalize_feedback_payload(result.data)
        return {
            **payload,
            "llm_used": True,
            "fallback_used": False,
            "schema_valid": True,
            "accepted_suggestions": [
                {"type": "feedback_interpretation", "reason": "schema_valid_advisory"}
            ],
            "rejected_suggestions": [],
            "rejection_reasons": [],
        }


class QueryQualityRule:
    """Accept or reject LLM query-critic suggestions as diagnostics only."""

    def evaluate(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return evaluate_critic_suggestions(payload)


class ClarificationPolicy:
    """Assumption-first clarification policy for novice questions."""

    BLOCKING_MARKERS = [
        "different retrieval direction",
        "mutually exclusive domain",
        "could mean clinical screening",
        "could mean materials screening",
        "cannot choose domain",
    ]

    def apply(
        self,
        clarification: dict[str, Any],
        ambiguities: list[str],
    ) -> dict[str, Any]:
        cleaned = {
            "needed": bool(clarification.get("needed", False)),
            "question": str(clarification.get("question") or ""),
            "reason": str(clarification.get("reason") or ""),
            "blocking": bool(clarification.get("blocking", False)),
        }
        evidence = " ".join([cleaned["reason"], cleaned["question"], *ambiguities]).lower()
        cleaned["blocking"] = any(marker in evidence for marker in self.BLOCKING_MARKERS)
        if not cleaned["blocking"] and cleaned["needed"]:
            cleaned["reason"] = cleaned["reason"] or "Non-blocking; continue with explicit assumptions."
        return cleaned


class LLMUserReportAdapter:
    """Create novice-friendly report text from verified artifacts only."""

    def generate(
        self,
        ranked_papers: list[RankedPaper],
        domain_assessments: list[Any],
        ranking_diagnostics: dict[str, Any],
        paper_roles: list[PaperRoleRecord],
        provider_status: dict[str, Any],
        exploration_quality: dict[str, Any],
        search_contract: SearchContract | None = None,
        research_gap_matrix: list[dict[str, Any]] | None = None,
        suggested_next_searches: list[dict[str, Any]] | None = None,
        llm_client: GenericLLMClient | None = None,
    ) -> tuple[str, dict[str, Any]]:
        artifact_input = verified_report_input(
            ranked_papers,
            domain_assessments,
            ranking_diagnostics,
            paper_roles,
            provider_status,
            exploration_quality,
            search_contract=search_contract,
            research_gap_matrix=research_gap_matrix,
            suggested_next_searches=suggested_next_searches,
        )
        fallback_md = render_user_report_markdown(artifact_input)
        fallback_trace = {
            "llm_used": False,
            "fallback_used": True,
            "schema_valid": True,
            "accepted_suggestions": [],
            "rejected_suggestions": [],
            "rejection_reasons": [],
            "input_artifacts": list(artifact_input.keys()),
        }
        if not llm_client or not llm_client.is_available:
            return fallback_md, fallback_trace
        result = llm_client.chat_json(
            "Return JSON with markdown only. Use only provided titles, evidence, domain, roles, status. Do not invent evidence.",
            json.dumps(artifact_input, ensure_ascii=False),
        )
        if result.invalid_llm_output:
            return fallback_md, {
                **fallback_trace,
                "schema_valid": False,
                "rejection_reasons": [result.error_type or "invalid_llm_output"],
            }
        markdown = str(result.data.get("markdown") or "").strip()
        if not markdown:
            return fallback_md, {
                **fallback_trace,
                "schema_valid": False,
                "rejection_reasons": ["missing_markdown"],
            }
        if not report_markdown_is_grounded(markdown, artifact_input):
            return fallback_md, {
                **fallback_trace,
                "schema_valid": False,
                "rejection_reasons": ["ungrounded_title_or_evidence_reference"],
            }
        return markdown, {
            "llm_used": True,
            "fallback_used": False,
            "schema_valid": True,
            "accepted_suggestions": [{"type": "user_report", "reason": "grounded_markdown"}],
            "rejected_suggestions": [],
            "rejection_reasons": [],
            "input_artifacts": list(artifact_input.keys()),
        }


def validate_intent_enhancement_payload(payload: dict[str, Any]) -> list[str]:
    required = {
        "interpreted_user_goal",
        "expert_rewritten_question",
        "user_level",
        "research_intent_frame",
        "concept_groups",
        "assumptions",
        "ambiguities",
        "clarification",
    }
    return [f"missing:{key}" for key in sorted(required) if key not in payload]


def normalize_intent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frame = payload.get("research_intent_frame") or {}
    normalized_frame = {key: _string_list(frame.get(key, [])) for key in INTENT_FRAME_FIELDS}
    concept_groups = []
    for group in payload.get("concept_groups", []) or []:
        if not isinstance(group, dict):
            continue
        concept_groups.append(
            {
                "group_name": str(group.get("group_name") or ""),
                "group_role": str(group.get("group_role") or "optional_aspect"),
                "operator_within_group": str(group.get("operator_within_group") or "OR"),
                "terms": _string_list(group.get("terms", [])),
                "source": str(group.get("source") or "llm_inferred"),
                "confidence": _float(group.get("confidence"), 0.0),
                "activation_reason": str(group.get("activation_reason") or ""),
            }
        )
    clarification = payload.get("clarification") or {}
    return {
        "interpreted_user_goal": str(payload.get("interpreted_user_goal") or ""),
        "expert_rewritten_question": str(payload.get("expert_rewritten_question") or ""),
        "user_level": str(payload.get("user_level") or "unknown"),
        "research_intent_frame": normalized_frame,
        "concept_groups": concept_groups,
        "assumptions": _string_list(payload.get("assumptions", [])),
        "ambiguities": _string_list(payload.get("ambiguities", [])),
        "clarification": {
            "needed": bool(clarification.get("needed", False)),
            "question": str(clarification.get("question") or ""),
            "reason": str(clarification.get("reason") or ""),
            "blocking": bool(clarification.get("blocking", False)),
        },
    }


def evaluate_intent_suggestions(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for group in payload.get("concept_groups", []):
        reasons: list[str] = []
        if group.get("group_role") not in ALLOWED_GROUP_ROLES:
            reasons.append("invalid_group_role")
        if group.get("group_role") in {"required", "strong_context"}:
            if group.get("source") != "user_text":
                reasons.append("llm_inferred_terms_not_allowed_as_required")
            if _float(group.get("confidence"), 0.0) < 0.8:
                reasons.append("low_confidence_required_group")
        if reasons:
            rejected.append({**group, "rejection_reasons": reasons})
        else:
            accepted.append({**group, "accepted_reason": "schema_valid_controlled_role"})
    return accepted, rejected


def frame_to_llm_frame(frame: GenericResearchIntentFrame | None) -> dict[str, list[str]]:
    if frame is None:
        return {key: [] for key in INTENT_FRAME_FIELDS}
    return {
        "core_object_terms": list(frame.research_object),
        "domain_context_terms": list(frame.domain_context),
        "process_or_property_terms": list(frame.target_process_or_property),
        "method_scope_terms": list(frame.method_scope),
        "mechanism_terms": list(frame.mechanism_terms),
        "case_or_system_terms": list(frame.material_or_case_terms),
        "application_or_metric_terms": list(frame.application_or_metric_terms),
        "failure_or_limitation_terms": list(frame.failure_or_limitation_terms),
        "controversy_terms": list(frame.controversy_terms),
        "review_background_terms": ["review_background"] if frame.review_background_need else [],
    }


def validate_query_critic_payload(payload: dict[str, Any]) -> list[str]:
    required = {
        "missing_user_aspects",
        "overbroad_queries",
        "overconstrained_queries",
        "repetition_issues",
        "cross_domain_pollution",
        "diversity_suggestions",
        "overall_assessment",
    }
    return [f"missing:{key}" for key in sorted(required) if key not in payload]


def normalize_query_critic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "missing_user_aspects": _string_list(payload.get("missing_user_aspects", [])),
        "overbroad_queries": _string_list(payload.get("overbroad_queries", [])),
        "overconstrained_queries": _string_list(payload.get("overconstrained_queries", [])),
        "repetition_issues": _string_list(payload.get("repetition_issues", [])),
        "cross_domain_pollution": _string_list(payload.get("cross_domain_pollution", [])),
        "diversity_suggestions": _string_list(payload.get("diversity_suggestions", [])),
        "overall_assessment": str(payload.get("overall_assessment") or ""),
    }


def rule_query_critic(
    enhanced_intent_frame: dict[str, Any],
    search_contract: SearchContract,
    final_provider_queries: dict[str, list[str]],
) -> dict[str, Any]:
    flat_queries = [
        query
        for queries in final_provider_queries.values()
        for query in queries
        if query
    ]
    text = "\n".join(flat_queries).lower()
    frame = search_contract.generic_intent_frame
    missing: list[str] = []
    pollution: list[str] = []
    diversity: list[str] = []
    if frame and frame.in_situ_or_operando_need and "in situ" not in text and "operando" not in text:
        missing.append("missing_in_situ_or_operando_query")
    if frame and frame.ex_situ_need and "ex situ" not in text and "post mortem" not in text:
        missing.append("missing_ex_situ_query")
    if ("llm" in text or "systematic review" in text) and (
        "representative materials" in text or "experimental characterization" in text
    ):
        pollution.append("ai_screening_query_contains_material_template_terms")
    if "water stability" in text and flat_queries and all("water stability" in query.lower() for query in flat_queries):
        diversity.append("mof_queries_overrequire_water_stability")
    if "atomic layer deposition" in text and "sputtering" not in text:
        diversity.append("thin_film_queries_ald_bias")
    if "oer" in text or "oxygen evolution reaction" in text:
        surface_spin_count = sum(
            1 for query in flat_queries if "surface spin state" in query.lower()
        )
        spin_or_electronic_count = sum(
            1
            for query in flat_queries
            if "spin state" in query.lower() or "electronic structure" in query.lower()
        )
        if flat_queries and surface_spin_count == len(flat_queries):
            diversity.append("oer_queries_overconstrained_to_surface_spin_state")
        if spin_or_electronic_count == 0:
            missing.append("missing_spin_state_or_electronic_structure_query")
    return {
        "missing_user_aspects": missing,
        "overbroad_queries": [
            query for query in flat_queries if len(query.split()) <= 2
        ],
        "overconstrained_queries": [],
        "repetition_issues": repeated_anchor_issues(flat_queries),
        "cross_domain_pollution": pollution,
        "diversity_suggestions": diversity,
        "overall_assessment": "rule_fallback_query_critic",
        "accepted_suggestions": [
            {"type": "rule_query_critic", "reason": item}
            for item in [*missing, *pollution, *diversity]
        ],
        "rejected_suggestions": [],
        "rejection_reasons": [],
    }


def evaluate_critic_suggestions(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = []
    for field in [
        "missing_user_aspects",
        "overbroad_queries",
        "overconstrained_queries",
        "repetition_issues",
        "cross_domain_pollution",
        "diversity_suggestions",
    ]:
        for item in payload.get(field, []):
            accepted.append({"type": field, "value": item, "accepted_reason": "diagnostic_only"})
    return accepted, []


def validate_feedback_payload(payload: dict[str, Any]) -> list[str]:
    required = {"feedback_summary", "updated_goal", "boost", "downrank", "exclude", "report_preferences"}
    return [f"missing:{key}" for key in sorted(required) if key not in payload]


def normalize_feedback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "feedback_summary": str(payload.get("feedback_summary") or ""),
        "updated_goal": str(payload.get("updated_goal") or ""),
        "boost": _string_list(payload.get("boost", [])),
        "downrank": _string_list(payload.get("downrank", [])),
        "exclude": _string_list(payload.get("exclude", [])),
        "report_preferences": _string_list(payload.get("report_preferences", [])),
    }


def rule_feedback_interpretation(feedback_text: str) -> dict[str, Any]:
    lowered = feedback_text.lower()
    downrank: list[str] = []
    boost: list[str] = []
    if any(marker in feedback_text for marker in ["钠电池", "钠离子", "钠"]) or "sodium" in lowered:
        downrank.extend(["sodium-ion battery", "potassium-ion battery", "zinc battery context"])
    if any(marker in feedback_text for marker in ["实验表征", "表征方法", "原位", "非原位"]):
        boost.extend(["characterization_method", "in_situ_operando", "ex_situ_post_mortem"])
    return {
        "feedback_summary": feedback_text.strip(),
        "updated_goal": "",
        "boost": _unique(boost, 12),
        "downrank": _unique(downrank, 12),
        "exclude": [],
        "report_preferences": [],
        "accepted_suggestions": [{"type": "rule_feedback_interpretation", "reason": "keyword_feedback"}]
        if boost or downrank
        else [],
    }


def verified_report_input(
    ranked_papers: list[RankedPaper],
    domain_assessments: list[Any],
    ranking_diagnostics: dict[str, Any],
    paper_roles: list[PaperRoleRecord],
    provider_status: dict[str, Any],
    exploration_quality: dict[str, Any],
    search_contract: SearchContract | None = None,
    research_gap_matrix: list[dict[str, Any]] | None = None,
    suggested_next_searches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    roles_by_id = {record.paper_id: record for record in paper_roles}
    retrieval_summary = report_retrieval_summary(provider_status, len(ranked_papers))
    task_profile = report_task_profile(search_contract, ranked_papers)
    rows = []
    for item in ranked_papers[:12]:
        role = roles_by_id.get(item.paper.paper_id)
        assessment = item.domain_assessment
        decision = item.screening_decision
        matched_groups = list(assessment.matched_groups) if assessment else []
        missing_groups = list(assessment.missing_groups) if assessment else []
        target_context = list(assessment.target_context_match) if assessment else []
        negative_context = list(assessment.negative_context_match) if assessment else []
        peripheral_reason = assessment.peripheral_context_reason if assessment else ""
        topic_focus = assessment.topic_focus_score if assessment else 0.0
        domain_decision = assessment.domain_decision if assessment else ""
        primary_role = role.primary_role if role else ""
        reading_priority = decision.reading_priority if decision else ""
        rows.append(
            {
                "rank": item.rank,
                "title": item.paper.title,
                "evidence_grounding": item.verification.support_level,
                "intent_match": item.scores.intent_centrality_score,
                "reading_priority": reading_priority,
                "domain_decision": domain_decision,
                "primary_role": primary_role,
                "matched_groups": matched_groups,
                "missing_groups": missing_groups,
                "target_context_match": target_context,
                "negative_context_match": negative_context,
                "topic_focus_score": topic_focus,
                "peripheral_context_reason": peripheral_reason,
                "why_read_now": why_read_now_summary(
                    primary_role,
                    reading_priority,
                    domain_decision,
                    matched_groups,
                    missing_groups,
                    target_context,
                    negative_context,
                    peripheral_reason,
                ),
                "evidence_sentence": item.evidence.evidence_sentence,
            }
        )
    return {
        "ranked_papers": rows,
        "domain_assessments": to_plain_data(domain_assessments[:20]),
        "ranking_diagnostics": ranking_diagnostics,
        "paper_roles": to_plain_data(paper_roles[:20]),
        "provider_status": provider_status,
        "retrieval_summary": retrieval_summary,
        "exploration_quality": exploration_quality,
        "research_gap_matrix": research_gap_matrix or [],
        "suggested_next_searches": suggested_next_searches or [],
        "task_profile": task_profile,
    }


def render_user_report_markdown(artifact_input: dict[str, Any]) -> str:
    retrieval_summary = artifact_input.get("retrieval_summary") or {}
    lines = [
        "# User-Friendly Literature Screening Summary",
        "",
        "## Retrieval status",
        "",
        f"- retrieval_status: {retrieval_summary.get('retrieval_status', 'unknown')}",
        f"- provider_summary: {retrieval_summary.get('provider_summary', 'not available')}",
        f"- ranked papers based on real retrieval: {retrieval_summary.get('ranked_papers_based_on_real_retrieval', False)}",
        "",
        "## How the system interpreted the task",
        "",
        "The system split the question into intent groups, query families, domain checks, and evidence-grounded paper summaries.",
        "",
    ]
    if retrieval_summary.get("retrieval_status") == "planning_only":
        lines.extend(
            [
                "本次只完成 query planning，未执行论文检索，因此不生成 research gaps，也不会把任何论文标成核心必读。",
                "",
            ]
        )
    lines.extend(
        [
        "## What to read first",
        "",
        "| Rank | Paper | Role | Evidence grounding | Intent match | Reading priority | Intent match summary | Matched groups | Missing groups | Context warning | Why read now |",
        "| ---: | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in artifact_input.get("ranked_papers", [])[:8]:
        context_warning = context_warning_summary(row)
        lines.append(
            "| "
            f"{row.get('rank', '')} | "
            f"{_escape_md(row.get('title', ''))} | "
            f"{_escape_md(row.get('primary_role', '') or row.get('domain_decision', ''))} | "
            f"{_escape_md(row.get('evidence_grounding', ''))} | "
            f"{float(row.get('intent_match') or 0):.3f} | "
            f"{_escape_md(row.get('reading_priority', ''))} | "
            f"{_escape_md(intent_match_summary(row))} | "
            f"{_escape_md('; '.join(row.get('matched_groups') or []))} | "
            f"{_escape_md('; '.join(row.get('missing_groups') or []))} | "
            f"{_escape_md(context_warning)} | "
            f"{_escape_md(row.get('why_read_now', ''))} |"
        )
    lines.extend(render_coverage_gap_section(artifact_input))
    lines.extend(
        [
            "",
            "## Important caveat",
            "",
            "`strict_support` means the evidence sentence is grounded in the abstract. It does not by itself mean the paper is highly relevant.",
            "A paper can have `strict_support` but still be background or peripheral if it misses required concept groups or has non-target context.",
            "",
            "## Provider status",
            "",
        ]
    )
    for provider, status in (artifact_input.get("provider_status") or {}).items():
        lines.append(
            f"- {provider}: {status.get('status', 'unknown')} "
            f"({status.get('returned_paper_count', 0)} papers returned)"
        )
    lines.extend(
        [
            "",
            "## How to narrow the next run",
            "",
        ]
    )
    for example in feedback_examples_for_task(str(artifact_input.get("task_profile") or "")):
        lines.append(f"- {example}")
    return "\n".join(lines) + "\n"


def render_coverage_gap_section(artifact_input: dict[str, Any]) -> list[str]:
    """Render novice-facing coverage, gap, and next-search guidance."""

    retrieval_summary = artifact_input.get("retrieval_summary") or {}
    gap_rows = artifact_input.get("research_gap_matrix") or []
    next_searches = artifact_input.get("suggested_next_searches") or []
    lines = ["", "## Coverage and remaining gaps", ""]
    if retrieval_summary.get("retrieval_status") == "planning_only":
        lines.append(
            "本次只完成 query planning，未执行论文检索，因此不生成 coverage 或 research gap 结论。"
        )
        return lines
    if gap_rows and gap_rows[0].get("gap_generation_status") == "skipped":
        lines.append(
            "Research gaps were not generated because retrieval was not performed or there were not enough screened papers."
        )
        return lines
    if gap_rows and gap_rows[0].get("gap_generation_status") == "no_clear_gap":
        row = gap_rows[0]
        lines.extend(
            [
                f"- Coverage summary: {row.get('coverage_summary', '')}",
                "- No clear research gap is asserted from this run alone; the screened set covers the configured markers reasonably well.",
                f"- Suggested action: {row.get('suggested_action', 'optional_depth_check')}",
            ]
        )
        if next_searches:
            lines.append("- Optional follow-up directions:")
            for item in next_searches[:5]:
                lines.append(
                    f"  - `{item.get('query', '')}`: {item.get('reason', 'Optional follow-up')}"
                )
        return lines
    if gap_rows:
        lines.append("- Low-coverage gaps inferred from this run:")
        for row in gap_rows[:5]:
            lines.append(
                f"  - {row.get('gap_label') or row.get('gap')}: "
                f"{row.get('evidence_or_reason') or row.get('why_gap_remains', '')}"
            )
        if next_searches:
            lines.append("- Suggested next searches:")
            for item in next_searches[:5]:
                lines.append(
                    f"  - `{item.get('query', '')}`: {item.get('reason', '')}"
                )
        return lines
    lines.append("No coverage or gap artifact was generated for this run.")
    return lines


def report_task_profile(
    search_contract: SearchContract | None,
    ranked_papers: list[RankedPaper],
) -> str:
    """Infer a lightweight report profile for feedback examples."""

    parts: list[str] = []
    if search_contract:
        frame = search_contract.generic_intent_frame
        parts.extend(search_contract.must_include_concepts)
        parts.extend(search_contract.optional_concepts)
        parts.extend(search_contract.required_aspects)
        if frame:
            parts.extend(frame.research_object)
            parts.extend(frame.domain_context)
            parts.extend(frame.target_process_or_property)
            parts.extend(frame.method_terms)
    for item in ranked_papers[:5]:
        parts.extend([item.paper.title, item.paper.abstract, item.evidence.evidence_sentence])
    text = " ".join(parts).lower()
    if "solid electrolyte interphase" in text or " sei" in f" {text}":
        return "sei_interface"
    if "oxygen evolution" in text or " oer" in f" {text}" or "water oxidation" in text:
        return "oer_catalysis"
    if "systematic review" in text or "literature screening" in text or "title abstract screening" in text:
        return "ai_literature_screening"
    if "mof" in text or "metal-organic framework" in text or "co2 capture" in text:
        return "mof_co2_capture"
    if "thin film" in text or "sputtering" in text or "atomic layer deposition" in text:
        return "thin_film_fabrication"
    return "generic_science"


def feedback_examples_for_task(task_profile: str) -> list[str]:
    """Return task-specific feedback examples for the next run."""

    examples = {
        "sei_interface": [
            "exclude non-lithium battery systems",
            "show more in situ / operando characterization",
            "focus on graphite / silicon / lithium metal anodes",
            "show fewer modeling papers",
        ],
        "oer_catalysis": [
            "focus on operando spectroscopy",
            "focus on perovskite oxides / cobalt oxyhydroxides",
            "compare spin-state effects with surface reconstruction",
            "exclude ORR-only / CO2-reduction papers",
            "show more controversy papers",
        ],
        "ai_literature_screening": [
            "focus on recall / precision evaluation",
            "exclude generic AI review papers",
            "show more human feedback workflows",
        ],
        "mof_co2_capture": [
            "focus on water stability",
            "focus on functionalized MOFs",
            "show more experimental CO2 adsorption papers",
        ],
        "thin_film_fabrication": [
            "compare ALD vs PLD vs sputtering vs CVD",
            "focus on review papers",
            "focus on deposition quality / scalability / cost",
        ],
    }
    return examples.get(
        task_profile,
        [
            "show more papers that cover all required concept groups",
            "exclude broad background papers",
            "focus on methods, mechanisms, or controversies",
        ],
    )


def report_retrieval_summary(
    provider_status: dict[str, Any],
    ranked_count: int,
) -> dict[str, Any]:
    states = [
        str(status.get("status") or "")
        for status in (provider_status or {}).values()
        if isinstance(status, dict)
    ]
    if states and all(state == "not_attempted" for state in states):
        retrieval_status = "planning_only"
    elif any(state in {"success", "success_no_results"} for state in states) and any(
        state in {"failed", "rate_limited", "partial_success", "partial_success_rate_limited"}
        for state in states
    ):
        retrieval_status = "partial_success"
    elif any(state in {"success", "success_no_results", "partial_success", "partial_success_rate_limited"} for state in states):
        retrieval_status = "success" if ranked_count else "failed"
    elif states:
        retrieval_status = "failed"
    else:
        retrieval_status = "planning_only"
    provider_bits = [
        f"{provider}:{status.get('status', 'unknown')}({status.get('returned_paper_count', 0)} papers)"
        for provider, status in (provider_status or {}).items()
        if isinstance(status, dict)
    ]
    return {
        "retrieval_status": retrieval_status,
        "provider_summary": "; ".join(provider_bits) or "no providers requested",
        "ranked_papers_based_on_real_retrieval": retrieval_status
        in {"success", "partial_success"}
        and ranked_count > 0,
    }


def intent_match_summary(row: dict[str, Any]) -> str:
    matched = row.get("matched_groups") or []
    missing = row.get("missing_groups") or []
    focus = _float(row.get("topic_focus_score"), 0.0)
    parts = []
    if matched:
        parts.append("covers " + ", ".join(matched[:4]))
    if missing:
        parts.append("missing " + ", ".join(missing[:3]))
    parts.append(f"topic focus {focus:.2f}")
    return "; ".join(parts)


def context_warning_summary(row: dict[str, Any]) -> str:
    negative = row.get("negative_context_match") or []
    target = row.get("target_context_match") or []
    peripheral = str(row.get("peripheral_context_reason") or "").strip()
    if negative or peripheral:
        return "peripheral/caution: " + "; ".join([*negative[:3], peripheral]).strip("; ")
    if target:
        return "target context: " + ", ".join(target[:3])
    return ""


def why_read_now_summary(
    primary_role: str,
    reading_priority: str,
    domain_decision: str,
    matched_groups: list[str],
    missing_groups: list[str],
    target_context: list[str],
    negative_context: list[str],
    peripheral_reason: str,
) -> str:
    if negative_context or peripheral_reason or domain_decision in {"borderline", "out_of_scope"}:
        return (
            "Use as peripheral/background only: it touches the topic but has non-target or borderline context."
        )
    if reading_priority == "must_read" and matched_groups and not missing_groups:
        return (
            "Read early: it covers the required concept groups and is treated as a core paper for this intent."
        )
    if "review" in primary_role or reading_priority == "read_later":
        return "Good background: useful for orientation, but not necessarily the central intersection paper."
    if target_context:
        return "Relevant candidate: it matches the target context and can support a focused reading pass."
    return "Shown because it has evidence or role signals, but verify fit before treating it as central."


def report_markdown_is_grounded(markdown: str, artifact_input: dict[str, Any]) -> bool:
    titles = [
        str(row.get("title") or "")
        for row in artifact_input.get("ranked_papers", [])
        if row.get("title")
    ]
    if not titles:
        return True
    mentioned_titles = [title for title in titles if title in markdown]
    return bool(mentioned_titles)


def repeated_anchor_issues(queries: list[str]) -> list[str]:
    if len(queries) < 4:
        return []
    first_tokens = []
    for query in queries:
        tokens = query.lower().replace('"', "").split()
        if tokens:
            first_tokens.append(tokens[0])
    if first_tokens and first_tokens.count(first_tokens[0]) / len(first_tokens) > 0.8:
        return [f"repeated_anchor:{first_tokens[0]}"]
    return []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique([str(item) for item in value if str(item).strip()], 64)
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unique(values: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen = set()
    for value in values:
        cleaned = " ".join(str(value or "").split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result[:limit] if limit else result


def _contains_any(value: str, markers: list[str]) -> bool:
    lowered = str(value or "").lower()
    return any(marker in lowered for marker in markers)


def _escape_md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:240]
