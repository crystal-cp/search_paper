"""Rule-based repair of novice research intent before query planning."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from lit_screening.agents.domain_router import DomainRouter
from lit_screening.agents.generic_intent import (
    build_generic_intent_frame,
    is_generic_noisy_term,
)
from lit_screening.domain_packs import list_domain_packs, load_domain_pack
from lit_screening.models import DomainPack, ExpertResearchIntent, IntentConcept, SeedHint


ALLOWED_CONCEPT_CATEGORIES = {
    "object",
    "property",
    "mechanism",
    "material",
    "method",
    "application",
    "exclusion",
    "generic_user_term",
}
ALLOWED_CONCEPT_SOURCES = {
    "user_text",
    "seed_title",
    "seed_metadata",
    "domain_pack",
    "llm_inferred",
    "feedback_profile",
    "pilot_repair",
}
ALLOWED_QUERY_ROLES = {"must", "optional", "exclude", "downweighted", "uncertain"}
LLM_CONFIDENCE_THRESHOLD = 0.55

GENERIC_DOWNWEIGHT_TERMS = [
    "importance",
    "background",
    "significance",
    "example",
    "review",
    "survey",
    "tutorial",
    "paper",
    "study",
    "research",
    "related work",
    "literature",
]

DETECTION_MARKERS = [
    "probe",
    "probing",
    "detect",
    "detection",
    "image",
    "imaging",
    "measurement",
    "measure",
    "探测",
    "检测",
    "成像",
    "测量",
]


class NoviceIntentInterpreter:
    """Convert underspecified user wording into an expert research task."""

    def repair(
        self,
        question: str,
        seed_hints: list[SeedHint] | None = None,
        domain_pack: DomainPack | None = None,
        llm_client: Any | None = None,
        use_llm: bool = True,
    ) -> ExpertResearchIntent:
        """Return an expert-level interpretation with optional LLM enhancement."""

        original_question = " ".join(str(question or "").split())
        seeds = seed_hints or []
        pack = domain_pack or _infer_domain_pack(original_question, seeds)
        metadata = _base_llm_metadata(pack.domain_name)
        if use_llm and _llm_available(llm_client):
            metadata["llm_attempted"] = True
            try:
                result = llm_client.chat_json(
                    _llm_system_prompt(),
                    _llm_user_prompt(original_question, seeds, pack),
                )
            except Exception as exc:
                metadata["fallback_reason"] = f"llm_exception:{exc.__class__.__name__}"
                return _with_llm_metadata(
                    _rule_repair(original_question, seeds, pack),
                    metadata,
                    fallback_used=True,
                )
            if getattr(result, "invalid_llm_output", False):
                metadata["invalid_json_count"] = 1
                metadata["fallback_reason"] = getattr(result, "error_type", "") or "invalid_json"
                return _with_llm_metadata(
                    _rule_repair(original_question, seeds, pack),
                    metadata,
                    fallback_used=True,
                )
            intent, errors = _intent_from_llm_payload(
                getattr(result, "data", {}),
                original_question,
                seeds,
                pack,
            )
            metadata["schema_validation_errors"] = errors
            metadata["llm_confidence"] = _safe_float(
                getattr(result, "data", {}).get("confidence"),
            )
            if errors:
                metadata["fallback_reason"] = "schema_validation_failed"
                return _with_llm_metadata(
                    _rule_repair(original_question, seeds, pack),
                    metadata,
                    fallback_used=True,
                )
            if intent is None or intent.confidence < LLM_CONFIDENCE_THRESHOLD:
                metadata["fallback_reason"] = "low_confidence"
                return _with_llm_metadata(
                    _rule_repair(original_question, seeds, pack),
                    metadata,
                    fallback_used=True,
                )
            validated = _validate_llm_intent_with_domain_pack(
                intent,
                original_question,
                seeds,
                pack,
            )
            metadata["llm_used"] = True
            metadata["fallback_used"] = False
            metadata["fallback_reason"] = ""
            metadata["domain_validation_events"] = validated.llm_metadata.get(
                "domain_validation_events",
                [],
            )
            metadata["llm_output_concepts"] = [
                concept.term for concept in intent.structured_concepts
            ]
            metadata["validated_concepts"] = [
                concept.term for concept in validated.structured_concepts
            ]
            return _with_llm_metadata(validated, metadata, fallback_used=False)
        if use_llm and llm_client is not None:
            metadata["fallback_reason"] = "llm_unavailable"
        else:
            metadata["fallback_reason"] = "llm_not_requested"
        return _with_llm_metadata(
            _rule_repair(original_question, seeds, pack),
            metadata,
            fallback_used=True,
        )


def _rule_repair(
    original_question: str,
    seeds: list[SeedHint],
    pack: DomainPack,
) -> ExpertResearchIntent:
    """Run the deterministic rule-based intent repair."""

    if pack.domain_name == "materials_magnetism":
        return _repair_materials_magnetism(original_question, seeds, pack)
    if pack.domain_name == "ferroelectric_polarization":
        return _repair_ferroelectric_polarization(original_question, pack)
    if pack.domain_name == "ai_literature_screening":
        return _repair_ai_literature_screening(original_question)
    return _fallback_repair(original_question)


def _llm_available(llm_client: Any | None) -> bool:
    return bool(llm_client is not None and getattr(llm_client, "is_available", False))


def _base_llm_metadata(domain_name: str) -> dict[str, Any]:
    return {
        "llm_attempted": False,
        "llm_used": False,
        "fallback_used": False,
        "invalid_json_count": 0,
        "schema_validation_errors": [],
        "llm_confidence": 0.0,
        "fallback_reason": "",
        "domain_pack_domain": domain_name,
        "domain_validation_events": [],
        "llm_output_concepts": [],
        "validated_concepts": [],
    }


def _with_llm_metadata(
    intent: ExpertResearchIntent,
    metadata: dict[str, Any],
    fallback_used: bool,
) -> ExpertResearchIntent:
    merged = {**metadata, "fallback_used": fallback_used}
    if fallback_used:
        merged["llm_used"] = False
    return replace(
        intent,
        llm_metadata=merged,
        needs_user_confirmation=intent.needs_user_confirmation
        or list(merged.get("needs_user_confirmation", [])),
    )


def _llm_system_prompt() -> str:
    return (
        "You are the intent-repair component of a scientific literature screening system. "
        "Return JSON only. The user's wording is not a search query; it is a fuzzy "
        "research intention and may be novice, incomplete, or partly mistaken. Rewrite "
        "it into an expert research task and output only ExpertResearchIntent fields. "
        "Do not treat every user word as a must term. Do not invent papers. Do not "
        "generate citation relations. Do not decide paper include/exclude outcomes. Do "
        "not perform evidence span validation, domain-guardrail final judgment, or final "
        "screening decisions. For every supplemented concept, provide source, confidence, "
        "activation_reason, query_role, and should_use_in_provider_query. If a concept is "
        "only inferred or uncertain, set query_role to optional or uncertain, not must."
    )


def _llm_user_prompt(
    question: str,
    seed_hints: list[SeedHint],
    pack: DomainPack,
) -> str:
    seed_payload = [
        {
            "title": hint.title,
            "authors": hint.authors,
            "doi": hint.doi,
            "arxiv_id": hint.arxiv_id,
            "confidence": hint.confidence,
        }
        for hint in seed_hints
    ]
    pack_terms = _domain_pack_terms(pack)[:80]
    return (
        "Output a JSON object with exactly this schema:\n"
        "{\n"
        '  "original_question": string,\n'
        '  "user_is_novice": boolean,\n'
        '  "expert_rewritten_question": string,\n'
        '  "inferred_goal": string,\n'
        '  "structured_concepts": [\n'
        "    {\n"
        '      "term": string,\n'
        '      "category": "object|property|mechanism|material|method|application|exclusion|generic_user_term",\n'
        '      "source": "user_text|seed_title|seed_metadata|domain_pack|llm_inferred|feedback_profile|pilot_repair",\n'
        '      "confidence": number,\n'
        '      "activation_reason": string,\n'
        '      "query_role": "must|optional|exclude|downweighted|uncertain",\n'
        '      "should_use_in_provider_query": boolean\n'
        "    }\n"
        "  ],\n"
        '  "likely_user_misconceptions": [string],\n'
        '  "downweighted_user_terms": [string],\n'
        '  "assumptions": [string],\n'
        '  "needs_user_confirmation": [string],\n'
        '  "confidence": number\n'
        "}\n\n"
        "Use only the schema above. Do not add paper rankings, include/exclude decisions, "
        "citation edges, or evidence validation results.\n\n"
        f"Detected domain pack: {pack.domain_name}\n"
        f"Domain-pack terms available for support: {pack_terms}\n"
        f"Seed hints: {seed_payload}\n"
        f"User question: {question}"
    )


def _intent_from_llm_payload(
    data: Any,
    question: str,
    seed_hints: list[SeedHint],
    pack: DomainPack,
) -> tuple[ExpertResearchIntent | None, list[str]]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return None, ["llm_output_not_object"]
    required = [
        "original_question",
        "user_is_novice",
        "expert_rewritten_question",
        "inferred_goal",
        "structured_concepts",
        "likely_user_misconceptions",
        "downweighted_user_terms",
        "assumptions",
        "needs_user_confirmation",
        "confidence",
    ]
    for key in required:
        if key not in data:
            errors.append(f"missing_field:{key}")
    forbidden_keys = {
        "paper_decisions",
        "include_decisions",
        "exclude_decisions",
        "ranked_papers",
        "citation_relations",
        "citation_edges",
        "evidence_span_validation",
        "domain_guardrail_decisions",
    }
    for key in forbidden_keys & set(data):
        errors.append(f"forbidden_field:{key}")
    if errors:
        return None, errors
    if not isinstance(data.get("structured_concepts"), list):
        errors.append("structured_concepts_not_list")
    for key in [
        "likely_user_misconceptions",
        "downweighted_user_terms",
        "assumptions",
        "needs_user_confirmation",
    ]:
        if not isinstance(data.get(key), list):
            errors.append(f"{key}_not_list")
    if not isinstance(data.get("user_is_novice"), bool):
        errors.append("user_is_novice_not_bool")
    if not isinstance(data.get("expert_rewritten_question"), str):
        errors.append("expert_rewritten_question_not_string")
    confidence = _safe_float(data.get("confidence"), default=-1.0)
    if confidence < 0 or confidence > 1:
        errors.append("confidence_out_of_range")
    concepts: list[IntentConcept] = []
    for index, item in enumerate(data.get("structured_concepts") or []):
        concept, concept_errors = _concept_from_llm_item(item, index)
        errors.extend(concept_errors)
        if concept is not None:
            concepts.append(concept)
    if errors:
        return None, errors
    downweighted_terms = [
        str(item)
        for item in _list_of_strings(data.get("downweighted_user_terms"))
    ]
    concepts.extend(_downweighted_concepts(" ".join([question, *downweighted_terms])))
    concepts = _merge_concepts(concepts)
    intent = _intent_from_concepts(
        question=str(data.get("original_question") or question),
        concepts=concepts,
        inferred_goal=str(data.get("inferred_goal") or ""),
        expert_rewritten_question=str(data.get("expert_rewritten_question") or question),
        assumptions=_list_of_strings(data.get("assumptions")),
        confidence=confidence,
    )
    return replace(
        intent,
        user_is_novice=bool(data.get("user_is_novice")),
        likely_user_misconceptions=_list_of_strings(
            data.get("likely_user_misconceptions"),
        ),
        ignored_or_downweighted_terms=_unique(
            [*intent.ignored_or_downweighted_terms, *downweighted_terms],
        ),
        needs_user_confirmation=_unique(
            [
                *intent.needs_user_confirmation,
                *_list_of_strings(data.get("needs_user_confirmation")),
            ]
        ),
        llm_metadata={
            "domain_pack_domain": pack.domain_name,
            "seed_hint_count": len(seed_hints),
        },
    ), []


def _concept_from_llm_item(
    item: Any,
    index: int,
) -> tuple[IntentConcept | None, list[str]]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return None, [f"structured_concepts[{index}]_not_object"]
    required = [
        "term",
        "category",
        "source",
        "confidence",
        "activation_reason",
        "query_role",
        "should_use_in_provider_query",
    ]
    for key in required:
        if key not in item:
            errors.append(f"structured_concepts[{index}].missing:{key}")
    if errors:
        return None, errors
    category = str(item.get("category") or "")
    source = str(item.get("source") or "")
    query_role = str(item.get("query_role") or "")
    confidence = _safe_float(item.get("confidence"), default=-1.0)
    if category not in ALLOWED_CONCEPT_CATEGORIES:
        errors.append(f"structured_concepts[{index}].invalid_category:{category}")
    if source not in ALLOWED_CONCEPT_SOURCES:
        errors.append(f"structured_concepts[{index}].invalid_source:{source}")
    if query_role not in ALLOWED_QUERY_ROLES:
        errors.append(f"structured_concepts[{index}].invalid_query_role:{query_role}")
    if confidence < 0 or confidence > 1:
        errors.append(f"structured_concepts[{index}].confidence_out_of_range")
    if not isinstance(item.get("should_use_in_provider_query"), bool):
        errors.append(f"structured_concepts[{index}].should_use_not_bool")
    term = " ".join(str(item.get("term") or "").split())
    if not term:
        errors.append(f"structured_concepts[{index}].empty_term")
    if errors:
        return None, errors
    return (
        _concept(
            term,
            category,
            source,
            confidence,
            str(item.get("activation_reason") or ""),
            query_role,
            bool(item.get("should_use_in_provider_query")),
        ),
        [],
    )


def _validate_llm_intent_with_domain_pack(
    intent: ExpertResearchIntent,
    question: str,
    seed_hints: list[SeedHint],
    pack: DomainPack,
) -> ExpertResearchIntent:
    support_text = _context(question, seed_hints)
    domain_terms = {term.lower() for term in _domain_pack_terms(pack)}
    has_active_pack = pack.domain_name not in {"generic", "general_science"}
    events: list[dict[str, Any]] = []
    validated: list[IntentConcept] = []
    for concept in intent.structured_concepts:
        term_lower = concept.term.lower()
        direct_support = term_lower in support_text
        domain_support = term_lower in domain_terms
        explicit_user_term = concept.source == "user_text" and concept.confidence >= 0.8
        if concept.query_role == "downweighted" or is_generic_noisy_term(concept.term):
            lowered = replace(
                concept,
                query_role="downweighted",
                should_use_in_provider_query=False,
            )
            validated.append(lowered)
            events.append(
                _validation_event(
                    concept,
                    lowered,
                    "generic_term_downweighted",
                )
            )
            continue
        event_reason = ""
        cross_domain_sensitive = _is_cross_domain_sensitive_term(concept.term, pack.domain_name)
        unsupported = not direct_support and not domain_support
        if cross_domain_sensitive and not direct_support:
            event_reason = "cross_domain_injection_blocked"
        elif explicit_user_term and unsupported:
            retained = replace(
                concept,
                should_use_in_provider_query=concept.query_role in {"must", "optional"},
            )
            validated.append(retained)
            events.append(
                _validation_event(
                    concept,
                    retained,
                    "unsupported_domain_pack_but_retained"
                    if has_active_pack
                    else "retained_explicit_user_term",
                )
            )
            continue
        elif (
            concept.source == "llm_inferred"
            and concept.confidence < 0.8
            and unsupported
        ):
            event_reason = "low_confidence_llm_inference_demoted"
        elif concept.source == "llm_inferred" and unsupported and has_active_pack:
            event_reason = "unsupported_domain_pack_and_demoted"
        elif concept.source == "llm_inferred" and unsupported and not has_active_pack:
            optional = replace(
                concept,
                query_role="optional",
                should_use_in_provider_query=False,
                confidence=min(concept.confidence, 0.68),
                activation_reason=(
                    f"{concept.activation_reason} "
                    "[Domain validation retained only as a non-query assumption]."
                ),
            )
            validated.append(optional)
            events.append(
                _validation_event(
                    concept,
                    optional,
                    "unsupported_domain_pack_and_demoted",
                )
            )
            continue
        elif unsupported and concept.source == "domain_pack" and not domain_support:
            event_reason = "forbidden_by_active_domain"
        if event_reason:
            lowered = replace(
                concept,
                query_role="uncertain",
                should_use_in_provider_query=False,
                confidence=min(concept.confidence, 0.49),
                activation_reason=(
                    f"{concept.activation_reason} "
                    f"[Domain validation downgraded: {event_reason}]."
                ),
            )
            validated.append(lowered)
            events.append(_validation_event(concept, lowered, event_reason))
            continue
        if concept.query_role == "uncertain":
            validated.append(replace(concept, should_use_in_provider_query=False))
        elif concept.query_role == "exclude":
            validated.append(replace(concept, should_use_in_provider_query=False))
        else:
            validated.append(concept)
    validated = _merge_concepts(validated)
    usable = [
        concept
        for concept in validated
        if concept.should_use_in_provider_query
        and concept.query_role in {"must", "optional"}
    ]
    return replace(
        intent,
        structured_concepts=validated,
        target_objects=_terms_by_category(usable, {"object"}),
        target_properties=_terms_by_category(usable, {"property"}),
        mechanisms=_terms_by_category(usable, {"mechanism"}),
        materials=_terms_by_category(usable, {"material"}),
        methods=_terms_by_category(usable, {"method"}),
        applications=_terms_by_category(usable, {"application"}),
        llm_metadata={
            **intent.llm_metadata,
            "domain_validation_events": events,
        },
    )


def _domain_pack_terms(pack: DomainPack) -> list[str]:
    terms = [pack.domain_name]
    terms.extend(pack.domain_anchors)
    for concept in pack.concepts.values():
        terms.extend(concept.synonyms)
        terms.extend(concept.related)
    terms.extend(pack.mechanisms)
    terms.extend(pack.materials)
    terms.extend(pack.methods)
    terms.extend(pack.applications)
    if pack.domain_name == "ai_literature_screening":
        terms.extend(
            [
                "LLM",
                "large language model",
                "literature screening",
                "human-in-the-loop",
                "multi-agent system",
                "evidence verification",
                "scientific literature",
            ]
        )
    if pack.domain_name == "generic":
        terms.extend(["scientific research topic"])
    return _unique(terms)


def _is_cross_domain_sensitive_term(term: str, domain_name: str) -> bool:
    """Block LLM-only concepts borrowed from non-selected domain packs."""

    lowered = " ".join(str(term or "").lower().split())
    if not lowered:
        return False
    for pack_name in list_domain_packs():
        if pack_name == domain_name:
            continue
        try:
            pack = load_domain_pack(pack_name)
        except ValueError:
            continue
        if lowered in {value.lower() for value in _domain_pack_terms(pack)}:
            return True
    return False


def _validation_event(
    original: IntentConcept,
    updated: IntentConcept,
    reason: str,
) -> dict[str, Any]:
    return {
        "term": original.term,
        "original_query_role": original.query_role,
        "new_query_role": updated.query_role,
        "original_should_use_in_provider_query": original.should_use_in_provider_query,
        "new_should_use_in_provider_query": updated.should_use_in_provider_query,
        "reason": reason,
        "source": original.source,
        "confidence": original.confidence,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _infer_domain_pack(question: str, seed_hints: list[SeedHint]) -> DomainPack:
    context = _context(question, seed_hints)
    routed = DomainRouter().route(context)
    if routed.selected_domain != "general_science":
        try:
            return load_domain_pack(routed.selected_domain)
        except ValueError:
            pass
    return DomainPack(domain_name="generic")


def _repair_materials_magnetism(
    question: str,
    seed_hints: list[SeedHint],
    pack: DomainPack,
) -> ExpertResearchIntent:
    context = _context(question, seed_hints)
    seed_context = _seed_context(seed_hints)
    concepts: list[IntentConcept] = _downweighted_concepts(question)
    detection_goal = _has_any(context, DETECTION_MARKERS)
    has_surface_magnetization = _has_any(
        context,
        ["surface magnetization", "surface magnetic moment", "表面磁化"],
    )
    has_antiferro_or_me = _has_any(
        context,
        [
            "antiferromagnet",
            "antiferromagnetic",
            "反铁磁",
            "magnetoelectric",
            "磁电",
        ],
    )
    has_local_me = _has_any(
        context,
        [
            "local magnetoelectric effects",
            "local magnetoelectric effect",
            "local magnetoelectric response",
            "surface magnetic order",
        ],
    )
    has_spaldin_seed = "spaldin" in context or _has_any(
        seed_context,
        [
            "surface magnetization in antiferromagnets",
            "local magnetoelectric effects",
        ],
    )
    has_surface_classification_seed = _has_any(
        seed_context,
        ["surface magnetization in antiferromagnets", "classification", "example materials"],
    )
    has_altermagnetism = _has_any(context, ["altermagnetism", "altermagnet"])

    if has_surface_magnetization:
        concepts.append(
            _concept(
                "surface magnetization",
                "object",
                "user_text",
                0.94,
                "User or seed mentions surface magnetization.",
                "must",
                True,
            )
        )
    if _has_any(context, ["surface spin polarization", "spin polarization", "自旋极化"]):
        concepts.append(
            _concept(
                "surface spin polarization",
                "property",
                "user_text",
                0.9,
                "User asks about surface spin polarization rather than generic spin.",
                "must",
                True,
            )
        )
    if _has_any(context, ["antiferromagnet", "antiferromagnetic", "反铁磁"]):
        concepts.append(
            _concept(
                "antiferromagnet",
                "object",
                "user_text",
                0.88,
                "User or seed anchors the topic in antiferromagnets.",
                "must",
                True,
            )
        )
    if _has_any(context, ["magnetoelectric", "磁电"]):
        concepts.append(
            _concept(
                "magnetoelectric antiferromagnet",
                "mechanism",
                "user_text",
                0.86,
                "Magnetoelectric wording narrows the magnetic-material mechanism.",
                "must",
                True,
            )
        )
    if has_local_me:
        concepts.extend(
            [
                _concept(
                    _pack_synonym(pack, "local_magnetoelectricity", "local magnetoelectric response"),
                    "mechanism",
                    "domain_pack",
                    0.9,
                    "Local magnetoelectric seed/user wording activates the local-response framework.",
                    "must",
                    True,
                ),
                _concept(
                    "surface magnetic order",
                    "property",
                    "seed_title" if "surface magnetic order" in seed_context else "user_text",
                    0.86,
                    "Surface magnetic order is explicit in the local magnetoelectric seed or user text.",
                    "must",
                    True,
                ),
            ]
        )
        multipole = _pack_synonym(pack, "local_magnetoelectricity", "magnetic multipole")
        if multipole:
            concepts.append(
                _concept(
                    multipole,
                    "mechanism",
                    "domain_pack",
                    0.78,
                    "Local magnetoelectric response often uses magnetic multipole descriptors.",
                    "optional",
                    True,
                )
            )
    if has_surface_magnetization and (has_antiferro_or_me or has_spaldin_seed):
        boundary = _pack_synonym(pack, "surface_magnetization", "boundary magnetization")
        if boundary:
            concepts.append(
                _concept(
                    boundary,
                    "object",
                    "domain_pack",
                    0.88,
                    "Surface magnetization plus antiferromagnet/magnetoelectric/seed context activates the boundary-magnetization synonym.",
                    "must",
                    True,
                )
            )

    material_expansion_trigger = has_surface_classification_seed or _has_any(
        context,
        ["Cr2O3", "chromia"],
    )
    if material_expansion_trigger:
        for material in _pack_items(pack.materials, ["Cr2O3", "chromia", "FeF2", "NiO", "CuMnAs"]):
            role = "must" if material in {"Cr2O3", "chromia"} else "optional"
            concepts.append(
                _concept(
                    material,
                    "material",
                    "domain_pack",
                    0.84 if role == "must" else 0.72,
                    "Spaldin-style classification/example-material seed activates representative antiferromagnets.",
                    role,
                    True,
                )
            )
    else:
        for material in _pack_items(pack.materials, ["Cr2O3", "chromia", "FeF2", "NiO", "CuMnAs"]):
            if material.lower() in context:
                concepts.append(
                    _concept(
                        material,
                        "material",
                        "user_text",
                        0.9,
                        "Material appears explicitly in user text or seed metadata.",
                        "must",
                        True,
                    )
                )

    material_context = material_expansion_trigger or _has_any(
        context,
        ["Cr2O3", "chromia", "magnetoelectric antiferromagnet"],
    )
    if detection_goal and (has_surface_magnetization or has_antiferro_or_me or has_altermagnetism):
        for method in _activated_magnetic_probe_methods(pack, material_context, has_altermagnetism):
            concepts.append(method)

    if has_altermagnetism:
        concepts.extend(
            [
                _concept(
                    "altermagnetism",
                    "object",
                    "user_text",
                    0.94,
                    "User explicitly asks about altermagnetism.",
                    "must",
                    True,
                ),
                _concept(
                    "spin splitting",
                    "property",
                    "domain_pack",
                    0.82,
                    "Altermagnetism is commonly detected through spin splitting.",
                    "optional",
                    True,
                ),
            ]
        )
        if detection_goal:
            concepts.append(
                _concept(
                    "SARPES",
                    "method",
                    "domain_pack",
                    0.82,
                    "Experimental altermagnetism detection activates spin-resolved ARPES/SARPES routes.",
                    "optional",
                    True,
                )
            )

    concepts.extend(_seed_title_concepts(seed_hints))
    concepts = _merge_concepts(concepts)
    return _intent_from_concepts(
        question=question,
        concepts=concepts,
        inferred_goal=_materials_goal(concepts),
        expert_rewritten_question=_materials_rewrite(concepts, has_spaldin_seed, has_altermagnetism),
        assumptions=[
            "Treat broad motivation words as context, not hard retrieval terms.",
            "Only activate domain-pack concepts when user text, seed titles, or explicit detection goals trigger them.",
            "Do not infer citation relations without citation-expansion evidence.",
        ],
        confidence=0.86 if has_spaldin_seed else 0.74,
    )


def _repair_ferroelectric_polarization(
    question: str,
    pack: DomainPack,
) -> ExpertResearchIntent:
    context = question.lower()
    concepts = _downweighted_concepts(question)
    concepts.extend(
        [
            _concept(
                "ferroelectric polarization",
                "property",
                "user_text",
                0.93,
                "User explicitly asks about ferroelectric polarization.",
                "must",
                True,
            ),
            _concept(
                "surface polarization",
                "property",
                "user_text",
                0.88,
                "User asks about surface polarization in thin films.",
                "must",
                True,
            ),
        ]
    )
    if _has_any(context, DETECTION_MARKERS):
        concepts.extend(
            [
                _concept("PFM", "method", "domain_pack", 0.82, "Probe/detection wording activates standard ferroelectric polarization microscopy.", "optional", True),
                _concept("piezoresponse force microscopy", "method", "domain_pack", 0.82, "PFM should be expanded for provider queries.", "optional", True),
                _concept("SHG", "method", "domain_pack", 0.74, "Probe/detection wording activates nonlinear optical polarization probes.", "optional", True),
                _concept("second harmonic generation", "method", "domain_pack", 0.74, "SHG should be expanded for provider queries.", "optional", True),
                _concept("KPFM", "method", "domain_pack", 0.7, "Surface-potential wording can be probed by Kelvin probe force microscopy.", "optional", True),
            ]
        )
    concepts.extend(
        [
            _concept("depolarization field", "mechanism", "domain_pack", 0.76, "Ferroelectric thin-film surface polarization commonly involves depolarization fields.", "optional", True),
            _concept("screening", "mechanism", "domain_pack", 0.7, "Surface polarization context can involve charge screening; keep it optional.", "optional", True),
            _concept("interface screening", "mechanism", "domain_pack", 0.72, "Surface/interface wording activates interface screening as a mechanism.", "optional", True),
        ]
    )
    if _has_any(context, ["material", "材料", "case", "案例", "thin film", "薄膜"]):
        for material in _pack_items(pack.materials, ["BaTiO3", "PZT", "PbTiO3", "BiFeO3", "HfO2", "HfZrO2", "LiNbO3"]):
            concepts.append(
                _concept(
                    material,
                    "material",
                    "domain_pack",
                    0.7,
                    "User asks for typical material cases; keep representative ferroelectrics optional.",
                    "optional",
                    True,
                )
            )
    if _has_any(context, ["application", "应用", "device", "器件", "memory", "important", "重要"]):
        for application in _pack_items(pack.applications, ["ferroelectric memory", "ferroelectric tunnel junction", "FeFET", "nonvolatile memory"]):
            concepts.append(
                _concept(
                    application,
                    "application",
                    "domain_pack",
                    0.68,
                    "Importance/application wording activates device motivation as optional context.",
                    "optional",
                    True,
                )
            )
    return _intent_from_concepts(
        question=question,
        concepts=_merge_concepts(concepts),
        inferred_goal="Map why surface or ferroelectric polarization matters in thin films and how it can be measured.",
        expert_rewritten_question=(
            "Find theoretical and experimental papers explaining why surface or "
            "ferroelectric polarization matters in thin films and which probes can measure it."
        ),
        assumptions=[
            "This is a ferroelectric-polarization question, not an antiferromagnetic surface-magnetization question.",
            "Magnetic probe methods should not be injected unless magnetic wording appears.",
        ],
        confidence=0.78,
    )


def _repair_ai_literature_screening(question: str) -> ExpertResearchIntent:
    concepts = _downweighted_concepts(question)
    concepts.extend(
        [
            _concept("LLM", "mechanism", "user_text", 0.92, "User explicitly asks about LLM-based systems.", "must", True),
            _concept("literature screening", "object", "user_text", 0.94, "User asks about literature screening.", "must", True),
            _concept("human-in-the-loop", "mechanism", "user_text", 0.86, "User asks about human collaboration.", "must", True),
            _concept("multi-agent system", "mechanism", "domain_pack", 0.7, "LLM screening systems may use multi-agent workflow; keep optional unless explicit.", "optional", True),
            _concept("evidence verification", "application", "domain_pack", 0.68, "Literature screening often needs evidence verification.", "optional", True),
        ]
    )
    return _intent_from_concepts(
        question=question,
        concepts=_merge_concepts(concepts),
        inferred_goal="Find AI and human-in-the-loop papers about LLM-assisted literature screening.",
        expert_rewritten_question=(
            "Find papers on human-in-the-loop LLM systems for scientific literature "
            "screening, including study selection, evidence verification, and collaboration workflows."
        ),
        assumptions=[
            "This is an AI literature-screening question, not a materials-magnetism question.",
        ],
        confidence=0.82,
    )


def _fallback_repair(question: str) -> ExpertResearchIntent:
    concepts = _downweighted_concepts(question)
    concepts.extend(_sei_lithium_normalized_concepts(question))
    frame = build_generic_intent_frame(question)
    for term in frame.research_object:
        concepts.append(
            _concept(
                term,
                "object",
                "user_text",
                0.86,
                "Explicit or translated research object appears in the user question.",
                "must",
                True,
            )
        )
    for term in frame.domain_context:
        concepts.append(
            _concept(
                term,
                "material",
                "user_text",
                0.82,
                "Explicit or translated domain context appears in the user question.",
                "optional",
                True,
            )
        )
    for term in frame.target_process_or_property:
        concepts.append(
            _concept(
                term,
                "property",
                "user_text",
                0.82,
                "Explicit or translated target property/process appears in the user question.",
                "optional",
                True,
            )
        )
    for term in frame.method_terms:
        concepts.append(
            _concept(
                term,
                "method",
                "user_text",
                0.8,
                "User asks for methods, characterization, workflow, or measurement evidence.",
                "optional",
                True,
            )
        )
    for term in frame.mechanism_terms:
        concepts.append(
            _concept(
                term,
                "mechanism",
                "user_text",
                0.8,
                "User asks for theory or mechanism evidence.",
                "optional",
                True,
            )
        )
    existing_terms = {concept.term.lower() for concept in concepts}
    for term, sources in frame.term_sources.items():
        if term.lower() in existing_terms or is_generic_noisy_term(term):
            continue
        if not ({"generic_glossary", "user_text"} & set(sources)):
            continue
        concepts.append(
            _concept(
                term,
                "object",
                "user_text",
                0.78,
                "Additional explicit or glossary-supported scientific term from the user question.",
                "optional",
                True,
            )
        )
        existing_terms.add(term.lower())
    for phrase in _candidate_user_phrases(question):
        if phrase.lower() not in {concept.term.lower() for concept in concepts}:
            concepts.append(
                _concept(
                    phrase,
                    "object",
                    "user_text",
                    0.72,
                    "Candidate topic phrase appears in the user question.",
                    "optional",
                    True,
                )
            )
    return _intent_from_concepts(
        question=question,
        concepts=_merge_concepts(concepts),
        inferred_goal="Find papers aligned with the user's research question.",
        expert_rewritten_question=_fallback_rewrite(question, concepts),
        assumptions=[
            "No domain-specific intent repair rules matched.",
            "Do not treat broad motivation words as hard retrieval requirements.",
        ],
        confidence=0.55,
    )


def _sei_lithium_normalized_concepts(question: str) -> list[IntentConcept]:
    """Normalize explicit Chinese lithium/SEI wording without changing query generation."""

    lowered = question.lower()
    concepts: list[IntentConcept] = []
    has_sei = _has_any(
        lowered,
        ["sei", "solid electrolyte interphase", "solid-electrolyte interphase"],
    )
    has_chinese_sei = "界面" in question or "固态电解质界面" in question
    if has_sei or has_chinese_sei:
        concepts.append(
            _concept(
                "solid electrolyte interphase",
                "object",
                "user_text",
                0.94,
                "User explicitly mentions SEI or an electrolyte interphase/interface.",
                "must",
                True,
            )
        )
    if "锂电池" in question or _has_any(lowered, ["lithium battery", "lithium-ion battery", "li-ion battery"]):
        concepts.extend(
            [
                _concept(
                    "lithium battery",
                    "material",
                    "user_text",
                    0.93,
                    "Chinese 锂电池 maps to lithium battery context.",
                    "must",
                    True,
                ),
                _concept(
                    "lithium-ion battery",
                    "material",
                    "user_text",
                    0.88,
                    "Lithium battery wording activates lithium-ion battery context for SEI screening.",
                    "must",
                    True,
                ),
            ]
        )
    if "锂金属电池" in question or _has_any(lowered, ["lithium metal battery", "lithium metal anode", "li metal"]):
        concepts.extend(
            [
                _concept(
                    "lithium metal battery",
                    "material",
                    "user_text",
                    0.95,
                    "Chinese 锂金属电池 maps to lithium metal battery context.",
                    "must",
                    True,
                ),
                _concept(
                    "lithium metal anode",
                    "material",
                    "user_text",
                    0.92,
                    "Lithium metal battery wording implies lithium metal anode context.",
                    "must",
                    True,
                ),
            ]
        )
    if "人工 sei" in lowered or "人工sei" in lowered or _has_any(lowered, ["artificial sei", "engineered sei", "artificial solid electrolyte interphase"]):
        concepts.extend(
            [
                _concept(
                    "artificial SEI",
                    "method",
                    "user_text",
                    0.94,
                    "Chinese 人工 SEI maps to artificial SEI.",
                    "must",
                    True,
                ),
                _concept(
                    "engineered SEI",
                    "method",
                    "user_text",
                    0.86,
                    "Artificial SEI wording activates engineered SEI terminology.",
                    "optional",
                    True,
                ),
                _concept(
                    "artificial solid electrolyte interphase",
                    "method",
                    "user_text",
                    0.86,
                    "Artificial SEI wording expands to the full technical phrase.",
                    "optional",
                    True,
                ),
            ]
        )
    if "枝晶" in question or _has_any(lowered, ["dendrite", "lithium dendrite"]):
        concepts.append(
            _concept(
                "lithium dendrite",
                "mechanism",
                "user_text",
                0.88,
                "Chinese 枝晶 in a lithium battery SEI question maps to lithium dendrite mechanisms.",
                "optional",
                True,
            )
        )
    return concepts


def _intent_from_concepts(
    question: str,
    concepts: list[IntentConcept],
    inferred_goal: str,
    expert_rewritten_question: str,
    assumptions: list[str],
    confidence: float,
) -> ExpertResearchIntent:
    downweighted = [
        concept.term
        for concept in concepts
        if concept.query_role == "downweighted"
    ]
    usable = [
        concept
        for concept in concepts
        if concept.should_use_in_provider_query
        and concept.query_role in {"must", "optional"}
    ]
    likely_misconceptions = [
        "Broad words such as importance, background, review, spin, or magnetization should guide interpretation but not dominate provider queries."
    ]
    ambiguity = _intent_ambiguity_profile(question, concepts)
    return ExpertResearchIntent(
        original_question=question,
        user_is_novice=_looks_novice(question, downweighted),
        inferred_goal=inferred_goal,
        expert_rewritten_question=expert_rewritten_question,
        structured_concepts=concepts,
        target_objects=_terms_by_category(usable, {"object"}),
        target_properties=_terms_by_category(usable, {"property"}),
        mechanisms=_terms_by_category(usable, {"mechanism"}),
        materials=_terms_by_category(usable, {"material"}),
        methods=_terms_by_category(usable, {"method"}),
        applications=_terms_by_category(usable, {"application"}),
        likely_user_misconceptions=likely_misconceptions,
        ignored_or_downweighted_terms=downweighted,
        must_not_overinterpret=[
            "Do not infer citation relations unless citation/snowballing artifacts verify them.",
            "Do not inject unrelated domain-pack terms without an activation rule.",
            *ambiguity["unsafe_or_overbroad_assumptions"],
        ],
        ambiguity_points=ambiguity["ambiguity_points"],
        possible_interpretations=ambiguity["possible_interpretations"],
        selected_interpretation=ambiguity["selected_interpretation"],
        selected_interpretation_reason=ambiguity["selected_interpretation_reason"],
        needs_user_confirmation=ambiguity["needs_user_confirmation"],
        unsafe_or_overbroad_assumptions=ambiguity["unsafe_or_overbroad_assumptions"],
        confidence=confidence,
        assumptions=_unique([*assumptions, *ambiguity["assumptions"]]),
    )


def _intent_ambiguity_profile(
    question: str,
    concepts: list[IntentConcept],
) -> dict[str, list[str] | str]:
    lowered = question.lower()
    terms = {concept.term.lower() for concept in concepts}
    ambiguity_points: list[str] = []
    possible_interpretations: list[str] = []
    needs_confirmation: list[str] = []
    unsafe: list[str] = []
    assumptions: list[str] = []
    selected = ""
    selected_reason = ""

    if "相关" in question or "related" in lowered:
        ambiguity_points.append(
            "The phrase 'related papers' is ambiguous: it may mean direct citation, same author, topical similarity, method overlap, application relevance, or theory lineage."
        )
        possible_interpretations.extend(
            [
                "direct citation or same-author related papers",
                "topic-related papers",
                "method-related papers",
                "application-related papers",
                "theory-lineage related papers",
            ]
        )
        selected = "topic + method + theory-lineage related papers"
        selected_reason = (
            "The question mentions research concepts and probe goals, but no verified citation relation; topical, methodological, and theory-lineage relevance is the safest default."
        )
        needs_confirmation.append(
            "Confirm whether 'related' should mean direct citation/same-author papers, or broader topical/method/theory-lineage relevance."
        )
        unsafe.append(
            "Do not treat 'related papers' as verified citation relation or same-author relation without seed/citation evidence."
        )
        assumptions.append(
            "Assume 'related papers' means topic + method + theory-lineage relevance unless the user asks for citation or author-only expansion."
        )

    if _has_any(lowered, ["importance", "significance", "重要", "意义"]):
        ambiguity_points.append(
            "The motivation word 'importance/significance' may ask for background, evidence papers, application motivation, or device relevance."
        )
        possible_interpretations.extend(
            [
                "background or review papers",
                "evidence papers that demonstrate why the effect matters",
                "application or device-motivation papers",
            ]
        )
        needs_confirmation.append(
            "Confirm whether the user wants broad background, evidence papers, application motivation, or all three."
        )
        unsafe.append(
            "Do not collapse 'importance' into only review/survey/tutorial queries."
        )
        assumptions.append(
            "Treat 'importance' as a request for background plus evidence and application/device motivation, not only review articles."
        )

    if _has_any(lowered, DETECTION_MARKERS):
        ambiguity_points.append(
            "Detection/probing wording could mean direct experimental probes, indirect readout methods, or surface-sensitive techniques."
        )
        possible_interpretations.extend(
            [
                "direct experimental probes",
                "indirect readout methods",
                "surface-sensitive techniques",
            ]
        )
        needs_confirmation.append(
            "Confirm whether detection should prioritize direct imaging/probing, indirect readout, or both."
        )
        assumptions.append(
            "Treat detection/probing as a method-and-evidence lane; concrete methods must come from the domain pack or validated LLM concepts."
        )

    if _has_any(lowered, ["spin polarization", "自旋极化"]):
        ambiguity_points.append(
            "Spin polarization may refer to surface spin polarization, spin-resolved electronic structure, spin-polarized probe methods, or magnetic moment/surface magnetization."
        )
        possible_interpretations.extend(
            [
                "surface spin polarization",
                "spin-resolved electronic structure",
                "spin-polarized probe method",
                "magnetic moment or surface magnetization",
            ]
        )
        needs_confirmation.append(
            "Confirm whether spin polarization means a surface signal, electronic-structure spin splitting, a probe method, or magnetic moment."
        )
        if "surface spin polarization" in terms:
            selected = selected or "surface spin polarization with method evidence"
            selected_reason = selected_reason or (
                "The question links spin polarization to surface/detection language, so surface spin polarization with evidence methods is the default interpretation."
            )
        assumptions.append(
            "Interpret spin polarization as surface spin polarization plus possible spin-resolved electronic/probe evidence, not automatically as bulk magnetization."
        )

    return {
        "ambiguity_points": _unique(ambiguity_points),
        "possible_interpretations": _unique(possible_interpretations),
        "selected_interpretation": selected,
        "selected_interpretation_reason": selected_reason,
        "needs_user_confirmation": _unique(needs_confirmation),
        "unsafe_or_overbroad_assumptions": _unique(unsafe),
        "assumptions": _unique(assumptions),
    }


def _activated_magnetic_probe_methods(
    pack: DomainPack,
    material_context: bool,
    has_altermagnetism: bool,
) -> list[IntentConcept]:
    methods: list[IntentConcept] = []
    if has_altermagnetism:
        for term in ["SARPES", "spin-resolved photoemission"]:
            if term in pack.methods or term == "spin-resolved photoemission":
                methods.append(
                    _concept(
                        term,
                        "method",
                        "domain_pack",
                        0.78,
                        "Altermagnetism detection wording activates spin-resolved photoemission routes.",
                        "optional",
                        True,
                    )
                )
        return methods
    method_terms = ["SPLEEM", "XMCD-PEEM", "SP-STM", "spin-resolved photoemission"]
    if material_context:
        method_terms.append("NV magnetometry")
    for term in method_terms:
        if term in pack.methods:
            methods.append(
                _concept(
                    term,
                    "method",
                    "domain_pack",
                    0.82 if term in {"SPLEEM", "XMCD-PEEM"} else 0.76,
                    "Surface-magnetization detection wording activates magnetic surface-probe methods.",
                    "optional",
                    True,
                )
            )
    return methods


def _seed_title_concepts(seed_hints: list[SeedHint]) -> list[IntentConcept]:
    concepts: list[IntentConcept] = []
    for hint in seed_hints:
        if hint.title:
            concepts.append(
                _concept(
                    hint.title,
                    "object",
                    "seed_title",
                    min(0.95, hint.confidence or 0.8),
                    "User explicitly mentioned this seed-paper title.",
                    "optional",
                    False,
                )
            )
        if hint.doi:
            concepts.append(
                _concept(
                    hint.doi,
                    "object",
                    "seed_metadata",
                    min(0.95, hint.confidence or 0.8),
                    "DOI appears in explicit seed-paper metadata.",
                    "optional",
                    False,
                )
            )
        if hint.arxiv_id:
            concepts.append(
                _concept(
                    hint.arxiv_id,
                    "object",
                    "seed_metadata",
                    min(0.9, hint.confidence or 0.75),
                    "arXiv ID appears in explicit seed-paper metadata.",
                    "optional",
                    False,
                )
            )
        for author in hint.authors:
            concepts.append(
                _concept(
                    author,
                    "object",
                    "seed_metadata",
                    0.75,
                    "Author appears in seed metadata or user text.",
                    "optional",
                    False,
                )
            )
    return concepts


def _downweighted_concepts(question: str) -> list[IntentConcept]:
    values: list[str] = []
    lowered = question.lower()
    for term in GENERIC_DOWNWEIGHT_TERMS:
        if term in lowered:
            values.append(term)
    if "重要" in question or "意义" in question:
        values.extend(["importance", "significance"])
    if "综述" in question:
        values.append("review")
    return [
        _concept(
            term,
            "generic_user_term",
            "user_text",
            0.95,
            "Generic novice wording helps interpret motivation but should not drive provider queries.",
            "downweighted",
            False,
        )
        for term in _unique(values)
    ]


def _concept(
    term: str,
    category: str,
    source: str,
    confidence: float,
    activation_reason: str,
    query_role: str,
    should_use_in_provider_query: bool,
) -> IntentConcept:
    return IntentConcept(
        term=" ".join(str(term or "").split()),
        category=category,
        source=source,
        confidence=round(float(confidence), 4),
        activation_reason=activation_reason,
        query_role=query_role,
        should_use_in_provider_query=bool(should_use_in_provider_query),
    )


def _merge_concepts(concepts: list[IntentConcept]) -> list[IntentConcept]:
    by_term: dict[str, IntentConcept] = {}
    role_rank = {"must": 4, "optional": 3, "uncertain": 2, "downweighted": 1, "exclude": 0}
    for concept in concepts:
        if not concept.term:
            continue
        key = concept.term.lower()
        existing = by_term.get(key)
        if existing is None:
            by_term[key] = concept
            continue
        if (
            role_rank.get(concept.query_role, 0) > role_rank.get(existing.query_role, 0)
            or concept.confidence > existing.confidence
        ):
            by_term[key] = replace(
                concept,
                activation_reason=f"{existing.activation_reason}; {concept.activation_reason}",
            )
    return list(by_term.values())


def _terms_by_category(concepts: list[IntentConcept], categories: set[str]) -> list[str]:
    return _unique([concept.term for concept in concepts if concept.category in categories])


def _context(question: str, seed_hints: list[SeedHint]) -> str:
    return " ".join(
        [
            question,
            _seed_context(seed_hints),
            " ".join(" ".join(hint.authors) for hint in seed_hints),
        ]
    ).lower()


def _seed_context(seed_hints: list[SeedHint]) -> str:
    return " ".join(str(hint.title or "") for hint in seed_hints).lower()


def _has_any(text: str, markers: list[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers if marker)


def _pack_synonym(pack: DomainPack, concept_name: str, preferred: str) -> str:
    concept = pack.concepts.get(concept_name)
    if not concept:
        return ""
    for term in [*concept.synonyms, *concept.related]:
        if term.lower() == preferred.lower():
            return term
    for term in [*concept.synonyms, *concept.related]:
        if preferred.lower() in term.lower() or term.lower() in preferred.lower():
            return term
    return ""


def _pack_items(items: list[str], wanted: list[str]) -> list[str]:
    available = {item.lower(): item for item in items}
    result: list[str] = []
    for term in wanted:
        value = available.get(term.lower())
        if value:
            result.append(value)
    return result


def _materials_goal(concepts: list[IntentConcept]) -> str:
    if any(concept.term == "altermagnetism" for concept in concepts):
        return "Find papers explaining experimental detection routes for altermagnetism."
    return (
        "Build a domain-aware literature map around antiferromagnetic surface "
        "magnetization, spin polarization detection, and local magnetoelectric predictors."
    )


def _materials_rewrite(
    concepts: list[IntentConcept],
    has_spaldin_seed: bool,
    has_altermagnetism: bool,
) -> str:
    if has_altermagnetism:
        return (
            "Find theoretical and experimental papers explaining how altermagnetism "
            "can be detected, especially through spin splitting and spin-resolved probes."
        )
    if has_spaldin_seed:
        return (
            "Find theoretical, experimental, and methodological papers explaining why "
            "probing surface magnetization and surface spin polarization is important "
            "in antiferromagnets, especially in relation to Spaldin's surface "
            "magnetization classification and local magnetoelectric response framework."
        )
    active_terms = [
        concept.term
        for concept in concepts
        if concept.should_use_in_provider_query and concept.query_role == "must"
    ]
    if active_terms:
        return "Find papers explaining " + ", ".join(active_terms[:4]) + "."
    return "Find papers aligned with the materials-magnetism research question."


def _candidate_user_phrases(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question)
    phrases: list[str] = []
    for size in (3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase = " ".join(words[index : index + size])
            if phrase.lower() not in set(GENERIC_DOWNWEIGHT_TERMS):
                phrases.append(phrase)
    return _unique(phrases[:4])


def _fallback_rewrite(question: str, concepts: list[IntentConcept]) -> str:
    usable = [
        concept.term
        for concept in concepts
        if concept.should_use_in_provider_query and concept.query_role in {"must", "optional"}
    ]
    if usable:
        return "Find papers about " + ", ".join(usable[:4]) + "."
    return question


def _looks_novice(question: str, downweighted_terms: list[str]) -> bool:
    novice_markers = ["有没有", "相关的", "重要性", "文章", "background", "importance", "我想"]
    return bool(downweighted_terms) or any(marker in question for marker in novice_markers)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value or "").split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
