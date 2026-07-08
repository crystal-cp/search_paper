"""Search-contract construction for intent-aware retrieval."""

from __future__ import annotations

from typing import Any

from lit_screening.agents.domain_router import DomainRouter
from lit_screening.agents.generic_intent import (
    build_generic_intent_frame,
    generic_aspect_groups,
)
from lit_screening.domain_packs import load_domain_pack
from lit_screening.llm_client import GenericLLMClient
from lit_screening.models import (
    DomainProfile,
    ExpertResearchIntent,
    GenericResearchIntentFrame,
    IntentConcept,
    SearchBrief,
    SearchConstraintGroup,
    SearchContract,
)
from lit_screening.utils import tokenize


AI_LITERATURE_TERMS = {
    "llm",
    "large language model",
    "multi-agent",
    "agentic",
    "human feedback",
    "literature screening",
    "claim extraction",
    "evidence verification",
    "scientific literature",
}

MATERIALS_MAGNETISM_TERMS = {
    "surface magnetization",
    "magnetization",
    "antiferromagnet",
    "antiferromagnetic",
    "magnetism",
    "spin",
    "surface spin",
    "magnetic material",
    "altermagnet",
}

BIOMEDICAL_TERMS = {
    "patient",
    "clinical",
    "drug",
    "biomarker",
    "disease",
    "diagnosis",
    "therapy",
}

FERROELECTRIC_TERMS = {
    "ferroelectric",
    "ferroelectricity",
    "ferroelectric thin film",
    "ferroelectric polarization",
    "surface polarization",
    "depolarization",
    "depolarization field",
    "screening charge",
    "interface screening",
    "pfm",
    "piezoresponse",
    "shg",
    "second harmonic generation",
    "铁电",
    "表面极化",
    "退极化",
    "屏蔽",
}

LITHIUM_TARGET_MARKERS = {
    "lithium battery",
    "lithium-ion battery",
    "lithium ion battery",
    "li-ion battery",
    "lithium metal battery",
    "lithium metal anode",
    "锂电池",
    "锂离子电池",
    "锂金属电池",
}

LITHIUM_TARGET_CHEMISTRY_TERMS = [
    "lithium",
    "lithium-ion",
    "Li-ion",
    "lithium metal",
    "lithium metal battery",
    "lithium metal anode",
    "lithium battery",
    "lithium-ion battery",
    "graphite anode",
    "silicon anode",
]

NON_TARGET_BATTERY_CHEMISTRY_TERMS = [
    "sodium",
    "sodium-ion",
    "Na-ion",
    "potassium",
    "potassium-ion",
    "K-ion",
    "zinc",
    "zinc-ion",
    "Zn",
    "AZIB",
    "magnesium",
    "aqueous zinc-ion",
    "beyond lithium-ion",
]


class SearchContractAgent:
    """Build a domain-aware SearchContract before query planning."""

    def __init__(
        self,
        mode: str = "rule",
        llm_client: GenericLLMClient | None = None,
    ) -> None:
        self.mode = mode
        self.llm_client = llm_client

    def build(
        self,
        question: str,
        search_brief: SearchBrief | None = None,
        ambiguity_analysis: list[dict[str, Any]] | None = None,
        expert_intent: ExpertResearchIntent | None = None,
    ) -> SearchContract:
        """Build a rule-based SearchContract with optional LLM left disabled."""

        fallback_brief = search_brief or _fallback_search_brief(question)
        ambiguity = ambiguity_analysis or []
        return self._build_rule(question, fallback_brief, ambiguity, expert_intent)

    def _build_rule(
        self,
        question: str,
        search_brief: SearchBrief,
        ambiguity_analysis: list[dict[str, Any]],
        expert_intent: ExpertResearchIntent | None = None,
    ) -> SearchContract:
        lowered = f"{question} {search_brief.refined_question}".lower()
        domain_profile = infer_domain_profile(
            lowered,
            domain_hint=_expert_domain_hint(expert_intent),
        )
        domain_pack = _load_pack_if_available(domain_profile.domain_name)
        generic_frame = build_generic_intent_frame(
            f"{question} {search_brief.refined_question}",
            expert_intent=expert_intent,
        )
        ambiguity_must = _flatten_ambiguity_terms(ambiguity_analysis, "recommended_must_terms")
        ambiguity_exclude = _flatten_ambiguity_terms(
            ambiguity_analysis,
            "recommended_exclude_terms",
        )
        if expert_intent is not None:
            must_include = _concept_terms_by_role(expert_intent.structured_concepts, "must")
            optional_concepts = _concept_terms_by_role(
                expert_intent.structured_concepts,
                "optional",
                require_query_use=False,
            )
            uncertain_concepts = _concept_terms_by_role(
                expert_intent.structured_concepts,
                "uncertain",
                require_query_use=False,
            )
            dropped_downweighted = _unique(
                [
                    *_concept_terms_by_role(
                        expert_intent.structured_concepts,
                        "downweighted",
                        require_query_use=False,
                    ),
                    *expert_intent.ignored_or_downweighted_terms,
                ],
                24,
            )
            pack_constraint_groups = _constraint_groups_from_domain_pack(domain_pack)
            intent_constraint_groups = _constraint_groups_from_intent(expert_intent)
            generic_constraint_groups = _constraint_groups_from_generic_frame(generic_frame)
            if any(group.required for group in pack_constraint_groups):
                intent_constraint_groups = _relaxed_constraint_groups(
                    intent_constraint_groups
                )
                generic_constraint_groups = _relaxed_constraint_groups(
                    generic_constraint_groups
                )
            constraint_groups = _unique_constraint_groups(
                [
                    *pack_constraint_groups,
                    *intent_constraint_groups,
                    *generic_constraint_groups,
                ]
            )
            assumptions = _unique(
                [
                    *expert_intent.assumptions,
                    *expert_intent.needs_user_confirmation,
                    *expert_intent.unsafe_or_overbroad_assumptions,
                ],
                24,
            )
        else:
            phrase_concepts = extract_question_concepts(search_brief.refined_question or question)
            must_include = _unique(
                [
                    *domain_profile.required_concepts,
                    *phrase_concepts,
                    *ambiguity_must,
                    *_specific_criteria(search_brief.inclusion_criteria),
                ],
                12,
            )
            optional_concepts = []
            uncertain_concepts = []
            dropped_downweighted = []
            pack_groups = _constraint_groups_from_domain_pack(domain_pack)
            generic_groups = _constraint_groups_from_generic_frame(generic_frame)
            fallback_groups = (
                []
                if any(group.required for group in [*pack_groups, *generic_groups])
                else _fallback_constraint_groups(must_include)
            )
            constraint_groups = _unique_constraint_groups(
                [*pack_groups, *generic_groups, *fallback_groups]
            )
            assumptions = []
        must_exclude = _unique(
            [
                *domain_profile.forbidden_concepts,
                *ambiguity_exclude,
                *_specific_criteria(search_brief.exclusion_criteria),
            ],
            16,
        )
        constraint_groups = _unique_constraint_groups(
            [
                *constraint_groups,
                *_lithium_target_chemistry_groups(
                    question,
                    search_brief,
                    expert_intent,
                ),
            ]
        )
        return SearchContract(
            original_question=search_brief.original_question or question,
            refined_question=search_brief.refined_question or question,
            user_goal=search_brief.user_goal,
            search_intent=search_brief.search_intent,
            domain_profile=domain_profile,
            must_include_concepts=must_include,
            must_exclude_concepts=must_exclude,
            optional_concepts=optional_concepts,
            uncertain_concepts=uncertain_concepts,
            dropped_downweighted_terms=dropped_downweighted,
            constraint_groups=constraint_groups,
            assumptions=assumptions,
            inclusion_criteria=search_brief.inclusion_criteria,
            exclusion_criteria=search_brief.exclusion_criteria,
            required_aspects=_unique(
                [
                    *_required_aspects_for_contract(domain_pack, search_brief, generic_frame),
                    *(must_include[:4] if expert_intent is not None else domain_profile.required_concepts[:4]),
                ],
                16,
            ),
            preferred_paper_types=search_brief.preferred_paper_types,
            time_window=search_brief.time_window,
            success_definition=search_brief.success_definition,
            generic_intent_frame=generic_frame,
            concept_validation_events=(expert_intent.llm_metadata or {}).get(
                "domain_validation_events",
                [],
            )
            if expert_intent is not None
            else [],
        )


def infer_domain_profile(lowered_text: str, domain_hint: str = "") -> DomainProfile:
    """Infer a coarse domain profile from user-visible terms."""

    routed = DomainRouter().route(lowered_text, domain_hint=domain_hint)
    if routed.selected_domain != "general_science":
        pack = _load_pack_if_available(routed.selected_domain)
        terms = _domain_pack_terms_for_profile(pack)
        return DomainProfile(
            domain_name=routed.selected_domain,
            positive_domains=[routed.selected_domain.replace("_", " ")],
            negative_domains=list(getattr(pack, "false_positive_terms", []) or []),
            required_concepts=_unique(terms[:8], 8),
            forbidden_concepts=list(getattr(pack, "false_positive_terms", []) or []),
            preferred_venues=list(getattr(pack, "preferred_venues", []) or []),
            field_of_study_whitelist=list(
                getattr(pack, "field_of_study_whitelist", []) or []
            ),
            field_of_study_blacklist=list(
                getattr(pack, "field_of_study_blacklist", []) or []
            ),
            terminology_map=dict(getattr(pack, "query_expansions", {}) or {}),
            candidate_domains=routed.candidate_domains,
            activation_evidence=routed.activation_evidence,
            negative_evidence=routed.negative_evidence,
            confidence=routed.confidence,
            fallback_reason=routed.fallback_reason,
        )
    fallback_profile = DomainProfile(
        domain_name="general_science",
        positive_domains=["science"],
        negative_domains=[],
        required_concepts=[],
        forbidden_concepts=[],
        preferred_venues=[],
        excluded_venues=[],
        field_of_study_whitelist=[],
        field_of_study_blacklist=[],
        terminology_map={},
        candidate_domains=routed.candidate_domains,
        activation_evidence=routed.activation_evidence,
        negative_evidence=routed.negative_evidence,
        confidence=routed.confidence,
        fallback_reason=routed.fallback_reason,
    )
    if not domain_hint or domain_hint in {"generic", "general_science"}:
        return fallback_profile

    if domain_hint and domain_hint != "generic":
        hinted = _domain_profile_for_hint(domain_hint, lowered_text)
        if hinted is not None:
            return hinted

    has_ai = any(term in lowered_text for term in AI_LITERATURE_TERMS)
    has_literature_screening = any(
        term in lowered_text
        for term in ["literature screening", "scientific literature", "abstract screening"]
    )
    has_materials = any(term in lowered_text for term in MATERIALS_MAGNETISM_TERMS)
    has_biomedical = any(term in lowered_text for term in BIOMEDICAL_TERMS)
    has_ferroelectric = any(term in lowered_text for term in FERROELECTRIC_TERMS)

    if has_ai and has_literature_screening:
        return DomainProfile(
            domain_name="ai_literature_screening",
            positive_domains=[
                "artificial intelligence",
                "information retrieval",
                "scientific literature screening",
            ],
            negative_domains=[
                "clinical screening",
                "drug screening",
                "high-throughput materials screening",
            ],
            required_concepts=_unique(
                [
                    "literature screening",
                    "scientific literature",
                    "LLM agents" if "agent" in lowered_text else "",
                    "evidence verification" if "evidence" in lowered_text else "",
                    "human feedback" if "human feedback" in lowered_text else "",
                    "claim extraction" if "claim" in lowered_text else "",
                ]
            ),
            forbidden_concepts=[
                "patient screening",
                "drug screening",
                "biomarker screening",
                *([] if has_materials else ["high-throughput materials screening"]),
                *([] if has_materials else ["materials screening"]),
                "biological agent",
                "chemical agent",
                "infectious agent",
            ],
            preferred_venues=[
                "ACL",
                "EMNLP",
                "NAACL",
                "NeurIPS",
                "ICLR",
                "JCDL",
                "Scientometrics",
            ],
            excluded_venues=[],
            field_of_study_whitelist=[
                "Computer Science",
                "Information Retrieval",
                "Artificial Intelligence",
                *(["Materials Science"] if has_materials else []),
            ],
            field_of_study_blacklist=[
                "Medicine",
                "Biology",
                *([] if has_materials else ["Materials Science"]),
            ],
            terminology_map={
                "LLM agents": ["large language model agents", "multi-agent LLM systems"],
                "literature screening": ["abstract screening", "study selection"],
                "evidence verification": ["claim verification", "grounded evidence"],
            },
        )

    if has_ferroelectric and not has_ai:
        return _ferroelectric_domain_profile(lowered_text)

    if has_materials and not has_ai:
        return DomainProfile(
            domain_name="materials_magnetism",
            positive_domains=[
                "materials science",
                "condensed matter physics",
                "magnetism",
            ],
            negative_domains=[
                "artificial intelligence literature screening",
                "clinical screening",
            ],
            required_concepts=_unique(
                [
                    "surface magnetization" if "surface magnetization" in lowered_text else "",
                    "magnetization" if "magnetization" in lowered_text else "",
                    "antiferromagnetism" if "antiferro" in lowered_text else "",
                    "surface spin" if "surface spin" in lowered_text else "",
                ]
            ),
            forbidden_concepts=[
                "LLM agent",
                "large language model",
                "human feedback",
                "literature screening",
                "patient screening",
                "drug screening",
            ],
            preferred_venues=[
                "Physical Review Letters",
                "Physical Review B",
                "Nature Materials",
                "Advanced Materials",
            ],
            excluded_venues=[],
            field_of_study_whitelist=["Materials Science", "Physics"],
            field_of_study_blacklist=["Computer Science", "Medicine"],
            terminology_map={
                "surface magnetization": [
                    "surface magnetic moment",
                    "surface spin polarization",
                ],
                "antiferromagnetism": ["antiferromagnetic materials", "Neel order"],
            },
        )

    if has_biomedical:
        return DomainProfile(
            domain_name="biomedical_screening",
            positive_domains=["medicine", "biomedicine", "clinical research"],
            negative_domains=["AI literature screening", "materials screening"],
            required_concepts=_unique(
                [
                    "patient screening" if "patient" in lowered_text else "",
                    "drug screening" if "drug" in lowered_text else "",
                    "biomarker screening" if "biomarker" in lowered_text else "",
                ]
            ),
            forbidden_concepts=["literature screening", "materials screening", "LLM agent"],
            preferred_venues=["The Lancet", "JAMA", "NEJM", "Nature Medicine"],
            excluded_venues=[],
            field_of_study_whitelist=["Medicine", "Biology"],
            field_of_study_blacklist=["Computer Science", "Materials Science"],
            terminology_map={
                "patient screening": ["clinical screening", "diagnostic screening"],
                "drug screening": ["compound screening", "pharmacological screening"],
            },
        )

    return fallback_profile


def _expert_domain_hint(expert_intent: ExpertResearchIntent | None) -> str:
    """Return the domain selected during intent repair, when available."""

    if expert_intent is None:
        return ""
    metadata = expert_intent.llm_metadata or {}
    return str(metadata.get("domain_pack_domain") or "")


def _domain_profile_for_hint(domain_hint: str, lowered_text: str) -> DomainProfile | None:
    """Return an explicit domain profile from an intent-repair/domain-pack hint."""

    if domain_hint == "ferroelectric_polarization":
        return _ferroelectric_domain_profile(lowered_text)
    if domain_hint == "materials_magnetism":
        return infer_domain_profile(
            f"{lowered_text} surface magnetization antiferromagnet",
            domain_hint="",
        )
    if domain_hint == "ai_literature_screening":
        return infer_domain_profile(
            f"{lowered_text} LLM scientific literature screening",
            domain_hint="",
        )
    return None


def _ferroelectric_domain_profile(lowered_text: str) -> DomainProfile:
    """Return a ferroelectric-polarization domain profile."""

    pack = _load_pack_if_available("ferroelectric_polarization")
    pack_terms = pack.false_positive_terms if pack else []
    required = _unique(
        [
            "ferroelectric polarization"
            if "ferroelectric" in lowered_text or "铁电" in lowered_text
            else "",
            "ferroelectric thin film"
            if "thin film" in lowered_text or "薄膜" in lowered_text
            else "",
            "surface polarization"
            if "surface polarization" in lowered_text or "表面极化" in lowered_text
            else "",
            "depolarization field" if "depolarization" in lowered_text or "退极化" in lowered_text else "",
            "interface screening" if "interface screening" in lowered_text or "界面" in lowered_text else "",
        ],
        8,
    )
    if not required:
        required = ["ferroelectric polarization"]
    return DomainProfile(
        domain_name="ferroelectric_polarization",
        positive_domains=[
            "materials science",
            "condensed matter physics",
            "ferroelectricity",
        ],
        negative_domains=[
            "biomedical screening",
            "clinical screening",
            "drug screening",
            "cognitive screening",
            "generic solvent screening",
            "cell surface polarization",
            "surface plasmon polaritons",
            "generic thin film deposition",
            "generic SHG plasmonics",
            "hydrogen evolution catalyst",
            *pack_terms,
        ],
        required_concepts=required,
        forbidden_concepts=[
            "drug screening",
            "clinical screening",
            "cognitive screening",
            "transcription factor screening",
            "transcription factor binding profiles",
            "cell surface polarization",
            "generic solvent screening",
            "COSMO solvent screening",
        ],
        preferred_venues=[
            "Physical Review Letters",
            "Physical Review B",
            "Applied Physics Letters",
            "Advanced Functional Materials",
            "Nature Materials",
        ],
        excluded_venues=[],
        field_of_study_whitelist=["Materials Science", "Physics"],
        field_of_study_blacklist=["Medicine", "Biology"],
        terminology_map={
            "PFM": ["piezoresponse force microscopy"],
            "SHG": ["second harmonic generation"],
            "KPFM": ["Kelvin probe force microscopy"],
            "surface polarization": ["surface charge", "interface polarization"],
            "screening": ["screening charge", "interface screening", "electrode screening"],
        },
    )


def extract_question_concepts(question: str) -> list[str]:
    """Extract concise topic concepts from a refined question."""

    lowered = question.lower()
    known_phrases = [
        "human-in-the-loop",
        "human feedback",
        "multi-agent LLM systems",
        "LLM agents",
        "large language model",
        "scientific literature screening",
        "literature screening",
        "evidence verification",
        "claim extraction",
        "surface magnetization",
        "surface spin",
        "boundary spin signals",
        "antiferromagnetic materials",
        "materials screening",
        "patient screening",
        "drug screening",
        "biomarker screening",
    ]
    concepts = [phrase for phrase in known_phrases if phrase.lower() in lowered]
    tokens = tokenize(question)
    if not concepts:
        concepts.extend(tokens[:4])
    return _unique(concepts, 8)


def _fallback_search_brief(question: str) -> SearchBrief:
    cleaned = " ".join(question.split())
    return SearchBrief(
        original_question=cleaned,
        refined_question=cleaned,
        search_intent="overview",
        user_goal="Find papers aligned with the research question.",
        inclusion_criteria=[cleaned] if cleaned else [],
        exclusion_criteria=[],
        required_aspects=tokenize(cleaned)[:4],
        preferred_paper_types=["article"],
        time_window="no strict time window",
        success_definition="A ranked list of relevant, evidence-grounded papers.",
    )


def _flatten_ambiguity_terms(
    ambiguity_analysis: list[dict[str, Any]],
    key: str,
) -> list[str]:
    values: list[str] = []
    for record in ambiguity_analysis:
        terms = record.get(key, [])
        if isinstance(terms, list):
            values.extend(str(term) for term in terms)
    return _unique(values, 16)


def _specific_criteria(criteria: list[str]) -> list[str]:
    selected: list[str] = []
    for item in criteria:
        cleaned = " ".join(str(item).split())
        if (
            cleaned
            and len(tokenize(cleaned)) <= 6
            and not _looks_like_natural_language_criterion(cleaned)
        ):
            selected.append(cleaned)
    return _unique(selected, 8)


def _looks_like_natural_language_criterion(text: str) -> bool:
    """Return True for prose criteria that should not become hard query terms."""

    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return False
    prose_prefixes = (
        "studies discussing",
        "papers discussing",
        "articles discussing",
        "studies that",
        "papers that",
        "articles that",
        "related to",
        "relevant to",
        "work that",
        "research that",
    )
    if lowered.startswith(prose_prefixes):
        return True
    tokens = tokenize(lowered)
    prose_markers = {
        "discussing",
        "describing",
        "explaining",
        "related",
        "relevant",
        "important",
        "importance",
        "significance",
    }
    return len(tokens) > 3 and bool(set(tokens) & prose_markers)


def _concept_terms_by_role(
    concepts: list[IntentConcept],
    role: str,
    *,
    require_query_use: bool = True,
) -> list[str]:
    """Return structured intent concepts for one query role."""

    return _unique(
        [
            concept.term
            for concept in concepts
            if concept.query_role == role
            and concept.term
            and (concept.should_use_in_provider_query or not require_query_use)
        ],
        24,
    )


def _constraint_groups_from_intent(
    expert_intent: ExpertResearchIntent,
) -> list[SearchConstraintGroup]:
    """Build generic OR groups from intent concept categories and roles."""

    must_concepts = [
        concept
        for concept in expert_intent.structured_concepts
        if concept.query_role == "must"
        and concept.should_use_in_provider_query
        and concept.term
    ]
    groups: list[SearchConstraintGroup] = []
    if must_concepts:
        groups.append(
            SearchConstraintGroup(
                group_name="must_concepts",
                operator="OR",
                terms=_unique([concept.term for concept in must_concepts], 24),
                source="expert_intent",
                required=True,
            )
        )
    optional_terms = _unique(
        [
            concept.term
            for concept in expert_intent.structured_concepts
            if concept.query_role == "optional"
            and concept.should_use_in_provider_query
            and concept.confidence >= 0.72
        ],
        16,
    )
    if optional_terms:
        groups.append(
            SearchConstraintGroup(
                group_name="optional_high_confidence",
                operator="OR",
                terms=optional_terms,
                source="expert_intent",
                required=False,
            )
        )
    return groups


def _constraint_groups_from_generic_frame(
    frame: GenericResearchIntentFrame | None,
) -> list[SearchConstraintGroup]:
    """Build reusable generic constraint groups from the intent frame."""

    if frame is None:
        return []
    core_groups = _semantic_required_core_groups(frame)
    groups: list[SearchConstraintGroup] = []
    for index, terms in enumerate(core_groups, start=1):
        groups.append(
            SearchConstraintGroup(
                group_name=(
                    "core_object_group"
                    if len(core_groups) == 1
                    else f"core_object_group_{index}"
                ),
                operator="OR",
                terms=terms,
                source="generic_intent_frame",
                required=True,
            )
        )
    context_terms = _constraint_context_terms(frame.domain_context)
    specs = [
        ("domain_context_group", context_terms, bool(context_terms)),
        ("process_or_property_group", frame.target_process_or_property, False),
        ("method_group", frame.method_terms, False),
        ("mechanism_group", frame.mechanism_terms, False),
        ("material_or_case_group", frame.material_or_case_terms, False),
        ("application_or_metric_group", frame.application_or_metric_terms, False),
        ("failure_or_limitation_group", frame.failure_or_limitation_terms, False),
        ("controversy_group", frame.controversy_terms, False),
    ]
    for name, terms, required in specs:
        cleaned = _unique(
            [
                term
                for term in terms
                if term.lower() not in {value.lower() for value in frame.downweighted_terms}
            ],
            24,
        )
        if not cleaned:
            continue
        groups.append(
            SearchConstraintGroup(
                group_name=name,
                operator="OR",
                terms=cleaned,
                source="generic_intent_frame",
                required=required,
            )
        )
    return groups


def _lithium_target_chemistry_groups(
    question: str,
    search_brief: SearchBrief,
    expert_intent: ExpertResearchIntent | None,
) -> list[SearchConstraintGroup]:
    """Create explicit chemistry context guardrails for lithium SEI questions."""

    text_parts = [
        question,
        search_brief.refined_question,
        " ".join(search_brief.inclusion_criteria),
        " ".join(search_brief.required_aspects),
    ]
    if expert_intent is not None:
        text_parts.extend(
            [
                expert_intent.expert_rewritten_question,
                " ".join(concept.term for concept in expert_intent.structured_concepts),
            ]
        )
    normalized = " ".join(str(part or "").lower() for part in text_parts)
    has_lithium_target = any(marker.lower() in normalized for marker in LITHIUM_TARGET_MARKERS)
    has_sei = (
        " sei" in f" {normalized}"
        or "solid electrolyte interphase" in normalized
        or "solid-electrolyte interphase" in normalized
        or "界面" in normalized
    )
    if not (has_lithium_target and has_sei):
        return []
    return [
        SearchConstraintGroup(
            group_name="target_chemistry_group",
            operator="OR",
            terms=LITHIUM_TARGET_CHEMISTRY_TERMS,
            source="search_contract_lithium_target_chemistry",
            required=True,
        ),
        SearchConstraintGroup(
            group_name="negative_context_group",
            operator="OR",
            terms=NON_TARGET_BATTERY_CHEMISTRY_TERMS,
            source="search_contract_lithium_target_chemistry",
            required=False,
        ),
    ]


def _semantic_required_core_groups(
    frame: GenericResearchIntentFrame,
) -> list[list[str]]:
    """Split core research axes into required groups while keeping aliases together."""

    terms = _unique(frame.research_object or frame.core_terms, 8)
    if not terms:
        return []
    target_terms = _unique(frame.target_process_or_property, 8)
    groups: list[list[str]] = []
    for term in terms:
        placed = False
        for group in groups:
            if _terms_are_aliases(term, group[0]):
                group.append(term)
                placed = True
                break
        if not placed:
            groups.append([term])
    for term in target_terms:
        if _is_generic_property_axis(term):
            continue
        for group in groups:
            if any(_terms_are_aliases(term, existing) for existing in group):
                group.append(term)
                break
    return [_unique(group, 8) for group in groups if group]


def _constraint_context_terms(terms: list[str]) -> list[str]:
    cleaned = _unique(terms, 12)
    if len(cleaned) <= 1:
        return cleaned
    weak_context = {"interface", "surface", "system", "material"}
    strong = [
        term
        for term in cleaned
        if " ".join(term.lower().split()) not in weak_context
    ]
    return strong or cleaned


def _is_generic_property_axis(term: str) -> bool:
    cleaned = " ".join(str(term or "").lower().split())
    return any(
        marker in cleaned
        for marker in [
            "activity",
            "accuracy",
            "recall",
            "performance",
            "composition",
            "structure",
            "evolution",
            "pore size",
            "functional group",
            "water stability",
            "stability",
        ]
    )


def _terms_are_aliases(left: str, right: str) -> bool:
    left_clean = " ".join(str(left or "").lower().split())
    right_clean = " ".join(str(right or "").lower().split())
    if not left_clean or not right_clean:
        return False
    if left_clean == right_clean:
        return True
    if left_clean in right_clean or right_clean in left_clean:
        return True
    return _initialism(left_clean) == right_clean or _initialism(right_clean) == left_clean


def _initialism(value: str) -> str:
    tokens = [token for token in value.replace("-", " ").split() if token]
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens).lower()


def _constraint_groups_from_domain_pack(
    domain_pack: Any | None,
) -> list[SearchConstraintGroup]:
    """Build SearchConstraintGroup objects from optional domain-pack rules."""

    if domain_pack is None:
        return []
    groups: list[SearchConstraintGroup] = []
    for item in getattr(domain_pack, "constraint_groups", []) or []:
        if not isinstance(item, dict):
            continue
        terms = _unique([str(term) for term in item.get("terms", [])], 32)
        if not terms:
            continue
        groups.append(
            SearchConstraintGroup(
                group_name=str(item.get("group_name") or "domain_pack_group"),
                operator=str(item.get("operator") or "OR").upper(),
                terms=terms,
                source=str(item.get("source") or "domain_pack"),
                required=bool(item.get("required", False)),
            )
        )
    return groups


def _unique_constraint_groups(
    groups: list[SearchConstraintGroup],
) -> list[SearchConstraintGroup]:
    """Deduplicate constraint groups while preserving order."""

    result: list[SearchConstraintGroup] = []
    seen: set[tuple[str, tuple[str, ...], bool]] = set()
    for group in groups:
        terms = tuple(_unique(group.terms, 32))
        key = (group.group_name, terms, group.required)
        if not terms or key in seen:
            continue
        result.append(
            SearchConstraintGroup(
                group_name=group.group_name,
                operator=group.operator,
                terms=list(terms),
                source=group.source,
                required=group.required,
            )
        )
        seen.add(key)
    return result


def _relaxed_constraint_groups(
    groups: list[SearchConstraintGroup],
) -> list[SearchConstraintGroup]:
    """Keep intent groups for diagnostics without making them hard guardrails."""

    return [
        SearchConstraintGroup(
            group_name=group.group_name,
            operator=group.operator,
            terms=list(group.terms),
            source=group.source,
            required=False,
        )
        for group in groups
    ]


def _aspect_groups_from_domain_pack(domain_pack: Any | None) -> list[str]:
    """Return compact aspect-group strings consumable by AspectCoverageAgent."""

    if domain_pack is None:
        return []
    aspect_groups = getattr(domain_pack, "aspect_groups", {}) or {}
    if not isinstance(aspect_groups, dict):
        return []
    groups: list[str] = []
    for name, terms in aspect_groups.items():
        cleaned_terms = _unique([str(term) for term in terms], 24)
        if name and cleaned_terms:
            groups.append(f"{name}: {'; '.join(cleaned_terms)}")
    return groups


def _required_aspects_for_contract(
    domain_pack: Any | None,
    search_brief: SearchBrief,
    generic_frame: GenericResearchIntentFrame | None = None,
) -> list[str]:
    """Prefer domain-pack aspect groups over noisy short SearchBrief tokens."""

    pack_aspects = _aspect_groups_from_domain_pack(domain_pack)
    if pack_aspects:
        return _unique([*pack_aspects, *generic_aspect_groups(generic_frame or GenericResearchIntentFrame())], 24)
    generic_aspects = generic_aspect_groups(generic_frame or GenericResearchIntentFrame())
    if generic_aspects:
        return generic_aspects
    return search_brief.required_aspects


def _fallback_constraint_groups(must_include: list[str]) -> list[SearchConstraintGroup]:
    """Use a loose group instead of implying every must term is jointly required."""

    if not must_include:
        return []
    return [
        SearchConstraintGroup(
            group_name="required_topic_terms",
            operator="OR",
            terms=_unique(must_include, 12),
            source="search_contract_fallback",
            required=True,
        )
    ]


def _load_pack_if_available(domain_name: str) -> Any | None:
    """Load a domain pack when it exists, otherwise return None."""

    if not domain_name:
        return None
    try:
        return load_domain_pack(domain_name)
    except ValueError:
        return None


def _domain_pack_terms_for_profile(domain_pack: Any | None) -> list[str]:
    """Return representative pack terms for a DomainProfile without hard guards."""

    if domain_pack is None:
        return []
    terms: list[str] = []
    terms.extend(getattr(domain_pack, "domain_anchors", []) or [])
    for concept in getattr(domain_pack, "concepts", {}).values():
        terms.extend(getattr(concept, "synonyms", []) or [])
    terms.extend(getattr(domain_pack, "mechanisms", []) or [])
    return _unique(terms, 16)


def _unique(values: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result[:limit] if limit else result
