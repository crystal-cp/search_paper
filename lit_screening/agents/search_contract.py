"""Search-contract construction for intent-aware retrieval."""

from __future__ import annotations

from typing import Any

from lit_screening.llm_client import GenericLLMClient
from lit_screening.models import DomainProfile, SearchBrief, SearchContract
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
    ) -> SearchContract:
        """Build a rule-based SearchContract with optional LLM left disabled."""

        fallback_brief = search_brief or _fallback_search_brief(question)
        ambiguity = ambiguity_analysis or []
        return self._build_rule(question, fallback_brief, ambiguity)

    def _build_rule(
        self,
        question: str,
        search_brief: SearchBrief,
        ambiguity_analysis: list[dict[str, Any]],
    ) -> SearchContract:
        lowered = f"{question} {search_brief.refined_question}".lower()
        domain_profile = infer_domain_profile(lowered)
        ambiguity_must = _flatten_ambiguity_terms(ambiguity_analysis, "recommended_must_terms")
        ambiguity_exclude = _flatten_ambiguity_terms(
            ambiguity_analysis,
            "recommended_exclude_terms",
        )
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
        must_exclude = _unique(
            [
                *domain_profile.forbidden_concepts,
                *ambiguity_exclude,
                *_specific_criteria(search_brief.exclusion_criteria),
            ],
            16,
        )
        return SearchContract(
            original_question=search_brief.original_question or question,
            refined_question=search_brief.refined_question or question,
            user_goal=search_brief.user_goal,
            search_intent=search_brief.search_intent,
            domain_profile=domain_profile,
            must_include_concepts=must_include,
            must_exclude_concepts=must_exclude,
            inclusion_criteria=search_brief.inclusion_criteria,
            exclusion_criteria=search_brief.exclusion_criteria,
            required_aspects=_unique(
                [*search_brief.required_aspects, *domain_profile.required_concepts[:4]],
                12,
            ),
            preferred_paper_types=search_brief.preferred_paper_types,
            time_window=search_brief.time_window,
            success_definition=search_brief.success_definition,
        )


def infer_domain_profile(lowered_text: str) -> DomainProfile:
    """Infer a coarse domain profile from user-visible terms."""

    has_ai = any(term in lowered_text for term in AI_LITERATURE_TERMS)
    has_literature_screening = any(
        term in lowered_text
        for term in ["literature screening", "scientific literature", "abstract screening"]
    )
    has_materials = any(term in lowered_text for term in MATERIALS_MAGNETISM_TERMS)
    has_biomedical = any(term in lowered_text for term in BIOMEDICAL_TERMS)

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
                    "boundary magnetization",
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

    return DomainProfile(
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
        if cleaned and len(tokenize(cleaned)) <= 6:
            selected.append(cleaned)
    return _unique(selected, 8)


def _unique(values: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result[:limit] if limit else result
