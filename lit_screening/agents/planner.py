"""Scholarly query planner with optional LLM enhancement."""

from __future__ import annotations

import re
from typing import Any

from lit_screening.llm_client import GenericLLMClient
from lit_screening.models import (
    ExpertResearchIntent,
    IntentConcept,
    QueryPlan,
    SearchBrief,
    SearchContract,
)
from lit_screening.utils import STOPWORDS, tokenize


CHINESE_TOPIC_GLOSSARY = [
    ("表面磁化", "surface magnetization"),
    ("表面磁矩", "surface magnetic moment"),
    ("反铁磁", "antiferromagnetic"),
    ("铁磁", "ferromagnetic"),
    ("磁电", "magnetoelectric"),
    ("自旋电子", "spintronics"),
    ("自旋", "spin"),
    ("磁化", "magnetization"),
    ("磁矩", "magnetic moment"),
    ("表面", "surface"),
    ("界面", "interface"),
    ("薄膜", "thin film"),
    ("材料", "materials"),
    ("二维", "two-dimensional"),
    ("第一性原理", "first-principles"),
    ("密度泛函", "density functional theory"),
    ("计算", "computational"),
    ("实验", "experimental"),
    ("理论", "theoretical"),
    ("机制", "mechanism"),
    ("意义", "significance"),
    ("重要性", "importance"),
    ("影响", "effect"),
    ("应用", "applications"),
    ("性质", "properties"),
    ("调控", "control"),
    ("检测", "detection"),
    ("综述", "review"),
    ("文献", "literature"),
    ("筛选", "screening"),
    ("证据", "evidence"),
    ("验证", "verification"),
    ("人工反馈", "human feedback"),
    ("大语言模型", "large language model"),
    ("语言模型", "language model"),
    ("大模型", "large language model"),
    ("多智能体", "multi-agent"),
]

KNOWN_MULTI_WORD_TERMS = [
    "surface magnetization",
    "surface magnetic moment",
    "antiferromagnetic materials",
    "first-principles",
    "density functional theory",
    "large language model",
    "language model",
    "llm agents",
    "multi-agent",
    "human feedback",
    "human-in-the-loop",
    "literature screening",
    "evidence verification",
    "claim extraction",
    "scientific literature",
    "systematic review",
]

LLM_RELATED_TERMS = {
    "llm",
    "agent",
    "agents",
    "multi-agent",
    "language",
    "model",
    "human",
    "feedback",
    "screening",
    "verification",
    "claim",
    "extraction",
}

GENERIC_TOPIC_MODIFIERS = {
    "importance",
    "significance",
    "effect",
    "effects",
    "impact",
    "role",
    "overview",
    "background",
}


def contains_cjk(text: str) -> bool:
    """Return True when text contains common Chinese/Japanese/Korean characters."""

    return any("\u4e00" <= char <= "\u9fff" for char in text or "")


def _append_non_redundant_term(terms: list[str], term: str) -> None:
    """Append a translated term unless it is already covered by existing terms."""

    cleaned = " ".join(term.split()).lower()
    if not cleaned:
        return
    existing_tokens = set(tokenize(" ".join(terms)))
    new_tokens = set(tokenize(cleaned))
    if cleaned in terms or (new_tokens and new_tokens <= existing_tokens):
        return
    terms.append(cleaned)


def fallback_translate_chinese_question(question: str) -> str:
    """Translate common Chinese research-topic words into an English query seed.

    This is intentionally conservative. It is not a general-purpose translator,
    but it keeps the offline rule-based pipeline usable when no LLM key exists.
    """

    terms: list[str] = []
    normalized = " ".join(question.split())
    for chinese, english in CHINESE_TOPIC_GLOSSARY:
        if chinese in normalized:
            _append_non_redundant_term(terms, english)

    embedded_english = re.findall(r"[A-Za-z][A-Za-z0-9+\-_/ ]*", normalized)
    for phrase in embedded_english:
        cleaned = " ".join(phrase.split())
        if cleaned:
            _append_non_redundant_term(terms, cleaned)

    if terms:
        return " ".join(terms[:10])
    return "scientific research topic"


def _unique(items: list[str], limit: int | None = None) -> list[str]:
    """Return normalized unique strings while preserving order."""

    values: list[str] = []
    for item in items:
        cleaned = " ".join(str(item).split()).strip()
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values[:limit] if limit else values


def _quote_phrase(term: str) -> str:
    """Quote multi-word terms for provider query strings."""

    cleaned = " ".join(term.split())
    if " " in cleaned and not (cleaned.startswith('"') and cleaned.endswith('"')):
        return f'"{cleaned}"'
    return cleaned


def _semantic_required(term: str) -> str:
    """Format a required Semantic Scholar term."""

    return f"+{_quote_phrase(term)}"


def _semantic_excluded(term: str) -> str:
    """Format an excluded Semantic Scholar term."""

    return f"-{_quote_phrase(term)}"


def _apply_openalex_excludes(query: str, exclude_terms: list[str]) -> str:
    if not exclude_terms:
        return query
    return f"{query} NOT {' NOT '.join(_quote_phrase(term) for term in exclude_terms)}"


def _plain_terms_query(terms: list[str], limit: int = 3) -> str:
    """Join terms without provider operators for broad recall-oriented search."""

    return " ".join(" ".join(term.split()) for term in terms[:limit])


def build_openalex_queries(plan: QueryPlan) -> list[str]:
    """Build OpenAlex-flavored queries from a structured query plan."""

    if _is_expert_materials_plan(plan):
        return _expert_materials_openalex_queries(plan)
    if _is_expert_plan(plan):
        return _expert_generic_openalex_queries(plan)

    core = plan.core_terms or plan.must_terms
    must = plan.must_terms or core
    optional = plan.optional_terms
    openalex_mode = plan.filters.get("openalex_mode", "keyword+semantic")
    primary_terms = [_quote_phrase(term) for term in (core[:3] or must[:3])]
    primary = " AND ".join(primary_terms) if primary_terms else plan.translated_question or plan.original_question
    natural_query = plan.translated_question or plan.original_question
    queries = []
    broad_core_query = _plain_terms_query(core[:3] or must[:3])
    single_core_query = _plain_terms_query(core[:1] or must[:1])
    quoted_core_query = " ".join(_quote_phrase(term) for term in (core[:2] or must[:2]))
    if natural_query:
        queries.append(natural_query)
    if single_core_query:
        queries.append(single_core_query)
    if broad_core_query:
        queries.append(broad_core_query)
    if quoted_core_query and quoted_core_query != broad_core_query:
        queries.append(quoted_core_query)
    if openalex_mode in {"semantic", "keyword+semantic"} and natural_query:
        queries.append(natural_query)
    if openalex_mode in {"keyword", "keyword+semantic"}:
        queries.append(primary)
    if must and must != core and openalex_mode in {"keyword", "keyword+semantic"}:
        queries.append(" AND ".join(_quote_phrase(term) for term in must[:4]))
    if optional:
        queries.append(f"{primary} AND ({' OR '.join(_quote_phrase(term) for term in optional[:4])})")
    if core:
        focused = " ".join(_quote_phrase(term) for term in core[:2])
        queries.append(f"{focused} review")
        queries.append(f"{focused} mechanism")
    if plan.filters.get("strictness") == "broad" and optional:
        queries.append(" OR ".join(_quote_phrase(term) for term in [*core[:2], *optional[:3]]))
    return _unique([_apply_openalex_excludes(query, plan.exclude_terms) for query in queries], 6)


def build_semantic_scholar_queries(plan: QueryPlan) -> list[str]:
    """Build Semantic Scholar-flavored queries from a structured query plan."""

    if _is_expert_materials_plan(plan):
        return _expert_materials_semantic_scholar_queries(plan)
    if _is_expert_plan(plan):
        return _expert_generic_semantic_scholar_queries(plan)

    core = plan.core_terms or plan.must_terms
    must = plan.must_terms or core
    optional = plan.optional_terms
    natural_query = plan.translated_question or plan.original_question
    required = " ".join(_semantic_required(term) for term in must[:4])
    excludes = " ".join(_semantic_excluded(term) for term in plan.exclude_terms[:4])
    optional_or = f"({' OR '.join(_quote_phrase(term) for term in optional[:4])})" if optional else ""
    primary = " ".join(part for part in [required, optional_or, excludes] if part)
    if not primary:
        primary = natural_query
    queries = []
    if natural_query:
        queries.append(natural_query)
    single_core_query = _plain_terms_query(core[:1] or must[:1])
    if single_core_query:
        queries.append(single_core_query)
    broad_core_query = _plain_terms_query(core[:3] or must[:3])
    if broad_core_query:
        queries.append(broad_core_query)
    queries.append(primary)
    if core:
        phrase = " ".join(_quote_phrase(term) for term in core[:2])
        queries.append(" ".join(part for part in [phrase, optional_or, excludes] if part))
        queries.append(" ".join(part for part in [phrase, "review", excludes] if part))
    if optional:
        queries.append(
            " ".join(
                part
                for part in [
                    required,
                    f"({' OR '.join(_quote_phrase(term) for term in optional[:5])})",
                    excludes,
                ]
                if part
            )
        )
    return _unique(queries, 6)


def _is_expert_materials_plan(plan: QueryPlan) -> bool:
    return bool(plan.expert_rewritten_question) and (
        plan.filters.get("search_contract_domain") == "materials_magnetism"
        or plan.filters.get("intent_domain") == "materials_magnetism"
    )


def _is_expert_plan(plan: QueryPlan) -> bool:
    return bool(plan.expert_rewritten_question and plan.filters.get("structured_concepts"))


def _expert_generic_openalex_queries(plan: QueryPlan) -> list[str]:
    """Short expert-intent OpenAlex queries from activated concepts only."""

    concepts = _concepts_from_filters(plan)
    must = _terms_by_role(concepts, "must")
    optional = _terms_by_role(concepts, "optional")
    base_terms = must or optional
    queries = _structured_concept_queries(base_terms, optional)
    excludes = plan.exclude_terms[:4]
    return _unique(
        [
            _apply_openalex_excludes(query, excludes)
            for query in queries
            if query and len(query) < 180
        ],
        6,
    )


def _expert_generic_semantic_scholar_queries(plan: QueryPlan) -> list[str]:
    """Short expert-intent Semantic Scholar queries from activated concepts only."""

    concepts = _concepts_from_filters(plan)
    must = _terms_by_role(concepts, "must")
    optional = _terms_by_role(concepts, "optional")
    base_terms = must or optional
    excludes = " ".join(_semantic_excluded(term) for term in plan.exclude_terms[:4])
    queries = [
        *_seed_title_queries(concepts),
        *_structured_concept_queries(base_terms, optional),
    ]
    if must:
        required = " ".join(_semantic_required(term) for term in must[:3])
        queries.append(required)
        if optional:
            queries.append(
                " ".join(
                    part
                    for part in [
                        required,
                        _quote_phrase(optional[0]),
                    ]
                    if part
                )
            )
    return _unique(
        [
            " ".join(part for part in [query, excludes] if part)
            for query in queries
            if query and len(query) < 220
        ],
        6,
    )


def _expert_materials_openalex_queries(plan: QueryPlan) -> list[str]:
    """Short OpenAlex queries from activated materials-magnetism concepts."""

    concepts = _concepts_from_filters(plan)
    terms = _active_terms(concepts)
    queries = _activated_materials_queries(concepts, include_seed_titles=False)
    if not queries:
        queries = _short_pair_queries(terms)
    return _unique([query for query in queries if len(query) < 180], 8)


def _expert_materials_semantic_scholar_queries(plan: QueryPlan) -> list[str]:
    """Short Semantic Scholar queries from activated concepts and seed hints."""

    concepts = _concepts_from_filters(plan)
    terms = _active_terms(concepts)
    queries = [
        *_seed_title_queries(concepts),
        *_activated_materials_queries(concepts, include_seed_titles=False),
    ]
    if not queries:
        queries = _short_pair_queries(terms)
    return _unique([query for query in queries if len(query) < 220], 8)


def _concepts_from_filters(plan: QueryPlan) -> list[IntentConcept]:
    raw_items = plan.filters.get("structured_concepts", [])
    concepts: list[IntentConcept] = []
    if not isinstance(raw_items, list):
        return concepts
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        concepts.append(
            IntentConcept(
                term=str(item.get("term") or ""),
                category=str(item.get("category") or ""),
                source=str(item.get("source") or ""),
                confidence=float(item.get("confidence") or 0.0),
                activation_reason=str(item.get("activation_reason") or ""),
                query_role=str(item.get("query_role") or "uncertain"),
                should_use_in_provider_query=bool(
                    item.get("should_use_in_provider_query")
                ),
            )
        )
    return concepts


def _query_eligible(concepts: list[IntentConcept]) -> list[IntentConcept]:
    return [
        concept
        for concept in concepts
        if concept.should_use_in_provider_query
        and (
            concept.query_role == "must"
            or (concept.query_role == "optional" and concept.confidence >= 0.72)
        )
        and concept.query_role not in {"downweighted", "exclude", "uncertain"}
    ]


def _active_terms(concepts: list[IntentConcept]) -> list[str]:
    return _unique([concept.term for concept in _query_eligible(concepts)], 24)


def _terms_by_role(concepts: list[IntentConcept], role: str) -> list[str]:
    return _unique(
        [
            concept.term
            for concept in _query_eligible(concepts)
            if concept.query_role == role
        ],
        12,
    )


def _terms_by_category_from_concepts(
    concepts: list[IntentConcept],
    category: str,
) -> list[str]:
    return _unique(
        [
            concept.term
            for concept in _query_eligible(concepts)
            if concept.category == category
        ],
        24,
    )


def _has_active(concepts: list[IntentConcept], term: str) -> bool:
    lowered = term.lower()
    return any(
        concept.term.lower() == lowered
        for concept in _query_eligible(concepts)
    )


def _find_active(concepts: list[IntentConcept], candidates: list[str]) -> str:
    candidate_set = {candidate.lower() for candidate in candidates}
    for concept in _query_eligible(concepts):
        if concept.term.lower() in candidate_set:
            return concept.term
    return ""


def _activated_materials_queries(
    concepts: list[IntentConcept],
    include_seed_titles: bool,
) -> list[str]:
    queries: list[str] = []
    materials = _terms_by_category_from_concepts(concepts, "material")
    methods = _terms_by_category_from_concepts(concepts, "method")
    active_terms = _active_terms(concepts)

    boundary = _find_active(concepts, ["boundary magnetization"])
    magnetoelectric = _find_active(
        concepts,
        ["magnetoelectric antiferromagnet", "magnetoelectric"],
    )
    surface_magnetization = _find_active(concepts, ["surface magnetization"])
    local_me = _find_active(
        concepts,
        ["local magnetoelectric response", "local magnetoelectric effect"],
    )
    surface_order = _find_active(concepts, ["surface magnetic order"])
    spin_polarization = _find_active(
        concepts,
        ["surface spin polarization", "spin polarization"],
    )
    altermagnetism = _find_active(concepts, ["altermagnetism"])
    spin_splitting = _find_active(concepts, ["spin splitting"])

    if boundary and magnetoelectric:
        queries.append(f"{_quote_phrase(boundary)} {_quote_phrase(magnetoelectric)}")
    if local_me and surface_order:
        queries.append(f"{_quote_phrase(local_me)} {_quote_phrase(surface_order)}")
    material_queries: list[str] = []
    method_queries: list[str] = []
    for material in materials[:3]:
        if surface_magnetization:
            material_queries.append(f"{material} {_quote_phrase(surface_magnetization)}")
        if boundary:
            material_queries.append(f"{material} {_quote_phrase(boundary)}")
    for method in methods:
        anchor = materials[0] if materials else ""
        if method in {"SPLEEM", "XMCD-PEEM", "SP-STM", "spin-resolved photoemission"}:
            signal = spin_polarization or surface_magnetization or surface_order
            if anchor and signal:
                method_queries.append(f"{anchor} {_quote_phrase(signal)} {method}")
            elif signal:
                method_queries.append(f"{_quote_phrase(signal)} {method}")
        elif method in {"NV magnetometry", "scanning diamond magnetometry"}:
            signal = boundary or surface_magnetization or surface_order
            if anchor and signal:
                method_queries.append(f"{method} {anchor} {_quote_phrase(signal)}")
        elif method in {"SARPES", "spin-resolved ARPES"} and altermagnetism:
            method_queries.append(f"{altermagnetism} {spin_splitting or 'spin splitting'} {method}")
    queries.extend(_unique(method_queries, 5))
    queries.extend(_unique(material_queries, 5))
    if altermagnetism and not any("altermagnet" in query.lower() for query in queries):
        queries.append(f"{altermagnetism} {spin_splitting or 'spin splitting'}")
    if include_seed_titles:
        queries.extend(_seed_title_queries(concepts))
    return _unique([query for query in queries if query.strip()], 12)


def _structured_concept_queries(
    base_terms: list[str],
    optional_terms: list[str],
) -> list[str]:
    """Compose short provider-neutral queries from structured concepts."""

    if not base_terms:
        return []
    queries = [
        _plain_terms_query(base_terms[:1]),
        _plain_terms_query(base_terms[:3]),
        " ".join(_quote_phrase(term) for term in base_terms[:2]),
    ]
    for optional in optional_terms[:4]:
        if base_terms:
            queries.append(f"{_quote_phrase(base_terms[0])} {_quote_phrase(optional)}")
        if len(base_terms) > 1:
            queries.append(
                f"{_quote_phrase(base_terms[0])} {_quote_phrase(base_terms[1])} {_quote_phrase(optional)}"
            )
    return _unique(
        [
            query
            for query in queries
            if query and not _looks_like_boundary_note(query)
        ],
        10,
    )


def _seed_title_queries(concepts: list[IntentConcept]) -> list[str]:
    titles = [
        concept.term
        for concept in concepts
        if concept.source == "seed_title" and concept.confidence >= 0.8
    ]
    identifiers = [
        concept.term
        for concept in concepts
        if concept.source == "seed_metadata"
        and concept.confidence >= 0.7
        and _looks_like_seed_identifier(concept.term)
    ]
    authors = [
        concept.term
        for concept in concepts
        if concept.source == "seed_metadata" and concept.confidence >= 0.7
        and not _looks_like_seed_identifier(concept.term)
    ]
    queries: list[str] = []
    queries.extend(identifiers[:3])
    for title in titles[:3]:
        if len(title) <= 160:
            queries.append(_quote_phrase(title))
        short = _seed_core_phrase(title)
        if short:
            queries.append(_quote_phrase(short))
            if authors:
                queries.append(f"{authors[0]} {_quote_phrase(short)}")
    return _unique(queries, 8)


def _looks_like_boundary_note(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("do not ") or "unless citation" in lowered


def _looks_like_seed_identifier(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return lowered.startswith("10.") or lowered.startswith("arxiv:") or bool(
        re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", lowered)
    )


def _seed_core_phrase(title: str) -> str:
    lower = title.lower()
    for marker in [
        "surface magnetization in antiferromagnets",
        "local magnetoelectric effects",
        "surface magnetic order",
        "altermagnetism",
    ]:
        if marker in lower:
            return marker
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", title) if len(word) > 3]
    return " ".join(words[:5])


def _short_pair_queries(terms: list[str]) -> list[str]:
    if not terms:
        return []
    queries = []
    for index in range(0, min(len(terms), 6), 2):
        pair = terms[index : index + 2]
        if pair:
            queries.append(" ".join(_quote_phrase(term) for term in pair))
    return queries


def _extract_exclude_terms(text: str) -> list[str]:
    """Extract simple user-stated exclusion terms."""

    excludes: list[str] = []
    patterns = [
        r"\bwithout\s+([a-zA-Z0-9][a-zA-Z0-9\- ]{1,40})",
        r"\bexcluding\s+([a-zA-Z0-9][a-zA-Z0-9\- ]{1,40})",
        r"\bnot\s+([a-zA-Z0-9][a-zA-Z0-9\- ]{1,30})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text.lower()):
            phrase = match.group(1).strip(" .,;:")
            if phrase:
                excludes.append(phrase)
    return _unique(excludes, 4)


def _extract_candidate_phrases(text: str) -> list[str]:
    """Extract topic-like multi-word terms from an English question."""

    lowered = text.lower()
    phrases = [term for term in KNOWN_MULTI_WORD_TERMS if term in lowered]
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", lowered)
    for size in (3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            chunk = words[index : index + size]
            if any(word in STOPWORDS for word in chunk):
                continue
            if any(word in GENERIC_TOPIC_MODIFIERS for word in chunk):
                continue
            phrase = " ".join(chunk)
            if len(tokenize(phrase)) >= 2:
                phrases.append(phrase)
    return _unique(phrases, 8)


def _is_term_redundant(term: str, existing_terms: list[str]) -> bool:
    """Return True when a single term is already covered by a phrase."""

    if " " in term:
        return False
    return any(term in tokenize(existing) for existing in existing_terms if " " in existing)


def _specific_inclusion_terms(
    inclusion_criteria: list[str],
    planning_question: str,
    limit: int = 3,
) -> list[str]:
    """Return short inclusion criteria that are safe to use as required terms."""

    planning_clean = " ".join(planning_question.lower().split())
    planning_tokens = set(tokenize(planning_question))
    selected: list[str] = []
    for criterion in inclusion_criteria:
        cleaned = " ".join(str(criterion).split())
        lowered = cleaned.lower()
        criterion_tokens = set(tokenize(cleaned))
        if not cleaned or not criterion_tokens:
            continue
        if lowered == planning_clean or criterion_tokens == planning_tokens:
            continue
        if len(criterion_tokens) > 5:
            continue
        selected.append(cleaned)
    return _unique(selected, limit)


def _intent_domain(search_contract: SearchContract | None) -> str:
    if not search_contract:
        return ""
    return search_contract.domain_profile.domain_name


def _expert_core_terms(
    expert_intent: ExpertResearchIntent,
    candidate_phrases: list[str],
) -> list[str]:
    """Return expert concepts without letting broad user words dominate."""

    concepts = _query_eligible(expert_intent.structured_concepts)
    values = [concept.term for concept in concepts]
    if not values:
        values = [
            *expert_intent.target_objects,
            *expert_intent.mechanisms,
            *expert_intent.materials,
            *expert_intent.methods,
            *candidate_phrases,
        ]
    downweighted = {term.lower() for term in expert_intent.ignored_or_downweighted_terms}
    return _unique(
        [
            value
            for value in values
            if value.lower() not in downweighted
            and value.lower() not in {"spin", "magnetization", "materials", "example", "review"}
        ],
        24,
    )


def _materials_must_groups(expert_intent: ExpertResearchIntent) -> list[list[str]]:
    """Any-one-of groups used for materials-magnetism retrieval/domain logic."""

    active = _query_eligible(expert_intent.structured_concepts)
    group_a = _unique(
        [
            concept.term
            for concept in active
            if concept.category in {"material"}
            or concept.term in {"antiferromagnet", "magnetoelectric antiferromagnet"}
        ],
        10,
    )
    group_b = _unique(
        [
            concept.term
            for concept in active
            if concept.category in {"object", "property", "mechanism"}
            and concept.term not in {"antiferromagnet", "magnetoelectric antiferromagnet"}
        ],
        10,
    )
    return [group for group in [group_a, group_b] if group]


def _concept_payloads(concepts: list[IntentConcept]) -> list[dict[str, Any]]:
    return [
        {
            "term": concept.term,
            "category": concept.category,
            "source": concept.source,
            "confidence": concept.confidence,
            "activation_reason": concept.activation_reason,
            "query_role": concept.query_role,
            "should_use_in_provider_query": concept.should_use_in_provider_query,
        }
        for concept in concepts
    ]


class PlannerAgent:
    """Create a compact set of academic search queries from a question."""

    expansion_terms = [
        "review",
        "recent advances",
        "mechanism",
        "experimental study",
        "theoretical study",
        "applications",
    ]

    def __init__(
        self,
        mode: str = "rule",
        llm_client: GenericLLMClient | None = None,
    ) -> None:
        self.mode = mode
        self.llm_client = llm_client
        self.last_llm_metadata: dict[str, Any] = {
            "planner_mode": mode,
            "llm_used": False,
            "invalid_llm_output": False,
            "llm_error_type": "",
        }

    def plan(self, question: str) -> list[str]:
        """Return 4 to 6 English search queries for the research question."""

        plan = self.plan_structured(question)
        return _unique(
            [
                plan.translated_question or plan.original_question,
                *plan.openalex_queries,
                *plan.semantic_scholar_queries,
            ],
            6,
        )

    def plan_structured(
        self,
        question: str,
        strictness: str = "balanced",
        openalex_mode: str = "keyword+semantic",
        sort_preference: str = "relevance",
        ranking_profile: str = "balanced",
        search_brief: SearchBrief | None = None,
        search_contract: SearchContract | None = None,
        expert_intent: ExpertResearchIntent | None = None,
    ) -> QueryPlan:
        """Return a structured, provider-aware query plan."""

        if expert_intent is not None:
            preprocessing = self._preprocess_question_rule(question)
            repaired_question = expert_intent.expert_rewritten_question or question
            preprocessing = {
                **preprocessing,
                "translation_used": preprocessing["translation_used"],
                "translation_mode": "intent_repair",
                "translated_question": repaired_question
                if contains_cjk(question)
                else preprocessing["translated_question"],
                "planning_question": repaired_question,
                "translation_warning": "",
                "intent_repair_used": True,
            }
            plan = self._plan_structured_rule(
                question=question,
                preprocessing=preprocessing,
                strictness=strictness,
                openalex_mode=openalex_mode,
                sort_preference=sort_preference,
                ranking_profile=ranking_profile,
                search_brief=search_brief,
                search_contract=search_contract,
                expert_intent=expert_intent,
            )
            self.last_llm_metadata = self._metadata(
                preprocessing,
                llm_used=False,
                invalid_llm_output=False,
                llm_error_type="",
            )
            self.last_llm_metadata.update(
                {
                    "intent_repair_used": True,
                    "expert_rewritten_question": expert_intent.expert_rewritten_question,
                    "downweighted_user_terms": expert_intent.ignored_or_downweighted_terms,
                }
            )
            return plan

        if self.mode == "llm" and self.llm_client and self.llm_client.is_available:
            return self._plan_structured_with_llm(
                question,
                strictness=strictness,
                openalex_mode=openalex_mode,
                sort_preference=sort_preference,
                ranking_profile=ranking_profile,
                search_brief=search_brief,
                search_contract=search_contract,
            )
        preprocessing = self._preprocess_question_rule(question)
        plan = self._plan_structured_rule(
            question=question,
            preprocessing=preprocessing,
            strictness=strictness,
            openalex_mode=openalex_mode,
            sort_preference=sort_preference,
            ranking_profile=ranking_profile,
            search_brief=search_brief,
            search_contract=search_contract,
        )
        self.last_llm_metadata = self._metadata(
            preprocessing,
            llm_used=False,
            invalid_llm_output=False,
            llm_error_type="llm_unavailable" if self.mode == "llm" else "",
        )
        return plan

    def _plan_rule(self, question: str) -> list[str]:
        """Rule-based query planning fallback."""

        base = " ".join(question.split())
        terms = tokenize(base)
        core = " ".join(terms[:8]) if terms else base
        compact = " ".join(terms[:5]) if terms else base
        queries = [
            base,
            f"{core} review",
            f"{core} recent advances",
            f"{compact} mechanism",
            f"{compact} experimental theoretical study",
            f"{compact} applications significance",
        ]
        unique: list[str] = []
        for query in queries:
            if query and query not in unique:
                unique.append(query)
        return unique[:6]

    def _plan_structured_rule(
        self,
        question: str,
        preprocessing: dict[str, Any],
        strictness: str,
        openalex_mode: str,
        sort_preference: str,
        ranking_profile: str,
        search_brief: SearchBrief | None = None,
        search_contract: SearchContract | None = None,
        expert_intent: ExpertResearchIntent | None = None,
    ) -> QueryPlan:
        """Build a topic-aware structured plan without requiring an LLM."""

        planning_question = (
            expert_intent.expert_rewritten_question
            if expert_intent
            else
            search_contract.refined_question
            if search_contract
            else search_brief.refined_question
            if search_brief
            else preprocessing["planning_question"]
        )
        detected_language = "zh" if preprocessing["question_language"] == "zh" else "en"
        tokens = tokenize(planning_question)
        candidate_phrases = _extract_candidate_phrases(planning_question)
        if expert_intent:
            core_terms = _expert_core_terms(expert_intent, candidate_phrases)
        else:
            core_terms = list(candidate_phrases)
            for token in tokens:
                if not _is_term_redundant(token, core_terms):
                    core_terms.append(token)
        if search_contract and not expert_intent:
            core_terms = [
                *search_contract.must_include_concepts[:6],
                *search_contract.domain_profile.required_concepts[:4],
                *core_terms,
            ]
        core_terms = _unique(core_terms, 24 if expert_intent else 8)
        if not core_terms and planning_question:
            core_terms = [planning_question]

        must_groups: list[list[str]] = []
        if expert_intent and _intent_domain(search_contract) == "materials_magnetism":
            must_groups = _materials_must_groups(expert_intent)
            must_terms = _unique(
                [
                    concept.term
                    for concept in _query_eligible(expert_intent.structured_concepts)
                    if concept.query_role == "must"
                ],
                5,
            )
            must_count = len(must_terms)
        else:
            must_count = 4 if strictness == "strict" else 3
            if strictness == "broad":
                must_count = 1
            must_terms = _unique(core_terms[:must_count], 5)
        if search_contract and not expert_intent:
            must_terms = _unique(
                [
                    *search_contract.must_include_concepts,
                    *must_terms,
                ],
                10,
            )
        if search_brief and not expert_intent:
            must_terms = _unique(
                [
                    *must_terms,
                    *_specific_inclusion_terms(
                        search_brief.inclusion_criteria,
                        planning_question,
                    ),
                ],
                8,
            )

        if expert_intent:
            topical_optional = _unique(
                [
                    concept.term
                    for concept in _query_eligible(expert_intent.structured_concepts)
                    if concept.query_role == "optional"
                ],
                10,
            )
        else:
            topical_optional = [
                token
                for token in tokens
                if token not in set(tokenize(" ".join(core_terms[:must_count])))
            ]
        generic_optional = ["review", "recent advances", "mechanism", "applications"]
        if ranking_profile == "high_quality_review":
            generic_optional = ["review", "systematic review", "survey", "meta-analysis"]
        optional_terms = _unique(
            topical_optional
            if expert_intent
            else [*topical_optional, *generic_optional],
            12 if expert_intent else 8,
        )
        if search_brief and not expert_intent:
            optional_terms = _unique(
                [
                    *optional_terms,
                    *search_brief.preferred_paper_types,
                    *search_brief.required_aspects,
                ],
                12,
            )
        if search_contract and not expert_intent:
            optional_terms = _unique(
                [
                    *optional_terms,
                    *search_contract.preferred_paper_types,
                    *search_contract.required_aspects,
                ],
                14,
            )

        if not (set(tokens) & LLM_RELATED_TERMS):
            optional_terms = [
                term
                for term in optional_terms
                if not (set(tokenize(term)) & LLM_RELATED_TERMS)
            ]

        plan = QueryPlan(
            original_question=preprocessing["original_question"],
            detected_language=detected_language,
            translated_question=preprocessing["translated_question"],
            core_terms=core_terms,
            must_terms=must_terms,
            optional_terms=optional_terms,
            exclude_terms=_unique(
                [
                    *_extract_exclude_terms(planning_question),
                    *(search_brief.exclusion_criteria if search_brief else []),
                    *(search_contract.must_exclude_concepts if search_contract else []),
                    *(
                        search_contract.domain_profile.forbidden_concepts
                        if search_contract
                        else []
                    ),
                ],
                16,
            ),
            required_aspects=(
                search_contract.required_aspects
                if search_contract
                else search_brief.required_aspects
                if search_brief
                else []
            ),
            filters={
                "strictness": strictness,
                "openalex_mode": openalex_mode,
                "sort_preference": sort_preference,
                "ranking_profile": ranking_profile,
                "search_contract_domain": search_contract.domain_profile.domain_name
                if search_contract
                else "",
                "intent_domain": _intent_domain(search_contract),
                "must_term_groups": must_groups,
                "intent_materials": expert_intent.materials if expert_intent else [],
                "intent_methods": expert_intent.methods if expert_intent else [],
                "intent_repair_used": bool(expert_intent),
                "structured_concepts": _concept_payloads(
                    expert_intent.structured_concepts
                )
                if expert_intent
                else [],
            },
            expert_rewritten_question=(
                expert_intent.expert_rewritten_question if expert_intent else ""
            ),
            intent_assumptions=expert_intent.assumptions if expert_intent else [],
            downweighted_user_terms=(
                expert_intent.ignored_or_downweighted_terms if expert_intent else []
            ),
        )
        plan.openalex_queries = build_openalex_queries(plan)
        plan.semantic_scholar_queries = build_semantic_scholar_queries(plan)
        return plan

    def _preprocess_question_rule(self, question: str) -> dict[str, Any]:
        """Detect Chinese input and produce an English planning question."""

        original_question = " ".join(question.split())
        is_chinese = contains_cjk(original_question)
        if not is_chinese:
            return {
                "original_question": original_question,
                "question_language": "en_or_other",
                "translation_used": False,
                "translation_mode": "none",
                "translated_question": "",
                "planning_question": original_question,
                "translation_warning": "",
            }

        translated_question = fallback_translate_chinese_question(original_question)
        warning = (
            "rule_glossary_translation_is_approximate"
            if translated_question != "scientific research topic"
            else "rule_glossary_missing_topic_terms"
        )
        return {
            "original_question": original_question,
            "question_language": "zh",
            "translation_used": True,
            "translation_mode": "rule_glossary",
            "translated_question": translated_question,
            "planning_question": translated_question,
            "translation_warning": warning,
        }

    def _metadata(
        self,
        preprocessing: dict[str, Any],
        *,
        llm_used: bool,
        invalid_llm_output: bool,
        llm_error_type: str,
    ) -> dict[str, Any]:
        """Build consistent planner metadata."""

        return {
            "planner_mode": self.mode,
            "llm_used": llm_used,
            "invalid_llm_output": invalid_llm_output,
            "llm_error_type": llm_error_type,
            **preprocessing,
        }

    def _plan_with_llm(self, question: str) -> list[str]:
        """Use an LLM to suggest queries, falling back safely."""

        plan = self._plan_structured_with_llm(
            question,
            strictness="balanced",
            openalex_mode="keyword+semantic",
            sort_preference="relevance",
            ranking_profile="balanced",
        )
        return _unique(
            [
                plan.translated_question or plan.original_question,
                *plan.openalex_queries,
                *plan.semantic_scholar_queries,
            ],
            6,
        )

    def _plan_structured_with_llm(
        self,
        question: str,
        strictness: str,
        openalex_mode: str,
        sort_preference: str,
        ranking_profile: str,
        search_brief: SearchBrief | None = None,
        search_contract: SearchContract | None = None,
    ) -> QueryPlan:
        """Use an LLM to enrich structured query planning, falling back safely."""

        preprocessing = self._preprocess_question_rule(question)
        system_prompt = (
            "You plan scholarly search queries for a literature-screening pipeline. "
            "Return JSON only with keys 'translated_question', 'core_terms', "
            "'must_terms', 'optional_terms', 'exclude_terms', and 'queries'. "
            "'translated_question' must be a concise English research question. "
            "If the input is already English, copy it with light cleanup. "
            "'queries' must be a list of 4 to 6 concise English academic search "
            "queries. Stay inside the user's scientific topic. Do not introduce "
            "AI, LLM, human-feedback, or literature-screening terms unless they "
            "are already part of the research question."
        )
        user_prompt = f"Research question:\n{question}"
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        translated_question = result.data.get("translated_question")
        if isinstance(translated_question, str) and translated_question.strip():
            cleaned_translation = " ".join(translated_question.split())
            if not contains_cjk(cleaned_translation):
                preprocessing = {
                    **preprocessing,
                    "translation_used": contains_cjk(question),
                    "translation_mode": "llm" if contains_cjk(question) else "none",
                    "translated_question": cleaned_translation
                    if contains_cjk(question)
                    else "",
                    "planning_question": cleaned_translation,
                    "translation_warning": "",
                }

        rule_plan = self._plan_structured_rule(
            question=question,
            preprocessing=preprocessing,
            strictness=strictness,
            openalex_mode=openalex_mode,
            sort_preference=sort_preference,
            ranking_profile=ranking_profile,
            search_brief=search_brief,
            search_contract=search_contract,
        )
        queries = result.data.get("queries")

        if result.invalid_llm_output or not isinstance(queries, list):
            self.last_llm_metadata = self._metadata(
                preprocessing,
                llm_used=True,
                invalid_llm_output=True,
                llm_error_type=result.error_type or "missing_queries",
            )
            return rule_plan

        def _list_from_result(key: str, fallback: list[str]) -> list[str]:
            value = result.data.get(key)
            if not isinstance(value, list):
                return fallback
            return _unique([item for item in value if isinstance(item, str)], 8) or fallback

        core_terms = _list_from_result("core_terms", rule_plan.core_terms)
        must_terms = _list_from_result("must_terms", rule_plan.must_terms)
        optional_terms = _list_from_result("optional_terms", rule_plan.optional_terms)
        exclude_terms = _list_from_result("exclude_terms", rule_plan.exclude_terms)
        if search_contract:
            core_terms = _unique([*search_contract.must_include_concepts[:6], *core_terms], 8)
            must_terms = _unique([*search_contract.must_include_concepts, *must_terms], 10)
            exclude_terms = _unique([*exclude_terms, *search_contract.must_exclude_concepts], 16)

        unique: list[str] = []
        for query in [preprocessing["planning_question"], *queries]:
            if not isinstance(query, str):
                continue
            cleaned = " ".join(query.split())
            if cleaned and not contains_cjk(cleaned) and cleaned not in unique:
                unique.append(cleaned)

        if len(unique) < 4:
            self.last_llm_metadata = self._metadata(
                preprocessing,
                llm_used=True,
                invalid_llm_output=True,
                llm_error_type="too_few_queries",
            )
            return rule_plan

        plan = QueryPlan(
            original_question=preprocessing["original_question"],
            detected_language="zh" if preprocessing["question_language"] == "zh" else "en",
            translated_question=preprocessing["translated_question"],
            core_terms=core_terms,
            must_terms=must_terms,
            optional_terms=optional_terms,
            exclude_terms=exclude_terms,
            required_aspects=(
                search_contract.required_aspects
                if search_contract
                else search_brief.required_aspects
                if search_brief
                else rule_plan.required_aspects
            ),
            filters={
                "strictness": strictness,
                "openalex_mode": openalex_mode,
                "sort_preference": sort_preference,
                "ranking_profile": ranking_profile,
                "search_contract_domain": search_contract.domain_profile.domain_name
                if search_contract
                else "",
            },
        )
        plan.openalex_queries = _unique([*unique, *build_openalex_queries(plan)], 6)
        plan.semantic_scholar_queries = _unique(
            [*unique, *build_semantic_scholar_queries(plan)],
            6,
        )

        self.last_llm_metadata = self._metadata(
            preprocessing,
            llm_used=True,
            invalid_llm_output=False,
            llm_error_type="",
        )
        return plan
