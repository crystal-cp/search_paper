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


def _plain_terms_query(terms: list[str], limit: int = 3) -> str:
    """Join terms without provider operators for broad recall-oriented search."""

    return " ".join(" ".join(term.split()) for term in terms[:limit])


def build_openalex_queries(plan: QueryPlan) -> list[str]:
    """Build OpenAlex-flavored queries from a structured query plan."""

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
    return _unique(queries, 6)


def build_semantic_scholar_queries(plan: QueryPlan) -> list[str]:
    """Build Semantic Scholar-flavored queries from a structured query plan."""

    if _is_expert_plan(plan):
        return _expert_generic_semantic_scholar_queries(plan)

    core = plan.core_terms or plan.must_terms
    must = plan.must_terms or core
    optional = plan.optional_terms
    natural_query = plan.translated_question or plan.original_question
    required = " ".join(_semantic_required(term) for term in must[:4])
    optional_or = f"({' OR '.join(_quote_phrase(term) for term in optional[:4])})" if optional else ""
    primary = " ".join(part for part in [required, optional_or] if part)
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
        queries.append(" ".join(part for part in [phrase, optional_or] if part))
        queries.append(" ".join(part for part in [phrase, "review"] if part))
    if optional:
        queries.append(
            " ".join(
                    part
                    for part in [
                        required,
                        f"({' OR '.join(_quote_phrase(term) for term in optional[:5])})",
                    ]
                    if part
                )
        )
    return _unique(queries, 6)


def _is_expert_plan(plan: QueryPlan) -> bool:
    return bool(plan.expert_rewritten_question and plan.filters.get("structured_concepts"))


def _expert_generic_openalex_queries(plan: QueryPlan) -> list[str]:
    """Short expert-intent OpenAlex queries from activated concepts only."""

    concepts = _concepts_from_filters(plan)
    queries = _provider_neutral_expert_queries(
        concepts,
        include_seed_titles=False,
        terminology_map=plan.filters.get("terminology_map", {}),
    )
    return _unique(
        [
            query
            for query in queries
            if query and len(query) < 180
        ],
        12,
    )


def _expert_generic_semantic_scholar_queries(plan: QueryPlan) -> list[str]:
    """Short expert-intent Semantic Scholar queries from activated concepts only."""

    concepts = _concepts_from_filters(plan)
    must = _terms_by_role(concepts, "must")
    queries = [
        *_seed_title_queries(concepts),
        *_provider_neutral_expert_queries(
            concepts,
            include_seed_titles=False,
            terminology_map=plan.filters.get("terminology_map", {}),
        ),
    ]
    if must:
        required = " ".join(_semantic_required(term) for term in must[:3])
        queries.append(required)
        optional = _terms_by_role(concepts, "optional")
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
            query
            for query in queries
            if query and len(query) < 220
        ],
        6,
    )


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


def _provider_neutral_expert_queries(
    concepts: list[IntentConcept],
    include_seed_titles: bool,
    terminology_map: dict[str, list[str]] | None = None,
) -> list[str]:
    """Compose short queries from structured concept categories, not domain strings."""

    queries: list[str] = []
    active_terms = _active_terms(concepts)
    prioritized_must_terms = _terms_by_role_priority(concepts, "must")
    conceptual_terms = _unique(
        [
            *prioritized_must_terms,
            *_terms_by_category_from_concepts(concepts, "object"),
            *_terms_by_category_from_concepts(concepts, "property"),
            *_terms_by_category_from_concepts(concepts, "mechanism"),
            *_terms_by_category_from_concepts(concepts, "concept"),
            *_terms_by_role(concepts, "must"),
        ],
        12,
    )
    case_terms = _terms_by_category_from_concepts(concepts, "material")
    methods = _terms_by_category_from_concepts(concepts, "method")
    applications = _terms_by_category_from_concepts(concepts, "application")
    base_terms = conceptual_terms or active_terms
    expanded_base_terms = _provider_terms(base_terms, terminology_map)
    expanded_optional = _provider_terms(_terms_by_role(concepts, "optional"), terminology_map)
    case_terms = _provider_terms(case_terms, terminology_map)
    methods = _provider_terms(methods, terminology_map)
    applications = _provider_terms(applications, terminology_map)
    queries.extend(
        _quote_phrase(term)
        for term in _provider_terms(prioritized_must_terms[:6], terminology_map)
        if term
    )
    queries.extend(_quote_phrase(term) for term in case_terms[:2] if term)
    for method in _selected_provider_methods(methods):
        if expanded_base_terms:
            queries.append(f"{_quote_phrase(method)} {_quote_phrase(expanded_base_terms[0])}")
    queries.extend(_structured_concept_queries(expanded_base_terms, expanded_optional))
    for case in case_terms[:4]:
        for concept in expanded_base_terms[:2]:
            queries.append(f"{_quote_phrase(case)} {_quote_phrase(concept)}")
    for method in methods[:5]:
        for concept in expanded_base_terms[:2]:
            queries.append(f"{_quote_phrase(method)} {_quote_phrase(concept)}")
        if case_terms and expanded_base_terms:
            queries.append(
                f"{_quote_phrase(method)} {_quote_phrase(case_terms[0])} {_quote_phrase(expanded_base_terms[0])}"
            )
    for application in applications[:3]:
        if expanded_base_terms:
            queries.append(f"{_quote_phrase(expanded_base_terms[0])} {_quote_phrase(application)}")
    if include_seed_titles:
        queries.extend(_seed_title_queries(concepts))
    if not queries:
        queries = _short_pair_queries(_provider_terms(active_terms, terminology_map))
    return _unique([query for query in queries if query.strip()], 14)


def _provider_terms(
    terms: list[str],
    terminology_map: dict[str, list[str]] | None,
) -> list[str]:
    """Use provider-friendly terminology-map expansions when available."""

    if not terminology_map:
        return list(terms)
    result: list[str] = []
    normalized_map = {
        str(key).lower(): [str(value) for value in values if str(value).strip()]
        for key, values in terminology_map.items()
        if isinstance(values, list)
    }
    for term in terms:
        expanded = normalized_map.get(str(term).lower(), [])
        if expanded and _should_expand_provider_term(term, expanded[0]):
            result.append(expanded[0])
        else:
            result.append(term)
    return _unique(result, 24)


def _selected_provider_methods(methods: list[str]) -> list[str]:
    """Keep early query budget diverse across activated methods."""

    if len(methods) <= 2:
        return methods
    return _unique([methods[0], methods[-1]], 2)


def _should_expand_provider_term(term: str, expansion: str) -> bool:
    """Expand compact method acronyms without replacing ordinary concept phrases."""

    cleaned = " ".join(str(term or "").split())
    expanded = " ".join(str(expansion or "").split())
    if not cleaned or not expanded:
        return False
    if len(cleaned) <= 5 and cleaned.upper() == cleaned:
        return True
    return False


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


def _terms_by_role_priority(concepts: list[IntentConcept], role: str) -> list[str]:
    """Return role terms ordered by confidence for early query budget slots."""

    ranked = sorted(
        [
            concept
            for concept in _query_eligible(concepts)
            if concept.query_role == role
        ],
        key=lambda concept: (-concept.confidence, concept.term.lower()),
    )
    return _unique([concept.term for concept in ranked], 12)


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
            and value.lower() not in GENERIC_TOPIC_MODIFIERS
        ],
        24,
    )


def _constraint_groups_from_contract(search_contract: SearchContract | None) -> list[list[str]]:
    """Return generic required search groups from a SearchContract."""

    if not search_contract:
        return []
    return [
        _unique(group.terms, 12)
        for group in search_contract.constraint_groups
        if group.required and group.terms
    ]


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
        if expert_intent:
            must_groups = _constraint_groups_from_contract(search_contract)
            must_terms = _unique(
                [
                    concept.term
                    for concept in _query_eligible(expert_intent.structured_concepts)
                    if concept.query_role == "must"
                ],
                12,
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
        12,
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
                "constraint_groups": [
                    {
                        "group_name": group.group_name,
                        "operator": group.operator,
                        "terms": list(group.terms),
                        "source": group.source,
                        "required": group.required,
                    }
                    for group in (search_contract.constraint_groups if search_contract else [])
                ],
                "terminology_map": (
                    search_contract.domain_profile.terminology_map
                    if search_contract
                    else {}
                ),
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
