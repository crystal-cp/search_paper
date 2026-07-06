"""Scholarly query planner with optional LLM enhancement."""

from __future__ import annotations

import re
from typing import Any

from lit_screening.llm_client import GenericLLMClient
from lit_screening.models import QueryPlan, SearchBrief
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
    ) -> QueryPlan:
        """Return a structured, provider-aware query plan."""

        if self.mode == "llm" and self.llm_client and self.llm_client.is_available:
            return self._plan_structured_with_llm(
                question,
                strictness=strictness,
                openalex_mode=openalex_mode,
                sort_preference=sort_preference,
                ranking_profile=ranking_profile,
                search_brief=search_brief,
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
    ) -> QueryPlan:
        """Build a topic-aware structured plan without requiring an LLM."""

        planning_question = search_brief.refined_question if search_brief else preprocessing["planning_question"]
        detected_language = "zh" if preprocessing["question_language"] == "zh" else "en"
        tokens = tokenize(planning_question)
        candidate_phrases = _extract_candidate_phrases(planning_question)
        core_terms = list(candidate_phrases)
        for token in tokens:
            if not _is_term_redundant(token, core_terms):
                core_terms.append(token)
        core_terms = _unique(core_terms, 8)
        if not core_terms and planning_question:
            core_terms = [planning_question]

        must_count = 4 if strictness == "strict" else 3
        if strictness == "broad":
            must_count = 1
        must_terms = _unique(core_terms[:must_count], 5)
        if search_brief:
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

        topical_optional = [
            token
            for token in tokens
            if token not in set(tokenize(" ".join(core_terms[:must_count])))
        ]
        generic_optional = ["review", "recent advances", "mechanism", "applications"]
        if ranking_profile == "high_quality_review":
            generic_optional = ["review", "systematic review", "survey", "meta-analysis"]
        optional_terms = _unique([*topical_optional, *generic_optional], 8)
        if search_brief:
            optional_terms = _unique(
                [
                    *optional_terms,
                    *search_brief.preferred_paper_types,
                    *search_brief.required_aspects,
                ],
                12,
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
                ],
                8,
            ),
            required_aspects=(search_brief.required_aspects if search_brief else []),
            filters={
                "strictness": strictness,
                "openalex_mode": openalex_mode,
                "sort_preference": sort_preference,
                "ranking_profile": ranking_profile,
            },
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
            required_aspects=search_brief.required_aspects if search_brief else rule_plan.required_aspects,
            filters={
                "strictness": strictness,
                "openalex_mode": openalex_mode,
                "sort_preference": sort_preference,
                "ranking_profile": ranking_profile,
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
