"""Scholarly query planner with optional LLM enhancement."""

from __future__ import annotations

import re
from typing import Any

from lit_screening.llm_client import GenericLLMClient
from lit_screening.utils import tokenize


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

        if self.mode == "llm" and self.llm_client and self.llm_client.is_available:
            return self._plan_with_llm(question)
        preprocessing = self._preprocess_question_rule(question)
        self.last_llm_metadata = {
            "planner_mode": self.mode,
            "llm_used": False,
            "invalid_llm_output": False,
            "llm_error_type": "llm_unavailable" if self.mode == "llm" else "",
            **preprocessing,
        }
        return self._plan_rule(preprocessing["planning_question"])

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

        preprocessing = self._preprocess_question_rule(question)
        system_prompt = (
            "You plan scholarly search queries for a literature-screening pipeline. "
            "Return JSON only with keys 'translated_question' and 'queries'. "
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

        rule_queries = self._plan_rule(preprocessing["planning_question"])
        queries = result.data.get("queries")

        if result.invalid_llm_output or not isinstance(queries, list):
            self.last_llm_metadata = self._metadata(
                preprocessing,
                llm_used=True,
                invalid_llm_output=True,
                llm_error_type=result.error_type or "missing_queries",
            )
            return rule_queries

        unique: list[str] = []
        for query in [preprocessing["planning_question"], *queries, *rule_queries]:
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
            return rule_queries

        self.last_llm_metadata = self._metadata(
            preprocessing,
            llm_used=True,
            invalid_llm_output=False,
            llm_error_type="",
        )
        return unique[:6]
