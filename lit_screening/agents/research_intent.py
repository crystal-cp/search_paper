"""Research-intent interpretation for literature sensemaking."""

from __future__ import annotations

from typing import Any

from lit_screening.agents.planner import contains_cjk, fallback_translate_chinese_question
from lit_screening.llm_client import GenericLLMClient
from lit_screening.models import SearchBrief
from lit_screening.utils import tokenize


INTENT_KEYWORDS = {
    "systematic_review": {"systematic review", "prisma", "meta-analysis", "screening protocol"},
    "frontier": {"frontier", "recent", "latest", "emerging", "state-of-the-art", "trend"},
    "implementation": {"implement", "implementation", "code", "pipeline", "system", "workflow", "method"},
    "evidence_verification": {"evidence", "verify", "verification", "grounded", "claim"},
    "proposal": {"proposal", "novelty", "gap", "future", "research direction", "phd"},
    "overview": {"overview", "introduction", "background", "significance", "importance", "review"},
}

INTENT_PAPER_TYPES = {
    "overview": ["review", "survey", "tutorial"],
    "frontier": ["recent article", "high-impact paper", "preprint"],
    "implementation": ["method paper", "system paper", "benchmark paper"],
    "evidence_verification": ["evaluation paper", "benchmark paper", "empirical study"],
    "proposal": ["position paper", "recent article", "review"],
    "systematic_review": ["systematic review", "meta-analysis", "review"],
}


class ResearchIntentAgent:
    """Create a SearchBrief that captures why the user is searching."""

    def __init__(
        self,
        mode: str = "rule",
        llm_client: GenericLLMClient | None = None,
    ) -> None:
        self.mode = mode
        self.llm_client = llm_client

    def analyze(self, question: str) -> SearchBrief:
        """Return a SearchBrief, using LLM only when explicitly available."""

        fallback = self._analyze_rule(question)
        if self.mode == "llm" and self.llm_client and self.llm_client.is_available:
            return self._analyze_with_llm(question, fallback)
        return fallback

    def _analyze_rule(self, question: str) -> SearchBrief:
        """Rule-based intent interpretation."""

        cleaned = " ".join(question.split())
        refined = fallback_translate_chinese_question(cleaned) if contains_cjk(cleaned) else cleaned
        lowered = refined.lower()
        tokens = set(tokenize(lowered))
        intent = "overview"
        best_score = 0
        for candidate, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in lowered or keyword in tokens)
            if score > best_score:
                intent = candidate
                best_score = score

        if best_score <= 0 and any(word in lowered for word in ["how", "can", "improve"]):
            intent = "proposal"

        topic_terms = [term for term in tokenize(refined)[:8]]
        inclusion = [refined]
        if intent == "frontier":
            inclusion.append("recent advances")
        elif intent == "implementation":
            inclusion.extend(["method", "implementation"])
        elif intent == "evidence_verification":
            inclusion.extend(["evidence", "evaluation"])
        elif intent == "systematic_review":
            inclusion.extend(["systematic review", "screening criteria"])

        required_aspects = _required_aspects_for_intent(intent, topic_terms)
        return SearchBrief(
            original_question=cleaned,
            refined_question=refined,
            search_intent=intent,
            user_goal=_user_goal_for_intent(intent),
            inclusion_criteria=_unique(inclusion),
            exclusion_criteria=[],
            required_aspects=required_aspects,
            preferred_paper_types=INTENT_PAPER_TYPES.get(intent, ["article"]),
            time_window="recent 5 years" if intent == "frontier" else "no strict time window",
            success_definition=_success_definition_for_intent(intent),
        )

    def _analyze_with_llm(self, question: str, fallback: SearchBrief) -> SearchBrief:
        """Optional LLM interpretation with safe fallback."""

        system_prompt = (
            "You convert a research question into a literature-search brief. "
            "Return JSON only with keys: refined_question, search_intent, user_goal, "
            "inclusion_criteria, exclusion_criteria, required_aspects, "
            "preferred_paper_types, time_window, success_definition. "
            "Allowed search_intent values: overview, frontier, implementation, "
            "evidence_verification, proposal, systematic_review. Do not introduce "
            "LLM, AI, agents, or human-feedback terms unless the user's question needs them."
        )
        result = self.llm_client.chat_json(system_prompt, f"Question:\n{question}")
        if result.invalid_llm_output:
            return fallback
        data = result.data
        intent = str(data.get("search_intent") or fallback.search_intent)
        if intent not in INTENT_PAPER_TYPES:
            intent = fallback.search_intent
        return SearchBrief(
            original_question=fallback.original_question,
            refined_question=_string(data.get("refined_question"), fallback.refined_question),
            search_intent=intent,
            user_goal=_string(data.get("user_goal"), fallback.user_goal),
            inclusion_criteria=_string_list(data.get("inclusion_criteria"), fallback.inclusion_criteria),
            exclusion_criteria=_string_list(data.get("exclusion_criteria"), fallback.exclusion_criteria),
            required_aspects=_string_list(data.get("required_aspects"), fallback.required_aspects),
            preferred_paper_types=_string_list(
                data.get("preferred_paper_types"),
                fallback.preferred_paper_types,
            ),
            time_window=_string(data.get("time_window"), fallback.time_window),
            success_definition=_string(
                data.get("success_definition"),
                fallback.success_definition,
            ),
        )


def _required_aspects_for_intent(intent: str, topic_terms: list[str]) -> list[str]:
    base = topic_terms[:3] or ["topic relevance"]
    if intent == "frontier":
        return _unique([*base, "recent advances", "open challenges"])
    if intent == "implementation":
        return _unique([*base, "method", "implementation details", "evaluation"])
    if intent == "evidence_verification":
        return _unique([*base, "claim", "evidence", "evaluation"])
    if intent == "proposal":
        return _unique([*base, "research gap", "future work", "novelty"])
    if intent == "systematic_review":
        return _unique([*base, "eligibility criteria", "screening", "evidence"])
    return _unique([*base, "background", "significance"])


def _user_goal_for_intent(intent: str) -> str:
    return {
        "overview": "Build a reliable background understanding of the topic.",
        "frontier": "Find recent papers that reveal active research directions.",
        "implementation": "Find methods and systems that can be implemented or compared.",
        "evidence_verification": "Find papers with evidence that can support or refute claims.",
        "proposal": "Find gaps, motivations, and future-work directions for a proposal.",
        "systematic_review": "Support systematic screening with explicit inclusion criteria.",
    }.get(intent, "Find papers aligned with the research question.")


def _success_definition_for_intent(intent: str) -> str:
    return {
        "overview": "A small set of background papers plus clear concepts.",
        "frontier": "Recent, high-signal papers and unresolved challenges.",
        "implementation": "Methods, baselines, and evaluation details worth reproducing.",
        "evidence_verification": "Grounded papers with auditable abstract evidence.",
        "proposal": "Motivation, gap, and method directions for a proposal.",
        "systematic_review": "Transparent inclusion/exclusion decisions and traceable evidence.",
    }.get(intent, "A ranked list of relevant, evidence-grounded papers.")


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _string(value: Any, fallback: str) -> str:
    return " ".join(str(value).split()) if isinstance(value, str) and value.strip() else fallback


def _string_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    parsed = [item for item in value if isinstance(item, str) and item.strip()]
    return _unique(parsed) or fallback
