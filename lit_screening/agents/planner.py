"""Scholarly query planner with optional LLM enhancement."""

from __future__ import annotations

from typing import Any

from lit_screening.llm_client import GenericLLMClient


class PlannerAgent:
    """Create a compact set of academic search queries from a question."""

    expansion_terms = [
        "systematic review",
        "literature screening",
        "evidence verification",
        "claim extraction",
        "human feedback",
        "LLM agents",
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
        """Return 4 to 6 search queries, including the original question."""

        if self.mode == "llm" and self.llm_client and self.llm_client.is_available:
            return self._plan_with_llm(question)
        self.last_llm_metadata = {
            "planner_mode": self.mode,
            "llm_used": False,
            "invalid_llm_output": False,
            "llm_error_type": "llm_unavailable" if self.mode == "llm" else "",
        }
        return self._plan_rule(question)

    def _plan_rule(self, question: str) -> list[str]:
        """Rule-based query planning fallback."""

        base = " ".join(question.split())
        queries = [
            base,
            f"{base} systematic review literature screening",
            "human-in-the-loop LLM agents scientific literature screening",
            "LLM agents evidence verification claim extraction scholarly abstracts",
            "human feedback relevance ranking evidence verification literature review",
            "multi-agent LLM systems claim extraction systematic review",
        ]
        unique: list[str] = []
        for query in queries:
            if query and query not in unique:
                unique.append(query)
        return unique[:6]

    def _plan_with_llm(self, question: str) -> list[str]:
        """Use an LLM to suggest queries, falling back safely."""

        system_prompt = (
            "You plan scholarly search queries for a literature-screening pipeline. "
            "Return JSON only with key 'queries', a list of 4 to 6 concise academic "
            "queries. Include the original question and terms such as systematic "
            "review, literature screening, evidence verification, claim extraction, "
            "human feedback, and LLM agents."
        )
        user_prompt = f"Research question:\n{question}"
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        rule_queries = self._plan_rule(question)
        queries = result.data.get("queries")

        if result.invalid_llm_output or not isinstance(queries, list):
            self.last_llm_metadata = {
                "planner_mode": self.mode,
                "llm_used": True,
                "invalid_llm_output": True,
                "llm_error_type": result.error_type or "missing_queries",
            }
            return rule_queries

        unique: list[str] = []
        for query in [question, *queries, *rule_queries]:
            if not isinstance(query, str):
                continue
            cleaned = " ".join(query.split())
            if cleaned and cleaned not in unique:
                unique.append(cleaned)

        if len(unique) < 4:
            self.last_llm_metadata = {
                "planner_mode": self.mode,
                "llm_used": True,
                "invalid_llm_output": True,
                "llm_error_type": "too_few_queries",
            }
            return rule_queries

        self.last_llm_metadata = {
            "planner_mode": self.mode,
            "llm_used": True,
            "invalid_llm_output": False,
            "llm_error_type": "",
        }
        return unique[:6]
