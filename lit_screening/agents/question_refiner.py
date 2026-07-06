"""Question refinement for broad or mixed literature-search needs."""

from __future__ import annotations

from lit_screening.models import SearchBrief
from lit_screening.utils import tokenize


class QuestionRefinementAgent:
    """Suggest subquestions while keeping the pipeline optional and robust."""

    def refine(self, question: str, search_brief: SearchBrief | None = None) -> dict:
        """Return broadness, mixed-goal flags, and suggested subquestions."""

        brief = search_brief
        refined = brief.refined_question if brief else " ".join(question.split())
        tokens = tokenize(refined)
        broad = len(tokens) <= 5 or any(
            marker in refined.lower()
            for marker in ["significance", "overview", "impact", "role", "importance"]
        )
        mixed_goals = _detect_mixed_goals(refined)
        suggestions: list[str] = []
        aspects = (brief.required_aspects if brief else [])[:4]
        if broad and aspects:
            suggestions = [f"What does the literature say about {aspect}?" for aspect in aspects]
        elif broad:
            suggestions = [
                f"What are the main concepts in {refined}?",
                f"What evidence supports {refined}?",
                f"What are recent developments in {refined}?",
            ]
        if mixed_goals:
            suggestions.append("Which papers address methods, evidence, and research gaps separately?")
        return {
            "original_question": question,
            "refined_question": refined,
            "is_broad": broad,
            "mixed_goals": mixed_goals,
            "subquestions": _unique(suggestions[:4]),
        }


def _detect_mixed_goals(question: str) -> bool:
    lowered = question.lower()
    goal_markers = [
        {"overview", "background", "significance"},
        {"implementation", "method", "system"},
        {"evidence", "verify", "evaluation"},
        {"proposal", "gap", "future"},
    ]
    return sum(1 for group in goal_markers if group & set(tokenize(lowered))) >= 2


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(value.split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
