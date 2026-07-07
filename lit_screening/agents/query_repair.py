"""Query repair suggestions after pilot-search drift diagnostics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from lit_screening.models import QueryPlan, SearchContract


class QueryRepairAgent:
    """Suggest query repairs using ambiguity and SearchContract constraints."""

    def suggest(
        self,
        query_plan: QueryPlan,
        search_contract: SearchContract,
        ambiguity_analysis: list[dict[str, Any]],
        pilot_diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        """Return repaired query suggestions without applying them."""

        suggestions: list[dict[str, Any]] = []
        for result in pilot_diagnostics.get("results", []):
            recommendation = result.get("recommendation", "keep")
            if recommendation == "keep":
                continue
            provider = result.get("provider", "")
            original_query = str(result.get("query") or "")
            repaired_query, details = repair_query_text(
                original_query,
                provider=provider,
                search_contract=search_contract,
                ambiguity_analysis=ambiguity_analysis,
                detected_drift=list(result.get("detected_drift", [])),
            )
            suggestions.append(
                {
                    "provider": provider,
                    "original_query": original_query,
                    "retrieval_stage": result.get("retrieval_stage", provider),
                    "recommendation": recommendation,
                    "repaired_query": repaired_query,
                    "added_required_phrases": details["added_required_phrases"],
                    "added_excluded_concepts": details["added_excluded_concepts"],
                    "replaced_terms": details["replaced_terms"],
                    "reason": repair_reason(result),
                }
            )
        repaired_plan = apply_query_repair_suggestions(query_plan, suggestions)
        return {
            "enabled": True,
            "applied": False,
            "suggestions": suggestions,
            "repaired_query_plan": repaired_plan,
        }


def repair_query_text(
    query: str,
    provider: str,
    search_contract: SearchContract,
    ambiguity_analysis: list[dict[str, Any]],
    detected_drift: list[str],
) -> tuple[str, dict[str, list[str]]]:
    """Return one repaired query and details about the edits."""

    repaired = " ".join(query.split())
    replaced_terms: list[str] = []
    for record in ambiguity_analysis:
        term = str(record.get("term") or "").lower()
        selected = str(record.get("selected_meaning") or "").lower()
        if term == "screening" and selected == "literature screening":
            if "literature screening" not in repaired.lower():
                repaired = replace_word(repaired, "screening", "literature screening")
                replaced_terms.append("screening -> literature screening")
        if term == "agent" and selected == "software/llm agent":
            if "llm agent" not in repaired.lower() and "multi-agent" not in repaired.lower():
                repaired = replace_word(repaired, "agent", "LLM agent")
                replaced_terms.append("agent -> LLM agent")

    required_phrases = select_required_phrases(search_contract, detected_drift)
    excluded_concepts = select_excluded_concepts(search_contract, detected_drift)
    repaired = add_required_phrases(repaired, provider, required_phrases)
    repaired = add_excluded_concepts(repaired, provider, excluded_concepts)
    return repaired, {
        "added_required_phrases": required_phrases,
        "added_excluded_concepts": excluded_concepts,
        "replaced_terms": replaced_terms,
    }


def replace_word(query: str, word: str, replacement: str) -> str:
    """Replace a standalone ambiguous word with a precise phrase."""

    parts = query.split()
    replaced = [replacement if part.lower().strip('"') == word else part for part in parts]
    return " ".join(replaced)


def select_required_phrases(
    search_contract: SearchContract,
    detected_drift: list[str],
) -> list[str]:
    """Select short required phrases to anchor a repaired query."""

    phrases = list(search_contract.must_include_concepts[:3])
    if "healthcare_screening_drift" in detected_drift and "literature screening" not in phrases:
        phrases.insert(0, "literature screening")
    if "biological_agent_drift" in detected_drift:
        for phrase in ["LLM agent", "software agent", "multi-agent system"]:
            if phrase not in phrases:
                phrases.append(phrase)
    return unique(phrases, 5)


def select_excluded_concepts(
    search_contract: SearchContract,
    detected_drift: list[str],
) -> list[str]:
    """Select excluded concepts to reduce detected drift."""

    excluded = list(search_contract.must_exclude_concepts)
    if "healthcare_screening_drift" in detected_drift:
        excluded.extend(["patient screening", "drug screening", "biomarker screening"])
    if "biological_agent_drift" in detected_drift:
        excluded.extend(["biological agent", "infectious agent", "chemical agent"])
    return unique(excluded, 8)


def add_required_phrases(query: str, provider: str, phrases: list[str]) -> str:
    """Add required phrases using provider-appropriate syntax."""

    parts = [query] if query else []
    lowered = query.lower()
    for phrase in phrases:
        if phrase.lower() in lowered:
            continue
        if provider == "semantic_scholar":
            parts.append(f'+{quote_phrase(phrase)}')
        else:
            parts.append(quote_phrase(phrase))
    return " ".join(parts)


def add_excluded_concepts(query: str, provider: str, concepts: list[str]) -> str:
    """Add excluded concepts using provider-appropriate syntax."""

    parts = [query] if query else []
    lowered = query.lower()
    for concept in concepts:
        quoted = quote_phrase(concept)
        if concept.lower() in lowered or f"-{quoted.lower()}" in lowered:
            continue
        if provider == "semantic_scholar":
            parts.append(f"-{quoted}")
        else:
            parts.append(f"NOT {quoted}")
    return " ".join(parts)


def apply_query_repair_suggestions(
    query_plan: QueryPlan,
    suggestions: list[dict[str, Any]],
) -> QueryPlan:
    """Apply repair/drop suggestions to a QueryPlan."""

    openalex = apply_provider_suggestions(
        query_plan.openalex_queries,
        suggestions,
        provider="openalex",
    )
    semantic = apply_provider_suggestions(
        query_plan.semantic_scholar_queries,
        suggestions,
        provider="semantic_scholar",
    )
    return replace(query_plan, openalex_queries=openalex, semantic_scholar_queries=semantic)


def apply_provider_suggestions(
    queries: list[str],
    suggestions: list[dict[str, Any]],
    provider: str,
) -> list[str]:
    """Apply suggestions for one provider."""

    by_query = {
        str(item.get("original_query") or ""): item
        for item in suggestions
        if item.get("provider") == provider
    }
    repaired: list[str] = []
    for query in queries:
        suggestion = by_query.get(query)
        if suggestion and suggestion.get("recommendation") == "drop":
            continue
        if suggestion and suggestion.get("repaired_query"):
            repaired.append(str(suggestion["repaired_query"]))
        else:
            repaired.append(query)
    for suggestion in suggestions:
        if suggestion.get("provider") != provider:
            continue
        repaired_query = str(suggestion.get("repaired_query") or "")
        if suggestion.get("recommendation") == "repair" and repaired_query not in repaired:
            repaired.append(repaired_query)
    return unique(repaired)


def repair_reason(pilot_result: dict[str, Any]) -> str:
    """Describe why a query was repaired or dropped."""

    drift = ", ".join(pilot_result.get("detected_drift", [])) or "high off-topic rate"
    rate = pilot_result.get("off_topic_rate_estimate", 0)
    return f"Pilot search estimated off-topic rate {rate}; detected drift: {drift}."


def quote_phrase(phrase: str) -> str:
    """Quote multi-word phrases for scholarly search providers."""

    cleaned = " ".join(str(phrase).split())
    return f'"{cleaned}"' if " " in cleaned else cleaned


def unique(values: list[str], limit: int | None = None) -> list[str]:
    """Return unique non-empty strings while preserving order."""

    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result[:limit] if limit else result
