"""Pilot search diagnostics before full retrieval."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from lit_screening.agents.domain_guardrail import DomainGuardrailAgent
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.config import PipelineConfig
from lit_screening.models import Paper, QueryPlan, SearchContract


class QueryPilotAgent:
    """Run small pilot retrievals and diagnose query drift."""

    def __init__(
        self,
        retriever_agent: RetrieverAgent | None = None,
        domain_guardrail: DomainGuardrailAgent | None = None,
    ) -> None:
        self.retriever_agent = retriever_agent
        self.domain_guardrail = domain_guardrail or DomainGuardrailAgent()

    def run(
        self,
        queries: dict[str, list[str]],
        providers: list[str],
        search_contract: SearchContract,
        query_plan: QueryPlan,
        max_per_query: int = 5,
        from_year: int | None = None,
        use_cache: bool = True,
        cache_dir: str = "data/cache",
        openalex_mode: str = "keyword+semantic",
        sort_mode: str = "relevance",
        progress_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run a low-volume pilot search and return drift diagnostics."""

        retriever = self.retriever_agent or RetrieverAgent(
            config=PipelineConfig(use_cache=use_cache, cache_dir=cache_dir)
        )
        pilot_limit = max(0, int(max_per_query))
        raw_papers, raw_by_provider, retrieval_counts = retriever.retrieve(
            queries=queries,
            providers=providers,
            max_per_query=pilot_limit,
            from_year=from_year,
            output_dir=None,
            progress_callback=progress_callback,
            sort_mode=sort_mode,
            openalex_mode=openalex_mode,
        )
        paper_by_id = {paper.paper_id: paper for paper in raw_papers}
        results: list[dict[str, Any]] = []
        for provider, bundles in raw_by_provider.items():
            for bundle in bundles:
                papers = [
                    paper_by_id[paper_id]
                    for paper_id in bundle.get("paper_ids", [])
                    if paper_id in paper_by_id
                ]
                assessments = self.domain_guardrail.assess_many(
                    papers,
                    search_contract,
                    query_plan=query_plan,
                )
                results.append(
                    build_pilot_result(
                        provider=provider,
                        bundle=bundle,
                        papers=papers,
                        assessments=assessments,
                    )
                )
        return {
            "enabled": True,
            "pilot_max_per_query": pilot_limit,
            "retrieval_counts": retrieval_counts,
            "results": results,
            "summary": summarize_pilot_results(results),
        }


def build_pilot_result(
    provider: str,
    bundle: dict[str, Any],
    papers: list[Paper],
    assessments: list[Any],
) -> dict[str, Any]:
    """Build one provider/query/stage diagnostic record."""

    non_scope = [
        assessment
        for assessment in assessments
        if assessment.domain_decision != "in_scope"
    ]
    off_topic_rate = len(non_scope) / len(assessments) if assessments else 0.0
    reasons = Counter(
        assessment.off_topic_reason
        for assessment in non_scope
        if assessment.off_topic_reason
    )
    drift = detect_drift_categories(assessments)
    recommendation = recommend_query_action(
        raw_count=int(bundle.get("paper_count") or 0),
        off_topic_rate=off_topic_rate,
        detected_drift=drift,
    )
    return {
        "provider": provider,
        "query": bundle.get("query", ""),
        "retrieval_stage": bundle.get("retrieval_stage", provider),
        "search_mode": bundle.get("search_mode", ""),
        "pilot_raw_count": int(bundle.get("paper_count") or 0),
        "pilot_top_titles": [paper.title for paper in papers[:5] if paper.title],
        "missing_abstract_count": int(bundle.get("missing_abstract_count") or 0),
        "off_topic_rate_estimate": round(off_topic_rate, 4),
        "common_off_topic_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common(5)
        ],
        "detected_drift": drift,
        "recommendation": recommendation,
    }


def detect_drift_categories(assessments: list[Any]) -> list[str]:
    """Infer human-readable query-drift categories from domain assessments."""

    categories: list[str] = []
    text = " ".join(
        [
            *[
                " ".join(assessment.negative_domain_matches)
                for assessment in assessments
            ],
            *[
                " ".join(assessment.forbidden_concepts_found)
                for assessment in assessments
            ],
            *[assessment.off_topic_reason for assessment in assessments],
        ]
    ).lower()
    drift_rules = [
        ("healthcare_screening_drift", ["patient screening", "drug screening", "biomarker", "medicine", "clinical"]),
        ("biological_agent_drift", ["biological agent", "infectious agent", "chemical agent"]),
        ("materials_screening_drift", ["materials screening", "high-throughput materials"]),
        ("llm_agent_drift", ["llm agent", "large language model", "software agent"]),
    ]
    for category, markers in drift_rules:
        if any(marker in text for marker in markers):
            categories.append(category)
    return categories


def recommend_query_action(
    raw_count: int,
    off_topic_rate: float,
    detected_drift: list[str],
) -> str:
    """Recommend whether a planned query should be kept, repaired, or dropped."""

    if raw_count <= 0:
        return "repair"
    if off_topic_rate >= 0.8:
        return "drop"
    if off_topic_rate >= 0.35 or detected_drift:
        return "repair"
    return "keep"


def summarize_pilot_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize pilot diagnostics across queries."""

    counts = Counter(result["recommendation"] for result in results)
    drift = Counter(
        category
        for result in results
        for category in result.get("detected_drift", [])
    )
    return {
        "query_count": len(results),
        "recommendation_counts": dict(counts),
        "detected_drift_counts": dict(drift),
        "mean_off_topic_rate": (
            round(
                sum(result["off_topic_rate_estimate"] for result in results) / len(results),
                4,
            )
            if results
            else 0.0
        ),
    }
