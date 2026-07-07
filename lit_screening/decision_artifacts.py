"""Decision-oriented report artifacts for literature screening."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from lit_screening.models import (
    AspectCoverageRecord,
    QueryPlan,
    RankedPaper,
    SearchContract,
)
from lit_screening.result_groups import recommended_role
from lit_screening.utils import ensure_dir, tokenize, write_csv, write_json


METHOD_COMPARISON_FIELDS = [
    "paper_id",
    "title",
    "problem",
    "method",
    "human_role",
    "agent_design",
    "retrieval_source",
    "evidence_verification",
    "evaluation",
    "limitation",
    "recommended_use",
]

RESEARCH_GAP_FIELDS = [
    "gap",
    "supporting_papers",
    "why_gap_remains",
    "possible_project_idea",
    "related_aspects",
]

NEXT_SEARCH_FIELDS = [
    "query",
    "reason",
    "source",
]


def write_decision_artifacts(
    output_dir: str | Path,
    ranked_papers: list[RankedPaper],
    aspect_coverage_records: list[AspectCoverageRecord],
    search_contract: SearchContract | None = None,
    query_plan: QueryPlan | None = None,
    query_pilot_diagnostics: dict[str, Any] | None = None,
    prisma_like_flow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write decision matrices and next-search suggestions."""

    destination = ensure_dir(output_dir)
    method_rows = build_method_comparison_matrix(
        ranked_papers,
        aspect_coverage_records,
    )
    gap_rows = build_research_gap_matrix(
        ranked_papers,
        aspect_coverage_records,
        search_contract,
        prisma_like_flow,
    )
    next_searches = build_suggested_next_searches(
        ranked_papers,
        gap_rows,
        search_contract,
        query_plan,
        query_pilot_diagnostics,
        prisma_like_flow,
    )

    write_csv(
        destination / "method_comparison_matrix.csv",
        method_rows,
        METHOD_COMPARISON_FIELDS,
    )
    write_markdown_table(
        destination / "method_comparison_matrix.md",
        "Method Comparison Matrix",
        method_rows,
        METHOD_COMPARISON_FIELDS,
    )
    write_csv(
        destination / "research_gap_matrix.csv",
        gap_rows,
        RESEARCH_GAP_FIELDS,
    )
    write_markdown_table(
        destination / "research_gap_matrix.md",
        "Research Gap Matrix",
        gap_rows,
        RESEARCH_GAP_FIELDS,
    )
    write_json(destination / "suggested_next_searches.json", next_searches)
    write_markdown_table(
        destination / "suggested_next_searches.md",
        "Suggested Next Searches",
        next_searches,
        NEXT_SEARCH_FIELDS,
    )
    return {
        "method_comparison_matrix": method_rows,
        "research_gap_matrix": gap_rows,
        "suggested_next_searches": next_searches,
    }


def build_method_comparison_matrix(
    ranked_papers: list[RankedPaper],
    aspect_coverage_records: list[AspectCoverageRecord] | None = None,
) -> list[dict[str, Any]]:
    """Build a rule-based comparison table across ranked papers."""

    aspect_by_id = {record.paper_id: record for record in aspect_coverage_records or []}
    rows: list[dict[str, Any]] = []
    for item in ranked_papers:
        aspect = aspect_by_id.get(item.paper.paper_id)
        aspect_score = aspect.aspect_coverage_score if aspect else item.scores.aspect_coverage_score
        text = " ".join(
            [
                item.paper.title,
                item.paper.abstract,
                item.paper.tldr,
                item.evidence.claim,
                item.evidence.evidence_sentence,
            ]
        )
        rows.append(
            {
                "paper_id": item.paper.paper_id,
                "title": item.paper.title,
                "problem": infer_problem(item, aspect),
                "method": infer_method(text),
                "human_role": infer_human_role(text),
                "agent_design": infer_agent_design(text),
                "retrieval_source": retrieval_source(item),
                "evidence_verification": evidence_verification_summary(item),
                "evaluation": infer_evaluation(text),
                "limitation": infer_limitation(item, aspect),
                "recommended_use": recommended_role(item, aspect_score),
            }
        )
    return rows


def build_research_gap_matrix(
    ranked_papers: list[RankedPaper],
    aspect_coverage_records: list[AspectCoverageRecord],
    search_contract: SearchContract | None = None,
    prisma_like_flow: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Infer undercovered research gaps from aspect coverage and exclusions."""

    aspect_records = aspect_coverage_records or []
    gap_counts: Counter[str] = Counter()
    related_aspects: dict[str, set[str]] = {}
    total = max(1, len(aspect_records))
    for record in aspect_records:
        for aspect in record.missing_aspects:
            key = gap_key(aspect)
            gap_counts[key] += 1
            related_aspects.setdefault(key, set()).add(aspect)

    for reason, count in (prisma_like_flow or {}).get("common_exclusion_reasons", {}).items():
        key = gap_key(reason)
        if key != "general_relevance":
            gap_counts[key] += int(count)
            related_aspects.setdefault(key, set()).add(str(reason))

    for aspect in (search_contract.required_aspects if search_contract else []):
        key = gap_key(aspect)
        related_aspects.setdefault(key, set()).add(aspect)
        if key not in gap_counts and not supporting_papers_for_gap(ranked_papers, key):
            gap_counts[key] += 1

    if not gap_counts:
        gap_counts["evaluation"] = 1
        related_aspects["evaluation"] = {"evaluation protocol"}

    rows: list[dict[str, Any]] = []
    for key, count in gap_counts.most_common(8):
        if key == "general_relevance":
            continue
        support = supporting_papers_for_gap(ranked_papers, key)
        rows.append(
            {
                "gap": gap_label(key),
                "supporting_papers": "; ".join(support) or "No strong supporting papers in this run",
                "why_gap_remains": (
                    f"{count} signal(s) suggest this aspect is undercovered across "
                    f"{total} screened paper(s)."
                ),
                "possible_project_idea": gap_project_idea(key),
                "related_aspects": "; ".join(sorted(related_aspects.get(key, {key}))),
            }
        )
    if not rows:
        rows.append(
            {
                "gap": gap_label("evaluation"),
                "supporting_papers": "No strong supporting papers in this run",
                "why_gap_remains": "The current run did not expose a specific undercovered aspect, so evaluation remains the safest follow-up gap.",
                "possible_project_idea": gap_project_idea("evaluation"),
                "related_aspects": "evaluation protocol",
            }
        )
    return rows


def build_suggested_next_searches(
    ranked_papers: list[RankedPaper],
    gap_rows: list[dict[str, Any]],
    search_contract: SearchContract | None = None,
    query_plan: QueryPlan | None = None,
    query_pilot_diagnostics: dict[str, Any] | None = None,
    prisma_like_flow: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Suggest follow-up searches from gaps, drift, and included-paper keywords."""

    searches: list[dict[str, str]] = []
    base_terms = contract_base_terms(search_contract, query_plan)
    for row in gap_rows:
        query = " ".join([base_terms, search_phrase_for_gap(row["gap"])]).strip()
        add_search(
            searches,
            query,
            f"Investigate undercovered gap: {row['gap']}",
            "research_gap_matrix",
        )

    for reason in (prisma_like_flow or {}).get("common_exclusion_reasons", {}):
        if reason == "ambiguous_screening_meaning":
            add_search(
                searches,
                f'{base_terms} "literature screening" -"patient screening" -"drug screening"',
                "Reduce drift toward clinical or drug screening meanings.",
                "common_exclusion_reasons",
            )
        elif reason == "no_evidence_verification_component":
            add_search(
                searches,
                f'{base_terms} "evidence verification" "claim extraction"',
                "Find papers with an explicit evidence-verification component.",
                "common_exclusion_reasons",
            )
        elif reason == "no_multi_agent_component":
            add_search(
                searches,
                f'{base_terms} "multi-agent" "task allocation"',
                "Find papers with explicit multi-agent coordination.",
                "common_exclusion_reasons",
            )

    pilot = query_pilot_diagnostics or {}
    for result in pilot.get("results", [])[:8]:
        if result.get("recommendation") in {"repair", "drop"}:
            drift = " ".join(result.get("detected_drift", []))
            add_search(
                searches,
                f"{base_terms} {drift}".strip(),
                f"Pilot search flagged drift for query: {result.get('query', '')}",
                "query_pilot_diagnostics",
            )

    included_keywords = top_included_keywords(ranked_papers)
    if included_keywords:
        add_search(
            searches,
            " ".join([base_terms, *included_keywords[:3]]).strip(),
            "Expand around keywords from currently included papers.",
            "included_paper_keywords",
        )

    if not searches:
        add_search(
            searches,
            base_terms or "scientific literature screening evidence verification",
            "No dominant gap was detected; rerun a focused search around the Search Contract.",
            "search_contract",
        )
    return searches[:12]


def infer_problem(item: RankedPaper, aspect: AspectCoverageRecord | None) -> str:
    """Describe the paper's apparent problem focus."""

    if item.evidence.claim:
        return item.evidence.claim
    if aspect and aspect.covered_aspects:
        return "Addresses " + ", ".join(aspect.covered_aspects[:3])
    return item.paper.title


def infer_method(text: str) -> str:
    """Infer a broad method category from metadata text."""

    lowered = text.lower()
    if "review" in lowered or "survey" in lowered:
        return "review or survey synthesis"
    if "benchmark" in lowered or "dataset" in lowered:
        return "benchmark or dataset study"
    if "framework" in lowered or "system" in lowered or "pipeline" in lowered:
        return "system or framework"
    if "experiment" in lowered or "empirical" in lowered or "evaluation" in lowered:
        return "empirical evaluation"
    if "model" in lowered or "algorithm" in lowered:
        return "model or algorithm"
    return "not explicit from title/abstract metadata"


def infer_human_role(text: str) -> str:
    """Infer how humans participate in the method."""

    lowered = text.lower()
    if "human-in-the-loop" in lowered or "human in the loop" in lowered:
        return "human-in-the-loop oversight"
    if "human feedback" in lowered or "feedback" in lowered:
        return "feedback, labeling, or relevance correction"
    if "expert" in lowered or "annotator" in lowered or "reviewer" in lowered:
        return "expert review or annotation"
    return "not explicit"


def infer_agent_design(text: str) -> str:
    """Infer agent architecture from text."""

    lowered = text.lower()
    if "multi-agent" in lowered or "multi agent" in lowered:
        return "multi-agent system"
    if "agent" in lowered and ("llm" in lowered or "large language model" in lowered):
        return "LLM agent"
    if "agent" in lowered:
        return "agent-based system"
    return "not explicit"


def infer_evaluation(text: str) -> str:
    """Infer evaluation style from text."""

    lowered = text.lower()
    if "precision" in lowered or "recall" in lowered or "f1" in lowered:
        return "retrieval or classification metrics"
    if "benchmark" in lowered or "dataset" in lowered:
        return "benchmark dataset"
    if "user study" in lowered or "human evaluation" in lowered:
        return "human evaluation"
    if "case study" in lowered or "experiment" in lowered:
        return "case study or experiment"
    return "not explicit in abstract metadata"


def infer_limitation(
    item: RankedPaper,
    aspect: AspectCoverageRecord | None,
) -> str:
    """Summarize the most visible limitation for a method row."""

    if item.evidence.limitation:
        return item.evidence.limitation
    if item.verification.support_level != "strict_support":
        return f"Evidence is {item.verification.support_level}."
    if aspect and aspect.missing_aspects:
        return "Missing aspects: " + ", ".join(aspect.missing_aspects[:4])
    if item.domain_assessment and item.domain_assessment.domain_decision != "in_scope":
        return item.domain_assessment.off_topic_reason
    return "No major limitation detected from metadata."


def retrieval_source(item: RankedPaper) -> str:
    """Summarize where the paper came from."""

    provider = item.paper.retrieval_provider or item.paper.source_provider or "unknown"
    stage = item.paper.retrieval_stage
    query = item.paper.retrieval_query
    parts = [provider]
    if stage:
        parts.append(stage)
    if query:
        parts.append(f"query={query}")
    return " | ".join(parts)


def evidence_verification_summary(item: RankedPaper) -> str:
    """Summarize grounding status for comparison tables."""

    return (
        f"{item.verification.support_level}; "
        f"{item.verification.span_match_type}; "
        f"confidence={item.verification.confidence:.2f}"
    )


def gap_key(text: str) -> str:
    """Map aspect or reason text into a stable research-gap bucket."""

    lowered = text.lower().replace("_", " ")
    if "human" in lowered and ("feedback" in lowered or "loop" in lowered):
        return "human_feedback"
    if "feedback" in lowered:
        return "human_feedback"
    if "evidence" in lowered or "verification" in lowered or "claim" in lowered or "ground" in lowered:
        return "evidence_verification"
    if "evaluation" in lowered or "benchmark" in lowered or "metric" in lowered or "protocol" in lowered:
        return "evaluation"
    if "multi" in lowered and "agent" in lowered:
        return "multi_agent"
    if "agent" in lowered and "coordination" in lowered:
        return "multi_agent"
    if "query" in lowered or "retrieval" in lowered or "search" in lowered:
        return "retrieval"
    return "general_relevance"


def gap_label(key: str) -> str:
    """Return a human-readable gap label."""

    labels = {
        "human_feedback": "Human feedback is undercovered",
        "evidence_verification": "Evidence verification is undercovered",
        "evaluation": "Evaluation protocol is undercovered",
        "multi_agent": "Multi-agent coordination is undercovered",
        "retrieval": "Query refinement and retrieval control are undercovered",
    }
    return labels.get(key, "Research relevance is undercovered")


def gap_project_idea(key: str) -> str:
    """Return a possible project idea for a gap bucket."""

    ideas = {
        "human_feedback": "Feedback-aware query refinement and reranking loop.",
        "evidence_verification": "Span-grounded verification for abstract-level evidence chains.",
        "evaluation": "Benchmarking and evaluation protocol for literature screening agents.",
        "multi_agent": "Agent coordination and task allocation for screening, extraction, and verification.",
        "retrieval": "Pilot-search query repair with ambiguity-aware provider-specific search.",
    }
    return ideas.get(key, "Run a narrower follow-up review around the missing concept.")


def search_phrase_for_gap(gap: str) -> str:
    """Return a query phrase for a gap label."""

    lowered = gap.lower()
    if "human feedback" in lowered:
        return '"human feedback" "query refinement"'
    if "evidence verification" in lowered:
        return '"span-grounded" "evidence verification"'
    if "evaluation" in lowered:
        return '"benchmark" "evaluation protocol"'
    if "multi-agent" in lowered:
        return '"multi-agent" "task allocation"'
    if "query" in lowered or "retrieval" in lowered:
        return '"query repair" "retrieval diagnostics"'
    return '"research gap"'


def supporting_papers_for_gap(
    ranked_papers: list[RankedPaper],
    key: str,
    limit: int = 3,
) -> list[str]:
    """Return top paper titles that mention a gap bucket."""

    terms = {
        "human_feedback": ["human", "feedback", "annotator", "reviewer"],
        "evidence_verification": ["evidence", "verification", "claim", "ground"],
        "evaluation": ["evaluation", "benchmark", "metric", "dataset"],
        "multi_agent": ["multi-agent", "multi agent", "agent coordination"],
        "retrieval": ["retrieval", "query", "search"],
    }.get(key, [])
    titles: list[str] = []
    for item in ranked_papers:
        text = " ".join([item.paper.title, item.paper.abstract, item.evidence.claim]).lower()
        if any(term in text for term in terms):
            titles.append(item.paper.title)
        if len(titles) >= limit:
            break
    return titles


def contract_base_terms(
    search_contract: SearchContract | None,
    query_plan: QueryPlan | None,
) -> str:
    """Build a compact base query from contract or query plan terms."""

    if search_contract:
        terms = search_contract.must_include_concepts[:3] or search_contract.required_aspects[:3]
        if terms:
            return " ".join(f'"{term}"' if " " in term else term for term in terms)
    if query_plan:
        terms = query_plan.must_terms[:2] + query_plan.core_terms[:2]
        if terms:
            return " ".join(f'"{term}"' if " " in term else term for term in terms[:4])
    return ""


def top_included_keywords(ranked_papers: list[RankedPaper]) -> list[str]:
    """Extract frequent non-stopword tokens from included papers."""

    counter: Counter[str] = Counter()
    for item in ranked_papers:
        decision = item.screening_decision.decision if item.screening_decision else ""
        if decision != "include":
            continue
        counter.update(tokenize(" ".join([item.paper.title, item.evidence.claim])))
    return [token for token, _ in counter.most_common(6)]


def add_search(
    searches: list[dict[str, str]],
    query: str,
    reason: str,
    source: str,
) -> None:
    """Append a unique next-search suggestion."""

    cleaned = " ".join(query.split())
    if not cleaned:
        return
    if any(item["query"].lower() == cleaned.lower() for item in searches):
        return
    searches.append({"query": cleaned, "reason": reason, "source": source})


def write_markdown_table(
    path: str | Path,
    title: str,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    """Write rows as a Markdown table."""

    destination = Path(path)
    ensure_dir(destination.parent)
    lines = [f"# {title}", ""]
    if rows:
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(escape_markdown_cell(row.get(field, "")) for field in fields)
                + " |"
            )
    else:
        lines.append("- No rows were generated.")
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def escape_markdown_cell(value: Any) -> str:
    """Escape a value for a Markdown table cell."""

    return str(value or "").replace("\n", " ").replace("|", "\\|")[:300]
