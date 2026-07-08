"""Markdown report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import (
    AspectCoverageRecord,
    DomainAssessment,
    EvidenceRecord,
    PreferenceLearningResult,
    QueryPlan,
    RankedPaper,
    RetrievalPath,
    ResearchTension,
    SearchBrief,
    SearchContract,
    SeedPaper,
    ScreeningDecision,
)
from .utils import ensure_dir, to_plain_data


SCORING_FORMULA = (
    "base_score = weighted relevance/evidence/recency/quality/diversity; "
    "pre_domain_score = 0.52 * intent_centrality_score + 0.48 * base_score "
    "when intent centrality is available; then domain penalty is applied"
)


def _escape_table(text: Any) -> str:
    value = str(text or "").replace("\n", " ").replace("|", "\\|")
    return value[:300]


PAPER_ROLE_ORDER = [
    "theory_origin",
    "conceptual_framework",
    "experimental_proof",
    "surface_probe_method",
    "nanoscale_readout",
    "application_bridge",
    "frontier_extension",
    "limitation_or_challenge",
    "review_background",
]

MISSING_ARTIFACT_MESSAGE = "Not generated in this run."


def _read_optional_json(artifact_dir: Path, filename: str) -> tuple[Any, bool]:
    path = artifact_dir / filename
    if not path.exists():
        return None, False
    try:
        return json.loads(path.read_text(encoding="utf-8")), True
    except (OSError, json.JSONDecodeError):
        return {"error": f"Could not read {filename}."}, True


def _read_optional_csv(artifact_dir: Path, filename: str) -> tuple[list[dict[str, Any]], bool]:
    path = artifact_dir / filename
    if not path.exists():
        return [], False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle)), True
    except OSError:
        return [{"error": f"Could not read {filename}."}], True


def _artifact_payload(
    artifact_dir: Path,
    filename: str,
    fallback: Any | None = None,
) -> tuple[Any, bool]:
    if fallback is not None:
        return to_plain_data(fallback), True
    return _read_optional_json(artifact_dir, filename)


def _gap_artifact_payload(
    artifact_dir: Path,
    fallback: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    if fallback is not None:
        return to_plain_data(fallback), True
    json_rows, has_json = _read_optional_json(artifact_dir, "research_gap_matrix.json")
    if has_json and isinstance(json_rows, list):
        return json_rows, True
    csv_rows, has_csv = _read_optional_csv(artifact_dir, "research_gap_matrix.csv")
    return csv_rows, has_csv


def _append_missing(lines: list[str]) -> None:
    lines.append(MISSING_ARTIFACT_MESSAGE)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _join(values: Any, limit: int = 8) -> str:
    items = [str(item) for item in _as_list(values) if str(item)]
    return ", ".join(items[:limit])


def _paper_title_by_id(ranked_papers: list[RankedPaper]) -> dict[str, str]:
    return {item.paper.paper_id: item.paper.title for item in ranked_papers}


def _paper_role_groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {role: [] for role in PAPER_ROLE_ORDER}
    for record in records:
        roles = _as_list(record.get("roles"))
        primary = str(record.get("primary_role") or "")
        candidate_roles = roles or ([primary] if primary else [])
        for role in candidate_roles:
            if role in grouped:
                grouped[role].append(record)
    return grouped


def _is_verified_support(item: RankedPaper) -> bool:
    verification = item.verification
    support_level = verification.support_level
    if support_level in {"strict_support", "strong_support"}:
        return True
    if verification.span_match_type in {"exact", "exact_match", "high_confidence_fuzzy", "fuzzy"}:
        return verification.span_match_confidence >= 0.8
    return False


def _seed_related_papers(
    seed: dict[str, Any],
    ranked_papers: list[RankedPaper],
) -> list[str]:
    title = str(seed.get("title") or "")
    seed_terms = {term for term in title.lower().replace(":", " ").split() if len(term) > 5}
    if not seed_terms:
        return []
    related: list[str] = []
    for item in ranked_papers:
        paper_terms = set(item.paper.title.lower().replace(":", " ").split())
        if len(seed_terms & paper_terms) >= 2:
            related.append(item.paper.title)
    return related[:5]


def _append_research_process_section(
    lines: list[str],
    artifact_dir: Path,
    research_question: str,
    planned_queries: list[str],
    ranked_papers: list[RankedPaper],
    evidence_records: list[EvidenceRecord],
    search_brief: SearchBrief | None,
    search_contract: SearchContract | None,
    query_plan: QueryPlan | None,
    screening_decisions: list[ScreeningDecision] | None,
    research_gap_matrix: list[dict[str, Any]] | None,
    suggested_next_searches: list[dict[str, Any]] | None,
    research_tensions: list[ResearchTension] | None,
) -> None:
    concept_map, concept_generated = _artifact_payload(artifact_dir, "concept_map.json")
    query_families, query_families_generated = _artifact_payload(
        artifact_dir,
        "query_families.json",
    )
    seed_hints, seed_hints_generated = _artifact_payload(artifact_dir, "seed_hints.json")
    paper_roles, paper_roles_generated = _artifact_payload(artifact_dir, "paper_roles.json")
    evidence_functions, evidence_functions_generated = _artifact_payload(
        artifact_dir,
        "evidence_functions.json",
    )
    tension_rows, tensions_generated = _artifact_payload(
        artifact_dir,
        "research_tensions.json",
        fallback=research_tensions,
    )
    gap_rows, gaps_generated = _gap_artifact_payload(
        artifact_dir,
        fallback=research_gap_matrix,
    )
    next_search_rows, next_searches_generated = _artifact_payload(
        artifact_dir,
        "suggested_next_searches.json",
        fallback=suggested_next_searches,
    )

    lines.extend(["", "# Research Process", ""])
    _append_research_question_interpretation(
        lines,
        research_question,
        search_brief,
        search_contract,
        query_plan,
    )
    _append_concept_decomposition(lines, concept_map, concept_generated)
    _append_search_lenses(lines, query_families, query_families_generated, planned_queries)
    _append_screening_criteria(lines, search_brief, search_contract, screening_decisions)
    _append_paper_roles(lines, paper_roles, paper_roles_generated)
    _append_research_lineage(
        lines,
        ranked_papers,
        paper_roles,
        paper_roles_generated,
        seed_hints,
        seed_hints_generated,
    )
    _append_controversies_and_gaps(
        lines,
        tension_rows,
        tensions_generated,
        gap_rows,
        gaps_generated,
    )
    _append_missing_keywords_methods_authors(
        lines,
        concept_map,
        concept_generated,
        gap_rows,
        gaps_generated,
        seed_hints,
        seed_hints_generated,
    )
    _append_process_next_searches(
        lines,
        next_search_rows,
        next_searches_generated,
        gap_rows,
        gaps_generated,
        tension_rows,
        tensions_generated,
    )
    _append_verified_vs_uncertain(
        lines,
        ranked_papers,
        evidence_records,
        evidence_functions,
        evidence_functions_generated,
    )


def _append_intent_repair_section(lines: list[str], artifact_dir: Path) -> None:
    lines.extend(["## How the system corrected the user’s question", ""])
    intent, generated = _artifact_payload(artifact_dir, "expert_research_intent.json")
    if not generated:
        _append_missing(lines)
        lines.append("")
        return
    payload = _as_dict(intent)
    lines.append(f"- User original wording: {payload.get('original_question', '')}")
    lines.append(
        f"- Expert rewritten question: {payload.get('expert_rewritten_question', '')}"
    )
    lines.append(
        f"- Downweighted user terms: {_join(payload.get('ignored_or_downweighted_terms'), 12)}"
    )
    supplemented = _unique(
        [
            *[str(item) for item in _as_list(payload.get("target_objects"))],
            *[str(item) for item in _as_list(payload.get("mechanisms"))],
            *[str(item) for item in _as_list(payload.get("materials"))],
            *[str(item) for item in _as_list(payload.get("methods"))],
        ]
    )
    lines.append(f"- Expert concepts added: {', '.join(supplemented[:24])}")
    lines.extend(["", "## Intent assumptions and possible misreadings", ""])
    ambiguity_points = _as_list(payload.get("ambiguity_points"))
    possible_interpretations = _as_list(payload.get("possible_interpretations"))
    selected_interpretation = payload.get("selected_interpretation", "")
    selected_reason = payload.get("selected_interpretation_reason", "")
    needs_confirmation_report = _as_list(payload.get("needs_user_confirmation"))
    unsafe_assumptions = _as_list(payload.get("unsafe_or_overbroad_assumptions"))
    if possible_interpretations:
        lines.append("- Possible interpretations of the user's wording:")
        for item in possible_interpretations[:10]:
            lines.append(f"  - {item}")
    else:
        lines.append("- Possible interpretations of the user's wording: none recorded")
    lines.append(f"- Selected interpretation: {selected_interpretation or 'Not explicitly selected.'}")
    lines.append(f"- Why this interpretation: {selected_reason or 'Not generated in this run.'}")
    if ambiguity_points:
        lines.append("- Ambiguity points:")
        for item in ambiguity_points[:8]:
            lines.append(f"  - {item}")
    if needs_confirmation_report:
        lines.append("- Needs user confirmation:")
        for item in needs_confirmation_report[:8]:
            lines.append(f"  - {item}")
    if unsafe_assumptions:
        lines.append("- Unsafe or overbroad assumptions avoided:")
        for item in unsafe_assumptions[:8]:
            lines.append(f"  - {item}")
    downweighted_for_report = _join(
        payload.get("ignored_or_downweighted_terms"),
        16,
    )
    lines.append(
        "- User words not mechanically used as hard query terms: "
        + (downweighted_for_report or "none recorded")
    )
    llm_metadata = _as_dict(payload.get("llm_metadata"))
    if llm_metadata:
        lines.extend(
            [
                "",
                "LLM-assisted intent repair:",
                f"- LLM attempted: {bool(llm_metadata.get('llm_attempted'))}",
                f"- LLM used: {bool(llm_metadata.get('llm_used'))}",
                f"- Fallback used: {bool(llm_metadata.get('fallback_used'))}",
                f"- LLM confidence: {llm_metadata.get('llm_confidence', 0)}",
                f"- Fallback reason: {llm_metadata.get('fallback_reason', '')}",
            ]
        )
        validation_events = _as_list(llm_metadata.get("domain_validation_events"))
        if validation_events:
            lines.extend(
                [
                    "",
                    "Concepts downgraded by domain-pack validation:",
                    "",
                    "| Concept | Source | Reason | New role |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for event in validation_events[:12]:
                event_dict = _as_dict(event)
                lines.append(
                    "| "
                    f"{_escape_table(event_dict.get('term'))} | "
                    f"{_escape_table(event_dict.get('source'))} | "
                    f"{_escape_table(event_dict.get('reason'))} | "
                    f"{_escape_table(event_dict.get('new_query_role'))} |"
                )
        schema_errors = _as_list(llm_metadata.get("schema_validation_errors"))
        if schema_errors:
            lines.append("")
            lines.append(f"- Schema validation errors: {_join(schema_errors, 8)}")
    structured_concepts = [
        _as_dict(item) for item in _as_list(payload.get("structured_concepts"))
    ]
    used_concepts = [
        item
        for item in structured_concepts
        if item.get("should_use_in_provider_query")
        and str(item.get("query_role") or "") in {"must", "optional"}
    ]
    if used_concepts:
        lines.extend(
            [
                "",
                "Concepts supplemented for expert-style planning:",
                "",
                "| Concept | Category | Source | Role | Used in provider query | Activation reason |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for item in used_concepts[:20]:
            lines.append(
                "| "
                f"{_escape_table(item.get('term'))} | "
                f"{_escape_table(item.get('category'))} | "
                f"{_escape_table(item.get('source'))} | "
                f"{_escape_table(item.get('query_role'))} | "
                f"{'yes' if item.get('should_use_in_provider_query') else 'no'} | "
                f"{_escape_table(item.get('activation_reason'))} |"
            )
    downweighted_concepts = [
        item
        for item in structured_concepts
        if str(item.get("query_role") or "") == "downweighted"
    ]
    if downweighted_concepts:
        lines.append("")
        lines.append(
            "Downweighted terms: "
            + ", ".join(_escape_table(item.get("term")) for item in downweighted_concepts[:16])
        )
    optional_or_uncertain = [
        item
        for item in structured_concepts
        if str(item.get("query_role") or "") in {"optional", "uncertain"}
    ]
    if optional_or_uncertain:
        lines.append("")
        lines.append(
            "Optional or uncertain concepts: "
            + ", ".join(_escape_table(item.get("term")) for item in optional_or_uncertain[:16])
        )
    needs_confirmation = _as_list(payload.get("needs_user_confirmation"))
    if needs_confirmation:
        lines.append("")
        lines.append(
            "Needs user confirmation: "
            + ", ".join(_escape_table(item) for item in needs_confirmation[:12])
        )
    assumptions = _as_list(payload.get("assumptions"))
    if assumptions:
        lines.append("- Assumptions needing confirmation or verification:")
        for assumption in assumptions[:8]:
            lines.append(f"  - {assumption}")
    else:
        lines.append("- Assumptions needing confirmation or verification: none recorded")
    must_not = _as_list(payload.get("must_not_overinterpret"))
    if must_not:
        lines.append("- Conservative boundaries:")
        for item in must_not[:6]:
            lines.append(f"  - {item}")
    lines.append("")


def _append_research_question_interpretation(
    lines: list[str],
    research_question: str,
    search_brief: SearchBrief | None,
    search_contract: SearchContract | None,
    query_plan: QueryPlan | None,
) -> None:
    lines.extend(["## 1. Research question interpretation", ""])
    if not (search_brief or search_contract or query_plan):
        _append_missing(lines)
        lines.append("")
        return
    lines.append(f"- Original question: {research_question}")
    if search_brief:
        lines.append(f"- Refined question: {search_brief.refined_question}")
        lines.append(f"- Search intent: {search_brief.search_intent}")
        lines.append(f"- User goal: {search_brief.user_goal}")
    if search_contract:
        lines.append(f"- Domain: {search_contract.domain_profile.domain_name}")
        lines.append(
            f"- Must include concepts: {_join(search_contract.must_include_concepts)}"
        )
        lines.append(
            f"- Must exclude concepts: {_join(search_contract.must_exclude_concepts)}"
        )
    if query_plan:
        lines.append(f"- Core terms: {_join(query_plan.core_terms)}")
        if query_plan.expert_rewritten_question:
            lines.append(f"- Expert rewrite: {query_plan.expert_rewritten_question}")
            lines.append(
                f"- Downweighted user terms: {_join(query_plan.downweighted_user_terms)}"
            )
    lines.append("")


def _append_concept_decomposition(
    lines: list[str],
    concept_map: Any,
    generated: bool,
) -> None:
    lines.extend(["## 2. Concept decomposition", ""])
    if not generated:
        _append_missing(lines)
        lines.append("")
        return
    payload = _as_dict(concept_map)
    lines.append(f"- Domain: {payload.get('domain', '')}")
    lines.append(f"- Central question: {payload.get('central_question', '')}")
    for lens in _as_list(payload.get("lenses"))[:6]:
        lens_data = _as_dict(lens)
        lines.append(
            "- "
            f"{lens_data.get('name', '')}: "
            f"concepts={_join(lens_data.get('core_concepts'), 5)}; "
            f"methods={_join(lens_data.get('methods'), 5)}; "
            f"materials={_join(lens_data.get('materials'), 5)}"
        )
    lines.append("")


def _append_search_lenses(
    lines: list[str],
    query_families: Any,
    generated: bool,
    planned_queries: list[str],
) -> None:
    lines.extend(["## 3. Search lenses and query families", ""])
    if not generated:
        _append_missing(lines)
        if planned_queries:
            lines.append(f"- Existing planner queries: {len(planned_queries)}")
        lines.append("")
        return
    payload = _as_dict(query_families)
    families = _as_list(payload.get("families"))
    lines.append(f"- Query families generated: {len(families)}")
    for family in families[:8]:
        family_data = _as_dict(family)
        queries_by_provider = _as_dict(family_data.get("queries_by_provider"))
        samples: list[str] = []
        for provider, queries in queries_by_provider.items():
            sample = _as_list(queries)[:2]
            if sample:
                samples.append(f"{provider}: {'; '.join(str(query) for query in sample)}")
        lines.append(
            "- "
            f"{family_data.get('name', '')} "
            f"(lens={family_data.get('lens_name', '')}): "
            f"{family_data.get('purpose', '')}; "
            f"sample queries={_escape_table(' | '.join(samples))}"
        )
    lines.append("")


def _append_screening_criteria(
    lines: list[str],
    search_brief: SearchBrief | None,
    search_contract: SearchContract | None,
    screening_decisions: list[ScreeningDecision] | None,
) -> None:
    lines.extend(["## 4. Screening and inclusion criteria", ""])
    if not (search_brief or search_contract or screening_decisions):
        _append_missing(lines)
        lines.append("")
        return
    if search_brief:
        lines.append(f"- Inclusion criteria: {_join(search_brief.inclusion_criteria)}")
        lines.append(f"- Exclusion criteria: {_join(search_brief.exclusion_criteria)}")
        lines.append(f"- Required aspects: {_join(search_brief.required_aspects)}")
    if search_contract:
        lines.append(
            f"- Contract inclusion criteria: {_join(search_contract.inclusion_criteria)}"
        )
        lines.append(
            f"- Contract exclusion criteria: {_join(search_contract.exclusion_criteria)}"
        )
    if screening_decisions:
        counts: dict[str, int] = {}
        for decision in screening_decisions:
            counts[decision.decision] = counts.get(decision.decision, 0) + 1
        lines.append(f"- Decision counts: {counts}")
    lines.append("")


def _append_paper_roles(
    lines: list[str],
    paper_roles: Any,
    generated: bool,
) -> None:
    lines.extend(["## 5. Paper roles and why they matter", ""])
    if not generated:
        _append_missing(lines)
        lines.append("")
        return
    records = [_as_dict(record) for record in _as_list(paper_roles)]
    grouped = _paper_role_groups(records)
    for role in PAPER_ROLE_ORDER:
        items = grouped.get(role, [])
        lines.append(f"- {role}: {len(items)} paper(s)")
        for item in items[:4]:
            reason = _join(item.get("reasons"), 2)
            lines.append(
                f"  - {item.get('title', item.get('paper_id', ''))}"
                f" ({item.get('paper_id', '')})"
                f"{': ' + reason if reason else ''}"
            )
    lines.append("")


def _append_research_lineage(
    lines: list[str],
    ranked_papers: list[RankedPaper],
    paper_roles: Any,
    paper_roles_generated: bool,
    seed_hints: Any,
    seed_hints_generated: bool,
) -> None:
    lines.extend(["## 6. Research lineage", ""])
    role_by_id: dict[str, list[str]] = {}
    if paper_roles_generated:
        for record in _as_list(paper_roles):
            data = _as_dict(record)
            role_by_id[str(data.get("paper_id") or "")] = _as_list(data.get("roles"))
    if ranked_papers:
        sorted_items = sorted(
            ranked_papers,
            key=lambda item: (item.paper.year is None, item.paper.year or 9999, item.rank),
        )
        for item in sorted_items[:12]:
            year = item.paper.year if item.paper.year is not None else "unknown year"
            roles = role_by_id.get(item.paper.paper_id) or ["role not generated"]
            lines.append(
                f"- {year}: {item.paper.title} "
                f"({', '.join(str(role) for role in roles[:3])}); "
                "citation relation not verified"
            )
    else:
        _append_missing(lines)
    if seed_hints_generated:
        hints = [_as_dict(seed) for seed in _as_list(seed_hints)]
        if hints:
            lines.append("- Seed context:")
            for seed in hints[:6]:
                related = _seed_related_papers(seed, ranked_papers)
                title = seed.get("title") or seed.get("raw_mention") or "untitled seed"
                if related:
                    lines.append(
                        f"  - {title}: possible title-overlap with "
                        f"{'; '.join(related)}; citation relation not verified"
                    )
                else:
                    lines.append(
                        f"  - {title}: no retrieved title-overlap found; "
                        "citation relation not verified"
                    )
    else:
        lines.append(f"- Seed hints: {MISSING_ARTIFACT_MESSAGE}")
    lines.append("- citation relation not verified unless explicit citation-expansion artifacts support it.")
    lines.append("")


def _append_controversies_and_gaps(
    lines: list[str],
    tension_rows: Any,
    tensions_generated: bool,
    gap_rows: list[dict[str, Any]],
    gaps_generated: bool,
) -> None:
    lines.extend(["## 7. Controversies, limitations, and gaps", ""])
    if not tensions_generated and not gaps_generated:
        _append_missing(lines)
        lines.append("")
        return
    if tensions_generated:
        for row in _as_list(tension_rows)[:6]:
            data = _as_dict(row)
            lines.append(
                "- "
                f"{data.get('tension_label', data.get('tension_key', ''))}: "
                f"{data.get('why_it_matters', data.get('description', ''))}"
            )
    else:
        lines.append(f"- Research tensions: {MISSING_ARTIFACT_MESSAGE}")
    if gaps_generated:
        for row in gap_rows[:6]:
            lines.append(
                "- Gap: "
                f"{row.get('gap_label') or row.get('gap') or row.get('gap_key')}: "
                f"{row.get('evidence_or_reason') or row.get('why_gap_remains', '')}"
            )
    else:
        lines.append(f"- Gap matrix: {MISSING_ARTIFACT_MESSAGE}")
    lines.append("")


def _append_missing_keywords_methods_authors(
    lines: list[str],
    concept_map: Any,
    concept_generated: bool,
    gap_rows: list[dict[str, Any]],
    gaps_generated: bool,
    seed_hints: Any,
    seed_hints_generated: bool,
) -> None:
    lines.extend(["## 8. Missing keywords, methods, authors, or schools", ""])
    if not (concept_generated or gaps_generated or seed_hints_generated):
        _append_missing(lines)
        lines.append("")
        return
    if gaps_generated and gap_rows:
        lines.append("- Gap-derived missing directions:")
        for row in gap_rows[:6]:
            suggestions = row.get("suggested_next_searches", "")
            lines.append(
                f"  - {row.get('gap_label') or row.get('gap') or row.get('gap_key')}: "
                f"{suggestions}"
            )
    else:
        lines.append(f"- Gap-derived missing directions: {MISSING_ARTIFACT_MESSAGE}")
    if concept_generated:
        payload = _as_dict(concept_map)
        methods: list[str] = []
        materials: list[str] = []
        for lens in _as_list(payload.get("lenses")):
            data = _as_dict(lens)
            methods.extend(str(item) for item in _as_list(data.get("methods")))
            materials.extend(str(item) for item in _as_list(data.get("materials")))
        lines.append(f"- Lens methods to audit: {_join(_unique(methods), 12)}")
        lines.append(f"- Lens materials to audit: {_join(_unique(materials), 12)}")
    if seed_hints_generated:
        authors: list[str] = []
        for seed in _as_list(seed_hints):
            authors.extend(str(item) for item in _as_list(_as_dict(seed).get("authors")))
        lines.append(f"- Author hints from seed mentions: {_join(_unique(authors), 8)}")
    lines.append("- School/lab inference was not generated; needs further verification.")
    lines.append("")


def _append_process_next_searches(
    lines: list[str],
    next_search_rows: Any,
    next_searches_generated: bool,
    gap_rows: list[dict[str, Any]],
    gaps_generated: bool,
    tension_rows: Any,
    tensions_generated: bool,
) -> None:
    lines.extend(["## 9. Suggested next searches", ""])
    queries: list[str] = []
    if next_searches_generated:
        for row in _as_list(next_search_rows):
            query = _as_dict(row).get("query")
            if query:
                queries.append(str(query))
    if gaps_generated:
        for row in gap_rows:
            suggestions = row.get("suggested_next_searches", "")
            if isinstance(suggestions, list):
                queries.extend(str(item) for item in suggestions)
            elif suggestions:
                queries.extend(part.strip() for part in str(suggestions).split(";") if part.strip())
    if tensions_generated:
        for row in _as_list(tension_rows):
            queries.extend(
                str(item)
                for item in _as_list(_as_dict(row).get("suggested_next_searches"))
            )
    queries = _unique(queries)
    if not queries:
        _append_missing(lines)
    for query in queries[:12]:
        lines.append(f"- {query}")
    lines.append("")


def _append_verified_vs_uncertain(
    lines: list[str],
    ranked_papers: list[RankedPaper],
    evidence_records: list[EvidenceRecord],
    evidence_functions: Any,
    evidence_functions_generated: bool,
) -> None:
    lines.extend(["## 10. Verified vs uncertain findings", ""])
    if not ranked_papers and not evidence_records and not evidence_functions_generated:
        _append_missing(lines)
        lines.append("")
        return
    verified = [item for item in ranked_papers if _is_verified_support(item)]
    uncertain = [item for item in ranked_papers if not _is_verified_support(item)]
    lines.append("- Verified:")
    if verified:
        for item in verified[:8]:
            lines.append(
                f"  - {item.paper.title}: {item.verification.support_level}; "
                f"span={item.verification.span_match_type or 'none'}"
            )
    else:
        lines.append("  - No strict or span-validated findings were available.")
    lines.append("- Uncertain:")
    if uncertain:
        for item in uncertain[:8]:
            reason = item.verification.support_level or "rule/query-derived"
            if not item.paper.abstract:
                reason = "missing abstract"
            lines.append(f"  - {item.paper.title}: {reason}; needs further verification")
    else:
        lines.append("  - No uncertain findings were identified.")
    if evidence_functions_generated:
        functions = [
            str(_as_dict(row).get("evidence_function"))
            for row in _as_list(evidence_functions)
            if _as_dict(row).get("evidence_function")
        ]
        lines.append(f"- Evidence functions observed: {_join(_unique(functions), 12)}")
    else:
        lines.append(f"- Evidence functions: {MISSING_ARTIFACT_MESSAGE}")
    lines.append("")


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def generate_report(
    path: str | Path,
    research_question: str,
    planned_queries: list[str],
    retrieval_statistics: dict[str, Any],
    ranked_papers: list[RankedPaper],
    evidence_records: list[EvidenceRecord],
    evaluation_metrics: dict[str, Any],
    feedback_applied: bool = False,
    search_brief: SearchBrief | None = None,
    search_contract: SearchContract | None = None,
    ambiguity_analysis: list[dict[str, Any]] | None = None,
    domain_assessments: list[DomainAssessment] | None = None,
    query_pilot_diagnostics: dict[str, Any] | None = None,
    query_repair_suggestions: dict[str, Any] | None = None,
    question_refinement: dict[str, Any] | None = None,
    query_plan: QueryPlan | None = None,
    aspect_coverage_records: list[AspectCoverageRecord] | None = None,
    result_groups: dict[str, Any] | None = None,
    reading_path_path: str | Path | None = None,
    paper_cards_path: str | Path | None = None,
    prisma_like_flow: dict[str, Any] | None = None,
    screening_decisions: list[ScreeningDecision] | None = None,
    method_comparison_matrix: list[dict[str, Any]] | None = None,
    research_gap_matrix: list[dict[str, Any]] | None = None,
    suggested_next_searches: list[dict[str, Any]] | None = None,
    preference_learning: PreferenceLearningResult | None = None,
    feedback_query_refinement: dict[str, Any] | None = None,
    seed_papers: list[SeedPaper] | None = None,
    retrieval_paths: list[RetrievalPath] | None = None,
    citation_expansion_papers: list[Any] | None = None,
    research_tensions: list[ResearchTension] | None = None,
) -> None:
    """Generate a human-readable Markdown report."""

    destination = Path(path)
    ensure_dir(destination.parent)
    planner_metadata = retrieval_statistics.get("llm", {}).get("planner", {})
    planning_question = planner_metadata.get("planning_question", research_question)
    translated_question = planner_metadata.get("translated_question", "")

    lines: list[str] = [
        "# Literature Screening Decision Report",
        "",
        "## Research Question",
        "",
        research_question,
        "",
    ]
    if planning_question and planning_question != research_question:
        lines.extend(
            [
                "## Question Preprocessing",
                "",
                f"Planning question: {planning_question}",
                "",
            ]
        )
    if translated_question:
        lines.extend(
            [
                f"Translated question: {translated_question}",
                "",
            ]
        )
    _append_intent_repair_section(lines, destination.parent)
    if search_brief:
        lines.extend(
            [
                "## What the System Thinks the User Is Looking For",
                "",
                f"- Refined question: {search_brief.refined_question}",
                f"- Search intent: {search_brief.search_intent}",
                f"- User goal: {search_brief.user_goal}",
                f"- Inclusion criteria: {', '.join(search_brief.inclusion_criteria)}",
                f"- Exclusion criteria: {', '.join(search_brief.exclusion_criteria)}",
                f"- Required aspects: {', '.join(search_brief.required_aspects)}",
                f"- Preferred paper types: {', '.join(search_brief.preferred_paper_types)}",
                f"- Time window: {search_brief.time_window}",
                f"- Success definition: {search_brief.success_definition}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## What the System Thinks the User Is Looking For",
                "",
                "- No search brief was generated for this run.",
                "",
            ]
        )
    if search_contract:
        domain = search_contract.domain_profile
        lines.extend(
            [
                "## Search Contract",
                "",
                f"- Domain: {domain.domain_name}",
                f"- Positive domains: {', '.join(domain.positive_domains)}",
                f"- Negative domains: {', '.join(domain.negative_domains)}",
                f"- Must include concepts: {', '.join(search_contract.must_include_concepts)}",
                f"- Must exclude concepts: {', '.join(search_contract.must_exclude_concepts)}",
                f"- Required aspects: {', '.join(search_contract.required_aspects)}",
                f"- Field whitelist: {', '.join(domain.field_of_study_whitelist)}",
                f"- Field blacklist: {', '.join(domain.field_of_study_blacklist)}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Search Contract",
                "",
                "- No Search Contract was generated for this run.",
                "",
            ]
        )
    if ambiguity_analysis:
        lines.extend(
            [
                "## Ambiguity Handling",
                "",
                "| Term | Selected meaning | Recommended excludes |",
                "| --- | --- | --- |",
            ]
        )
        for record in ambiguity_analysis:
            lines.append(
                "| "
                f"{_escape_table(record.get('term', ''))} | "
                f"{_escape_table(record.get('selected_meaning', ''))} | "
                f"{_escape_table(', '.join(record.get('recommended_exclude_terms', [])))} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Ambiguity Handling",
                "",
                "- No ambiguous terms were detected.",
                "",
            ]
        )
    if question_refinement:
        lines.extend(["## Refined Subquestions", ""])
        subquestions = question_refinement.get("subquestions", [])
        if subquestions:
            lines.extend([f"- {item}" for item in subquestions])
        else:
            lines.append("- No subquestions were needed for this run.")
        lines.append("")
    if query_plan:
        lines.extend(
            [
                "## Query Strategy",
                "",
                f"- Core terms: {', '.join(query_plan.core_terms)}",
                f"- Must terms: {', '.join(query_plan.must_terms)}",
                f"- Optional terms: {', '.join(query_plan.optional_terms)}",
                f"- Exclude terms: {', '.join(query_plan.exclude_terms)}",
                f"- Required aspects: {', '.join(query_plan.required_aspects)}",
                f"- Filters: {query_plan.filters}",
                "",
            ]
        )
    _append_research_process_section(
        lines,
        destination.parent,
        research_question,
        planned_queries,
        ranked_papers,
        evidence_records,
        search_brief,
        search_contract,
        query_plan,
        screening_decisions,
        research_gap_matrix,
        suggested_next_searches,
        research_tensions,
    )
    year_filter = retrieval_statistics.get("year_filter", {})
    imported_library = retrieval_statistics.get("imported_library", {})
    lines.extend(
        [
            "## Planned Queries",
            "",
        ]
    )
    lines.extend([f"- {query}" for query in planned_queries])
    domain_guardrails = retrieval_statistics.get("domain_guardrails", {})
    domain_counts = domain_guardrails.get("counts", {})
    demoted_examples = domain_guardrails.get("demoted_examples", [])
    common_reasons = domain_guardrails.get("common_off_topic_reasons", [])
    lines.extend(
        [
            "",
            "## Domain Guardrails",
            "",
            f"- In scope: {domain_counts.get('in_scope', 0)}",
            f"- Borderline: {domain_counts.get('borderline', 0)}",
            f"- Out of scope: {domain_counts.get('out_of_scope', 0)}",
            "",
        ]
    )
    if demoted_examples:
        lines.extend(
            [
                "| Rank | Decision | Penalty | Domain score | Paper | Reason |",
                "| ---: | --- | ---: | ---: | --- | --- |",
            ]
        )
        for example in demoted_examples[:8]:
            lines.append(
                "| "
                f"{example.get('rank', '')} | "
                f"{_escape_table(example.get('domain_decision', ''))} | "
                f"{example.get('domain_penalty_multiplier', '')} | "
                f"{example.get('domain_match_score', '')} | "
                f"{_escape_table(example.get('title', ''))} | "
                f"{_escape_table(example.get('off_topic_reason', ''))} |"
            )
        lines.append("")
    else:
        lines.extend(["- No papers were demoted by domain guardrails.", ""])
    if common_reasons:
        lines.append("Common off-topic reasons:")
        for item in common_reasons[:5]:
            lines.append(f"- {item.get('reason', '')} ({item.get('count', 0)})")
        lines.append("")

    pilot = query_pilot_diagnostics or {}
    repairs = query_repair_suggestions or {}
    lines.extend(["## Query Pilot Diagnostics", ""])
    if pilot.get("enabled"):
        summary = pilot.get("summary", {})
        lines.extend(
            [
                f"- Pilot max per query: {pilot.get('pilot_max_per_query', '')}",
                f"- Pilot records: {len(pilot.get('results', []))}",
                f"- Mean off-topic rate: {summary.get('mean_off_topic_rate', 0)}",
                f"- Recommendation counts: {summary.get('recommendation_counts', {})}",
                f"- Detected drift counts: {summary.get('detected_drift_counts', {})}",
                "",
                "| Provider | Stage | Recommendation | Off-topic rate | Query | Drift |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for result in pilot.get("results", [])[:12]:
            lines.append(
                "| "
                f"{_escape_table(result.get('provider', ''))} | "
                f"{_escape_table(result.get('retrieval_stage', ''))} | "
                f"{_escape_table(result.get('recommendation', ''))} | "
                f"{result.get('off_topic_rate_estimate', 0)} | "
                f"{_escape_table(result.get('query', ''))} | "
                f"{_escape_table(', '.join(result.get('detected_drift', [])))} |"
            )
        lines.append("")
    else:
        lines.append("- Pilot search was not run for this report.")
        lines.append("")

    lines.extend(["## Query Repairs Applied", ""])
    if repairs.get("enabled"):
        lines.append(f"- Auto repair applied: {repairs.get('applied', False)}")
        suggestions = repairs.get("suggestions", [])
        if suggestions:
            lines.extend(
                [
                    "",
                    "| Provider | Recommendation | Original query | Repaired query | Reason |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for suggestion in suggestions[:12]:
                lines.append(
                    "| "
                    f"{_escape_table(suggestion.get('provider', ''))} | "
                    f"{_escape_table(suggestion.get('recommendation', ''))} | "
                    f"{_escape_table(suggestion.get('original_query', ''))} | "
                    f"{_escape_table(suggestion.get('repaired_query', ''))} | "
                    f"{_escape_table(suggestion.get('reason', ''))} |"
                )
            lines.append("")
        else:
            lines.append("- No query repairs were suggested.")
            lines.append("")
    else:
        lines.append("- Query repair was not run.")
        lines.append("")

    lines.extend(["## Seed Paper Expansion", ""])
    seed_rows = seed_papers or []
    path_rows = retrieval_paths or []
    expansion_rows = citation_expansion_papers or []
    if seed_rows:
        lines.extend(
            [
                f"- Seed papers: {len(seed_rows)}",
                f"- Expanded papers kept for ranking: {len(expansion_rows)}",
                f"- Retrieval paths recorded: {len(path_rows)}",
                "",
                "| Seed type | Seed ID | Title | Note |",
                "| --- | --- | --- | --- |",
            ]
        )
        for seed in seed_rows[:10]:
            lines.append(
                "| "
                f"{_escape_table(seed.seed_type)} | "
                f"{_escape_table(seed.seed_id)} | "
                f"{_escape_table(seed.title)} | "
                f"{_escape_table(seed.note)} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "- No user-provided seed papers were supplied. If snowballing was enabled, seeds were selected from high-confidence ranked papers when possible.",
                "",
            ]
        )
    if path_rows:
        lines.extend(
            [
                "| Stage | Expanded paper ID | Seed | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for path in path_rows[:20]:
            lines.append(
                "| "
                f"{_escape_table(path.source_stage)} | "
                f"{_escape_table(path.paper_id)} | "
                f"{_escape_table(path.seed_title or path.seed_paper_id)} | "
                f"{_escape_table(path.reason)} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "- No citation/reference/recommendation expansion paths were recorded for this run.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Retrieval Summary",
            "",
            f"- Raw retrieved paper count: {retrieval_statistics.get('raw_retrieved_paper_count', 0)}",
            f"- Merged paper count: {retrieval_statistics.get('merged_paper_count', 0)}",
            f"- Duplicate count: {retrieval_statistics.get('duplicate_count', 0)}",
            f"- Counts by provider: {retrieval_statistics.get('retrieval_counts_by_provider', {})}",
            f"- Imported library papers: {imported_library.get('paper_count', 0)}",
            f"- Imported library format: {imported_library.get('detected_format', 'none')}",
            f"- Year filter enabled: {year_filter.get('enabled', False)}",
            f"- From year: {year_filter.get('from_year')}",
            f"- Records kept after year filter: {year_filter.get('kept_count', retrieval_statistics.get('raw_retrieved_paper_count', 0))}",
            f"- Records excluded before from-year: {year_filter.get('excluded_before_year_count', 0)}",
            f"- Records excluded because year is missing: {year_filter.get('excluded_missing_year_count', 0)}",
            "",
            "## PRISMA-like Screening Flow",
            "",
            f"- Records identified by OpenAlex: {(prisma_like_flow or {}).get('records_identified_by_openalex', 0)}",
            f"- Records identified by Semantic Scholar: {(prisma_like_flow or {}).get('records_identified_by_semantic_scholar', 0)}",
            f"- Duplicate records removed: {(prisma_like_flow or {}).get('duplicate_records_removed', 0)}",
            f"- Records with missing abstracts: {(prisma_like_flow or {}).get('records_with_missing_abstracts', 0)}",
            f"- Records screened: {(prisma_like_flow or {}).get('records_screened', 0)}",
            f"- Records included: {(prisma_like_flow or {}).get('records_included', 0)}",
            f"- Records maybe: {(prisma_like_flow or {}).get('records_maybe', 0)}",
            f"- Records excluded: {(prisma_like_flow or {}).get('records_excluded', 0)}",
            f"- Out-of-domain records: {(prisma_like_flow or {}).get('out_of_domain_records', 0)}",
            f"- Included in top ranked results: {(prisma_like_flow or {}).get('records_included_in_top_ranked_results', 0)}",
            f"- Excluded or low confidence: {(prisma_like_flow or {}).get('records_excluded_or_low_confidence', 0)}",
            f"- Common exclusion reasons: {(prisma_like_flow or {}).get('common_exclusion_reasons', {})}",
            "",
            "## Provider Status",
            "",
        ]
    )
    provider_status = retrieval_statistics.get("provider_status", {})
    if provider_status:
        lines.extend(["| Provider | Status | Queries attempted | Papers returned | Notes |", "| --- | --- | ---: | ---: | --- |"])
        for provider, status in provider_status.items():
            notes = "rate limited" if status.get("rate_limited") else ""
            stopped = status.get("stopped_after_query_count")
            if stopped is not None:
                notes = (notes + f"; stopped after {stopped} queries").strip("; ")
            lines.append(
                "| "
                f"{_escape_table(provider)} | "
                f"{_escape_table(status.get('status', ''))} | "
                f"{status.get('attempted_query_count', 0)} | "
                f"{status.get('returned_paper_count', 0)} | "
                f"{_escape_table(notes)} |"
            )
    else:
        lines.append("- Provider status was not generated for this run.")
    lines.extend(
        [
            "",
            "## LLM Settings",
            "",
            f"- Backend requested: {retrieval_statistics.get('llm', {}).get('backend_requested', 'none')}",
            f"- Backend active: {retrieval_statistics.get('llm', {}).get('backend_active', 'none')}",
            f"- Planner mode: {retrieval_statistics.get('llm', {}).get('planner_mode', 'rule')}",
            f"- Extractor mode: {retrieval_statistics.get('llm', {}).get('extractor_mode', 'rule')}",
            f"- Verifier mode: {retrieval_statistics.get('llm', {}).get('verifier_mode', 'rule')}",
            f"- Invalid LLM output count: {retrieval_statistics.get('llm', {}).get('invalid_llm_output_count', 0)}",
            "",
            "## Evidence Validation",
            "",
            "`strict_support` means the evidence sentence was grounded in the abstract. It is not a claim that the paper is highly relevant; use intent match and reading priority separately.",
            "",
            f"- Strict supported count: {retrieval_statistics.get('strict_supported_count', 0)}",
            f"- Weak support count: {retrieval_statistics.get('weak_support_count', 0)}",
            f"- Unverified count: {retrieval_statistics.get('unverified_count', 0)}",
            f"- LLM invalid evidence count: {retrieval_statistics.get('llm_invalid_evidence_count', 0)}",
            f"- Grounding accuracy: {retrieval_statistics.get('grounding_accuracy', 0):.3f}",
            f"- Strict support rate: {retrieval_statistics.get('strict_support_rate', 0):.3f}",
            f"- Weak support rate: {retrieval_statistics.get('weak_support_rate', 0):.3f}",
            f"- LLM invalid evidence rate: {retrieval_statistics.get('llm_invalid_evidence_rate', 0):.3f}",
            f"- Average aspect coverage: {retrieval_statistics.get('average_aspect_coverage', 0):.3f}",
            "",
            "## Scoring Formula",
            "",
            f"`{SCORING_FORMULA}`",
            "",
            "Current weights:",
            "",
            f"- Relevance: {evaluation_metrics.get('scoring_weights', {}).get('relevance', 0.40)}",
            f"- Evidence: {evaluation_metrics.get('scoring_weights', {}).get('evidence', 0.25)}",
            f"- Recency: {evaluation_metrics.get('scoring_weights', {}).get('recency', 0.15)}",
            f"- Quality: {evaluation_metrics.get('scoring_weights', {}).get('quality', 0.15)}",
            f"- Diversity: {evaluation_metrics.get('scoring_weights', {}).get('diversity', 0.05)}",
            "- Domain penalty: in_scope x1.0, borderline x0.7, out_of_scope x0.3",
            "",
            "## Top 10 Ranked Papers",
            "",
            "| Rank | Decision | Score | Evidence grounding | Intent match | Reading priority | Domain | Year | Title | Venue | DOI |",
            "| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |",
        ]
    )

    for item in ranked_papers[:10]:
        domain = item.domain_assessment
        decision = item.screening_decision
        lines.append(
            "| "
            f"{item.rank} | "
            f"{_escape_table(decision.decision if decision else '')} | "
            f"{item.scores.final_score:.3f} | "
            f"{_escape_table(item.verification.support_level)} | "
            f"{item.scores.intent_centrality_score:.3f} | "
            f"{_escape_table(decision.reading_priority if decision else '')} | "
            f"{_escape_table(domain.domain_decision if domain else '')} | "
            f"{item.paper.year or ''} | "
            f"{_escape_table(item.paper.title)} | "
            f"{_escape_table(item.paper.venue)} | "
            f"{_escape_table(item.paper.doi)} |"
        )

    lines.extend(["", "## Included Papers", ""])
    decision_records = screening_decisions or [
        item.screening_decision
        for item in ranked_papers
        if item.screening_decision is not None
    ]
    ranked_by_id = {item.paper.paper_id: item for item in ranked_papers}
    included = [record for record in decision_records if record.decision == "include"]
    maybe = [record for record in decision_records if record.decision == "maybe"]
    excluded = [record for record in decision_records if record.decision == "exclude"]
    if included:
        lines.extend(
            [
                "| Rank | Score | Paper | Primary reason | Reading priority |",
                "| ---: | ---: | --- | --- | --- |",
            ]
        )
        for record in included[:10]:
            item = ranked_by_id.get(record.paper_id)
            score_text = f"{item.scores.final_score:.3f}" if item else ""
            lines.append(
                "| "
                f"{item.rank if item else ''} | "
                f"{score_text} | "
                f"{_escape_table(item.paper.title if item else record.paper_id)} | "
                f"{_escape_table(record.primary_reason)} | "
                f"{_escape_table(record.reading_priority)} |"
            )
    else:
        lines.append("- No papers were automatically marked include.")

    lines.extend(["", "## Maybe / Needs Human Inspection", ""])
    if maybe:
        lines.extend(
            [
                "| Rank | Score | Paper | Primary reason | Suggested action |",
                "| ---: | ---: | --- | --- | --- |",
            ]
        )
        for record in maybe[:15]:
            item = ranked_by_id.get(record.paper_id)
            score_text = f"{item.scores.final_score:.3f}" if item else ""
            lines.append(
                "| "
                f"{item.rank if item else ''} | "
                f"{score_text} | "
                f"{_escape_table(item.paper.title if item else record.paper_id)} | "
                f"{_escape_table(record.primary_reason)} | "
                f"{_escape_table(record.suggested_action)} |"
            )
    else:
        lines.append("- No papers were marked maybe.")

    lines.extend(["", "## Excluded Papers And Reasons", ""])
    if excluded:
        lines.extend(
            [
                "| Rank | Score | Paper | Primary reason | Exclusion reasons |",
                "| ---: | ---: | --- | --- | --- |",
            ]
        )
        for record in excluded[:20]:
            item = ranked_by_id.get(record.paper_id)
            score_text = f"{item.scores.final_score:.3f}" if item else ""
            lines.append(
                "| "
                f"{item.rank if item else ''} | "
                f"{score_text} | "
                f"{_escape_table(item.paper.title if item else record.paper_id)} | "
                f"{_escape_table(record.primary_reason)} | "
                f"{_escape_table(', '.join(record.exclusion_reasons))} |"
            )
    else:
        lines.append("- No papers were automatically excluded.")

    lines.extend(["", "## Common Exclusion Reasons", ""])
    reason_counts = (
        evaluation_metrics.get("screening_decisions", {})
        .get("common_exclusion_reasons", {})
    )
    if not reason_counts:
        reason_counts = (prisma_like_flow or {}).get("common_exclusion_reasons", {})
    if reason_counts:
        for reason, count in reason_counts.items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- No exclusion reasons were recorded.")

    method_rows = method_comparison_matrix or []
    lines.extend(["", "## Method Comparison Matrix", ""])
    if method_rows:
        lines.extend(
            [
                "| Paper | Method | Human role | Agent design | Evidence verification | Evaluation | Recommended use |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in method_rows[:12]:
            lines.append(
                "| "
                f"{_escape_table(row.get('title', ''))} | "
                f"{_escape_table(row.get('method', ''))} | "
                f"{_escape_table(row.get('human_role', ''))} | "
                f"{_escape_table(row.get('agent_design', ''))} | "
                f"{_escape_table(row.get('evidence_verification', ''))} | "
                f"{_escape_table(row.get('evaluation', ''))} | "
                f"{_escape_table(row.get('recommended_use', ''))} |"
            )
        lines.append("")
        lines.append("- Full matrix: `method_comparison_matrix.csv` and `method_comparison_matrix.md`.")
    else:
        lines.append("- No method comparison rows were generated.")

    gap_rows = research_gap_matrix or []
    lines.extend(["", "## Research Gap Matrix", ""])
    if gap_rows:
        lines.extend(
            [
                "| Gap | Supporting papers | Why gap remains | Possible project idea | Related aspects |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in gap_rows[:10]:
            lines.append(
                "| "
                f"{_escape_table(row.get('gap', ''))} | "
                f"{_escape_table(row.get('supporting_papers', ''))} | "
                f"{_escape_table(row.get('why_gap_remains', ''))} | "
                f"{_escape_table(row.get('possible_project_idea', ''))} | "
                f"{_escape_table(row.get('related_aspects', ''))} |"
            )
        lines.append("")
        lines.append("- Full matrix: `research_gap_matrix.csv` and `research_gap_matrix.md`.")
    else:
        lines.append("- No research gaps were generated from this run.")

    next_searches = suggested_next_searches or []
    lines.extend(["", "## Suggested Next Searches", ""])
    if next_searches:
        lines.extend(
            [
                "| Query | Reason | Source |",
                "| --- | --- | --- |",
            ]
        )
        for row in next_searches[:12]:
            lines.append(
                "| "
                f"{_escape_table(row.get('query', ''))} | "
                f"{_escape_table(row.get('reason', ''))} | "
                f"{_escape_table(row.get('source', ''))} |"
            )
        lines.append("")
        lines.append("- Full list: `suggested_next_searches.json` and `suggested_next_searches.md`.")
    else:
        lines.append("- No follow-up searches were suggested.")

    tension_rows = research_tensions or []
    lines.extend(["", "## Research Tensions And Boundary Conditions", ""])
    if tension_rows:
        lines.extend(
            [
                "| Tension | Why it matters | Supporting papers | Suggested searches | Confidence |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for record in tension_rows[:10]:
            lines.append(
                "| "
                f"{_escape_table(record.tension_label)} | "
                f"{_escape_table(record.why_it_matters)} | "
                f"{_escape_table(', '.join(record.supporting_paper_ids))} | "
                f"{_escape_table('; '.join(record.suggested_next_searches[:2]))} | "
                f"{record.confidence:.2f} |"
            )
        lines.append("")
        lines.append("- Full list: `research_tensions.json`.")
    else:
        lines.append("- No research tensions were identified from this run.")

    lines.extend(["", "## Aspect Coverage Summary", ""])
    aspect_records = aspect_coverage_records or []
    if aspect_records:
        lines.extend(
            [
                "| Paper | Covered aspects | Missing aspects | Score |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for record in aspect_records[:10]:
            lines.append(
                "| "
                f"{_escape_table(record.title)} | "
                f"{_escape_table(', '.join(record.covered_aspects))} | "
                f"{_escape_table(', '.join(record.missing_aspects))} | "
                f"{record.aspect_coverage_score:.2f} |"
            )
    else:
        lines.append("- No required aspects were classified.")

    lines.extend(["", "## Recommended Reading Path", ""])
    if reading_path_path:
        lines.append(f"- See `{Path(reading_path_path).name}`.")
    else:
        lines.append("- No reading path was generated.")

    lines.extend(["", "## Result Groups", ""])
    for group_name, rows in (result_groups or {}).items():
        lines.append(f"- {group_name}: {len(rows)} papers")

    lines.extend(["", "## Top Paper Evidence Cards", ""])
    if paper_cards_path:
        lines.append(f"- See `{Path(paper_cards_path).name}`.")
    else:
        lines.append("- No paper cards were generated.")

    lines.extend(
        [
            "",
            "## Evidence Chain Table",
            "",
            "| Rank | Support level | Span match | Span confidence | LLM invalid | Missing abstract | Claim | Evidence | Matched text |",
            "| ---: | --- | --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    evidence_by_id = {record.paper_id: record for record in evidence_records}
    for item in ranked_papers[:10]:
        evidence = evidence_by_id.get(item.paper.paper_id, item.evidence)
        lines.append(
            "| "
            f"{item.rank} | "
            f"{item.verification.support_level} | "
            f"{item.verification.span_match_type} | "
            f"{item.verification.span_match_confidence:.2f} | "
            f"{item.verification.support_level == 'llm_invalid_evidence'} | "
            f"{item.verification.support_level == 'missing_abstract'} | "
            f"{_escape_table(evidence.claim)} | "
            f"{_escape_table(evidence.evidence_sentence)} | "
            f"{_escape_table(item.verification.matched_text)} |"
        )

    lines.extend(["", "## Human Feedback Preference Learning", ""])
    if preference_learning and preference_learning.enabled:
        lines.extend(
            [
                f"- Model type: {preference_learning.model_type}",
                f"- Labeled papers: {preference_learning.labeled_paper_count}",
                f"- Include labels: {preference_learning.include_count}",
                f"- Exclude labels: {preference_learning.exclude_count}",
                f"- Learned positive terms: {', '.join(preference_learning.positive_terms[:12])}",
                f"- Learned negative terms: {', '.join(preference_learning.negative_terms[:12])}",
                f"- Note: {preference_learning.note}",
                "",
            ]
        )
    else:
        note = preference_learning.note if preference_learning else "No feedback model was available."
        lines.extend([f"- Preference learning inactive: {note}", ""])

    lines.extend(["## How feedback changed ranking", ""])
    if feedback_applied:
        changes = evaluation_metrics.get("ranking_changes", {})
        lines.append(f"- Feedback applied: yes")
        lines.append(f"- Papers with rank changes: {changes.get('moved_count', 0)}")
    else:
        lines.append("- Feedback applied: no")

    lines.extend(["", "## Suggested query refinements from feedback", ""])
    refinement = feedback_query_refinement or {}
    if refinement.get("enabled"):
        lines.extend(
            [
                f"- Suggested must terms: {', '.join(refinement.get('suggested_must_terms', []))}",
                f"- Suggested optional terms: {', '.join(refinement.get('suggested_optional_terms', []))}",
                f"- Suggested exclude terms: {', '.join(refinement.get('suggested_exclude_terms', []))}",
            ]
        )
    else:
        lines.append("- No feedback-derived query refinements were generated.")

    lines.extend(
        [
            "",
            "## Evaluation Metrics",
            "",
            f"- Missing abstract ratio: {evaluation_metrics.get('missing_abstract_ratio', 0):.3f}",
            f"- Unsupported claim rate: {evaluation_metrics.get('unsupported_claim_rate', 0):.3f}",
            f"- Precision@10: {evaluation_metrics.get('precision_at_10')}",
            f"- nDCG@10: {evaluation_metrics.get('ndcg_at_10')}",
            f"- MAP: {evaluation_metrics.get('map')}",
            f"- Recall@10: {evaluation_metrics.get('recall_at_10')}",
            f"- Feedback mean absolute rank delta: {evaluation_metrics.get('feedback_before_after_ranking_delta', {}).get('mean_abs_rank_delta', 0)}",
            "",
            "## Agent Trace Summary",
            "",
            f"- Intent agent: {search_brief.search_intent if search_brief else 'not available'}",
            f"- Planner: {len(planned_queries)} planned query strings.",
            f"- Retriever: {retrieval_statistics.get('raw_retrieved_paper_count', 0)} raw records, {retrieval_statistics.get('merged_paper_count', 0)} merged records.",
            f"- Domain guardrail: {domain_counts.get('in_scope', 0)} in_scope, {domain_counts.get('borderline', 0)} borderline, {domain_counts.get('out_of_scope', 0)} out_of_scope.",
            f"- Screening decision agent: {(prisma_like_flow or {}).get('records_included', 0)} include, {(prisma_like_flow or {}).get('records_maybe', 0)} maybe, {(prisma_like_flow or {}).get('records_excluded', 0)} exclude.",
            "- Full trace: `agent_trace.json`.",
            "",
            "## Limitations",
            "",
            "- Retrieval depends on provider metadata quality and availability.",
            "- Evidence extraction is lexical and abstract-only.",
            "- Verification checks grounding in abstracts, not full-text claims.",
            "- Ranking weights are transparent but manually chosen for the MVP.",
            "",
            "## Future Work",
            "",
            "- Add richer query expansion and provider diagnostics.",
            "- Compare rule-based extraction with optional LLM extraction.",
            "- Add a small human-labeling loop for iterative ranking studies.",
            "- Add full-text or PDF support only after the abstract-level baseline is validated.",
            "",
        ]
    )

    destination.write_text("\n".join(lines), encoding="utf-8")
