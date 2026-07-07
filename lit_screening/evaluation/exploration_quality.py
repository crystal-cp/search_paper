"""Exploration-quality metrics for research-process artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lit_screening.utils import to_plain_data, write_json


PAPER_ROLE_TARGETS = {
    "theory_origin",
    "conceptual_framework",
    "experimental_proof",
    "surface_probe_method",
    "nanoscale_readout",
    "material_case",
    "application_bridge",
    "frontier_extension",
    "limitation_or_challenge",
    "review_background",
}

EVIDENCE_FUNCTION_TARGETS = {
    "defines_concept",
    "predicts_effect",
    "reports_experiment",
    "directly_images_signal",
    "measures_spin_polarization",
    "reports_surface_probe",
    "connects_to_application",
    "reports_limitation",
    "review_background",
}


def compute_exploration_quality(
    concept_map: Any = None,
    query_families: Any = None,
    paper_roles: Any = None,
    evidence_functions: Any = None,
    gap_matrix: Any = None,
    research_tensions: Any = None,
    seed_hints: Any = None,
) -> dict[str, Any]:
    """Compute lightweight exploration-quality metrics from optional artifacts."""

    concept_payload = as_dict(concept_map)
    family_payload = as_dict(query_families)
    lenses = as_records(concept_payload.get("lenses"))
    families = as_records(family_payload.get("families"))
    role_records = as_records(paper_roles)
    evidence_records = as_records(evidence_functions)
    gap_rows = as_records(gap_matrix)
    tension_rows = as_records(research_tensions)
    seed_rows = as_records(seed_hints)

    lens_names = {
        str(lens.get("name") or "").strip()
        for lens in lenses
        if str(lens.get("name") or "").strip()
    }
    family_lens_names = {
        str(family.get("lens_name") or "").strip()
        for family in families
        if str(family.get("lens_name") or "").strip()
    }
    role_values = unique_role_values(role_records)
    evidence_function_values = {
        str(record.get("evidence_function") or "").strip()
        for record in evidence_records
        if str(record.get("evidence_function") or "").strip()
        and str(record.get("evidence_function") or "").strip() != "unknown"
    }

    concept_coverage = coverage_ratio(
        sum(1 for lens in lenses if lens_has_concepts(lens)),
        len(lenses),
    )
    query_family_coverage = (
        coverage_ratio(len(lens_names & family_lens_names), len(lens_names))
        if lens_names
        else (1.0 if families else 0.0)
    )
    paper_role_diversity = min(
        len(role_values & PAPER_ROLE_TARGETS) / len(PAPER_ROLE_TARGETS),
        1.0,
    )
    seed_hint_utilization = compute_seed_hint_utilization(seed_rows, families)
    evidence_function_diversity = min(
        len(evidence_function_values & EVIDENCE_FUNCTION_TARGETS)
        / len(EVIDENCE_FUNCTION_TARGETS),
        1.0,
    )
    gap_specificity = coverage_ratio(
        sum(1 for row in gap_rows if gap_row_is_specific(row)),
        len(gap_rows),
    )

    return {
        "concept_coverage": round(concept_coverage, 4),
        "query_family_coverage": round(query_family_coverage, 4),
        "paper_role_diversity": round(paper_role_diversity, 4),
        "seed_hint_utilization": round(seed_hint_utilization, 4),
        "evidence_function_diversity": round(evidence_function_diversity, 4),
        "gap_specificity": round(gap_specificity, 4),
        "research_tension_count": len(tension_rows),
        "details": {
            "lens_count": len(lenses),
            "conceptualized_lens_count": sum(
                1 for lens in lenses if lens_has_concepts(lens)
            ),
            "query_family_count": len(families),
            "covered_lens_count": len(lens_names & family_lens_names)
            if lens_names
            else len(families),
            "paper_role_count": len(role_values),
            "evidence_function_count": len(evidence_function_values),
            "gap_count": len(gap_rows),
            "specific_gap_count": sum(1 for row in gap_rows if gap_row_is_specific(row)),
            "seed_hint_count": len(seed_rows),
            "used_seed_hint_count": seed_hint_used_count(seed_rows, families),
        },
    }


def save_exploration_quality(path: str | Path, metrics: dict[str, Any]) -> None:
    """Write exploration-quality metrics to JSON."""

    write_json(path, metrics)


def as_dict(value: Any) -> dict[str, Any]:
    payload = to_plain_data(value)
    return payload if isinstance(payload, dict) else {}


def as_records(value: Any) -> list[dict[str, Any]]:
    payload = to_plain_data(value)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def lens_has_concepts(lens: dict[str, Any]) -> bool:
    for field in [
        "core_concepts",
        "synonyms",
        "materials",
        "methods",
        "applications",
        "seed_paper_hints",
        "expected_evidence_types",
    ]:
        if non_empty_values(lens.get(field)):
            return True
    return False


def unique_role_values(records: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for record in records:
        for role in non_empty_values(record.get("roles")):
            values.add(role)
        primary = str(record.get("primary_role") or "").strip()
        if primary:
            values.add(primary)
    return values


def compute_seed_hint_utilization(
    seed_rows: list[dict[str, Any]],
    families: list[dict[str, Any]],
) -> float:
    return coverage_ratio(seed_hint_used_count(seed_rows, families), len(seed_rows))


def seed_hint_used_count(
    seed_rows: list[dict[str, Any]],
    families: list[dict[str, Any]],
) -> int:
    if not seed_rows:
        return 0
    family_text = normalized_family_text(families)
    used = 0
    for seed in seed_rows:
        title = str(seed.get("title") or seed.get("raw_mention") or "").strip()
        authors = " ".join(non_empty_values(seed.get("authors")))
        needles = [title, title_core_phrase(title), authors]
        if any(normalize_text(needle) and normalize_text(needle) in family_text for needle in needles):
            used += 1
    return used


def normalized_family_text(families: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for family in families:
        parts.extend(
            [
                str(family.get("name") or ""),
                str(family.get("purpose") or ""),
                str(family.get("lens_name") or ""),
                " ".join(non_empty_values(family.get("linked_seed_titles"))),
            ]
        )
        queries_by_provider = family.get("queries_by_provider")
        if isinstance(queries_by_provider, dict):
            for queries in queries_by_provider.values():
                parts.extend(non_empty_values(queries))
    return normalize_text(" ".join(parts))


def gap_row_is_specific(row: dict[str, Any]) -> bool:
    has_identity = any(
        str(row.get(field) or "").strip()
        for field in ["gap_key", "gap_label", "gap"]
    )
    has_reason = any(
        str(row.get(field) or "").strip()
        for field in ["evidence_or_reason", "why_gap_remains", "supporting_papers"]
    )
    has_search = bool(non_empty_values(row.get("suggested_next_searches")))
    return has_identity and has_reason and has_search


def non_empty_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if str(key).strip() and str(item).strip()
        ]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    return [text]


def title_core_phrase(title: str) -> str:
    words = [word for word in normalize_text(title).split() if len(word) > 4]
    return " ".join(words[:5])


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def coverage_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))
