import json
from pathlib import Path

from lit_screening.agents.concept_mapper import ConceptMapper
from lit_screening.agents.query_family_planner import QueryFamilyPlanner
from lit_screening.agents.seed_extraction import SeedExtractionAgent


CASE_PATH = Path(__file__).parent / "fixtures" / "spaldin_surface_magnetization_case.json"


def test_spaldin_case_concept_mapper_covers_researcher_concepts():
    case = load_case()
    lens_plan = build_lens_plan(case)
    text = normalize(" ".join(flatten_lens_values(lens_plan.lenses)))

    matched = matched_terms(case["expected_concepts"], text)

    assert len(matched) >= 5
    assert "surface magnetization" in matched
    assert "boundary magnetization" in matched
    assert "local magnetoelectric response" in matched


def test_spaldin_case_concept_mapper_covers_materials_and_methods():
    case = load_case()
    lens_plan = build_lens_plan(case)
    text = normalize(" ".join(flatten_lens_values(lens_plan.lenses)))

    matched_materials = matched_terms(case["expected_materials"], text)
    matched_methods = matched_terms(case["expected_methods"], text)

    assert len(matched_materials) >= 4
    assert len(matched_methods) >= 4
    assert "Cr2O3" in matched_materials
    assert "SPLEEM" in matched_methods
    assert "NV magnetometry" in matched_methods


def test_spaldin_case_query_family_planner_covers_expected_lenses():
    case = load_case()
    seed_hints = SeedExtractionAgent().extract(case["user_question"])
    lens_plan = ConceptMapper().map_question(
        case["user_question"],
        seed_hints=seed_hints,
    )
    family_plan = QueryFamilyPlanner().plan(lens_plan, seed_hints=seed_hints)

    lens_names = {lens.name for lens in lens_plan.lenses}
    family_names = {family.name for family in family_plan.families}
    expected_lenses = set(case["expected_lenses"])

    assert len(expected_lenses & lens_names) >= 5
    assert len(expected_lenses & family_names) >= 5
    assert "direct_surface_detection" in family_names
    assert "local_magnetoelectric_predictor" in family_names
    assert "spaldin_framework" in family_names


def test_spaldin_case_query_family_queries_cover_expected_phrases():
    case = load_case()
    seed_hints = SeedExtractionAgent().extract(case["user_question"])
    lens_plan = ConceptMapper().map_question(
        case["user_question"],
        seed_hints=seed_hints,
    )
    family_plan = QueryFamilyPlanner().plan(lens_plan, seed_hints=seed_hints)
    query_text = normalize(
        " ".join(
            query
            for family in family_plan.families
            for queries in family.queries_by_provider.values()
            for query in queries
        )
    )

    matched = matched_terms(case["expected_query_phrases"], query_text)

    assert len(matched) >= 6
    assert "boundary magnetization" in matched
    assert "local magnetoelectric effects" in matched
    assert "NV magnetometry" in matched


def load_case() -> dict:
    return json.loads(CASE_PATH.read_text(encoding="utf-8"))


def build_lens_plan(case: dict):
    seed_hints = SeedExtractionAgent().extract(case["user_question"])
    return ConceptMapper().map_question(
        case["user_question"],
        seed_hints=seed_hints,
    )


def flatten_lens_values(lenses) -> list[str]:
    values: list[str] = []
    for lens in lenses:
        values.extend(
            [
                lens.name,
                lens.role,
                lens.question,
                *lens.core_concepts,
                *lens.synonyms,
                *lens.materials,
                *lens.methods,
                *lens.applications,
                *lens.seed_paper_hints,
                *lens.expected_evidence_types,
                *lens.exclusion_risks,
            ]
        )
    return values


def matched_terms(expected_terms: list[str], text: str) -> set[str]:
    return {
        term
        for term in expected_terms
        if normalize(term) in text
    }


def normalize(text: str) -> str:
    return " ".join(str(text).lower().replace("+", " ").replace("\"", " ").split())
