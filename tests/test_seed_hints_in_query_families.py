import json

from lit_screening.agents.concept_mapper import ConceptMapper
from lit_screening.agents.query_family_planner import QueryFamilyPlanner
from lit_screening.agents.seed_extraction import SeedExtractionAgent
from lit_screening.pipeline import run_pipeline


SPALDIN_QUESTION = (
    "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets: "
    "Classification, Example Materials, and Relation to Magnetoelectric Responses "
    "和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order "
    "相关的有关探测表面磁化和自旋极化的重要性的文章"
)


def test_seed_hints_add_lenses_and_seed_context_family():
    seed_hints = SeedExtractionAgent().extract(SPALDIN_QUESTION)
    lens_plan = ConceptMapper().map_question(
        SPALDIN_QUESTION,
        seed_hints=seed_hints,
    )
    family_plan = QueryFamilyPlanner().plan(lens_plan, seed_hints=seed_hints)
    lens_names = {lens.name for lens in lens_plan.lenses}
    family_names = {family.name for family in family_plan.families}
    seed_context = next(
        family for family in family_plan.families if family.name == "seed_context"
    )
    joined_queries = " ".join(
        query
        for queries in seed_context.queries_by_provider.values()
        for query in queries
    )

    assert {
        "spaldin_framework",
        "theory_origin",
        "surface_magnetization_classification",
        "local_magnetoelectric_predictor",
        "local_magnetic_order",
    }.issubset(lens_names)
    assert "seed_context" in family_names
    assert "Spaldin" in joined_queries
    assert "surface magnetization" in joined_queries
    assert "local magnetoelectric effects" in joined_queries
    assert seed_context.linked_seed_titles
    assert seed_context.seed_hint_confidence > 0


def test_pipeline_query_families_artifact_preserves_seed_metadata(tmp_path):
    output_dir = tmp_path / "outputs"

    run_pipeline(
        question=SPALDIN_QUESTION,
        providers=[],
        max_per_query=0,
        output_dir=str(output_dir),
    )
    payload = json.loads((output_dir / "query_families.json").read_text())
    seed_context = next(
        family
        for family in payload["families"]
        if family["name"] == "seed_context"
    )
    joined_queries = " ".join(
        query
        for queries in seed_context["queries_by_provider"].values()
        for query in queries
    )

    assert seed_context["linked_seed_titles"]
    assert seed_context["seed_hint_confidence"] > 0
    assert "Spaldin" in joined_queries
    assert "surface magnetization" in joined_queries
    assert "local magnetoelectric effects" in joined_queries
