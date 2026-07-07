from lit_screening.agents.concept_mapper import ConceptMapper
from lit_screening.agents.query_family_planner import QueryFamilyPlanner


SPALDIN_QUESTION = (
    "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets: "
    "Classification, Example Materials, and Relation to Magnetoelectric Responses "
    "和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order "
    "相关的有关探测表面磁化和自旋极化的重要性的文章"
)


def test_query_family_planner_generates_materials_families_from_lenses():
    lens_plan = ConceptMapper().map_question(SPALDIN_QUESTION)
    family_plan = QueryFamilyPlanner().plan(lens_plan)
    family_names = {family.name for family in family_plan.families}

    assert family_plan.domain == "materials_magnetism"
    assert family_plan.central_question == lens_plan.central_question
    assert len(family_plan.families) >= 4
    assert "direct_surface_detection" in family_names
    assert {"theory_origin", "spaldin_framework"} & family_names


def test_direct_surface_detection_family_contains_surface_probe_queries():
    family_plan = QueryFamilyPlanner().plan(ConceptMapper().map_question(SPALDIN_QUESTION))
    family = _family(family_plan, "direct_surface_detection")
    joined = _joined_queries(family)

    assert "SPLEEM" in joined or "XMCD" in joined
    assert family.purpose
    assert family.lens_name == "direct_surface_detection"
    assert family.stop_condition


def test_theory_origin_family_contains_boundary_magnetization():
    family_plan = QueryFamilyPlanner().plan(ConceptMapper().map_question(SPALDIN_QUESTION))
    family = _family(family_plan, "theory_origin")
    joined = _joined_queries(family)

    assert "boundary magnetization" in joined


def test_spaldin_or_local_family_contains_local_magnetoelectric_effects():
    family_plan = QueryFamilyPlanner().plan(ConceptMapper().map_question(SPALDIN_QUESTION))
    candidates = [
        family
        for family in family_plan.families
        if family.name in {"spaldin_framework", "local_magnetoelectric_predictor"}
    ]
    joined = " ".join(_joined_queries(family) for family in candidates)

    assert "local magnetoelectric effects" in joined


def test_frontier_family_contains_altermagnetism_or_spin_splitting_when_present():
    family_plan = QueryFamilyPlanner().plan(ConceptMapper().map_question(SPALDIN_QUESTION))
    frontier = [family for family in family_plan.families if family.name == "frontier"]
    if frontier:
        joined = _joined_queries(frontier[0])
        assert "surface altermagnetism" in joined or "surface spin splitting" in joined


def _family(family_plan, name):
    return next(family for family in family_plan.families if family.name == name)


def _joined_queries(family):
    return " ".join(
        query
        for queries in family.queries_by_provider.values()
        for query in queries
    )
