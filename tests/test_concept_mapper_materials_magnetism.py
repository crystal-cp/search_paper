from lit_screening.agents.concept_mapper import ConceptMapper


MATERIALS_QUESTION = (
    "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets: "
    "Classification, Example Materials, and Relation to Magnetoelectric Responses "
    "和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order "
    "相关的有关探测表面磁化和自旋极化的重要性的文章"
)


def test_concept_mapper_builds_materials_magnetism_lenses():
    plan = ConceptMapper().map_question(MATERIALS_QUESTION)
    lens_names = {lens.name for lens in plan.lenses}

    assert plan.central_question
    assert "direct_surface_detection" in lens_names
    assert "local_magnetoelectric_predictor" in lens_names
    assert {"theory_origin", "spaldin_framework"} & lens_names


def test_concept_mapper_materials_lens_content_contains_expected_terms():
    plan = ConceptMapper().map_question(MATERIALS_QUESTION)
    concepts = {
        term
        for lens in plan.lenses
        for term in [*lens.core_concepts, *lens.synonyms]
    }
    methods = {method for lens in plan.lenses for method in lens.methods}
    materials = {material for lens in plan.lenses for material in lens.materials}

    assert "surface magnetization" in concepts
    assert "boundary magnetization" in concepts
    assert "spin polarization" in concepts
    assert "antiferromagnet" in concepts
    assert "local magnetoelectric response" in concepts
    assert "magnetic multipole" in concepts
    assert "SPLEEM" in methods
    assert "XMCD-PEEM" in methods
    assert "NV magnetometry" in methods
    assert {"Cr2O3", "chromia"} & materials


def test_direct_surface_detection_lens_has_required_methods():
    plan = ConceptMapper().map_question(MATERIALS_QUESTION)
    lens = next(item for item in plan.lenses if item.name == "direct_surface_detection")

    assert "SPLEEM" in lens.methods
    assert "XMCD-PEEM" in lens.methods
    assert "spin-resolved photoemission" in lens.methods
    assert "SP-STM" in lens.methods


def test_nanoscale_readout_lens_has_required_methods():
    plan = ConceptMapper().map_question(MATERIALS_QUESTION)
    lens = next(item for item in plan.lenses if item.name == "nanoscale_readout")

    assert "NV magnetometry" in lens.methods
    assert "scanning diamond magnetometry" in lens.methods
