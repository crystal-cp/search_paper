from lit_screening.agents.intent_repair import NoviceIntentInterpreter
from lit_screening.agents.seed_extraction import SeedExtractionAgent
from lit_screening.pipeline import plan_screening_queries


SPALDIN_QUESTION = (
    "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets: "
    "Classification, Example Materials, and Relation to Magnetoelectric Responses "
    "和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order "
    "相关的有关探测表面磁化和自旋极化的重要性的文章"
)


def _intent(question: str):
    seed_hints = SeedExtractionAgent().extract(question)
    return NoviceIntentInterpreter().repair(question, seed_hints=seed_hints)


def _terms(question: str) -> set[str]:
    return {concept.term.lower() for concept in _intent(question).structured_concepts}


def _provider_query_text(question: str) -> str:
    plan = plan_screening_queries(question)["query_plan"]
    return " ".join([*plan.openalex_queries, *plan.semantic_scholar_queries]).lower()


def _provider_queries(question: str) -> list[str]:
    plan = plan_screening_queries(question)["query_plan"]
    return [*plan.openalex_queries, *plan.semantic_scholar_queries]


def test_spaldin_case_activates_pack_terms_without_long_generic_query():
    terms = _terms(SPALDIN_QUESTION)
    queries = _provider_query_text(SPALDIN_QUESTION)

    assert "surface magnetization" in terms
    assert "boundary magnetization" in terms
    assert any("spin polarization" in term for term in terms)
    assert "antiferromagnet" in terms
    assert "local magnetoelectric response" in terms
    assert "magnetic multipole" in terms
    assert "cr2o3" in terms or "chromia" in terms
    assert "spleem" in terms
    assert "xmcd-peem" in terms
    assert "nv magnetometry" in terms
    assert "surface magnetization magnetization antiferromagnetism" not in queries
    assert "do not infer citation" not in queries


def test_ferroelectric_surface_polarization_does_not_get_magnetic_pack_terms():
    question = "我想调研铁电薄膜表面极化为什么重要，以及怎么探测"
    terms = _terms(question)
    queries = _provider_query_text(question)
    query_items = _provider_queries(question)

    assert "ferroelectric polarization" in terms
    assert "surface polarization" in terms
    assert "pfm" in terms
    assert "boundary magnetization" not in terms
    assert "cr2o3" not in terms
    assert "spleem" not in terms
    assert "boundary magnetization" not in queries
    assert "cr2o3" not in queries
    assert "spleem" not in queries
    assert "importance" not in queries
    assert "significance" not in queries
    assert "review" not in queries
    assert "do not infer citation" not in queries
    assert all(len(query) < 180 for query in query_items)


def test_altermagnetism_detection_does_not_import_spaldin_material_case():
    question = "我想找关于 altermagnetism 如何被实验探测的文章"
    terms = _terms(question)
    queries = _provider_query_text(question)

    assert "altermagnetism" in terms
    assert "spin splitting" in terms
    assert "sarpes" in terms or "spin-resolved photoemission" in terms
    assert "spaldin" not in terms
    assert "cr2o3" not in terms
    assert "chromia" not in terms
    assert "surface magnetization in antiferromagnets" not in queries
    assert "cr2o3" not in queries
    assert "chromia" not in queries


def test_llm_literature_screening_keeps_ai_domain_terms():
    question = "我想找 LLM 做文献筛选的人机协作论文"
    terms = _terms(question)
    queries = _provider_query_text(question)
    query_items = _provider_queries(question)

    assert "llm" in terms
    assert "literature screening" in terms
    assert "human-in-the-loop" in terms
    assert "surface magnetization" not in terms
    assert "antiferromagnet" not in terms
    assert "cr2o3" not in terms
    assert "surface magnetization" not in queries
    assert "antiferromagnet" not in queries
    assert "cr2o3" not in queries
    assert "do not infer citation" not in queries
    assert "review" not in queries
    assert all(len(query) < 220 for query in query_items)
