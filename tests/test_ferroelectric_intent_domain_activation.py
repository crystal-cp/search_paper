from lit_screening.agents.intent_repair import NoviceIntentInterpreter
from lit_screening.pipeline import plan_screening_queries


FERROELECTRIC_QUESTION = (
    "我想了解铁电薄膜表面极化为什么重要，以及有哪些实验方法可以直接探测或表征它。"
    "最好能帮我找到理论背景、实验探测方法、典型材料案例、器件应用，"
    "以及表面/界面屏蔽效应相关的论文。"
)


def test_ferroelectric_intent_activates_domain_pack_without_magnetic_terms():
    intent = NoviceIntentInterpreter().repair(FERROELECTRIC_QUESTION)
    terms = {concept.term.lower() for concept in intent.structured_concepts}

    assert intent.llm_metadata["domain_pack_domain"] == "ferroelectric_polarization"
    assert "ferroelectric polarization" in terms
    assert "surface polarization" in terms
    assert "piezoresponse force microscopy" in terms
    assert "second harmonic generation" in terms
    assert "boundary magnetization" not in terms
    assert "cr2o3" not in terms
    assert "spleem" not in terms


def test_search_contract_uses_ferroelectric_domain_not_general_science():
    payload = plan_screening_queries(FERROELECTRIC_QUESTION)
    contract = payload["search_contract"]
    query_text = " ".join(
        [
            *payload["query_plan"].openalex_queries,
            *payload["query_plan"].semantic_scholar_queries,
        ]
    ).lower()

    assert contract.domain_profile.domain_name == "ferroelectric_polarization"
    assert contract.constraint_groups
    assert any(group.group_name == "ferroelectric_context" for group in contract.constraint_groups)
    assert "boundary magnetization" not in query_text
    assert "cr2o3" not in query_text
    assert "spleem" not in query_text
