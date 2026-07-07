import re

from lit_screening.pipeline import (
    _queries_by_provider,
    build_query_provenance,
    build_research_lens_artifacts,
    plan_screening_queries,
)
from lit_screening.utils import ensure_dir


FERROELECTRIC_QUESTION = (
    "我想了解铁电薄膜表面极化为什么重要，以及有哪些实验方法可以直接探测或表征它。"
    "最好能帮我找到理论背景、实验探测方法、典型材料案例、器件应用，"
    "以及表面/界面屏蔽效应相关的论文。"
)


def test_ferroelectric_query_family_plan_and_final_provider_queries(tmp_path):
    payload = plan_screening_queries(FERROELECTRIC_QUESTION)
    concept_map, family_plan, _trace = build_research_lens_artifacts(
        FERROELECTRIC_QUESTION,
        payload["search_brief"],
        payload["search_contract"],
        ensure_dir(tmp_path / "outputs"),
        seed_hints=payload["seed_hints"],
    )

    assert concept_map is not None
    assert family_plan is not None
    family_names = {family.name for family in family_plan.families}
    assert {
        "theory_origin",
        "direct_probe_methods",
        "interface_screening",
        "materials_cases",
        "device_applications",
    } <= family_names

    providers = ["openalex", "semantic_scholar"]
    query_plan = payload["query_plan"]
    queries_by_provider = _queries_by_provider(providers, query_plan, payload["queries"])
    final_queries, provenance = build_query_provenance(
        providers=providers,
        queries_by_provider=queries_by_provider,
        query_family_plan=family_plan,
        use_query_families=True,
        max_family_queries_per_provider=18,
    )
    all_queries = [
        query
        for provider_queries in final_queries.values()
        for query in provider_queries
    ]
    query_text = "\n".join(all_queries)

    assert provenance["applied"] is True
    assert provenance["query_family_queries"]
    assert provenance["provider_queries_by_family"]
    assert '"piezoresponse force microscopy" "ferroelectric thin film"' in query_text
    assert '"second harmonic generation" ferroelectric surface polarization' in query_text
    assert not re.search(r"[\u4e00-\u9fff]", "\n".join(final_queries["semantic_scholar"]))
    assert " -非铁电材料" not in query_text
    assert all(query not in {"thin film", "screening", "PFM", "SHG"} for query in all_queries)
