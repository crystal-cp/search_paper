import json

import pytest

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.agents.generic_intent import is_single_acronym_query
from lit_screening.pipeline import (
    _queries_by_provider,
    build_query_provenance,
    build_research_lens_artifacts,
    plan_screening_queries,
    run_pipeline,
)
from lit_screening.retrieval.base import RetrievalResult
from lit_screening.utils import ensure_dir


SEI_QUESTION = (
    "我想了解锂离子电池中 SEI 界面为什么重要，以及有哪些实验方法可以表征 SEI "
    "的组成、结构和演化。最好能找到理论背景、原位/非原位表征方法、"
    "典型材料体系和失效机制相关论文。"
)

OER_QUESTION = (
    "我想了解过渡金属氧化物催化剂表面自旋态对析氧反应 OER 活性的影响，"
    "想找理论机制、实验表征方法、典型材料和争议点相关论文。"
)

AI_SCREENING_QUESTION = (
    "我想找 LLM 用于系统综述文献筛选的人机协作论文，重点关注人工反馈、"
    "证据验证、召回率和筛选准确率。"
)

THIN_FILM_QUESTION = (
    "我想找关于薄膜沉积方法比较的综述，比如 ALD、PLD、sputtering 和 CVD 的优缺点。"
)

MAGNETISM_QUESTION = "探测表面磁化和自旋极化的重要性\n10.1103/physrevx.14.021033"

FERROELECTRIC_QUESTION = (
    "我想了解铁电薄膜表面极化为什么重要，以及有哪些实验方法可以直接探测或表征它。"
    "最好能帮我找到理论背景、实验探测方法、典型材料案例、器件应用，"
    "以及表面/界面屏蔽效应相关的论文。"
)


class EmptyOpenAlexClient:
    provider_name = "openalex"

    def search(self, *args, **kwargs):
        return RetrievalResult(raw={"results": []}, papers=[])


def _plan_with_families(question, tmp_path):
    payload = plan_screening_queries(question)
    concept_map, family_plan, trace = build_research_lens_artifacts(
        question,
        payload["search_brief"],
        payload["search_contract"],
        ensure_dir(tmp_path / "outputs"),
        seed_hints=payload["seed_hints"],
    )
    providers = ["openalex", "semantic_scholar"]
    queries_by_provider = _queries_by_provider(
        providers,
        payload["query_plan"],
        payload["queries"],
    )
    final_queries, provenance = build_query_provenance(
        providers=providers,
        queries_by_provider=queries_by_provider,
        query_family_plan=family_plan,
        use_query_families=True,
        max_family_queries_per_provider=18,
    )
    return payload, concept_map, family_plan, final_queries, provenance, trace


def _all_queries(final_queries):
    return [
        query
        for provider_queries in final_queries.values()
        for query in provider_queries
    ]


def _query_text(final_queries):
    return "\n".join(_all_queries(final_queries)).lower()


def _assert_generic_family_applied(final_queries, provenance):
    queries = _all_queries(final_queries)
    assert provenance["applied"] is True
    assert sum(len(values) for values in final_queries.values()) >= 8
    assert not any(is_single_acronym_query(query) for query in queries)


def test_sei_general_science_generates_generic_query_family(tmp_path):
    payload, concept_map, _family_plan, final_queries, provenance, trace = _plan_with_families(
        SEI_QUESTION,
        tmp_path,
    )
    lens_names = {lens.name for lens in concept_map.lenses}
    query_text = _query_text(final_queries)

    assert payload["search_contract"].domain_profile.domain_name == "general_science"
    assert trace["generic_fallback_used"] is True
    _assert_generic_family_applied(final_queries, provenance)
    assert {"background_review", "theory_mechanism", "characterization_methods"} <= lens_names
    assert {"in_situ_or_operando_methods", "ex_situ_methods"} <= lens_names
    assert {"materials_or_cases", "failure_or_limitation"} <= lens_names
    assert "sei" in query_text
    assert "lithium-ion battery" in query_text or "battery" in query_text


def test_oer_spin_state_does_not_activate_magnetism_pack(tmp_path):
    payload, concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        OER_QUESTION,
        tmp_path,
    )
    lens_names = {lens.name for lens in concept_map.lenses}
    query_text = _query_text(final_queries)

    assert payload["search_contract"].domain_profile.domain_name != "materials_magnetism"
    _assert_generic_family_applied(final_queries, provenance)
    assert {"theory_mechanism", "characterization_methods"} <= lens_names
    assert {"materials_or_cases", "application_or_performance", "controversy_debate"} <= lens_names
    assert "oer" in query_text or "oxygen evolution reaction" in query_text
    assert "transition metal oxide" in query_text or "catalyst" in query_text
    assert "spin state" in query_text or "electronic structure" in query_text


def test_ai_literature_screening_stays_out_of_material_domain_packs(tmp_path):
    payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        AI_SCREENING_QUESTION,
        tmp_path,
    )
    domain = payload["search_contract"].domain_profile.domain_name
    query_text = _query_text(final_queries)

    assert domain not in {
        "materials_magnetism",
        "ferroelectric_polarization",
        "battery_interface",
        "oxide_catalysis",
    }
    _assert_generic_family_applied(final_queries, provenance)
    assert "llm" in query_text or "large language model" in query_text
    assert "systematic review" in query_text or "literature screening" in query_text
    assert "human feedback" in query_text or "human-in-the-loop" in query_text
    assert "evidence validation" in query_text
    assert "pfm" not in query_text
    assert "shg" not in query_text
    assert "cr2o3" not in query_text
    assert "sei" not in query_text
    assert "oer" not in query_text


def test_general_thin_film_deposition_does_not_activate_ferroelectric_pack(tmp_path):
    payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        THIN_FILM_QUESTION,
        tmp_path,
    )
    query_text = _query_text(final_queries)

    assert payload["search_contract"].domain_profile.domain_name != "ferroelectric_polarization"
    _assert_generic_family_applied(final_queries, provenance)
    assert "thin film deposition" in query_text
    assert "ald" in query_text
    assert "pld" in query_text
    assert "sputtering" in query_text
    assert "cvd" in query_text
    assert "comparison" in query_text or "advantages disadvantages" in query_text
    assert "pfm" not in query_text
    assert "shg" not in query_text
    assert "batio3" not in query_text
    assert "depolarization field" not in query_text


@pytest.mark.parametrize(
    "question,expected_terms",
    [
        (
            "我想了解 MOF 材料用于 CO2 捕集时孔径、官能团和水稳定性对吸附性能的影响，"
            "想找理论机制、实验表征、典型材料和应用限制相关论文。",
            ["mof", "co2", "pore size", "functional groups", "adsorption performance"],
        ),
        (
            "我想了解钙钛矿太阳能电池中缺陷钝化为什么重要，以及有哪些实验方法可以表征"
            "缺陷态和载流子复合。",
            ["perovskite solar cell", "defect passivation", "defect states", "carrier recombination"],
        ),
    ],
)
def test_unknown_domains_still_generate_generic_query_family(question, expected_terms, tmp_path):
    payload, concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        question,
        tmp_path,
    )
    query_text = _query_text(final_queries)
    lens_names = {lens.name for lens in concept_map.lenses}

    assert payload["search_contract"].domain_profile.domain_name == "general_science"
    _assert_generic_family_applied(final_queries, provenance)
    assert len(lens_names & {"background_review", "theory_mechanism", "characterization_methods"}) >= 2
    assert any(term in query_text for term in expected_terms)
    assert "surface magnetization" not in query_text
    assert "ferroelectric polarization" not in query_text
    assert "sei" not in query_text
    assert "oer" not in query_text


def test_magnetism_pack_regression_not_polluted_by_other_domains(tmp_path):
    payload, concept_map, family_plan, final_queries, provenance, _trace = _plan_with_families(
        MAGNETISM_QUESTION,
        tmp_path,
    )
    family_names = {family.name for family in family_plan.families}
    query_text = _query_text(final_queries)

    assert payload["search_contract"].domain_profile.domain_name == "materials_magnetism"
    assert provenance["applied"] is True
    assert {"theory_origin", "direct_surface_detection", "nanoscale_readout"} <= family_names
    assert "pfm" not in query_text
    assert "batio3" not in query_text
    assert "sei" not in query_text
    assert "oer" not in query_text
    assert concept_map.domain == "materials_magnetism"


def test_ferroelectric_pack_regression_not_polluted_by_other_domains(tmp_path):
    payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        FERROELECTRIC_QUESTION,
        tmp_path,
    )
    query_text = _query_text(final_queries)

    assert payload["search_contract"].domain_profile.domain_name == "ferroelectric_polarization"
    assert provenance["applied"] is True
    assert "sei" not in query_text
    assert "oer" not in query_text
    assert "cr2o3" not in query_text
    assert "spleem" not in query_text


def test_generic_run_writes_auditable_diagnostics(tmp_path):
    output_dir = tmp_path / "outputs"
    run_pipeline(
        SEI_QUESTION,
        providers=["openalex"],
        max_per_query=0,
        output_dir=str(output_dir),
        use_query_families=True,
        retriever_agent=RetrieverAgent(clients={"openalex": EmptyOpenAlexClient()}),
    )

    planned = json.loads((output_dir / "planned_queries.json").read_text())
    provenance = json.loads((output_dir / "query_provenance.json").read_text())
    trace = json.loads((output_dir / "agent_trace.json").read_text())
    exploration = json.loads((output_dir / "exploration_quality.json").read_text())
    ranking = json.loads((output_dir / "ranking_diagnostics.json").read_text())

    assert planned["selected_domain"] == "general_science"
    assert planned["query_family_applied"] is True
    assert planned["generic_fallback_used"] is True
    assert planned["active_research_lenses"]
    assert provenance["records"]
    assert all("concepts" in record for record in provenance["records"])
    assert all("provider_serializer_changes" in record for record in provenance["records"])
    assert provenance["generic_fallback_used"] is True
    assert trace["generic_research_intent_frame"]["research_object"]
    assert trace["domain_activation"]["selected_domain"] == "general_science"
    assert "trigger_reasons" in trace["query_repair"]
    assert exploration["provider_query_count"] >= 8
    assert "aspect_coverage_distribution" in exploration
    assert "single_acronym_query_count" in exploration
    assert "role_assignment_evidence" in ranking
    assert "lane_aspect_contribution" in ranking
    assert "acronym_disambiguation_penalties" in ranking
