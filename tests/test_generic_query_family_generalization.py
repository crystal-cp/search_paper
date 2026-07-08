import json

import pytest

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.agents.generic_intent import is_single_acronym_query
from lit_screening.pipeline import (
    _queries_by_provider,
    build_auto_query_repair_suggestions,
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
        search_contract=payload["search_contract"],
        concept_map=concept_map,
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


def _contains_any(text, terms):
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _assert_generic_family_applied(final_queries, provenance):
    queries = _all_queries(final_queries)
    assert provenance["applied"] is True
    assert sum(len(values) for values in final_queries.values()) >= 8
    assert not any(is_single_acronym_query(query) for query in queries)


def _assert_no_standalone_query(final_queries, forbidden_terms):
    forbidden = {term.lower() for term in forbidden_terms}
    for query in _all_queries(final_queries):
        normalized = query.strip().strip("+\"'").lower()
        assert normalized not in forbidden


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
    _assert_no_standalone_query(final_queries, {"sei", "+sei"})
    assert all(
        _contains_any(query, {"sei", "solid electrolyte interphase"})
        and _contains_any(query, {"lithium-ion battery", "battery", "interface"})
        for query in _all_queries(final_queries)
    )


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
    _assert_no_standalone_query(final_queries, {"oer", "+oer"})
    assert any(
        _contains_any(query, {"oer", "oxygen evolution reaction"})
        and _contains_any(query, {"transition metal oxide", "catalyst"})
        and _contains_any(query, {"spin state", "electronic structure"})
        for query in _all_queries(final_queries)
    )
    assert any(
        _contains_any(query, {"controversy", "competing mechanism"})
        and _contains_any(query, {"spin state", "electronic structure"})
        and _contains_any(query, {"oer", "oxygen evolution reaction"})
        for query in _all_queries(final_queries)
    )


def test_oer_final_queries_have_three_group_intersection(tmp_path):
    _payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        OER_QUESTION,
        tmp_path,
    )
    unique_queries = sorted(set(_all_queries(final_queries)))
    intersection_queries = [
        query
        for query in unique_queries
        if _contains_any(query, {"oer", "oxygen evolution reaction"})
        and _contains_any(
            query,
            {"spin state", "surface spin state", "spin polarization", "electronic structure", "orbital occupancy"},
        )
        and _contains_any(
            query,
            {"transition metal oxide", "oxide catalyst", "catalyst", "electrocatalyst"},
        )
    ]

    assert provenance["applied"] is True
    assert len(intersection_queries) >= 5
    assert any("controversy" in query.lower() for query in intersection_queries)
    _assert_no_standalone_query(final_queries, {"oer", "+oer"})


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


def test_ai_screening_no_characterization_template(tmp_path):
    _payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        AI_SCREENING_QUESTION,
        tmp_path,
    )
    query_text = _query_text(final_queries)
    multi_agent_queries = [
        query for query in _all_queries(final_queries) if "multi-agent" in query.lower()
    ]

    assert provenance["applied"] is True
    assert "characterization" not in query_text
    assert "materials" not in query_text
    assert "systematic review" in query_text
    assert "literature screening" in query_text
    assert "human feedback" in query_text or "human-in-the-loop" in query_text
    assert "evidence validation" in query_text
    assert len(multi_agent_queries) <= 1


def test_ai_screening_query_count_and_coverage(tmp_path):
    _payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        AI_SCREENING_QUESTION,
        tmp_path,
    )
    queries = sorted(set(_all_queries(final_queries)))
    query_text = "\n".join(queries).lower()

    assert provenance["applied"] is True
    assert len(queries) >= 8
    assert "systematic review screening" in query_text
    assert "title abstract screening" in query_text
    assert "study selection" in query_text
    assert "human-in-the-loop" in query_text
    assert "human feedback" in query_text
    assert "evidence validation" in query_text or "evidence verification" in query_text
    assert "recall accuracy" in query_text or "recall precision" in query_text
    assert "active learning" in query_text


def test_ai_screening_no_material_template_terms(tmp_path):
    _payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        AI_SCREENING_QUESTION,
        tmp_path,
    )
    query_text = _query_text(final_queries)

    assert provenance["applied"] is True
    assert "representative materials" not in query_text
    assert "experimental characterization" not in query_text
    assert "materials_or_cases" not in query_text


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
    assert "pld" in query_text or "pulsed laser deposition" in query_text
    assert "sputtering" in query_text
    assert "cvd" in query_text
    assert "comparison" in query_text or "advantages disadvantages" in query_text
    _assert_no_standalone_query(final_queries, {"sputtering", "+sputtering"})
    assert any(
        "thin film deposition" in query.lower()
        and "sputtering" in query.lower()
        and "cvd" in query.lower()
        and _contains_any(query, {"ald", "atomic layer deposition"})
        and _contains_any(query, {"pld", "pulsed laser deposition"})
        and _contains_any(query, {"comparison", "advantages disadvantages"})
        for query in _all_queries(final_queries)
    )
    assert "pfm" not in query_text
    assert "shg" not in query_text
    assert "batio3" not in query_text
    assert "depolarization field" not in query_text


def test_thin_film_method_comparison_query_count_and_no_long_duplicates(tmp_path):
    _payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        THIN_FILM_QUESTION,
        tmp_path,
    )
    unique_queries = sorted(set(_all_queries(final_queries)))
    query_text = "\n".join(unique_queries).lower()

    assert provenance["applied"] is True
    assert len(unique_queries) >= 8
    assert all(len(query.split()) <= 12 for query in unique_queries)
    assert any(
        "ald" in query.lower()
        and _contains_any(query, {"pld", "pulsed laser deposition"})
        and "sputtering" in query.lower()
        and "cvd" in query.lower()
        and _contains_any(query, {"comparison", "review", "advantages disadvantages"})
        for query in unique_queries
    )
    assert "thin film deposition" in query_text
    assert "ald" in query_text
    assert "pld" in query_text or "pulsed laser deposition" in query_text
    assert "sputtering" in query_text
    assert "cvd" in query_text
    _assert_no_standalone_query(final_queries, {"sputtering", "+sputtering"})


def test_thin_film_queries_cover_multiple_methods(tmp_path):
    _payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        THIN_FILM_QUESTION,
        tmp_path,
    )
    queries = sorted(set(_all_queries(final_queries)))
    query_text = "\n".join(queries).lower()

    assert provenance["applied"] is True
    assert len(queries) >= 8
    assert "ald" in query_text or "atomic layer deposition" in query_text
    assert "pld" in query_text or "pulsed laser deposition" in query_text
    assert "sputtering" in query_text
    assert "cvd" in query_text or "chemical vapor deposition" in query_text
    assert sum(
        1
        for query in queries
        if _contains_any(query, {"comparison", "review", "advantages", "disadvantages", "pros cons", "methods", "techniques"})
        and _contains_any(query, {"thin film", "deposition", "fabrication"})
    ) >= 6


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
    assert not any(is_single_acronym_query(query) for query in _all_queries(final_queries))
    assert "surface magnetization" not in query_text
    assert "ferroelectric polarization" not in query_text
    assert "sei" not in query_text
    assert "oer" not in query_text
    if "钙钛矿" in question:
        assert "battery" not in query_text
        assert "lithium" not in query_text


def test_mof_queries_require_mof_co2_anchor(tmp_path):
    question = (
        "我想了解 MOF 材料用于 CO2 捕集时孔径、官能团和水稳定性对吸附性能的影响，"
        "想找理论机制、实验表征、典型材料和应用限制相关论文。"
    )
    _payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        question,
        tmp_path,
    )
    queries = sorted(set(_all_queries(final_queries)))

    assert provenance["applied"] is True
    assert len(queries) >= 8
    for query in queries:
        assert _contains_any(query, {"mof", "metal-organic framework"})
        assert _contains_any(query, {"co2", "carbon dioxide", "capture", "adsorption"})
    assert not any(
        "water stability" in query.lower()
        and "adsorption performance" in query.lower()
        and not _contains_any(query, {"mof", "metal-organic framework", "co2", "carbon dioxide"})
        for query in queries
    )


def test_mof_queries_require_mof_co2_adsorption_anchor(tmp_path):
    question = (
        "我想了解 MOF 材料用于 CO2 捕集时孔径、官能团和水稳定性对吸附性能的影响，"
        "想找理论机制、实验表征、典型材料和应用限制相关论文。"
    )
    _payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        question,
        tmp_path,
    )
    queries = sorted(set(_all_queries(final_queries)))

    assert provenance["applied"] is True
    assert len(queries) >= 8
    assert all(_contains_any(query, {"mof", "metal-organic framework"}) for query in queries)
    assert all(
        _contains_any(query, {"co2", "carbon dioxide", "carbon capture", "adsorption", "capture"})
        for query in queries
    )
    assert any("co2 adsorption" in query.lower() for query in queries)


def test_mof_overbroad_queries_dropped(tmp_path):
    question = (
        "我想了解 MOF 材料用于 CO2 捕集时孔径、官能团和水稳定性对吸附性能的影响，"
        "想找理论机制、实验表征、典型材料和应用限制相关论文。"
    )
    _payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        question,
        tmp_path,
    )
    queries = {query.strip().strip('"').lower() for query in _all_queries(final_queries)}
    dropped = provenance["dropped_queries"]
    dropped_text = "\n".join(str(item) for item in dropped).lower()

    assert "capture mof" not in queries
    assert "mof capture" not in queries
    assert "mof co2" not in queries
    assert not any("representative materials" in query for query in queries)
    assert "accepted_critic_issue" in dropped_text
    assert "mof_short_anchor_only" in dropped_text or "mof_missing" in dropped_text


def test_query_critic_issues_affect_final_queries(tmp_path):
    question = (
        "我想了解 MOF 材料用于 CO2 捕集时孔径、官能团和水稳定性对吸附性能的影响，"
        "想找理论机制、实验表征、典型材料和应用限制相关论文。"
    )
    payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        question,
        tmp_path,
    )
    repair = build_auto_query_repair_suggestions(payload["query_plan"], provenance)
    final_query_text = "\n".join(_all_queries(final_queries)).lower()
    critic_actions = [
        action
        for action in repair["query_actions"]
        if action["action"] == "drop" and action.get("accepted_critic_issue")
    ]

    assert critic_actions
    assert all(action["original_query"].lower() not in final_query_text for action in critic_actions)
    assert any("overbroad_queries" in action["accepted_critic_issue"] for action in critic_actions)


def test_planned_queries_records_dropped_and_repaired_queries(tmp_path):
    output_dir = tmp_path / "outputs"
    question = (
        "我想了解 MOF 材料用于 CO2 捕集时孔径、官能团和水稳定性对吸附性能的影响，"
        "想找理论机制、实验表征、典型材料和应用限制相关论文。"
    )
    run_pipeline(
        question,
        providers=["openalex"],
        max_per_query=0,
        output_dir=str(output_dir),
        use_query_families=True,
        retriever_agent=RetrieverAgent(clients={"openalex": EmptyOpenAlexClient()}),
    )
    planned = json.loads((output_dir / "planned_queries.json").read_text())
    provenance = json.loads((output_dir / "query_provenance.json").read_text())
    repairs = json.loads((output_dir / "query_repair_suggestions.json").read_text())

    assert planned["dropped_queries"] == provenance["dropped_queries"]
    assert "repaired_queries" in planned
    assert "query_repair_actions" in planned
    assert repairs["query_actions"]
    assert any(action["action"] == "drop" for action in repairs["query_actions"])
    assert any(action.get("accepted_critic_issue") for action in repairs["query_actions"])


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


def test_query_repair_replaces_dropped_overbroad_old_planner_queries(tmp_path):
    payload, _concept_map, _family_plan, final_queries, provenance, _trace = _plan_with_families(
        SEI_QUESTION,
        tmp_path,
    )
    repair = build_auto_query_repair_suggestions(payload["query_plan"], provenance)
    final_query_set = set(_all_queries(final_queries))

    assert provenance["dropped_query_count"] > 0
    assert "static_query_quality_repair" in repair["trigger_reasons"]
    assert repair["removed_queries"]
    assert repair["repaired_query_count"] == len(final_query_set)
    assert not final_query_set.intersection(repair["removed_queries"])


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
    repairs = json.loads((output_dir / "query_repair_suggestions.json").read_text())
    trace = json.loads((output_dir / "agent_trace.json").read_text())
    exploration = json.loads((output_dir / "exploration_quality.json").read_text())
    ranking = json.loads((output_dir / "ranking_diagnostics.json").read_text())

    assert planned["selected_domain"] == "general_science"
    assert planned["query_family_applied"] is True
    assert planned["generic_fallback_used"] is True
    assert planned["active_research_lenses"]
    assert planned["final_openalex_queries"] == provenance["final_openalex_queries"]
    assert planned["queries_by_provider"]["openalex"] == provenance["final_openalex_queries"]
    assert planned["queries"] == provenance["final_openalex_queries"]
    assert planned["dropped_queries"] == provenance["dropped_queries"]
    assert provenance["records"]
    assert all("concepts" in record for record in provenance["records"])
    assert all("provider_serializer_changes" in record for record in provenance["records"])
    assert provenance["generic_fallback_used"] is True
    assert trace["generic_research_intent_frame"]["research_object"]
    assert trace["domain_activation"]["selected_domain"] == "general_science"
    assert "trigger_reasons" in trace["query_repair"]
    assert "static_query_quality_repair" in repairs["trigger_reasons"]
    assert repairs["removed_queries"]
    assert not set(planned["queries"]).intersection(repairs["removed_queries"])
    assert exploration["provider_query_count"] >= 8
    assert "aspect_coverage_distribution" in exploration
    assert "single_acronym_query_count" in exploration
    assert "role_assignment_evidence" in ranking
    assert "lane_aspect_contribution" in ranking
    assert "acronym_disambiguation_penalties" in ranking
