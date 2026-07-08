from pathlib import Path

from lit_screening.agents.ambiguity_detector import AmbiguityDetectorAgent
from lit_screening.agents.paper_role_classifier import PaperRoleClassifier
from lit_screening.agents.research_intent import ResearchIntentAgent
from lit_screening.agents.search_contract import SearchContractAgent
from lit_screening.decision_artifacts import build_research_gap_matrix
from lit_screening.models import AspectCoverageRecord, Paper
from lit_screening.pipeline import (
    _queries_by_provider,
    build_provider_status,
    build_query_provenance,
    build_research_lens_artifacts,
    plan_screening_queries,
)
from lit_screening.utils import ensure_dir


def _contract(question: str):
    return SearchContractAgent().build(
        question,
        search_brief=ResearchIntentAgent().analyze(question),
        ambiguity_analysis=AmbiguityDetectorAgent().analyze(question),
    )


def _final_query_text(question: str, tmp_path: Path) -> str:
    payload = plan_screening_queries(question)
    concept_map, family_plan, _trace = build_research_lens_artifacts(
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
    final_queries, _provenance = build_query_provenance(
        providers=providers,
        queries_by_provider=queries_by_provider,
        query_family_plan=family_plan,
        use_query_families=True,
        search_contract=payload["search_contract"],
        concept_map=concept_map,
    )
    return "\n".join(
        query
        for provider_queries in final_queries.values()
        for query in provider_queries
    ).lower()


def test_in_situ_ex_situ_detected_from_chinese_query():
    contract = _contract(
        "我想了解锂离子电池中 SEI 界面，以及原位/非原位表征方法。"
    )
    frame = contract.generic_intent_frame

    assert frame.in_situ_or_operando_need is True
    assert frame.ex_situ_need is True
    assert "in situ characterization" in frame.method_scope
    assert "ex situ characterization" in frame.method_scope


def test_ai_query_no_material_template_terms(tmp_path):
    query_text = _final_query_text(
        "我想找 LLM 用于系统综述文献筛选的人机协作论文，重点关注人工反馈、证据验证、召回率和筛选准确率。",
        tmp_path,
    )

    assert "experimental characterization" not in query_text
    assert "representative materials" not in query_text
    assert "pfm" not in query_text
    assert "oer" not in query_text
    assert "sei" not in query_text


def test_mof_queries_split_factors(tmp_path):
    query_text = _final_query_text(
        "我想了解 MOF 材料用于 CO2 捕集时孔径、官能团和水稳定性对吸附性能的影响，想找理论机制、实验表征、典型材料和应用限制相关论文。",
        tmp_path,
    )
    lines = [line for line in query_text.splitlines() if line.strip()]

    assert any("pore size" in line for line in lines)
    assert any("functional groups" in line for line in lines)
    assert any("water stability" in line for line in lines)
    assert any("adsorption performance" in line for line in lines)
    assert not all("water stability" in line for line in lines)


def test_thin_film_no_ald_bias(tmp_path):
    query_text = _final_query_text(
        "我想找关于薄膜沉积方法比较的综述，比如 ALD、PLD、sputtering 和 CVD 的优缺点。",
        tmp_path,
    )

    assert "atomic layer deposition" in query_text or "ald" in query_text
    assert "pulsed laser deposition" in query_text or "pld" in query_text
    assert "sputtering" in query_text
    assert "chemical vapor deposition" in query_text or "cvd" in query_text
    assert "comparison" in query_text or "advantages disadvantages" in query_text


def test_paper_role_no_magnetism_roles_in_oer():
    paper = Paper(
        paper_id="oer",
        title="Surface spin state controls oxygen evolution reaction activity",
        abstract="The OER activity of oxide catalysts depends on spin state and electronic structure.",
    )
    record = PaperRoleClassifier().classify(
        paper,
        query_provenance={"domain": "general_science"},
    )

    assert "surface_probe_method" not in record.roles
    assert "nanoscale_readout" not in record.roles
    assert "local_ME" not in record.roles
    assert record.primary_role in {"theory_mechanism", "application_performance", "unclassified"}


def test_research_gap_no_ai_template_in_material_runs():
    contract = _contract(
        "我想了解锂离子电池中 SEI 界面的原位表征、组成结构演化和失效机制。"
    )
    gaps = build_research_gap_matrix(
        ranked_papers=[],
        aspect_coverage_records=[
            AspectCoverageRecord(
                paper_id="p1",
                title="missing",
                missing_aspects=["in_situ_operando", "failure_limitation"],
                aspect_coverage_score=0.0,
            )
        ],
        search_contract=contract,
    )
    text = " ".join(str(value) for row in gaps for value in row.values()).lower()

    assert "evidence verification" not in text
    assert "span-grounded" not in text
    assert "evaluation protocol" not in text
    assert "in situ" in text or "operando" in text


def test_provider_status_reports_semantic_scholar_429():
    status = build_provider_status(
        providers=["openalex", "semantic_scholar"],
        queries_by_provider={
            "openalex": ["q1"],
            "semantic_scholar": ["q1", "q2"],
        },
        raw_by_provider={
            "openalex": [
                {"query": "q1", "paper_count": 2, "response": {"results": [{}, {}]}},
            ],
            "semantic_scholar": [
                {
                    "query": "q1",
                    "paper_count": 0,
                    "response": {
                        "error": "HTTPError",
                        "status_code": 429,
                        "error_message": "rate limit",
                    },
                }
            ],
        },
        retrieval_counts={"openalex": 2, "semantic_scholar": 0},
    )

    assert status["openalex"]["status"] == "success"
    assert status["semantic_scholar"]["rate_limited"] is True
    assert status["semantic_scholar"]["stopped_after_query_count"] == 1
