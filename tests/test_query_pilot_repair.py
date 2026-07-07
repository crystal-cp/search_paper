import json

from lit_screening.agents.ambiguity_detector import AmbiguityDetectorAgent
from lit_screening.agents.query_pilot import QueryPilotAgent
from lit_screening.agents.query_repair import QueryRepairAgent
from lit_screening.agents.research_intent import ResearchIntentAgent
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.agents.search_contract import SearchContractAgent
from lit_screening.models import Paper, QueryPlan
from lit_screening.pipeline import run_pipeline
from lit_screening.retrieval.base import RetrievalResult


class PilotFakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        if "patient" in query.lower() or "screening" in query.lower():
            paper = Paper(
                paper_id="patient-screening",
                title="Patient screening and biomarker screening in clinical workflows",
                abstract="Drug screening and patient screening are evaluated in medicine.",
                year=2024,
                venue="Medical Journal",
                fields_of_study=["Medicine"],
                source_provider="fake",
            )
        else:
            paper = Paper(
                paper_id="literature-screening",
                title="LLM agents for scientific literature screening",
                abstract="LLM agents support scientific literature screening.",
                year=2024,
                venue="ACL",
                fields_of_study=["Computer Science"],
                source_provider="fake",
            )
        return RetrievalResult(
            raw={"data": [{"title": paper.title}]},
            papers=[paper],
        )


def build_ai_contract(question: str):
    brief = ResearchIntentAgent().analyze(question)
    ambiguity = AmbiguityDetectorAgent().analyze(question)
    contract = SearchContractAgent().build(
        question,
        search_brief=brief,
        ambiguity_analysis=ambiguity,
    )
    return contract, ambiguity


def test_query_pilot_agent_runs_with_fake_retriever():
    question = "How can LLM agents improve literature screening?"
    contract, _ = build_ai_contract(question)
    query_plan = QueryPlan(
        original_question=question,
        must_terms=["literature screening", "LLM agents"],
        exclude_terms=["patient screening"],
        openalex_queries=["screening"],
        semantic_scholar_queries=[],
    )
    retriever = RetrieverAgent(clients={"fake": PilotFakeClient()})

    diagnostics = QueryPilotAgent(retriever_agent=retriever).run(
        queries={"fake": ["screening"]},
        providers=["fake"],
        search_contract=contract,
        query_plan=query_plan,
        max_per_query=1,
    )

    result = diagnostics["results"][0]
    assert result["provider"] == "fake"
    assert result["pilot_raw_count"] == 1
    assert result["off_topic_rate_estimate"] == 1.0
    assert result["recommendation"] in {"repair", "drop"}
    assert "healthcare_screening_drift" in result["detected_drift"]


def test_query_repair_adds_excludes_for_healthcare_drift():
    question = "How can LLM agents improve literature screening?"
    contract, ambiguity = build_ai_contract(question)
    query_plan = QueryPlan(
        original_question=question,
        openalex_queries=["screening"],
        semantic_scholar_queries=[],
    )
    pilot = {
        "results": [
            {
                "provider": "openalex",
                "query": "screening",
                "retrieval_stage": "openalex_keyword",
                "recommendation": "repair",
                "off_topic_rate_estimate": 0.75,
                "detected_drift": ["healthcare_screening_drift"],
            }
        ]
    }

    suggestions = QueryRepairAgent().suggest(query_plan, contract, ambiguity, pilot)
    repaired = suggestions["suggestions"][0]["repaired_query"]

    assert "literature screening" in repaired
    assert 'NOT "patient screening"' in repaired
    assert 'NOT "drug screening"' in repaired
    assert 'NOT "biomarker screening"' in repaired


def test_query_repair_makes_ambiguous_screening_more_precise():
    question = "How can LLM agents improve literature screening?"
    contract, ambiguity = build_ai_contract(question)
    query_plan = QueryPlan(
        original_question=question,
        openalex_queries=[],
        semantic_scholar_queries=["screening"],
    )
    pilot = {
        "results": [
            {
                "provider": "semantic_scholar",
                "query": "screening",
                "retrieval_stage": "semantic_scholar",
                "recommendation": "repair",
                "off_topic_rate_estimate": 0.5,
                "detected_drift": ["healthcare_screening_drift"],
            }
        ]
    }

    suggestions = QueryRepairAgent().suggest(query_plan, contract, ambiguity, pilot)
    repaired = suggestions["suggestions"][0]["repaired_query"]

    assert "literature screening" in repaired
    assert '-"patient screening"' in repaired


def test_pipeline_writes_pilot_and_repair_outputs(tmp_path):
    retriever = RetrieverAgent(clients={"fake": PilotFakeClient()})

    run_pipeline(
        question="How can LLM agents improve literature screening?",
        providers=["fake"],
        max_per_query=1,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=retriever,
        planned_queries_override=["screening"],
        pilot_search=True,
        pilot_max_per_query=1,
    )

    output_dir = tmp_path / "outputs"
    pilot = json.loads((output_dir / "query_pilot_diagnostics.json").read_text())
    repairs = json.loads((output_dir / "query_repair_suggestions.json").read_text())
    report = (output_dir / "report.md").read_text()

    assert pilot["enabled"] is True
    assert pilot["results"]
    assert repairs["enabled"] is True
    assert "suggestions" in repairs
    assert "## Query Pilot Diagnostics" in report
    assert "## Query Repairs Applied" in report
