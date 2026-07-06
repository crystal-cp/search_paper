import csv
import json

import pytest

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.models import Paper
from lit_screening.pipeline import plan_screening_queries, run_pipeline
from lit_screening.retrieval.base import RetrievalResult


class FakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        paper = Paper(
            paper_id="paper-1",
            title="Human feedback improves LLM literature screening",
            abstract=(
                "Human feedback can improve literature screening by helping "
                "LLM agents verify evidence in scientific abstracts."
            ),
            authors=["A. Researcher"],
            year=2024,
            venue="Demo Journal",
            doi="10.1234/demo",
            url="https://example.test/demo",
            source_provider="fake",
            citation_count=5,
        )
        return RetrievalResult(raw={"query": query, "data": [{"title": paper.title}]}, papers=[paper])


class SurfaceMagnetizationFakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        paper = Paper(
            paper_id="surface-paper-1",
            title="Surface magnetization in antiferromagnetic materials",
            abstract=(
                "Surface magnetization is important for boundary spin signals "
                "in antiferromagnetic materials."
            ),
            authors=["A. Researcher"],
            year=2024,
            venue="Demo Journal",
            doi="10.1234/surface",
            url="https://example.test/surface",
            source_provider="fake",
            citation_count=7,
        )
        return RetrievalResult(raw={"query": query, "data": [{"title": paper.title}]}, papers=[paper])


class RecordingFakeClient(SurfaceMagnetizationFakeClient):
    def __init__(self):
        self.seen_queries = []

    def search(self, query, max_results, from_year=None):
        self.seen_queries.append(query)
        return super().search(query, max_results, from_year)


class ExplodingClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        raise TypeError("'NoneType' object is not iterable")


class MixedYearFakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        old_paper = Paper(
            paper_id="old-paper",
            title="An early surface magnetization study",
            abstract="Surface magnetization was discussed in an early materials paper.",
            authors=["A. Early"],
            year=1988,
            venue="Archive Journal",
            doi="10.1234/old",
            url="https://example.test/old",
            source_provider="fake",
            citation_count=20,
        )
        new_paper = Paper(
            paper_id="new-paper",
            title="Recent surface magnetization in antiferromagnetic materials",
            abstract=(
                "Recent work studies surface magnetization in antiferromagnetic "
                "materials and boundary spin signals."
            ),
            authors=["A. Recent"],
            year=2024,
            venue="Demo Journal",
            doi="10.1234/new",
            url="https://example.test/new",
            source_provider="fake",
            citation_count=8,
        )
        return RetrievalResult(
            raw={
                "query": query,
                "data": [
                    {"title": old_paper.title, "year": old_paper.year},
                    {"title": new_paper.title, "year": new_paper.year},
                ],
            },
            papers=[old_paper, new_paper],
        )


def test_pipeline_with_fake_retrievers_and_no_internet(tmp_path):
    feedback_path = tmp_path / "feedback.csv"
    with feedback_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "label", "adjustment", "note"])
        writer.writeheader()
        writer.writerow(
            {
                "paper_id": "paper-1",
                "label": "include",
                "adjustment": "0.1",
                "note": "Good fit.",
            }
        )

    retriever = RetrieverAgent(clients={"fake": FakeClient()})
    result = run_pipeline(
        question="How can human feedback improve LLM literature screening?",
        providers=["fake"],
        max_per_query=1,
        from_year=2020,
        feedback_path=str(feedback_path),
        gold_labels_path=None,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=retriever,
    )

    output_dir = tmp_path / "outputs"
    assert (output_dir / "planned_queries.json").exists()
    assert (output_dir / "raw_fake_results.json").exists()
    assert (output_dir / "merged_papers.csv").exists()
    assert (output_dir / "evidence_table.csv").exists()
    assert (output_dir / "aspect_coverage.csv").exists()
    assert (output_dir / "ranked_papers_before_feedback.csv").exists()
    assert (output_dir / "ranked_papers_after_feedback.csv").exists()
    assert (output_dir / "ranked_papers.csv").exists()
    assert (output_dir / "evaluation.json").exists()
    assert (output_dir / "agent_trace.json").exists()
    assert (output_dir / "run_events.jsonl").exists()
    assert (output_dir / "search_brief.json").exists()
    assert (output_dir / "question_refinement.json").exists()
    assert (output_dir / "result_groups.json").exists()
    assert (output_dir / "prisma_like_flow.json").exists()
    assert (output_dir / "paper_cards.md").exists()
    assert (output_dir / "reading_path.md").exists()
    assert (output_dir / "retrieval_diagnostics.json").exists()
    assert (output_dir / "report.md").exists()
    assert result.merged_paper_count == 1
    assert result.duplicate_count == result.raw_paper_count - result.merged_paper_count
    assert result.search_brief is not None
    assert result.aspect_coverage_records
    assert result.result_groups
    assert "research_intent" in result.agent_trace
    assert "aspect_coverage" in result.agent_trace
    assert result.agent_trace["planner"]["queries"]
    events = (output_dir / "run_events.jsonl").read_text()
    assert "Pipeline started" in events
    assert "Literature screening complete" in events


def test_pipeline_applies_hard_local_from_year_filter(tmp_path):
    retriever = RetrieverAgent(clients={"fake": MixedYearFakeClient()})

    result = run_pipeline(
        question="surface magnetization",
        providers=["fake"],
        max_per_query=2,
        from_year=2020,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=retriever,
        planned_queries_override=["surface magnetization"],
    )

    output_dir = tmp_path / "outputs"
    diagnostics = json.loads((output_dir / "retrieval_diagnostics.json").read_text())
    evaluation = json.loads((output_dir / "evaluation.json").read_text())
    merged_csv = (output_dir / "merged_papers.csv").read_text()

    assert result.raw_paper_count == 2
    assert result.merged_paper_count == 1
    assert all((paper.year or 0) >= 2020 for paper in result.merged_papers)
    assert result.ranked_final[0].paper.paper_id == "new-paper"
    assert "old-paper" not in merged_csv
    assert diagnostics["year_filter"]["enabled"] is True
    assert diagnostics["year_filter"]["excluded_before_year_count"] == 1
    assert diagnostics["year_filter"]["kept_count"] == 1
    assert evaluation["year_filter"]["excluded_before_year_count"] == 1


def test_pipeline_ranks_imported_csv_without_provider_calls(tmp_path):
    input_path = tmp_path / "library.csv"
    input_path.write_text(
        (
            "title,abstract,authors,year,venue,doi,url,citation_count\n"
            "Imported surface magnetization paper,"
            "Surface magnetization controls boundary spin signals.,"
            "Ada Researcher,2024,Demo Journal,10.2468/imported,https://example.test/imported,9\n"
        ),
        encoding="utf-8",
    )

    result = run_pipeline(
        question="surface magnetization boundary spin signals",
        providers=[],
        max_per_query=0,
        from_year=2020,
        output_dir=str(tmp_path / "outputs"),
        input_file=str(input_path),
        input_format="csv",
    )

    output_dir = tmp_path / "outputs"
    diagnostics = json.loads((output_dir / "retrieval_diagnostics.json").read_text())
    evaluation = json.loads((output_dir / "evaluation.json").read_text())

    assert result.raw_paper_count == 1
    assert result.merged_paper_count == 1
    assert result.ranked_final[0].paper.doi == "10.2468/imported"
    assert (output_dir / "imported_papers.csv").exists()
    assert (output_dir / "import_diagnostics.json").exists()
    assert diagnostics["imported_library"]["paper_count"] == 1
    assert evaluation["retrieval_counts_by_provider"]["imported"] == 1


def test_pipeline_uses_english_planning_question_for_chinese_input(tmp_path):
    retriever = RetrieverAgent(clients={"fake": SurfaceMagnetizationFakeClient()})

    result = run_pipeline(
        question="表面磁化的重要性",
        providers=["fake"],
        max_per_query=1,
        from_year=2020,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=retriever,
    )

    planned = json.loads((tmp_path / "outputs" / "planned_queries.json").read_text())

    assert result.planning_question == "surface magnetization importance"
    assert result.planned_queries[0] == "surface magnetization importance"
    assert planned["planner_metadata"]["translation_mode"] == "rule_glossary"
    assert result.evidence_records[0].keyword_overlap > 0
    assert result.agent_trace["planning_question"] == "surface magnetization importance"


def test_plan_screening_queries_runs_without_retrieval():
    plan = plan_screening_queries(
        question="表面磁化的重要性",
        llm_backend="none",
        planner_mode="rule",
    )

    assert plan["queries"][0] == "surface magnetization importance"
    assert plan["planner_metadata"]["translation_mode"] == "rule_glossary"


def test_pipeline_uses_user_confirmed_query_override(tmp_path):
    client = RecordingFakeClient()
    retriever = RetrieverAgent(clients={"fake": client})
    metadata = {
        "planner_mode": "rule",
        "llm_used": False,
        "invalid_llm_output": False,
        "llm_error_type": "",
        "original_question": "表面磁化的重要性",
        "question_language": "zh",
        "translation_used": True,
        "translation_mode": "rule_glossary",
        "translated_question": "surface magnetization importance",
        "planning_question": "surface magnetization importance",
        "translation_warning": "rule_glossary_translation_is_approximate",
    }

    result = run_pipeline(
        question="表面磁化的重要性",
        providers=["fake"],
        max_per_query=1,
        from_year=2020,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=retriever,
        planned_queries_override=["surface magnetization antiferromagnetic boundary"],
        planner_metadata_override=metadata,
    )

    assert client.seen_queries == ["surface magnetization antiferromagnetic boundary"]
    assert "surface magnetization antiferromagnetic boundary" in result.planned_queries
    assert result.agent_trace["planner"]["metadata"]["query_source"] == "user_confirmed"


def test_pipeline_emits_progress_events(tmp_path):
    events = []
    retriever = RetrieverAgent(clients={"fake": SurfaceMagnetizationFakeClient()})

    run_pipeline(
        question="表面磁化的重要性",
        providers=["fake"],
        max_per_query=1,
        from_year=2020,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=retriever,
        progress_callback=lambda stage, message, details: events.append(
            (stage, message, details)
        ),
    )

    stages = [event[0] for event in events]
    assert "retrieval" in stages
    assert "verification" in stages
    assert stages[-1] == "complete"
    assert any(event[2].get("provider") == "fake" for event in events)


def test_retrieval_diagnostics_json_is_produced(tmp_path):
    retriever = RetrieverAgent(clients={"fake": SurfaceMagnetizationFakeClient()})

    run_pipeline(
        question="surface magnetization",
        providers=["fake"],
        max_per_query=1,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=retriever,
    )

    diagnostics = json.loads(
        (tmp_path / "outputs" / "retrieval_diagnostics.json").read_text()
    )

    assert diagnostics["question"] == "surface magnetization"
    assert diagnostics["merged_count"] == 1
    assert diagnostics["raw_count_per_query"]["fake"]
    assert "provider_errors" in diagnostics
    assert diagnostics["top_10_score_breakdown"]


def test_retriever_writes_raw_error_when_provider_raises(tmp_path):
    retriever = RetrieverAgent(clients={"fake": ExplodingClient()})

    with pytest.raises(TypeError):
        retriever.retrieve(
            queries={"fake": ["surface magnetization"]},
            providers=["fake"],
            max_per_query=1,
            from_year=None,
            output_dir=tmp_path,
        )

    raw = json.loads((tmp_path / "raw_fake_results.json").read_text())
    assert raw[0]["response"]["error"] == "TypeError"
    assert "NoneType" in raw[0]["response"]["error_message"]


def test_fake_pipeline_produces_sensemaking_outputs(tmp_path):
    retriever = RetrieverAgent(clients={"fake": SurfaceMagnetizationFakeClient()})

    result = run_pipeline(
        question="Give me an overview of the significance of surface magnetization",
        providers=["fake"],
        max_per_query=1,
        output_dir=str(tmp_path / "outputs"),
        retriever_agent=retriever,
    )

    output_dir = tmp_path / "outputs"
    assert result.search_brief.search_intent == "overview"
    assert result.question_refinement["subquestions"]
    assert (output_dir / "aspect_coverage.csv").read_text()
    assert "Surface magnetization" in (output_dir / "paper_cards.md").read_text()
    assert "Recommended Reading Path" in (output_dir / "reading_path.md").read_text()
    assert json.loads((output_dir / "prisma_like_flow.json").read_text())[
        "records_screened"
    ] == 1
