import csv

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.models import Paper
from lit_screening.pipeline import run_pipeline
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
    assert (output_dir / "ranked_papers_before_feedback.csv").exists()
    assert (output_dir / "ranked_papers_after_feedback.csv").exists()
    assert (output_dir / "ranked_papers.csv").exists()
    assert (output_dir / "evaluation.json").exists()
    assert (output_dir / "agent_trace.json").exists()
    assert (output_dir / "report.md").exists()
    assert result.merged_paper_count == 1
    assert result.duplicate_count == 5
    assert result.agent_trace["planner"]["queries"]
