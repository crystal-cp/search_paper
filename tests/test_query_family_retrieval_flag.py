import json

import lit_screening.pipeline as pipeline
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.models import Paper
from lit_screening.retrieval.base import RetrievalResult


MATERIALS_QUESTION = (
    "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets: "
    "Classification, Example Materials, and Relation to Magnetoelectric Responses "
    "和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order "
    "相关的有关探测表面磁化和自旋极化的重要性的文章"
)


class RecordingClient:
    provider_name = "fake"

    def __init__(self):
        self.seen_queries = []

    def search(self, query, max_results, from_year=None):
        self.seen_queries.append(query)
        paper_index = len(self.seen_queries)
        paper = Paper(
            paper_id=f"paper-{paper_index}",
            title=f"Surface magnetization record {paper_index}",
            abstract=(
                "Surface magnetization creates boundary spin signals in "
                "antiferromagnetic materials."
            ),
            year=2024,
            venue="Demo Journal",
            doi=f"10.1234/query-family-{paper_index}",
            source_provider="fake",
            citation_count=5,
        )
        return RetrievalResult(
            raw={"data": [{"title": paper.title}]},
            papers=[paper],
        )


def test_query_families_disabled_keeps_retrieval_queries_unchanged(tmp_path):
    client = RecordingClient()
    output_dir = tmp_path / "outputs"

    result = pipeline.run_pipeline(
        question=MATERIALS_QUESTION,
        providers=["fake"],
        max_per_query=1,
        output_dir=str(output_dir),
        planned_queries_override=["surface magnetization importance"],
        retriever_agent=RetrieverAgent(clients={"fake": client}),
    )
    provenance = json.loads((output_dir / "query_provenance.json").read_text())

    assert client.seen_queries == ["surface magnetization importance"]
    assert "surface magnetization importance" in result.planned_queries
    assert provenance["enabled"] is False
    assert provenance["applied"] is False
    assert provenance["family_query_count"] == 0
    assert provenance["provider_query_counts"]["fake"] == len(client.seen_queries)


def test_query_families_enabled_adds_family_queries_and_provenance(tmp_path):
    client = RecordingClient()
    output_dir = tmp_path / "outputs"

    result = pipeline.run_pipeline(
        question=MATERIALS_QUESTION,
        providers=["fake"],
        max_per_query=1,
        output_dir=str(output_dir),
        planned_queries_override=["surface magnetization importance"],
        use_query_families=True,
        retriever_agent=RetrieverAgent(clients={"fake": client}),
    )
    provenance = json.loads((output_dir / "query_provenance.json").read_text())
    family_records = [
        record
        for record in provenance["records"]
        if record["source"] == "query_family"
    ]

    assert "surface magnetization importance" in result.planned_queries
    assert len(client.seen_queries) > 1
    assert any("SPLEEM" in query or "XMCD" in query for query in client.seen_queries)
    assert provenance["enabled"] is True
    assert provenance["applied"] is True
    assert provenance["family_query_count"] == len(family_records)
    assert provenance["provider_query_counts"]["fake"] == len(client.seen_queries)
    assert any(record["family_name"] for record in family_records)
    assert any(record["lens_name"] for record in family_records)
    assert any(
        paper.raw.get("matched_query_family")
        for paper in result.merged_papers
        if paper.raw.get("matched_query_source") == "query_family"
    )
