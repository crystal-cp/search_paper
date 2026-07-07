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


class RecordingMaterialsClient:
    provider_name = "fake"

    def __init__(self):
        self.seen_queries = []

    def search(self, query, max_results, from_year=None):
        self.seen_queries.append(query)
        paper = Paper(
            paper_id="surface-paper",
            title="Surface magnetization in antiferromagnetic materials",
            abstract=(
                "Surface magnetization creates boundary spin signals in "
                "antiferromagnetic materials."
            ),
            year=2024,
            venue="Demo Journal",
            doi="10.1234/surface",
            source_provider="fake",
            citation_count=5,
        )
        return RetrievalResult(raw={"data": [{"title": paper.title}]}, papers=[paper])


def test_pipeline_writes_research_lens_artifacts_without_changing_query_plan(tmp_path):
    client = RecordingMaterialsClient()
    retriever = RetrieverAgent(clients={"fake": client})
    output_dir = tmp_path / "outputs"

    result = pipeline.run_pipeline(
        question=MATERIALS_QUESTION,
        providers=["fake"],
        max_per_query=1,
        output_dir=str(output_dir),
        retriever_agent=retriever,
    )

    concept_map_path = output_dir / "concept_map.json"
    query_families_path = output_dir / "query_families.json"
    concept_map = json.loads(concept_map_path.read_text())
    query_families = json.loads(query_families_path.read_text())
    trace = json.loads((output_dir / "agent_trace.json").read_text())
    provenance = json.loads((output_dir / "query_provenance.json").read_text())

    assert concept_map_path.exists()
    assert query_families_path.exists()
    assert concept_map["domain"] == "materials_magnetism"
    assert query_families["domain"] == "materials_magnetism"
    assert result.concept_map is not None
    assert result.query_family_plan is not None
    assert result.query_plan is not None
    assert result.planned_queries
    assert client.seen_queries
    assert any(query in result.planned_queries for query in client.seen_queries)
    assert any("SPLEEM" in query or "XMCD" in query for query in client.seen_queries)
    assert provenance["enabled"] is True
    assert provenance["applied"] is True
    assert trace["concept_mapper"]["executed"] is True
    assert trace["concept_mapper"]["lens_count"] >= 1
    assert trace["query_family_planner"]["executed"] is True
    assert trace["query_family_planner"]["query_family_count"] >= 1


def test_pipeline_continues_when_concept_mapper_fails(tmp_path, monkeypatch):
    class FailingConceptMapper:
        def map_question(self, *args, **kwargs):
            raise RuntimeError("forced mapper failure")

    monkeypatch.setattr(pipeline, "ConceptMapper", FailingConceptMapper)
    output_dir = tmp_path / "outputs"

    result = pipeline.run_pipeline(
        question=MATERIALS_QUESTION,
        providers=[],
        max_per_query=0,
        output_dir=str(output_dir),
    )
    trace = json.loads((output_dir / "agent_trace.json").read_text())

    assert result.query_plan is not None
    assert result.concept_map is None
    assert result.query_family_plan is None
    assert not (output_dir / "concept_map.json").exists()
    assert not (output_dir / "query_families.json").exists()
    assert trace["concept_mapper"]["skipped"] is True
    assert trace["concept_mapper"]["warning"] == "forced mapper failure"
    assert trace["query_family_planner"]["skipped"] is True
