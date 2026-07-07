import csv
import json

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.models import Paper
from lit_screening.pipeline import run_pipeline
from lit_screening.retrieval.base import RetrievalResult


SURFACE_SEED = (
    "Surface Magnetization in Antiferromagnets: Classification, Example Materials, "
    "and Relation to Magnetoelectric Responses"
)
LOCAL_ME_SEED = "Local Magnetoelectric Effects as Predictors of Surface Magnetic Order"


class EmptyOpenAlexClient:
    provider_name = "openalex"

    def search(
        self,
        query,
        max_results,
        from_year=None,
        sort_mode="relevance",
        search_mode="keyword",
    ):
        return RetrievalResult(raw={"results": []}, papers=[])


class SeedSemanticScholarClient:
    provider_name = "semantic_scholar"
    api_key = "fake"

    def search(self, query, max_results, from_year=None):
        if query == SURFACE_SEED:
            paper = Paper(
                paper_id="surface-seed",
                title=SURFACE_SEED,
                abstract=(
                    "Surface magnetization in antiferromagnets can be classified "
                    "and related to magnetoelectric responses."
                ),
                authors=["Nicola A. Spaldin"],
                year=2024,
                venue="Physical Review Research",
                doi="10.1103/fake-surface",
                source_provider="semantic_scholar",
                provider_ids={"semantic_scholar": "S2SURFACE"},
                citation_count=5,
            )
            return RetrievalResult(raw={"data": [{"title": SURFACE_SEED}]}, papers=[paper])
        return RetrievalResult(raw={"data": []}, papers=[])

    def get_paper(self, paper_id):
        return RetrievalResult(raw={"data": []}, papers=[])


class LocalMagnetoelectricCrossrefClient:
    provider_name = "crossref"

    def works(self, query, rows=20):
        if query == LOCAL_ME_SEED:
            paper = Paper(
                paper_id="local-me-seed",
                title=LOCAL_ME_SEED,
                abstract="",
                authors=["Nicola A. Spaldin"],
                year=2025,
                venue="Physical Review Letters",
                doi="10.1103/7brd-lynv",
                url="https://doi.org/10.1103/7brd-lynv",
                source_provider="crossref",
                provider_ids={"crossref_doi": "10.1103/7brd-lynv"},
            )
            return RetrievalResult(raw={"message": {"items": [{"title": [LOCAL_ME_SEED]}]}}, papers=[paper])
        return RetrievalResult(raw={"message": {"items": []}}, papers=[])

    def resolve_doi(self, doi):
        return RetrievalResult(raw={"message": {}}, papers=[])


class EmptyArxivClient:
    provider_name = "arxiv"

    def search(self, query, max_results=20):
        return RetrievalResult(raw={"entries": []}, papers=[])


def test_spaldin_seed_mode_keeps_manual_seeds_as_anchors(tmp_path):
    retriever = RetrieverAgent(
        clients={
            "openalex": EmptyOpenAlexClient(),
            "semantic_scholar": SeedSemanticScholarClient(),
            "crossref": LocalMagnetoelectricCrossrefClient(),
            "arxiv": EmptyArxivClient(),
        }
    )
    output_dir = tmp_path / "outputs"

    result = run_pipeline(
        question=(
            "有没有和 Nicola A. Spaldin 的 "
            f"{SURFACE_SEED} 和 {LOCAL_ME_SEED} "
            "相关的有关探测表面磁化和自旋极化的重要性的文章"
        ),
        providers=["openalex", "semantic_scholar"],
        max_per_query=0,
        output_dir=str(output_dir),
        retriever_agent=retriever,
        seed_papers=[SURFACE_SEED, LOCAL_ME_SEED],
        enable_snowballing=False,
    )

    by_title = {item.paper.title: item for item in result.ranked_final}
    assert SURFACE_SEED in by_title
    assert LOCAL_ME_SEED in by_title

    for title in [SURFACE_SEED, LOCAL_ME_SEED]:
        item = by_title[title]
        assert item.paper.source_stage in {"seed_exact", "manual_seed"}
        assert item.paper.seed_title
        assert item.paper.seed_reason
        assert item.paper.seed_relation == "self"
        assert item.paper.seed_confidence > 0
        assert item.screening_decision is not None
        assert item.screening_decision.decision == "include"
        assert item.screening_decision.reading_priority == "must_read"
        assert item.screening_decision.domain_decision != "out_of_scope"

    rows = list(csv.DictReader((output_dir / "ranked_papers.csv").open(encoding="utf-8")))
    exported_by_title = {row["title"]: row for row in rows}
    for title in [SURFACE_SEED, LOCAL_ME_SEED]:
        row = exported_by_title[title]
        assert row["source_stage"] in {"seed_exact", "manual_seed"}
        assert row["seed_title"]
        assert row["seed_reason"]
        assert row["seed_relation"] == "self"
        assert row["seed_confidence"]
        assert row["decision"] == "include"
        assert row["reading_priority"] == "must_read"
        assert row["domain_decision"] != "out_of_scope"

    resolution = json.loads((output_dir / "seed_resolution_report.json").read_text())
    expansion = json.loads((output_dir / "seed_expansion_report.json").read_text())
    assert resolution["seed_input_count"] == 2
    assert resolution["seed_resolved_count"] == 2
    assert resolution["seed_unresolved_count"] == 0
    assert expansion["seed_input_count"] == 2
    assert expansion["seed_resolved_count"] == 2
    assert expansion["references_retrieved"] == 0
