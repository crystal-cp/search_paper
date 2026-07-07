import csv
import json

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.agents.snowball import (
    CitationSnowballAgent,
    parse_seed_file,
    parse_seed_values,
)
from lit_screening.models import (
    DomainAssessment,
    EvidenceRecord,
    Paper,
    RankedPaper,
    RetrievalPath,
    ScoreBreakdown,
    SeedPaper,
    VerificationResult,
)
from lit_screening.pipeline import run_pipeline
from lit_screening.retrieval.base import RetrievalResult


class FakeSemanticScholarSnowballClient:
    provider_name = "semantic_scholar"
    api_key = "fake"

    def __init__(self):
        self.calls = []

    def search(self, query, max_results, from_year=None):
        self.calls.append(("search", query, max_results))
        return RetrievalResult(raw={"data": []}, papers=[seed_paper()])

    def get_paper(self, paper_id):
        self.calls.append(("get_paper", paper_id, 1))
        return RetrievalResult(raw={"paperId": "S2SEED"}, papers=[seed_paper()])

    def get_references(self, paper_id, limit=10):
        self.calls.append(("references", paper_id, limit))
        return RetrievalResult(raw={"data": []}, papers=[expanded_paper("reference")])

    def get_citations(self, paper_id, limit=10):
        self.calls.append(("citations", paper_id, limit))
        return RetrievalResult(raw={"data": []}, papers=[expanded_paper("citation")])

    def get_recommendations(self, paper_id, limit=10):
        self.calls.append(("recommendations", paper_id, limit))
        return RetrievalResult(raw={"data": []}, papers=[expanded_paper("recommendation")])


class PipelineFakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        paper = Paper(
            paper_id="base-paper",
            title="Seed literature screening paper",
            abstract="Human feedback improves literature screening with grounded evidence.",
            year=2024,
            venue="Demo Journal",
            doi="10.1000/base",
            source_provider="fake",
            citation_count=5,
        )
        return RetrievalResult(raw={"data": [{"title": paper.title}]}, papers=[paper])


class FakeSnowballAgent:
    def expand(self, existing_papers, ranked_papers, seed_papers=None, top_n=3):
        seeds = seed_papers or [
            SeedPaper(
                seed_id="10.1000/base",
                seed_type="doi",
                title="Seed literature screening paper",
                doi="10.1000/base",
                note="fake auto seed",
            )
        ]
        paper = Paper(
            paper_id="expanded-paper",
            title="Citation-expanded human feedback screening paper",
            abstract=(
                "Human feedback improves literature screening by refining evidence "
                "verification and ranking."
            ),
            year=2025,
            venue="Demo Journal",
            doi="10.1000/expanded",
            source_provider="semantic_scholar",
            retrieval_provider="semantic_scholar",
            retrieval_stage="snowball_citation",
            retrieval_query=seeds[0].seed_id,
            source_stage="citation",
            seed_paper_id="base-paper",
            seed_title=seeds[0].title or seeds[0].seed_id,
            seed_reason="Found via citation expansion from seed.",
            citation_count=3,
        )
        path = RetrievalPath(
            paper_id=paper.paper_id,
            source_stage="citation",
            seed_paper_id="base-paper",
            seed_title=seeds[0].title or seeds[0].seed_id,
            reason=paper.seed_reason,
        )
        return [paper], [path], seeds


def seed_paper():
    return Paper(
        paper_id="seed-paper",
        title="Known seed paper",
        abstract="A known seed paper about literature screening.",
        year=2023,
        doi="10.1000/seed",
        provider_ids={"semantic_scholar": "S2SEED"},
        source_provider="semantic_scholar",
    )


def expanded_paper(stage):
    return Paper(
        paper_id=f"{stage}-paper",
        title=f"{stage.title()} expanded literature screening paper",
        abstract="This related paper studies evidence-grounded literature screening.",
        year=2024,
        doi=f"10.1000/{stage}",
        provider_ids={"semantic_scholar": f"S2{stage.upper()}"},
        source_provider="semantic_scholar",
    )


def ranked_seed():
    paper = seed_paper()
    evidence = EvidenceRecord(
        paper_id=paper.paper_id,
        title=paper.title,
        claim="Known seed paper about literature screening.",
        evidence_sentence="A known seed paper about literature screening.",
        relevance_reason="seed",
    )
    verification = VerificationResult(
        paper_id=paper.paper_id,
        supported=True,
        confidence=0.95,
        error_type="",
        rationale="strict",
        support_level="strict_support",
    )
    scores = ScoreBreakdown(
        relevance_score=1.0,
        evidence_score=1.0,
        recency_score=1.0,
        quality_score=0.5,
        diversity_score=1.0,
        human_feedback_adjustment=0.0,
        final_score=0.9,
    )
    return RankedPaper(
        rank=1,
        paper=paper,
        evidence=evidence,
        verification=verification,
        scores=scores,
        domain_assessment=DomainAssessment(
            paper_id=paper.paper_id,
            domain_match_score=1.0,
            domain_decision="in_scope",
            off_topic_reason="",
        ),
    )


def test_seed_file_is_parsed(tmp_path):
    seed_path = tmp_path / "seed_papers.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed_id", "seed_type", "title", "doi", "note"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed_id": "10.1000/seed",
                "seed_type": "doi",
                "title": "Known seed paper",
                "doi": "https://doi.org/10.1000/seed",
                "note": "important",
            }
        )

    seeds = parse_seed_file(seed_path)

    assert len(seeds) == 1
    assert seeds[0].seed_type == "doi"
    assert seeds[0].doi == "10.1000/seed"
    assert seeds[0].title == "Known seed paper"


def test_seed_values_are_parsed():
    seeds = parse_seed_values(["https://doi.org/10.1000/seed", "Known seed title"])

    assert seeds[0].seed_type == "doi"
    assert seeds[0].doi == "10.1000/seed"
    assert seeds[1].seed_type == "title"
    assert seeds[1].title == "Known seed title"


def test_snowballing_is_disabled_by_default():
    agent = CitationSnowballAgent(
        semantic_scholar_client=FakeSemanticScholarSnowballClient(),
        enabled=False,
    )

    expanded, paths, seeds = agent.expand(
        existing_papers=[],
        ranked_papers=[ranked_seed()],
        seed_papers=[SeedPaper(seed_id="10.1000/seed", seed_type="doi")],
        top_n=1,
    )

    assert expanded == []
    assert paths == []
    assert len(seeds) == 1


def test_snowballing_runs_with_fake_semantic_scholar_responses():
    client = FakeSemanticScholarSnowballClient()
    agent = CitationSnowballAgent(semantic_scholar_client=client, enabled=True)

    expanded, paths, seeds = agent.expand(
        existing_papers=[],
        ranked_papers=[ranked_seed()],
        seed_papers=[SeedPaper(seed_id="10.1000/seed", seed_type="doi")],
        top_n=2,
    )

    assert len(seeds) == 1
    assert {paper.source_stage for paper in expanded} == {
        "reference",
        "citation",
        "recommendation",
    }
    assert all(path.seed_paper_id == "seed-paper" for path in paths)
    assert all(paper.retrieval_stage.startswith("snowball_") for paper in expanded)
    assert ("references", "S2SEED", 2) in client.calls
    assert ("citations", "S2SEED", 2) in client.calls
    assert ("recommendations", "S2SEED", 2) in client.calls


def test_pipeline_generates_snowball_outputs_when_enabled(tmp_path):
    retriever = RetrieverAgent(clients={"fake": PipelineFakeClient()})
    output_dir = tmp_path / "outputs"

    result = run_pipeline(
        question="human feedback literature screening evidence verification",
        providers=["fake"],
        max_per_query=1,
        output_dir=str(output_dir),
        retriever_agent=retriever,
        seed_papers=["10.1000/base"],
        enable_snowballing=True,
        snowball_top_n=1,
        snowball_agent=FakeSnowballAgent(),
    )

    ranked_csv = (output_dir / "ranked_papers.csv").read_text()
    report = (output_dir / "report.md").read_text()
    retrieval_paths = (output_dir / "retrieval_paths.csv").read_text()
    seed_json = json.loads((output_dir / "seed_papers.json").read_text())

    assert (output_dir / "citation_expansion.csv").exists()
    assert (output_dir / "retrieval_paths.csv").exists()
    assert (output_dir / "seed_papers.json").exists()
    assert result.citation_expansion_papers
    assert result.retrieval_paths
    assert result.retrieval_counts["citation_snowball"] == 1
    assert "expanded-paper" in ranked_csv
    assert "source_stage" in ranked_csv.splitlines()[0]
    assert "seed_title" in ranked_csv.splitlines()[0]
    assert "citation" in retrieval_paths
    assert seed_json[0]["doi"] == "10.1000/base"
    assert "## Seed Paper Expansion" in report
