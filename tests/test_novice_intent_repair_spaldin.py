import csv
import json

from lit_screening.agents.intent_repair import NoviceIntentInterpreter
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.agents.seed_extraction import SeedExtractionAgent
from lit_screening.pipeline import plan_screening_queries, run_pipeline
from lit_screening.models import Paper
from lit_screening.retrieval.base import RetrievalResult


SPALDIN_QUESTION = (
    "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets: "
    "Classification, Example Materials, and Relation to Magnetoelectric Responses "
    "和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order "
    "相关的有关探测表面磁化和自旋极化的重要性的文章"
)
SURFACE_SEED = (
    "Surface Magnetization in Antiferromagnets: Classification, Example Materials, "
    "and Relation to Magnetoelectric Responses"
)
LOCAL_ME_SEED = "Local Magnetoelectric Effects as Predictors of Surface Magnetic Order"


def test_novice_intent_repair_spaldin_adds_expert_concepts():
    seed_hints = SeedExtractionAgent().extract(SPALDIN_QUESTION)
    intent = NoviceIntentInterpreter().repair(SPALDIN_QUESTION, seed_hints=seed_hints)
    joined = " ".join(
        [
            intent.expert_rewritten_question,
            *intent.target_objects,
            *intent.mechanisms,
            *intent.materials,
            *intent.methods,
        ]
    )

    assert intent.expert_rewritten_question.endswith(".")
    assert "Find theoretical, experimental, and methodological papers" in intent.expert_rewritten_question
    assert "boundary magnetization" in joined
    assert "local magnetoelectric response" in joined
    assert "Cr2O3" in joined or "chromia" in joined
    assert "SPLEEM" in joined
    assert "XMCD-PEEM" in joined
    assert "NV magnetometry" in joined
    assert set(intent.ignored_or_downweighted_terms) & {
        "importance",
        "background",
        "significance",
        "review",
    }


def test_repaired_intent_drives_short_provider_queries():
    payload = plan_screening_queries(SPALDIN_QUESTION)
    plan = payload["query_plan"]
    openalex_queries = plan.openalex_queries
    all_queries = " ".join([*openalex_queries, *plan.semantic_scholar_queries])

    assert plan.expert_rewritten_question
    assert "boundary magnetization" in all_queries
    assert "local magnetoelectric" in all_queries
    assert "Cr2O3" in all_queries or "chromia" in all_queries
    assert "SPLEEM" in all_queries
    assert "NV magnetometry" in all_queries
    assert all(SPALDIN_QUESTION not in query for query in openalex_queries)
    assert all(len(query) < 180 for query in openalex_queries)
    assert "surface magnetization magnetization antiferromagnetism" not in all_queries.lower()


class SpaldinOpenAlexClient:
    provider_name = "openalex"

    def __init__(self):
        self.seen_queries = []

    def search(
        self,
        query,
        max_results,
        from_year=None,
        sort_mode="relevance",
        search_mode="keyword",
    ):
        self.seen_queries.append(query)
        if query == LOCAL_ME_SEED or "Local Magnetoelectric Effects" in query:
            return RetrievalResult(
                raw={"results": [{"title": LOCAL_ME_SEED}]},
                papers=[
                    Paper(
                        paper_id="local-me-seed",
                        title=LOCAL_ME_SEED,
                        abstract=(
                            "Local magnetoelectric response and magnetic multipoles "
                            "predict surface magnetic order in antiferromagnets."
                        ),
                        authors=["Nicola A. Spaldin"],
                        year=2025,
                        venue="Physical Review Letters",
                        doi="10.1103/7brd-lynv",
                        source_provider="openalex",
                        citation_count=3,
                    )
                ],
            )
        if "SPLEEM" in query or "XMCD" in query or "surface magnetization domains" in query:
            return RetrievalResult(
                raw={"results": [{"title": "Imaging and Control of Surface Magnetization Domains in a Magnetoelectric Antiferromagnet"}]},
                papers=[
                    Paper(
                        paper_id="wu-2011",
                        title="Imaging and Control of Surface Magnetization Domains in a Magnetoelectric Antiferromagnet",
                        abstract=(
                            "We image surface magnetization domains in the "
                            "magnetoelectric antiferromagnet Cr2O3 using surface-sensitive microscopy."
                        ),
                        authors=["Wu"],
                        year=2011,
                        venue="Physical Review Letters",
                        doi="10.1103/mock-wu",
                        source_provider="openalex",
                        citation_count=120,
                    )
                ],
            )
        return RetrievalResult(raw={"results": []}, papers=[])


class SpaldinSemanticScholarClient:
    provider_name = "semantic_scholar"
    api_key = "fake"

    def __init__(self):
        self.seen_queries = []

    def search(self, query, max_results, from_year=None):
        self.seen_queries.append(query)
        if query == SURFACE_SEED or "Surface Magnetization in Antiferromagnets" in query:
            return RetrievalResult(
                raw={"data": [{"title": SURFACE_SEED}]},
                papers=[
                    Paper(
                        paper_id="surface-seed",
                        title=SURFACE_SEED,
                        abstract=(
                            "Surface magnetization in antiferromagnets is classified "
                            "and related to magnetoelectric responses."
                        ),
                        authors=["Nicola A. Spaldin"],
                        year=2024,
                        venue="Physical Review X",
                        doi="10.1103/physrevx.14.021033",
                        source_provider="semantic_scholar",
                        provider_ids={"semantic_scholar": "S2SURFACE"},
                        citation_count=5,
                    )
                ],
            )
        return RetrievalResult(raw={"data": []}, papers=[])

    def get_paper(self, paper_id):
        return RetrievalResult(raw={"data": []}, papers=[])


def test_spaldin_pipeline_repaired_intent_small_mock_run(tmp_path):
    openalex = SpaldinOpenAlexClient()
    semantic = SpaldinSemanticScholarClient()
    output_dir = tmp_path / "outputs"

    result = run_pipeline(
        question=SPALDIN_QUESTION,
        providers=["openalex", "semantic_scholar"],
        max_per_query=1,
        output_dir=str(output_dir),
        seed_papers=[SURFACE_SEED, LOCAL_ME_SEED],
        retriever_agent=RetrieverAgent(
            clients={
                "openalex": openalex,
                "semantic_scholar": semantic,
            }
        ),
    )

    rows = list(csv.DictReader((output_dir / "ranked_papers.csv").open(encoding="utf-8")))
    by_title = {row["title"]: row for row in rows}
    assert by_title[SURFACE_SEED]["rank"] == "1"
    assert by_title[LOCAL_ME_SEED]["rank"] == "2"
    assert by_title[SURFACE_SEED]["decision"] == "include"
    assert by_title[LOCAL_ME_SEED]["reading_priority"] == "must_read"

    wu_row = by_title["Imaging and Control of Surface Magnetization Domains in a Magnetoelectric Antiferromagnet"]
    assert wu_row["domain_decision"] != "out_of_scope"

    sent_queries = [*openalex.seen_queries, *semantic.seen_queries]
    assert any("SPLEEM" in query or "XMCD" in query for query in sent_queries)

    provenance = json.loads((output_dir / "query_provenance.json").read_text())
    family_names = {
        record["family_name"]
        for record in provenance["records"]
        if record["source"] == "query_family"
    }
    assert "direct_surface_detection" in family_names
    assert "nanoscale_readout" in family_names
    assert "local_magnetoelectric_predictor" in family_names

    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "How the system corrected the user’s question" in report
    assert "Expert rewritten question" in report
    assert result.expert_research_intent is not None
