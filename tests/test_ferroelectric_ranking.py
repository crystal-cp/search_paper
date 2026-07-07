import json

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.models import Paper
from lit_screening.pipeline import run_pipeline
from lit_screening.retrieval.base import RetrievalResult


FERROELECTRIC_QUESTION = (
    "我想了解铁电薄膜表面极化为什么重要，以及有哪些实验方法可以直接探测或表征它。"
    "最好能帮我找到理论背景、实验探测方法、典型材料案例、器件应用，"
    "以及表面/界面屏蔽效应相关的论文。"
)


class FerroelectricRankingClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        papers = [
            Paper(
                paper_id="surface-screening",
                title="Surface-screening mechanisms in ferroelectric thin films",
                abstract=(
                    "Interface screening and screening charge control the depolarization "
                    "field in ferroelectric thin films."
                ),
                year=2024,
                venue="Physical Review B",
                citation_count=20,
                source_provider="fake",
            ),
            Paper(
                paper_id="physics-oxides",
                title="Physics of thin-film ferroelectric oxides",
                abstract=(
                    "This review explains ferroelectric thin films, surface charge, "
                    "depolarization field, and polarization switching."
                ),
                year=2007,
                venue="Reviews of Modern Physics",
                citation_count=500,
                source_provider="fake",
            ),
            Paper(
                paper_id="ultrathin",
                title="Ferroelectricity in ultrathin perovskite films",
                abstract=(
                    "Ferroelectric thin films show polarization switching, surface charge, "
                    "and depolarization effects at small thickness."
                ),
                year=2004,
                venue="Nature",
                citation_count=300,
                source_provider="fake",
            ),
            Paper(
                paper_id="batio3-imprint",
                title="Imprint Control of BaTiO3 Thin Films by Interface Screening",
                abstract=(
                    "BaTiO3 ferroelectric thin film polarization is controlled by "
                    "interface screening, imprint, and surface charge."
                ),
                year=2020,
                venue="Applied Physics Letters",
                citation_count=60,
                source_provider="fake",
            ),
            Paper(
                paper_id="bulk-surface-charge",
                title="Electric polarization as a bulk quantity and its relation to surface charge",
                abstract=(
                    "In ferroelectric crystals, electric polarization relates to "
                    "surface charge and the modern theory of polarization."
                ),
                year=1993,
                venue="Physical Review B",
                citation_count=800,
                source_provider="fake",
            ),
            Paper(
                paper_id="interface-proximity",
                title="In-situ monitoring of interface proximity effects in ultrathin ferroelectrics",
                abstract=(
                    "Second harmonic generation monitors ferroelectric interface "
                    "polarization and screening effects in ultrathin films."
                ),
                year=2023,
                venue="Nature Communications",
                citation_count=30,
                source_provider="fake",
            ),
            Paper(
                paper_id="generic-thin-film",
                title="Thin Films - Deposition Methods and Applications",
                abstract="This chapter discusses generic thin film deposition methods.",
                year=2022,
                venue="Engineering Handbook",
                citation_count=10,
                source_provider="fake",
            ),
            Paper(
                paper_id="plasmon",
                title="Polarization-Controlled Tunable Directional Coupling of Surface Plasmon Polaritons",
                abstract="Generic SHG plasmonics controls surface plasmon polaritons.",
                year=2022,
                venue="Optics Letters",
                citation_count=15,
                source_provider="fake",
            ),
            Paper(
                paper_id="cosmo",
                title="COSMO solvent screening for molecular design",
                abstract="Generic solvent screening is used for molecular design.",
                year=2021,
                venue="Journal of Chemical Information",
                citation_count=12,
                source_provider="fake",
            ),
        ]
        return RetrievalResult(
            raw={"data": [{"title": paper.title} for paper in papers]},
            papers=papers,
        )


def test_ferroelectric_ranking_prioritizes_core_papers_and_aspect_coverage(tmp_path):
    output_dir = tmp_path / "outputs"
    result = run_pipeline(
        question=FERROELECTRIC_QUESTION,
        providers=["fake"],
        max_per_query=3,
        output_dir=str(output_dir),
        retriever_agent=RetrieverAgent(clients={"fake": FerroelectricRankingClient()}),
        use_query_families=True,
    )

    ranks = {item.paper.paper_id: item.rank for item in result.ranked_final}
    core_ids = {
        "surface-screening",
        "physics-oxides",
        "ultrathin",
        "batio3-imprint",
        "bulk-surface-charge",
        "interface-proximity",
    }
    generic_ids = {"generic-thin-film", "plasmon", "cosmo"}
    core_best = min(ranks[paper_id] for paper_id in core_ids)
    generic_best = min(ranks[paper_id] for paper_id in generic_ids)
    diagnostics = json.loads((output_dir / "ranking_diagnostics.json").read_text())
    exploration = json.loads((output_dir / "exploration_quality.json").read_text())

    assert result.search_contract.domain_profile.domain_name == "ferroelectric_polarization"
    assert any(record.aspect_coverage_score > 0 for record in result.aspect_coverage_records)
    assert core_best < generic_best
    assert diagnostics["domain_pack_used"] == "ferroelectric_polarization"
    assert diagnostics["query_family_applied"] is True
    assert diagnostics["aspect_coverage_distribution"]["nonzero_count"] > 0
    assert diagnostics["top20_false_positive_count"] >= 1
    assert exploration["query_family_coverage"] > 0
