import json

from lit_screening.agents.controversy_boundary import ControversyAndBoundaryAgent
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.models import Paper
from lit_screening.pipeline import run_pipeline
from lit_screening.retrieval.base import RetrievalResult


MATERIALS_QUESTION = (
    "有没有和 Surface Magnetization in Antiferromagnets 以及 Local "
    "Magnetoelectric Effects as Predictors of Surface Magnetic Order 相关的"
    "探测表面磁化和自旋极化重要性的文章"
)


def _materials_papers():
    return [
        Paper(
            paper_id="defect-boundary",
            title="Defects and parasitic magnetization in Cr2O3 thin films",
            abstract=(
                "Oxygen vacancy defects and impurity phases can create parasitic "
                "magnetization that competes with intrinsic boundary magnetization."
            ),
            venue="Physical Review B",
            year=2019,
            source_provider="fake",
        ),
        Paper(
            paper_id="rough-temperature",
            title="Surface roughness and finite temperature effects in chromia",
            abstract=(
                "Surface termination, step edge roughness, and thermal disorder near "
                "the Neel temperature limit ideal-surface predictions."
            ),
            venue="Journal of Magnetism",
            year=2021,
            source_provider="fake",
        ),
        Paper(
            paper_id="probe-comparison",
            title="SPLEEM and NV magnetometry of antiferromagnetic surface order",
            abstract=(
                "SPLEEM measures spin polarization, while NV magnetometry and MFM "
                "detect stray field contrast from antiferromagnetic domains."
            ),
            venue="Nature Materials",
            year=2023,
            source_provider="fake",
        ),
        Paper(
            paper_id="local-order",
            title="Local magnetic order without net magnetization at compensated surfaces",
            abstract=(
                "Local spin order and surface spin polarization may appear at a "
                "compensated antiferromagnetic surface with no net magnetization."
            ),
            venue="Physical Review Letters",
            year=2024,
            source_provider="fake",
        ),
    ]


def test_materials_controversy_agent_identifies_multiple_tensions():
    tensions = ControversyAndBoundaryAgent().analyze(
        _materials_papers(),
        domain="materials_magnetism",
    )
    keys = {tension.tension_key for tension in tensions}

    assert len(keys) >= 3
    assert "intrinsic_vs_defect_magnetism" in keys
    assert "ideal_surface_vs_real_rough_surface" in keys
    assert "zero_kelvin_dft_vs_finite_temperature" in keys
    assert "direct_spin_polarization_vs_stray_field_probe" in keys
    assert "net_surface_magnetization_vs_local_spin_order" in keys
    assert all(tension.confidence > 0 for tension in tensions)
    assert any(
        "defects parasitic magnetization" in " ".join(tension.suggested_next_searches)
        for tension in tensions
    )


def test_non_materials_domain_returns_no_tensions():
    tensions = ControversyAndBoundaryAgent().analyze(
        _materials_papers(),
        domain="ai_literature_screening",
    )

    assert tensions == []


class ControversyFakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        papers = _materials_papers()
        return RetrievalResult(
            raw={"data": [{"title": paper.title} for paper in papers]},
            papers=papers,
        )


def test_pipeline_writes_research_tensions_artifact_without_ranking_changes(tmp_path):
    output_dir = tmp_path / "outputs"

    result = run_pipeline(
        question=MATERIALS_QUESTION,
        providers=["fake"],
        max_per_query=1,
        output_dir=str(output_dir),
        retriever_agent=RetrieverAgent(clients={"fake": ControversyFakeClient()}),
    )
    payload = json.loads((output_dir / "research_tensions.json").read_text())
    trace = json.loads((output_dir / "agent_trace.json").read_text())

    assert payload
    assert result.research_tensions
    assert result.ranked_final
    assert trace["controversy_boundary"]["ranking_unchanged"] is True
    assert trace["controversy_boundary"]["tension_count"] == len(payload)
    assert any(
        row["tension_key"] == "direct_spin_polarization_vs_stray_field_probe"
        for row in payload
    )
    assert "Research Tensions And Boundary Conditions" in (
        output_dir / "report.md"
    ).read_text()
