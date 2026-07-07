import json

from lit_screening.agents.paper_role_classifier import PaperRoleClassifier
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.models import Paper
from lit_screening.pipeline import run_pipeline
from lit_screening.retrieval.base import RetrievalResult


def test_belashchenko_like_title_maps_to_theory_origin():
    record = PaperRoleClassifier().classify(
        Paper(
            paper_id="theory",
            title="Equilibrium magnetization of magnetoelectric antiferromagnets",
            abstract="Boundary magnetization appears in a magnetoelectric antiferromagnet.",
            venue="Physical Review Letters",
            year=2010,
        )
    )

    assert record.primary_role == "theory_origin"
    assert "theory_origin" in record.roles
    assert record.confidence > 0.4


def test_wu_like_imaging_surface_domains_maps_to_experimental_proof():
    record = PaperRoleClassifier().classify(
        Paper(
            paper_id="imaging",
            title="Imaging surface magnetization domains in chromia",
            abstract="XMCD-PEEM reveals spin polarization asymmetry at antiferromagnetic surfaces.",
            venue="Nature Materials",
            year=2011,
        )
    )

    assert record.primary_role == "experimental_proof"
    assert "experimental_proof" in record.roles
    assert "surface_probe_method" in record.roles


def test_nv_magnetometry_title_maps_to_nanoscale_readout():
    record = PaperRoleClassifier().classify(
        Paper(
            paper_id="nv",
            title="NV magnetometry of antiferromagnetic domains in Cr2O3",
            abstract="Scanning diamond magnetometry enables nanoscale readout.",
            venue="Nano Letters",
            year=2021,
        )
    )

    assert record.primary_role == "nanoscale_readout"
    assert "nanoscale_readout" in record.roles


def test_exchange_bias_title_maps_to_application_bridge():
    record = PaperRoleClassifier().classify(
        Paper(
            paper_id="application",
            title="Exchange bias and magnetoelectric memory in Cr2O3",
            abstract="Surface magnetization supports antiferromagnetic spintronics readout.",
            venue="Applied Physics Letters",
            year=2018,
        )
    )

    assert "application_bridge" in record.roles


def test_roughness_finite_temperature_title_maps_to_limitation():
    record = PaperRoleClassifier().classify(
        Paper(
            paper_id="limitations",
            title="Surface roughness and finite temperature effects in antiferromagnets",
            abstract="Defects and paramagnetism can obscure surface magnetization.",
            venue="Physical Review B",
            year=2019,
        )
    )

    assert "limitation_or_challenge" in record.roles


def test_query_provenance_links_lenses_and_families():
    paper = Paper(
        paper_id="query-linked",
        title="Surface magnetization in antiferromagnets",
        abstract="A classification of magnetoelectric responses.",
        source_provider="fake",
        retrieval_provider="fake",
        retrieval_query='"surface magnetization" antiferromagnets classification',
    )
    provenance = {
        "records": [
            {
                "provider": "fake",
                "raw_query": '"surface magnetization" antiferromagnets classification',
                "source": "query_family",
                "family_name": "spaldin_framework",
                "lens_name": "spaldin_framework",
                "purpose": "trace the Spaldin framework",
            }
        ]
    }
    record = PaperRoleClassifier().classify(paper, query_provenance=provenance)

    assert "spaldin_framework" in record.linked_lenses
    assert "spaldin_framework" in record.linked_query_families
    assert "conceptual_framework" in record.roles


class RoleFakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        paper = Paper(
            paper_id="role-paper",
            title="Equilibrium magnetization of magnetoelectric antiferromagnets",
            abstract="Boundary magnetization appears in a magnetoelectric antiferromagnet.",
            venue="Physical Review Letters",
            year=2010,
            source_provider="fake",
        )
        return RetrievalResult(
            raw={"data": [{"title": paper.title}]},
            papers=[paper],
        )


def test_pipeline_writes_paper_roles_artifact(tmp_path):
    output_dir = tmp_path / "outputs"
    result = run_pipeline(
        question="surface magnetization in magnetoelectric antiferromagnets",
        providers=["fake"],
        max_per_query=1,
        output_dir=str(output_dir),
        retriever_agent=RetrieverAgent(clients={"fake": RoleFakeClient()}),
    )
    payload = json.loads((output_dir / "paper_roles.json").read_text())

    assert payload
    assert result.paper_role_records
    assert payload[0]["primary_role"] == "theory_origin"
