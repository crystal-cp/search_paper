import csv
import json

from lit_screening.agents.evidence_function_classifier import classify_evidence_function
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.models import EvidenceFunction, Paper
from lit_screening.pipeline import run_pipeline
from lit_screening.retrieval.base import RetrievalResult


def test_predicts_effect_from_theory_markers():
    assert (
        classify_evidence_function(
            "We predict boundary magnetization from first-principles calculation and symmetry."
        )
        == EvidenceFunction.PREDICTS_EFFECT
    )


def test_directly_images_signal_from_domain_imaging():
    assert (
        classify_evidence_function("PEEM imaging reveals surface magnetization domains.")
        == EvidenceFunction.DIRECTLY_IMAGES_SIGNAL
    )


def test_surface_probe_from_probe_method_without_image_marker():
    assert (
        classify_evidence_function("SPLEEM detects an antiferromagnetic surface signal.")
        == EvidenceFunction.REPORTS_SURFACE_PROBE
    )


def test_measures_spin_polarization_from_photoemission():
    assert (
        classify_evidence_function(
            "Spin-polarized photoemission measures surface spin polarization."
        )
        == EvidenceFunction.MEASURES_SPIN_POLARIZATION
    )


def test_application_and_limitation_functions():
    assert (
        classify_evidence_function("Exchange bias enables memory readout in spintronics.")
        == EvidenceFunction.CONNECTS_TO_APPLICATION
    )
    assert (
        classify_evidence_function(
            "Surface roughness and finite temperature defects are a limitation."
        )
        == EvidenceFunction.REPORTS_LIMITATION
    )


def test_review_background_and_unknown():
    assert (
        classify_evidence_function("This review provides an overview of the field.")
        == EvidenceFunction.REVIEW_BACKGROUND
    )
    assert classify_evidence_function("") == EvidenceFunction.UNKNOWN


class EvidenceFunctionFakeClient:
    provider_name = "fake"

    def search(self, query, max_results, from_year=None):
        paper = Paper(
            paper_id="evidence-function-paper",
            title="Imaging surface magnetization domains with PEEM",
            abstract=(
                "PEEM imaging reveals surface magnetization domains in chromia. "
                "Exchange bias enables device readout."
            ),
            venue="Demo Journal",
            year=2024,
            source_provider="fake",
        )
        return RetrievalResult(
            raw={"data": [{"title": paper.title}]},
            papers=[paper],
        )


def test_pipeline_writes_evidence_function_artifacts_without_changing_verifier(tmp_path):
    output_dir = tmp_path / "outputs"
    result = run_pipeline(
        question="PEEM imaging surface magnetization domains",
        providers=["fake"],
        max_per_query=1,
        output_dir=str(output_dir),
        retriever_agent=RetrieverAgent(clients={"fake": EvidenceFunctionFakeClient()}),
    )

    payload = json.loads((output_dir / "evidence_functions.json").read_text())
    with (output_dir / "evidence_table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert payload[0]["evidence_function"] == "directly_images_signal"
    assert rows[0]["evidence_function"] == "directly_images_signal"
    assert result.evidence_records[0].evidence_function == EvidenceFunction.DIRECTLY_IMAGES_SIGNAL
    assert result.verification_results[0].support_level == "strict_support"
    assert result.verification_results[0].span_match_type == "exact"
