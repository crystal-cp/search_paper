import json

from lit_screening.evaluation.exploration_quality import compute_exploration_quality
from lit_screening.pipeline import run_pipeline


def test_exploration_quality_scores_mock_artifacts():
    metrics = compute_exploration_quality(
        concept_map={
            "domain": "materials_magnetism",
            "lenses": [
                {
                    "name": "theory_origin",
                    "core_concepts": ["boundary magnetization"],
                    "methods": [],
                },
                {
                    "name": "direct_surface_detection",
                    "core_concepts": ["surface spin polarization"],
                    "methods": ["SPLEEM"],
                },
            ],
        },
        query_families={
            "families": [
                {
                    "name": "direct_surface_detection",
                    "lens_name": "direct_surface_detection",
                    "purpose": "find direct probe papers",
                    "queries_by_provider": {
                        "openalex": [
                            "SPLEEM chromia spin polarization asymmetry",
                        ]
                    },
                    "linked_seed_titles": [
                        "Surface Magnetization in Antiferromagnets",
                    ],
                }
            ]
        },
        paper_roles=[
            {"paper_id": "p1", "roles": ["theory_origin"]},
            {"paper_id": "p2", "roles": ["surface_probe_method"]},
            {"paper_id": "p3", "primary_role": "nanoscale_readout"},
        ],
        evidence_functions=[
            {"paper_id": "p1", "evidence_function": "defines_concept"},
            {"paper_id": "p2", "evidence_function": "directly_images_signal"},
        ],
        gap_matrix=[
            {
                "gap_key": "finite_temperature_effect",
                "gap_label": "Finite-temperature effects are undercovered",
                "evidence_or_reason": "No finite-temperature marker was found.",
                "suggested_next_searches": [
                    "Cr2O3 surface paramagnetism finite temperature magnetoelectric antiferromagnet",
                ],
            }
        ],
        research_tensions=[
            {"tension_key": "zero_kelvin_dft_vs_finite_temperature"},
            {"tension_key": "ideal_surface_vs_real_rough_surface"},
        ],
        seed_hints=[
            {
                "title": "Surface Magnetization in Antiferromagnets",
                "authors": ["Nicola A. Spaldin"],
            }
        ],
    )

    assert metrics["concept_coverage"] == 1.0
    assert metrics["query_family_coverage"] > 0
    assert metrics["paper_role_diversity"] > 0
    assert metrics["seed_hint_utilization"] > 0
    assert metrics["evidence_function_diversity"] > 0
    assert metrics["gap_specificity"] > 0
    assert metrics["research_tension_count"] == 2


def test_exploration_quality_handles_missing_artifacts():
    metrics = compute_exploration_quality()

    assert metrics["concept_coverage"] == 0
    assert metrics["query_family_coverage"] == 0
    assert metrics["paper_role_diversity"] == 0
    assert metrics["seed_hint_utilization"] == 0
    assert metrics["evidence_function_diversity"] == 0
    assert metrics["gap_specificity"] == 0
    assert metrics["research_tension_count"] == 0


def test_pipeline_writes_exploration_quality_artifact(tmp_path):
    output_dir = tmp_path / "outputs"

    run_pipeline(
        question="surface magnetization in antiferromagnets",
        providers=[],
        max_per_query=0,
        output_dir=str(output_dir),
    )
    payload = json.loads((output_dir / "exploration_quality.json").read_text())

    assert (output_dir / "evaluation.json").exists()
    assert "concept_coverage" in payload
    assert "query_family_coverage" in payload
    assert "gap_specificity" in payload
