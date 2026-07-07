import json

from lit_screening.models import (
    QueryFamily,
    QueryFamilyPlan,
    ResearchLens,
    ResearchLensPlan,
)
from lit_screening.utils import to_plain_data


def test_research_lens_plan_serializes_to_json():
    plan = ResearchLensPlan(
        domain="materials_magnetism",
        central_question="How can surface magnetization be detected?",
        lenses=[
            ResearchLens(
                name="surface_detection",
                role="find experimental probes of boundary magnetization",
                question="Which probes detect surface magnetization or spin polarization?",
                core_concepts=["surface magnetization", "surface spin polarization"],
                synonyms=["boundary magnetization", "spin-polarized surface state"],
                materials=["Cr2O3", "NiO"],
                methods=["SPLEEM", "XMCD-PEEM", "NV magnetometry"],
                applications=["antiferromagnetic spintronics"],
                seed_paper_hints=["Surface Magnetization in Antiferromagnets"],
                expected_evidence_types=["experimental probe", "surface-sensitive measurement"],
                exclusion_risks=["generic ferromagnetic thin films"],
            )
        ],
    )

    payload = json.loads(json.dumps(to_plain_data(plan), ensure_ascii=False))

    assert payload["domain"] == "materials_magnetism"
    assert payload["central_question"] == "How can surface magnetization be detected?"
    assert payload["lenses"][0]["name"] == "surface_detection"
    assert "boundary magnetization" in payload["lenses"][0]["synonyms"]
    assert "NV magnetometry" in payload["lenses"][0]["methods"]


def test_query_family_plan_serializes_to_json():
    plan = QueryFamilyPlan(
        domain="materials_magnetism",
        central_question="How can surface magnetization be detected?",
        families=[
            QueryFamily(
                name="surface_probe_queries",
                purpose="retrieve surface-sensitive experimental probe papers",
                lens_name="surface_detection",
                queries_by_provider={
                    "openalex": [
                        '"surface magnetization" SPLEEM antiferromagnet',
                        '"surface spin polarization" XMCD-PEEM',
                    ],
                    "semantic_scholar": [
                        '+"surface magnetization" +SPLEEM +antiferromagnet',
                    ],
                },
                expected_paper_roles=["experimental evidence", "method reference"],
                expected_evidence_types=["measurement", "imaging"],
                exclusion_terms=["magnetic nanoparticles"],
                stop_condition=None,
            )
        ],
    )

    payload = json.loads(json.dumps(to_plain_data(plan), ensure_ascii=False))

    assert payload["domain"] == "materials_magnetism"
    assert payload["families"][0]["lens_name"] == "surface_detection"
    assert "openalex" in payload["families"][0]["queries_by_provider"]
    assert payload["families"][0]["queries_by_provider"]["openalex"][0].startswith(
        '"surface magnetization"'
    )
    assert payload["families"][0]["stop_condition"] is None
