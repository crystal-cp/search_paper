from lit_screening.models import (
    DomainProfile,
    EvidenceRecord,
    Paper,
    QueryPlan,
    RankedPaper,
    SearchBrief,
    SearchContract,
    ScreeningDecision,
    VerificationResult,
)
from lit_screening.report import generate_report
from lit_screening.scoring import compute_final_score
from lit_screening.utils import write_json


SECTION_TITLES = [
    "# Research Process",
    "## 1. Research question interpretation",
    "## 2. Concept decomposition",
    "## 3. Search lenses and query families",
    "## 4. Screening and inclusion criteria",
    "## 5. Paper roles and why they matter",
    "## 6. Research lineage",
    "## 7. Controversies, limitations, and gaps",
    "## 8. Missing keywords, methods, authors, or schools",
    "## 9. Suggested next searches",
    "## 10. Verified vs uncertain findings",
]


def test_research_process_report_renders_mock_artifacts(tmp_path):
    _write_mock_artifacts(tmp_path)
    ranked = [
        _ranked_paper(
            paper_id="p1",
            title="Surface Magnetization in Antiferromagnets",
            year=2019,
            support_level="strict_support",
            abstract="Surface magnetization is an intrinsic boundary signal.",
            span_match_type="exact",
            span_match_confidence=1.0,
        ),
        _ranked_paper(
            paper_id="p2",
            title="Roughness effects in chromia surfaces",
            year=2024,
            support_level="missing_abstract",
            abstract="",
        ),
    ]
    path = tmp_path / "report.md"

    generate_report(
        path=path,
        research_question="surface magnetization and spin polarization",
        planned_queries=["surface magnetization antiferromagnets"],
        retrieval_statistics={"raw_retrieved_paper_count": 2, "merged_paper_count": 2},
        ranked_papers=ranked,
        evidence_records=[item.evidence for item in ranked],
        evaluation_metrics={},
        search_brief=_brief(),
        search_contract=_contract(),
        query_plan=QueryPlan(core_terms=["surface magnetization", "spin polarization"]),
        screening_decisions=[
            ScreeningDecision(
                paper_id="p1",
                decision="include",
                decision_confidence=0.9,
                primary_reason="strict evidence",
            )
        ],
    )
    report = path.read_text()

    for title in SECTION_TITLES:
        assert title in report
    assert "# Literature Screening Decision Report" in report
    assert "## Top 10 Ranked Papers" in report
    assert "- theory_origin: 1 paper(s)" in report
    assert "citation relation not verified" in report
    assert "needs further verification" in report
    assert "Cr2O3 surface paramagnetism finite temperature" in report


def test_research_process_report_handles_missing_artifacts(tmp_path):
    path = tmp_path / "report.md"

    generate_report(
        path=path,
        research_question="surface magnetization",
        planned_queries=[],
        retrieval_statistics={},
        ranked_papers=[],
        evidence_records=[],
        evaluation_metrics={},
    )
    report = path.read_text()

    for title in SECTION_TITLES:
        assert title in report
    assert "Not generated in this run." in report
    assert "citation relation not verified" in report


def _write_mock_artifacts(output_dir):
    write_json(
        output_dir / "concept_map.json",
        {
            "domain": "materials_magnetism",
            "central_question": "surface magnetization and spin polarization",
            "lenses": [
                {
                    "name": "direct_surface_detection",
                    "core_concepts": ["boundary magnetization"],
                    "materials": ["Cr2O3", "chromia"],
                    "methods": ["SPLEEM", "XMCD-PEEM"],
                    "applications": [],
                }
            ],
        },
    )
    write_json(
        output_dir / "query_families.json",
        {
            "domain": "materials_magnetism",
            "central_question": "surface magnetization and spin polarization",
            "families": [
                {
                    "name": "direct_surface_detection",
                    "lens_name": "direct_surface_detection",
                    "purpose": "find direct surface probe papers",
                    "queries_by_provider": {
                        "openalex": ["SPLEEM chromia spin polarization asymmetry"]
                    },
                }
            ],
        },
    )
    write_json(
        output_dir / "seed_hints.json",
        [
            {
                "title": "Surface Magnetization in Antiferromagnets",
                "authors": ["Nicola A. Spaldin"],
                "confidence": 0.9,
            }
        ],
    )
    write_json(
        output_dir / "paper_roles.json",
        [
            {
                "paper_id": "p1",
                "title": "Surface Magnetization in Antiferromagnets",
                "roles": ["theory_origin"],
                "primary_role": "theory_origin",
                "reasons": ["matched boundary magnetization"],
            },
            {
                "paper_id": "p2",
                "title": "Roughness effects in chromia surfaces",
                "roles": ["limitation_or_challenge"],
                "primary_role": "limitation_or_challenge",
                "reasons": ["matched roughness"],
            },
        ],
    )
    write_json(
        output_dir / "evidence_functions.json",
        [
            {
                "paper_id": "p1",
                "title": "Surface Magnetization in Antiferromagnets",
                "evidence_function": "defines_concept",
            }
        ],
    )
    write_json(
        output_dir / "research_tensions.json",
        [
            {
                "tension_key": "zero_kelvin_dft_vs_finite_temperature",
                "tension_label": "Zero-temperature calculations vs finite-temperature surfaces",
                "why_it_matters": "Temperature controls surface order.",
                "suggested_next_searches": [
                    "Cr2O3 surface paramagnetism finite temperature magnetoelectric antiferromagnet"
                ],
                "confidence": 0.8,
            }
        ],
    )
    write_json(
        output_dir / "suggested_next_searches.json",
        [
            {
                "query": "Cr2O3 surface paramagnetism finite temperature magnetoelectric antiferromagnet",
                "reason": "finite-temperature gap",
                "source": "research_gap_matrix",
            }
        ],
    )
    (output_dir / "research_gap_matrix.csv").write_text(
        "gap_key,gap_label,evidence_or_reason,suggested_next_searches,confidence\n"
        "finite_temperature_effect,Finite-temperature effects are undercovered,"
        "No finite-temperature marker was found,"
        "Cr2O3 surface paramagnetism finite temperature magnetoelectric antiferromagnet,"
        "0.86\n",
        encoding="utf-8",
    )


def _brief():
    return SearchBrief(
        original_question="surface magnetization",
        refined_question="surface magnetization and spin polarization",
        search_intent="overview",
        user_goal="map research process",
        inclusion_criteria=["antiferromagnets"],
        exclusion_criteria=["clinical screening"],
        required_aspects=["direct probes"],
    )


def _contract():
    return SearchContract(
        original_question="surface magnetization",
        refined_question="surface magnetization and spin polarization",
        user_goal="map research process",
        search_intent="overview",
        domain_profile=DomainProfile(domain_name="materials_magnetism"),
        must_include_concepts=["surface magnetization"],
        must_exclude_concepts=["drug screening"],
        inclusion_criteria=["antiferromagnets"],
        exclusion_criteria=["clinical screening"],
        required_aspects=["direct probes"],
    )


def _ranked_paper(
    paper_id,
    title,
    year,
    support_level,
    abstract,
    span_match_type="none",
    span_match_confidence=0.0,
):
    paper = Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=year,
    )
    evidence = EvidenceRecord(
        paper_id=paper_id,
        title=title,
        claim="Surface magnetization is relevant.",
        evidence_sentence=abstract or title,
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id=paper_id,
        supported=support_level == "strict_support",
        confidence=0.9 if support_level == "strict_support" else 0.2,
        error_type="" if support_level == "strict_support" else support_level,
        rationale="test",
        support_level=support_level,
        span_match_type=span_match_type,
        span_match_confidence=span_match_confidence,
    )
    scores = compute_final_score(
        relevance_score=0.8,
        evidence_score=0.8,
        recency_score=0.8,
        quality_score=0.8,
        diversity_score=0.8,
    )
    return RankedPaper(
        rank=1,
        paper=paper,
        evidence=evidence,
        verification=verification,
        scores=scores,
    )
