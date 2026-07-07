from lit_screening.decision_artifacts import (
    build_research_gap_matrix,
    build_suggested_next_searches,
)
from lit_screening.models import (
    AspectCoverageRecord,
    DomainProfile,
    EvidenceRecord,
    Paper,
    QueryPlan,
    RankedPaper,
    SearchContract,
    VerificationResult,
)
from lit_screening.scoring import compute_final_score


def test_materials_gap_matrix_flags_finite_temperature_gap():
    gaps = build_research_gap_matrix(
        ranked_papers=[
            _ranked_materials_paper(
                title="Boundary magnetization in Cr2O3 magnetoelectric antiferromagnets",
                abstract=(
                    "Symmetry classification predicts surface magnetization in Cr2O3 "
                    "as a magnetoelectric antiferromagnet."
                ),
            )
        ],
        aspect_coverage_records=[],
        search_contract=_materials_contract(),
    )
    gap_by_key = {row["gap_key"]: row for row in gaps}

    assert "finite_temperature_effect" in gap_by_key
    assert "finite temperature" in gap_by_key["finite_temperature_effect"]["suggested_next_searches"]
    assert gap_by_key["finite_temperature_effect"]["confidence"]


def test_materials_gap_matrix_flags_direct_surface_probe_gap():
    gaps = build_research_gap_matrix(
        ranked_papers=[
            _ranked_materials_paper(
                title="Surface magnetization classification in Cr2O3",
                abstract="Boundary magnetization and local magnetoelectric responses are predicted by symmetry.",
            )
        ],
        aspect_coverage_records=[],
        search_contract=_materials_contract(),
    )
    gap_by_key = {row["gap_key"]: row for row in gaps}

    assert "direct_surface_probe_gap" in gap_by_key
    assert "SPLEEM chromia spin polarization asymmetry" in gap_by_key[
        "direct_surface_probe_gap"
    ]["suggested_next_searches"]


def test_materials_suggested_next_searches_use_domain_queries():
    gaps = build_research_gap_matrix(
        ranked_papers=[
            _ranked_materials_paper(
                title="Surface magnetization theory in Cr2O3",
                abstract="Boundary magnetization is predicted in a magnetoelectric antiferromagnet.",
            )
        ],
        aspect_coverage_records=[],
        search_contract=_materials_contract(),
    )
    searches = build_suggested_next_searches(
        ranked_papers=[],
        gap_rows=gaps,
        search_contract=_materials_contract(),
        query_plan=QueryPlan(must_terms=["surface magnetization"]),
    )
    queries = {row["query"] for row in searches}

    assert "Cr2O3 surface paramagnetism finite temperature magnetoelectric antiferromagnet" in queries
    assert "SPLEEM chromia spin polarization asymmetry" in queries
    assert "NV magnetometry Cr2O3 boundary magnetization" in queries
    assert "defects parasitic magnetization antiferromagnetic thin films" in queries


def test_default_domain_keeps_legacy_gap_rules():
    gaps = build_research_gap_matrix(
        ranked_papers=[],
        aspect_coverage_records=[
            AspectCoverageRecord(
                paper_id="p1",
                title="Paper",
                missing_aspects=["evidence verification"],
            )
        ],
        search_contract=None,
    )

    assert gaps[0]["gap"] == "Evidence verification is undercovered"
    assert gaps[0]["possible_project_idea"] == (
        "Span-grounded verification for abstract-level evidence chains."
    )


def _materials_contract() -> SearchContract:
    return SearchContract(
        original_question="surface magnetization in antiferromagnets",
        refined_question="surface magnetization in antiferromagnets",
        user_goal="Find materials magnetism papers.",
        search_intent="overview",
        domain_profile=DomainProfile(domain_name="materials_magnetism"),
        required_aspects=["surface magnetization", "direct surface probe"],
    )


def _ranked_materials_paper(title: str, abstract: str) -> RankedPaper:
    paper = Paper(
        paper_id=title.lower().replace(" ", "-")[:40],
        title=title,
        abstract=abstract,
        year=2024,
    )
    evidence = EvidenceRecord(
        paper_id=paper.paper_id,
        title=title,
        claim=abstract,
        evidence_sentence=abstract,
        relevance_reason="test",
    )
    verification = VerificationResult(
        paper_id=paper.paper_id,
        supported=True,
        confidence=1.0,
        error_type="",
        rationale="test",
        support_level="strict_support",
        span_match_type="exact",
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
