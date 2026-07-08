import json

from lit_screening.agents.llm_enhancement import (
    render_user_report_markdown,
    verified_report_input,
)
from lit_screening.decision_artifacts import (
    build_research_gap_matrix,
    build_suggested_next_searches,
    write_decision_artifacts,
)
from lit_screening.models import (
    AspectCoverageRecord,
    DomainAssessment,
    DomainProfile,
    EvidenceRecord,
    GenericResearchIntentFrame,
    Paper,
    PaperRoleRecord,
    QueryPlan,
    RankedPaper,
    SearchContract,
    ScreeningDecision,
    VerificationResult,
)
from lit_screening.pipeline import (
    build_auto_query_repair_suggestions,
    sync_query_critic_with_repair,
)
from lit_screening.report import generate_report
from lit_screening.scoring import compute_final_score


def test_no_research_gaps_when_retrieval_not_attempted(tmp_path):
    decision_artifacts = write_decision_artifacts(
        tmp_path,
        [],
        [],
        search_contract=_contract(["MOF", "CO2 capture"], ["CO2 adsorption"]),
        query_plan=QueryPlan(core_terms=["MOF", "CO2 capture"]),
        retrieval_context={
            "retrieval_status": "planning_only",
            "reason": "retrieval_not_performed",
            "merged_paper_count": 0,
            "ranked_papers_based_on_real_retrieval": False,
        },
    )

    assert decision_artifacts["research_gap_matrix"][0]["gap_generation_status"] == "skipped"
    assert decision_artifacts["research_gap_matrix"][0]["reason"] == "retrieval_not_performed"
    csv_text = (tmp_path / "research_gap_matrix.csv").read_text()
    assert "undercovered" not in csv_text


def test_suggested_next_searches_not_generated_from_fake_gaps():
    gap_rows = [
        {
            "gap_generation_status": "skipped",
            "reason": "retrieval_not_performed",
            "gap": "gap_generation_status=skipped",
        }
    ]

    searches = build_suggested_next_searches(
        [],
        gap_rows,
        search_contract=_contract(["MOF", "CO2 capture"], ["CO2 adsorption"]),
        retrieval_context={
            "retrieval_status": "planning_only",
            "ranked_papers_based_on_real_retrieval": False,
        },
    )

    assert searches == []


def test_user_report_explains_planning_only():
    artifact_input = verified_report_input(
        ranked_papers=[],
        domain_assessments=[],
        ranking_diagnostics={},
        paper_roles=[],
        provider_status={
            "openalex": {"status": "not_attempted", "returned_paper_count": 0},
            "semantic_scholar": {"status": "not_attempted", "returned_paper_count": 0},
        },
        exploration_quality={},
    )

    report = render_user_report_markdown(artifact_input)

    assert "retrieval_status: planning_only" in report
    assert "本次只完成 query planning" in report
    assert "不生成 research gaps" in report


def test_gap_specificity_for_sei_full():
    gaps = build_research_gap_matrix(
        [_ranked_paper("p1", "Solid electrolyte interphase review", "SEI formation in lithium-ion batteries.")],
        [],
        search_contract=_contract(
            ["SEI", "solid electrolyte interphase"],
            ["lithium-ion battery", "composition", "structure", "evolution"],
        ),
        retrieval_context={
            "retrieval_status": "success",
            "merged_paper_count": 1,
            "ranked_papers_based_on_real_retrieval": True,
        },
    )
    gap_text = json.dumps(gaps, ensure_ascii=False)

    assert "SEI" in gap_text
    assert "operando" in gap_text or "composition-structure-function" in gap_text
    assert "Characterization methods are undercovered" not in gap_text


def test_gap_specificity_for_oer_full():
    gaps = build_research_gap_matrix(
        [
            _ranked_paper(
                "p1",
                "Spin-state transition improves oxygen evolution",
                "OER activity in transition metal oxide catalysts depends on spin state.",
            )
        ],
        [],
        search_contract=_contract(
            ["oxygen evolution reaction", "spin state"],
            ["transition metal oxide catalyst", "electronic structure"],
        ),
        retrieval_context={
            "retrieval_status": "success",
            "merged_paper_count": 1,
            "ranked_papers_based_on_real_retrieval": True,
        },
    )
    gap_text = json.dumps(gaps, ensure_ascii=False)

    assert "OER" in gap_text or "oxygen evolution" in gap_text
    assert "surface reconstruction" in gap_text or "lattice oxygen" in gap_text
    assert "Human feedback is undercovered" not in gap_text


def test_user_report_shows_matched_and_missing_groups():
    report = _render_single_paper_user_report(
        domain_assessment=DomainAssessment(
            paper_id="p1",
            domain_match_score=0.9,
            domain_decision="in_scope",
            off_topic_reason="",
            matched_groups=["core_object_group", "domain_context_group"],
            missing_groups=["method_group"],
            topic_focus_score=0.82,
            target_context_match=["lithium-ion battery"],
        ),
        decision=ScreeningDecision(
            paper_id="p1",
            decision="include",
            decision_confidence=0.9,
            primary_reason="matches intent",
            reading_priority="must_read",
        ),
    )

    assert "Matched groups" in report
    assert "core_object_group" in report
    assert "Missing groups" in report
    assert "method_group" in report
    assert "topic focus" in report


def test_user_report_flags_peripheral_context():
    report = _render_single_paper_user_report(
        title="Beyond lithium-ion batteries",
        abstract="This review focuses on zinc-ion and beyond lithium-ion contexts with only brief SEI mentions.",
        domain_assessment=DomainAssessment(
            paper_id="p1",
            domain_match_score=0.5,
            domain_decision="borderline",
            off_topic_reason="non-target chemistry",
            matched_groups=["core_object_group"],
            missing_groups=["domain_context_group"],
            negative_context_match=["zinc-ion", "beyond lithium-ion"],
            peripheral_context_reason="non-target battery chemistry",
            topic_focus_score=0.35,
        ),
        decision=ScreeningDecision(
            paper_id="p1",
            decision="maybe",
            decision_confidence=0.5,
            primary_reason="peripheral context",
            reading_priority="read_later",
        ),
    )

    assert "peripheral/caution" in report
    assert "zinc-ion" in report
    assert "Use as peripheral/background only" in report


def test_user_report_distinguishes_strict_support_from_relevance():
    report = _render_single_paper_user_report()

    assert "`strict_support` means the evidence sentence is grounded in the abstract" in report
    assert "does not by itself mean the paper is highly relevant" in report


def test_provider_status_summary_in_user_report():
    report = _render_single_paper_user_report(
        provider_status={
            "openalex": {"status": "success", "returned_paper_count": 1},
            "semantic_scholar": {"status": "failed", "returned_paper_count": 0},
        }
    )

    assert "retrieval_status: partial_success" in report
    assert "openalex:success(1 papers)" in report
    assert "semantic_scholar:failed(0 papers)" in report


def test_query_repair_applied_matches_final_query_changes():
    repair = build_auto_query_repair_suggestions(
        QueryPlan(openalex_queries=["MOF CO2"], semantic_scholar_queries=[]),
        {
            "applied": True,
            "provider_query_counts": {"openalex": 8},
            "final_openalex_queries": ['MOF "CO2 capture" "adsorption performance"'],
            "final_semantic_scholar_queries": [],
            "dropped_query_count": 1,
            "dropped_queries": [
                {
                    "query": "MOF CO2",
                    "reason": "short_anchor_only",
                    "accepted_critic_issue": "overbroad_queries:mof_short_anchor_only",
                }
            ],
        },
    )

    assert repair["applied"] is True
    assert repair["dropped_queries"]


def test_query_critic_records_repair_accepted_issues():
    critic = {
        "missing_user_aspects": [],
        "overbroad_queries": [],
        "overconstrained_queries": [],
        "repetition_issues": [],
        "cross_domain_pollution": [],
        "diversity_suggestions": [],
        "accepted_suggestions": [],
    }
    synced = sync_query_critic_with_repair(
        critic,
        {
            "query_actions": [
                {
                    "original_query": "MOF CO2",
                    "action": "drop",
                    "accepted_critic_issue": "overbroad_queries:mof_short_anchor_only",
                },
                {
                    "original_query": '"atomic layer deposition" CVD',
                    "action": "drop",
                    "accepted_critic_issue": "diversity_suggestions:thin_film_ald_bias",
                },
            ]
        },
    )

    assert synced["overbroad_queries"]
    assert synced["diversity_suggestions"]
    assert synced["accepted_repair_issues"]


def test_full_run_no_clear_gap_not_rendered_as_gap(tmp_path):
    gaps = _sei_no_clear_gap_rows()
    report_path = tmp_path / "report.md"

    generate_report(
        path=report_path,
        research_question="SEI characterization in lithium-ion batteries",
        planned_queries=[],
        retrieval_statistics={
            "retrieval_status": "success",
            "provider_summary": "openalex:success(3 papers)",
            "ranked_papers_based_on_real_retrieval": True,
            "raw_retrieved_paper_count": 3,
            "merged_paper_count": 3,
        },
        ranked_papers=[],
        evidence_records=[],
        evaluation_metrics={},
        research_gap_matrix=gaps,
        suggested_next_searches=[],
    )
    report = report_path.read_text()

    assert "Focused follow-up remains useful" not in report
    assert "- Gap: Coverage summary" not in report
    assert "Coverage summary:" in report
    assert "No clear research gap was inferred" in report


def test_suggested_next_searches_no_undercovered_gap_when_no_clear_gap():
    gaps = _sei_no_clear_gap_rows()
    searches = build_suggested_next_searches(
        [_ranked_paper("p1", "SEI characterization", "SEI formation in lithium-ion batteries.")],
        gaps,
        search_contract=_contract(
            ["SEI", "solid electrolyte interphase"],
            ["lithium-ion battery", "composition", "structure"],
        ),
        retrieval_context={
            "retrieval_status": "success",
            "merged_paper_count": 1,
            "ranked_papers_based_on_real_retrieval": True,
        },
    )

    assert searches
    assert all("undercovered gap" not in item["reason"].lower() for item in searches)
    assert any("optional follow-up" in item["reason"].lower() for item in searches)


def test_sei_full_suggested_searches_are_aspect_specific():
    searches = build_suggested_next_searches(
        [_ranked_paper("p1", "SEI characterization", "SEI formation in lithium-ion batteries.")],
        _sei_no_clear_gap_rows(),
        search_contract=_contract(
            ["SEI", "solid electrolyte interphase"],
            ["lithium-ion battery", "composition", "structure"],
        ),
        retrieval_context={
            "retrieval_status": "success",
            "merged_paper_count": 1,
            "ranked_papers_based_on_real_retrieval": True,
        },
    )
    query_text = " ".join(item["query"] for item in searches)

    assert 'SEI "in situ" characterization "lithium-ion battery"' in query_text
    assert 'SEI "ex situ" characterization "lithium-ion battery"' in query_text
    assert "failure mechanism" in query_text


def test_oer_full_suggested_searches_are_aspect_specific():
    gaps = _oer_no_clear_gap_rows()
    searches = build_suggested_next_searches(
        [
            _ranked_paper(
                "p1",
                "OER spin state catalyst",
                "OER spin state and electronic structure in perovskite catalysts.",
            )
        ],
        gaps,
        search_contract=_contract(
            ["oxygen evolution reaction", "spin state"],
            ["transition metal catalyst", "electronic structure"],
        ),
        retrieval_context={
            "retrieval_status": "success",
            "merged_paper_count": 1,
            "ranked_papers_based_on_real_retrieval": True,
        },
    )
    query_text = " ".join(item["query"] for item in searches)

    assert 'OER "spin state" operando spectroscopy' in query_text
    assert '"surface reconstruction"' in query_text
    assert '"lattice oxygen mechanism"' in query_text
    assert '"electronic structure" perovskite oxide' in query_text


def test_user_report_has_coverage_and_remaining_gaps_section():
    report = _render_single_paper_user_report(
        search_contract=_contract(
            ["SEI", "solid electrolyte interphase"],
            ["lithium-ion battery", "composition", "structure"],
        ),
        research_gap_matrix=_sei_no_clear_gap_rows(),
        suggested_next_searches=[
            {
                "query": 'SEI "in situ" characterization "lithium-ion battery"',
                "reason": "Optional follow-up to check depth and recency",
                "source": "research_gap_matrix",
            }
        ],
    )

    assert "## Coverage and remaining gaps" in report
    assert "Coverage summary:" in report
    assert "No clear research gap is asserted" in report
    assert "Optional follow-up directions" in report


def test_user_report_feedback_examples_are_task_specific():
    sei_report = _render_single_paper_user_report(
        search_contract=_contract(
            ["SEI", "solid electrolyte interphase"],
            ["lithium-ion battery"],
        )
    )
    oer_report = _render_single_paper_user_report(
        title="OER spin state catalyst",
        abstract="OER spin state in transition metal catalysts.",
        search_contract=_contract(
            ["oxygen evolution reaction", "spin state"],
            ["transition metal catalyst"],
        ),
    )

    assert "exclude non-lithium battery systems" in sei_report
    assert "focus on graphite / silicon / lithium metal anodes" in sei_report
    assert "focus on operando spectroscopy" in oer_report
    assert "exclude ORR-only / CO2-reduction papers" in oer_report


def test_plan_report_does_not_render_skipped_gap_as_gap(tmp_path):
    report_path = tmp_path / "report.md"
    skipped = [
        {
            "gap_generation_status": "skipped",
            "reason": "retrieval_not_performed",
            "gap_key": "gap_generation_skipped",
            "gap_label": "Research gap generation skipped",
            "gap": "gap_generation_status=skipped",
            "evidence_or_reason": "retrieval_not_performed",
            "suggested_next_searches": "",
        }
    ]

    generate_report(
        path=report_path,
        research_question="MOF CO2 capture",
        planned_queries=[],
        retrieval_statistics={
            "retrieval_status": "planning_only",
            "provider_summary": "openalex:not_attempted(0 papers)",
            "ranked_papers_based_on_real_retrieval": False,
            "provider_status": {
                "openalex": {"status": "not_attempted", "returned_paper_count": 0}
            },
        },
        ranked_papers=[],
        evidence_records=[],
        evaluation_metrics={},
        research_gap_matrix=skipped,
        suggested_next_searches=[],
    )
    report = report_path.read_text()

    assert "- Gap: Research gap generation skipped" not in report
    assert "Gap-derived missing directions: Research gap generation skipped" not in report
    assert "Research gaps were not generated in this planning-only run" in report


def _contract(core_terms, context_terms):
    frame = GenericResearchIntentFrame(
        research_object=list(core_terms),
        domain_context=list(context_terms),
        core_terms=list(core_terms),
        target_process_or_property=list(context_terms),
        method_need=True,
        mechanism_need=True,
        material_case_need=True,
        failure_or_limitation_need=True,
        controversy_need=True,
        review_background_need=True,
    )
    return SearchContract(
        original_question="test question",
        refined_question="test question",
        user_goal="test",
        search_intent="overview",
        domain_profile=DomainProfile(domain_name="general_science"),
        must_include_concepts=list(core_terms),
        optional_concepts=list(context_terms),
        required_aspects=list(context_terms),
        generic_intent_frame=frame,
    )


def _ranked_paper(paper_id, title, abstract, domain_assessment=None, decision=None):
    scores = compute_final_score(
        relevance_score=0.8,
        evidence_score=0.8,
        recency_score=0.8,
        quality_score=0.8,
        diversity_score=0.8,
    )
    scores.intent_centrality_score = (
        domain_assessment.intent_centrality_score if domain_assessment else 0.8
    )
    return RankedPaper(
        rank=1,
        paper=Paper(paper_id=paper_id, title=title, abstract=abstract, year=2024),
        evidence=EvidenceRecord(
            paper_id=paper_id,
            title=title,
            claim=abstract,
            evidence_sentence=abstract,
            relevance_reason="test",
        ),
        verification=VerificationResult(
            paper_id=paper_id,
            supported=True,
            confidence=0.9,
            error_type="",
            rationale="test",
            support_level="strict_support",
            span_match_type="exact",
            span_match_confidence=1.0,
        ),
        scores=scores,
        domain_assessment=domain_assessment,
        screening_decision=decision,
    )


def _render_single_paper_user_report(
    title="Core SEI characterization paper",
    abstract="This paper studies SEI composition and structure in lithium-ion batteries.",
    domain_assessment=None,
    decision=None,
    provider_status=None,
    search_contract=None,
    research_gap_matrix=None,
    suggested_next_searches=None,
):
    assessment = domain_assessment or DomainAssessment(
        paper_id="p1",
        domain_match_score=0.9,
        domain_decision="in_scope",
        off_topic_reason="",
        matched_groups=["core_object_group", "domain_context_group", "method_group"],
        missing_groups=[],
        target_context_match=["lithium-ion battery"],
        topic_focus_score=0.9,
        intent_centrality_score=0.9,
    )
    decision = decision or ScreeningDecision(
        paper_id="p1",
        decision="include",
        decision_confidence=0.9,
        primary_reason="matches intent",
        reading_priority="must_read",
    )
    ranked = [_ranked_paper("p1", title, abstract, assessment, decision)]
    artifact_input = verified_report_input(
        ranked_papers=ranked,
        domain_assessments=[assessment],
        ranking_diagnostics={},
        paper_roles=[
            PaperRoleRecord(
                paper_id="p1",
                title=title,
                primary_role="experimental_characterization",
                roles=["experimental_characterization"],
            )
        ],
        provider_status=provider_status
        or {"openalex": {"status": "success", "returned_paper_count": 1}},
        exploration_quality={},
        search_contract=search_contract,
        research_gap_matrix=research_gap_matrix,
        suggested_next_searches=suggested_next_searches,
    )
    return render_user_report_markdown(artifact_input)


def _sei_no_clear_gap_rows():
    return build_research_gap_matrix(
        [
            _ranked_paper(
                "p1",
                "SEI formation and operando characterization in lithium-ion batteries",
                "SEI formation, progress, in situ, operando, evolution, ex situ, limitation, composition, structure, graphite, silicon, lithium metal, failure, degradation, cycling.",
            )
        ],
        [],
        search_contract=_contract(
            ["SEI", "solid electrolyte interphase"],
            ["lithium-ion battery", "composition", "structure", "evolution"],
        ),
        retrieval_context={
            "retrieval_status": "success",
            "merged_paper_count": 1,
            "ranked_papers_based_on_real_retrieval": True,
        },
    )


def _oer_no_clear_gap_rows():
    return build_research_gap_matrix(
        [
            _ranked_paper(
                "p1",
                "OER spin-state and electronic structure in perovskite catalysts",
                "OER oxygen evolution operando spin state electronic structure cobalt CoOOH perovskite transition metal oxyhydroxide mechanism activity performance surface reconstruction lattice oxygen mechanism adsorbate descriptor transfer.",
            )
        ],
        [],
        search_contract=_contract(
            ["oxygen evolution reaction", "spin state"],
            ["transition metal catalyst", "electronic structure"],
        ),
        retrieval_context={
            "retrieval_status": "success",
            "merged_paper_count": 1,
            "ranked_papers_based_on_real_retrieval": True,
        },
    )
