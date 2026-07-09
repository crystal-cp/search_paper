import csv
import json

import pytest

from tools.compare_ablations import SUMMARY_COLUMNS, compare_ablations
from tools.evaluate_run import evaluate_run, load_benchmark_cases
from tools.run_ablations import (
    ABLATION_CONFIGS,
    main as run_ablations_main,
    select_cases,
)


@pytest.fixture(scope="module")
def plan_ablation_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ablations")
    exit_code = run_ablations_main(
        [
            "--mode",
            "plan",
            "--configs",
            "full_system,no_query_family,no_query_repair,no_domain_guardrail,no_group_coverage_ranking",
            "--case-id",
            "ai_screening",
            "--output-root",
            str(root),
        ]
    )
    assert exit_code == 0
    return root


def _run_dir(root, config_name):
    return root / "ai_literature_screening" / config_name


@pytest.fixture(scope="module")
def weak_sei_repair_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("weak-sei-repair")
    exit_code = run_ablations_main(
        [
            "--mode",
            "plan",
            "--configs",
            "llm_query_critic_diagnostic_only,llm_query_critic_repair_applied",
            "--case-id",
            "weak_sei_acronym_query",
            "--output-root",
            str(root),
        ]
    )
    assert exit_code == 0
    return root


def _weak_sei_repair_run_dir(root):
    return root / "weak_sei_acronym_query" / "llm_query_critic_repair_applied"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_ranked_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "title",
        "abstract",
        "evidence_sentence",
        "decision",
        "reading_priority",
        "domain_decision",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_query_run(run_dir, case_id, config_name, queries):
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "ablation_config.json",
        {
            "ablation_config_name": config_name,
            "mode": "plan",
            "disabled_modules": [] if config_name == "full_system" else ["query_family"],
            "support_status": {},
        },
    )
    _write_json(
        run_dir / "planned_queries.json",
        {
            "query_family_applied": config_name != "no_query_family",
            "final_provider_queries": {"openalex": queries},
        },
    )
    return evaluate_run(run_dir, case_id=case_id)


def test_ablation_configs_load():
    assert "full_system" in ABLATION_CONFIGS
    assert ABLATION_CONFIGS["no_query_family"]["flags"] == ["--skip-query-families"]
    assert "--disable-query-repair" in ABLATION_CONFIGS["no_query_repair"]["flags"]
    assert "--disable-domain-guardrail" in ABLATION_CONFIGS["no_domain_guardrail"]["flags"]


def test_run_ablations_plan_mode_builds_outputs(plan_ablation_root):
    run_dir = _run_dir(plan_ablation_root, "full_system")

    assert (run_dir / "planned_queries.json").exists()
    assert (run_dir / "ablation_config.json").exists()
    assert (run_dir / "ablation_command.json").exists()


def test_ablation_runs_json_appends_multiple_invocations(tmp_path):
    root = tmp_path / "ablations"
    first = run_ablations_main(
        [
            "--mode",
            "plan",
            "--configs",
            "full_system",
            "--case-id",
            "ai_screening",
            "--output-root",
            str(root),
        ]
    )
    second = run_ablations_main(
        [
            "--mode",
            "plan",
            "--configs",
            "full_system",
            "--case-id",
            "mof_co2",
            "--output-root",
            str(root),
        ]
    )

    assert first == 0
    assert second == 0
    records = json.loads((root / "ablation_runs.json").read_text(encoding="utf-8"))
    observed = {(record["case_id"], record["config_name"]) for record in records}
    assert ("ai_literature_screening", "full_system") in observed
    assert ("mof_co2_capture", "full_system") in observed
    for record in records:
        assert "command" in record
        assert "returncode" in record
        assert "stdout_tail" in record
        assert "stderr_tail" in record


def test_compare_ablations_generates_csv_and_md(plan_ablation_root, tmp_path):
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"

    rows = compare_ablations(plan_ablation_root, csv_path=csv_path, markdown_path=md_path)

    assert rows
    assert csv_path.exists()
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8").lower()
    assert "pilot / diagnostic" in text
    assert "not a final experimental conclusion" in text


def test_ablation_summary_contains_forbidden_pattern_counts(plan_ablation_root, tmp_path):
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"
    compare_ablations(plan_ablation_root, csv_path=csv_path, markdown_path=md_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert "forbidden_pattern_top10_count" in reader.fieldnames
        assert "forbidden_pattern_top20_count" in reader.fieldnames
        assert "forbidden_pattern_must_read_count" in reader.fieldnames
        assert "forbidden_pattern_include_count" in reader.fieldnames


def test_forbidden_patterns_detect_sodium_potassium_in_sei_titles(tmp_path):
    run_dir = tmp_path / "sei_lithium_battery" / "full_system"
    _write_ranked_csv(
        run_dir / "ranked_papers.csv",
        [
            {
                "title": "Solubility of the Solid Electrolyte Interphase in Sodium Ion Batteries",
                "decision": "include",
                "reading_priority": "must_read",
                "domain_decision": "in_scope",
            },
            {
                "title": "Artificial SEI Membranes for Potassium-Ion Battery Anodes",
                "decision": "include",
                "reading_priority": "must_read",
                "domain_decision": "in_scope",
            },
        ],
    )

    metrics = evaluate_run(run_dir, case_id="sei_lithium_battery")

    assert metrics["forbidden_pattern_top10_count"] == 2
    assert metrics["forbidden_pattern_top20_count"] == 2
    assert metrics["forbidden_pattern_include_count"] == 2


def test_forbidden_patterns_detect_mixed_lithium_sodium_potassium_title(tmp_path):
    run_dir = tmp_path / "sei_lithium_battery" / "full_system"
    _write_ranked_csv(
        run_dir / "ranked_papers.csv",
        [
            {
                "title": "Review of Emerging Concepts in SEI Analysis and Artificial SEI Membranes for Lithium, Sodium, and Potassium Metal Battery Anodes",
                "decision": "include",
                "reading_priority": "must_read",
                "domain_decision": "in_scope",
            }
        ],
    )

    metrics = evaluate_run(run_dir, case_id="sei_lithium_battery")

    assert metrics["forbidden_pattern_top10_count"] == 1
    assert metrics["forbidden_pattern_must_read_count"] == 1


def test_forbidden_must_read_count_nonzero_when_must_read_matches_forbidden(tmp_path):
    run_dir = tmp_path / "sei_lithium_battery" / "full_system"
    _write_ranked_csv(
        run_dir / "ranked_papers.csv",
        [
            {
                "title": "HTML <b>Sodium-Ion</b> Battery SEI Review",
                "abstract": "Matched evidence discusses sodium ion interphase chemistry.",
                "evidence_sentence": "Sodium-ion SEI chemistry differs from lithium metal systems.",
                "decision": "include",
                "reading_priority": "must_read",
                "domain_decision": "in_scope",
            }
        ],
    )

    metrics = evaluate_run(run_dir, case_id="sei_lithium_battery")

    assert metrics["forbidden_pattern_must_read_count"] == 1
    assert metrics["forbidden_pattern_include_count"] == 1


def test_query_quality_flags_mof_overbroad_queries(tmp_path):
    run_dir = tmp_path / "mof_co2_capture" / "no_query_family"

    metrics = _write_query_run(
        run_dir,
        "mof_co2_capture",
        "no_query_family",
        ["CO2 MOF", "MOF capture", "MOF MOF CO2 capture MOF CO2"],
    )

    assert metrics["overbroad_query_count"] >= 2
    assert metrics["weak_query_count"] >= 3
    assert metrics["repeated_phrase_query_count"] >= 1


def test_query_quality_flags_thin_film_weak_queries(tmp_path):
    run_dir = tmp_path / "thin_film_deposition" / "no_query_family"

    metrics = _write_query_run(
        run_dir,
        "thin_film_deposition",
        "no_query_family",
        ["thin film deposition", "ALD thin film deposition"],
    )

    assert metrics["weak_query_count"] >= 1
    assert metrics["single_axis_query_count"] >= 1


def test_query_quality_flags_ai_weak_queries(tmp_path):
    run_dir = tmp_path / "ai_literature_screening" / "no_query_family"

    metrics = _write_query_run(
        run_dir,
        "ai_literature_screening",
        "no_query_family",
        ["literature screening", "human-in-the-loop"],
    )

    assert metrics["weak_query_count"] == 2
    assert metrics["overbroad_query_count"] >= 2


def test_ablation_summary_uses_query_quality_not_only_query_count(tmp_path):
    root = tmp_path / "ablations"
    _write_query_run(
        root / "mof_co2_capture" / "full_system",
        "mof_co2_capture",
        "full_system",
        [
            'MOF "CO2 adsorption" "pore size"',
            'MOF "CO2 capture" "functional groups"',
            'MOF "CO2 adsorption" "water stability"',
        ],
    )
    _write_query_run(
        root / "mof_co2_capture" / "no_query_family",
        "mof_co2_capture",
        "no_query_family",
        ["CO2 MOF", "MOF capture", "MOF MOF CO2 capture MOF CO2"],
    )
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"

    compare_ablations(root, csv_path=csv_path, markdown_path=md_path)

    text = md_path.read_text(encoding="utf-8")
    assert "`no_query_family` degraded query quality" in text
    assert "weak_query_count" in text


def test_no_query_repair_summary_marks_non_conclusive_when_sanitizer_active(tmp_path):
    root = tmp_path / "ablations"
    full_run = root / "ai_literature_screening" / "full_system"
    no_repair_run = root / "ai_literature_screening" / "no_query_repair"
    _write_query_run(
        full_run,
        "ai_literature_screening",
        "full_system",
        ['LLM "literature screening" "human feedback"'],
    )
    _write_query_run(
        no_repair_run,
        "ai_literature_screening",
        "no_query_repair",
        ['LLM "literature screening" "human feedback"'],
    )
    _write_json(
        no_repair_run / "ablation_config.json",
        {
            "ablation_config_name": "no_query_repair",
            "mode": "plan",
            "disabled_modules": ["query_repair"],
            "support_status": {"query_repair": "supported"},
        },
    )
    _write_json(
        no_repair_run / "query_repair_stage_status.json",
        {
            "repair_enabled": False,
            "repair_applied": False,
            "disabled_by_ablation": True,
            "raw_candidate_query_count": 1,
            "final_query_count": 1,
            "dropped_query_count": 0,
            "repaired_query_count": 0,
            "added_query_count": 0,
            "reason_if_no_difference": "repair stage disabled but upstream sanitizer still applied",
        },
    )
    _write_json(
        no_repair_run / "raw_candidate_queries_before_repair.json",
        {"queries": ['LLM "literature screening" "human feedback"']},
    )
    _write_json(
        no_repair_run / "final_queries_after_repair.json",
        {"queries": ['LLM "literature screening" "human feedback"']},
    )
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"

    rows = compare_ablations(root, csv_path=csv_path, markdown_path=md_path)

    no_repair = next(row for row in rows if row["config_name"] == "no_query_repair")
    assert no_repair["repair_disabled_but_sanitizer_active"] is True
    assert no_repair["no_query_repair_conclusive"] is False
    text = md_path.read_text(encoding="utf-8")
    assert "diagnostic only because upstream sanitizer remains active" in text
    assert "do not conclude QueryRepair has no value" in text


def test_ablation_config_name_consistency(tmp_path):
    root = tmp_path / "ablations"
    run_dir = root / "mof_co2_capture" / "directory_name"
    _write_query_run(
        run_dir,
        "mof_co2_capture",
        "directory_name",
        ['MOF "CO2 adsorption" "pore size"'],
    )
    _write_json(
        run_dir / "ablation_config.json",
        {
            "ablation_config_name": "no_query_family",
            "config_name": "legacy_wrong_name",
            "mode": "plan",
            "disabled_modules": ["query_family"],
        },
    )

    rows = compare_ablations(
        root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=tmp_path / "reports" / "ablation_summary.md",
    )

    assert rows[0]["config_name"] == "no_query_family"


def test_disable_query_family_records_query_family_applied_false(plan_ablation_root):
    planned = json.loads(
        (_run_dir(plan_ablation_root, "no_query_family") / "planned_queries.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(
        (_run_dir(plan_ablation_root, "no_query_family") / "ablation_config.json").read_text(
            encoding="utf-8"
        )
    )

    assert planned["query_family_applied"] is False
    assert "query_family" in config["disabled_modules"]


def test_query_repair_stage_status_written(plan_ablation_root):
    status_path = _run_dir(plan_ablation_root, "full_system") / "query_repair_stage_status.json"
    assert status_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))

    for key in [
        "repair_enabled",
        "repair_applied",
        "disabled_by_ablation",
        "raw_candidate_query_count",
        "final_query_count",
        "dropped_query_count",
        "repaired_query_count",
        "added_query_count",
        "reason_if_no_difference",
    ]:
        assert key in status


def test_no_query_repair_records_disabled_by_ablation(plan_ablation_root):
    run_dir = _run_dir(plan_ablation_root, "no_query_repair")
    config = json.loads((run_dir / "ablation_config.json").read_text(encoding="utf-8"))
    repair = json.loads((run_dir / "query_repair_suggestions.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "query_repair_stage_status.json").read_text(encoding="utf-8"))

    assert "--disable-query-repair" in config["cli_flags_used"]
    assert "query_repair" in config["disabled_modules"]
    assert repair["disabled_by_ablation"] is True
    assert status["disabled_by_ablation"] is True
    assert status["repair_enabled"] is False


def test_disable_query_repair_flag_is_recorded(plan_ablation_root):
    run_dir = _run_dir(plan_ablation_root, "no_query_repair")
    config = json.loads((run_dir / "ablation_config.json").read_text(encoding="utf-8"))
    repair = json.loads((run_dir / "query_repair_suggestions.json").read_text(encoding="utf-8"))

    assert "--disable-query-repair" in config["cli_flags_used"]
    assert "query_repair" in config["disabled_modules"]
    assert repair["disabled_by_ablation"] is True


def test_disable_domain_guardrail_flag_is_recorded(plan_ablation_root):
    config = json.loads(
        (_run_dir(plan_ablation_root, "no_domain_guardrail") / "ablation_config.json").read_text(
            encoding="utf-8"
        )
    )

    assert "--disable-domain-guardrail" in config["cli_flags_used"]
    assert "domain_guardrail" in config["disabled_modules"]


def test_ablation_config_json_written(plan_ablation_root):
    config = json.loads(
        (_run_dir(plan_ablation_root, "full_system") / "ablation_config.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["ablation_config_name"] == "full_system"
    assert config["disabled_modules"] == []
    assert "query_family" in config["enabled_modules"]


def test_partially_supported_ablations_marked_in_summary(plan_ablation_root, tmp_path):
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"
    compare_ablations(plan_ablation_root, csv_path=csv_path, markdown_path=md_path)

    text = md_path.read_text(encoding="utf-8")
    assert "Partially supported ablations" in text
    assert "no_group_coverage_ranking" in text
    assert "partially_supported" in text


def test_compare_ablations_handles_missing_or_partial_metrics(tmp_path):
    root = tmp_path / "ablations"
    run_dir = root / "synthetic_case" / "no_group_coverage_ranking"
    run_dir.mkdir(parents=True)
    (run_dir / "ablation_config.json").write_text(
        json.dumps(
            {
                "ablation_config_name": "no_group_coverage_ranking",
                "mode": "full",
                "disabled_modules": ["group_coverage_ranking"],
                "support_status": {"group_coverage_ranking": "partially_supported"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"

    rows = compare_ablations(root, csv_path=csv_path, markdown_path=md_path)

    assert len(rows) == 1
    assert csv_path.exists()
    assert md_path.exists()
    assert "partially_supported" in rows[0]["notes"]


def test_ablation_summary_contains_expected_columns(plan_ablation_root, tmp_path):
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"
    compare_ablations(plan_ablation_root, csv_path=csv_path, markdown_path=md_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == SUMMARY_COLUMNS


def test_compare_ablations_includes_reading_path_diagnostics(tmp_path):
    root = tmp_path / "ablations"
    run_dir = root / "sei_lithium_battery" / "full_system"
    _write_query_run(
        run_dir,
        "sei_lithium_battery",
        "full_system",
        ['SEI "lithium metal battery" "artificial SEI"'],
    )
    _write_json(
        run_dir / "ablation_config.json",
        {
            "ablation_config_name": "full_system",
            "mode": "full",
            "disabled_modules": [],
            "support_status": {},
        },
    )
    _write_json(
        run_dir / "evaluation.json",
        {
            "reading_path_diagnostics": {
                "reading_path_paper_count": 12,
                "reading_path_exclude_count": 0,
                "reading_path_out_of_scope_count": 0,
                "reading_path_duplicate_count": 0,
                "reading_path_negative_context_count": 0,
            },
            "reading_priority_policy": {
                "target_context_required_for_priority": True,
            },
        },
    )
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"

    rows = compare_ablations(root, csv_path=csv_path, markdown_path=md_path)

    assert rows[0]["reading_path_paper_count"] == 12
    assert rows[0]["reading_path_exclude_count"] == 0
    assert rows[0]["reading_path_out_of_scope_count"] == 0
    assert rows[0]["reading_path_duplicate_count"] == 0
    assert rows[0]["reading_path_negative_context_count"] == 0
    assert rows[0]["target_context_required_for_priority"] is True
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["reading_path_paper_count"] == "12"
    assert csv_row["reading_path_exclude_count"] == "0"
    assert csv_row["target_context_required_for_priority"] == "true"
    text = md_path.read_text(encoding="utf-8")
    assert "| full_system | 12 | 0 | 0 | 0 | 0 | true |" in text


def test_plan_only_summary_leaves_reading_path_diagnostics_empty(tmp_path):
    root = tmp_path / "ablations"
    run_dir = root / "ai_literature_screening" / "full_system"
    _write_query_run(
        run_dir,
        "ai_literature_screening",
        "full_system",
        ['LLM "literature screening" "human feedback"'],
    )
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"

    rows = compare_ablations(root, csv_path=csv_path, markdown_path=md_path)

    assert rows[0]["reading_path_paper_count"] is None
    assert rows[0]["reading_path_exclude_count"] is None
    assert rows[0]["reading_path_out_of_scope_count"] is None
    assert rows[0]["reading_path_duplicate_count"] is None
    assert rows[0]["reading_path_negative_context_count"] is None
    assert rows[0]["target_context_required_for_priority"] is None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["reading_path_paper_count"] == ""
    assert csv_row["reading_path_exclude_count"] == ""
    assert csv_row["target_context_required_for_priority"] == ""


def test_full_summary_reads_target_context_required_for_priority(tmp_path):
    root = tmp_path / "ablations"
    run_dir = root / "oer_spin_state" / "full_system"
    _write_query_run(
        run_dir,
        "oer_spin_state",
        "full_system",
        ['OER "spin state" "electronic structure"'],
    )
    _write_json(
        run_dir / "ablation_config.json",
        {
            "ablation_config_name": "full_system",
            "mode": "full",
            "disabled_modules": [],
            "support_status": {},
        },
    )
    _write_json(
        run_dir / "evaluation.json",
        {
            "reading_path_paper_count": 11,
            "reading_path_exclude_count": 0,
            "reading_path_out_of_scope_count": 0,
            "reading_path_duplicate_count": 0,
            "reading_path_negative_context_count": 0,
            "target_context_required_for_priority": False,
        },
    )
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"

    rows = compare_ablations(root, csv_path=csv_path, markdown_path=md_path)

    assert rows[0]["reading_path_paper_count"] == 11
    assert rows[0]["target_context_required_for_priority"] is False
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["reading_path_paper_count"] == "11"
    assert csv_row["target_context_required_for_priority"] == "false"


def _write_llm_plan_run(
    root,
    *,
    case_id="ai_literature_screening",
    config_name="llm_query_critic_diagnostic_only",
    intent_trace=None,
    critic_trace=None,
    repair_artifact=None,
    case_metadata=None,
    before_queries=None,
    after_queries=None,
):
    run_dir = root / case_id / config_name
    _write_query_run(
        run_dir,
        case_id,
        config_name,
        ['LLM "literature screening"', 'LLM "literature screening"'],
    )
    config = ABLATION_CONFIGS[config_name]
    _write_json(
        run_dir / "ablation_config.json",
        {
            "ablation_config_name": config_name,
            "mode": "plan",
            "disabled_modules": config.get("disabled_modules", []),
            "enabled_modules": config.get("enabled_modules_extra", []),
            "support_status": config.get("support_status", {}),
            "mode_support": config.get("mode_support"),
            "applies_query_changes": config.get("applies_query_changes"),
            "uses_real_llm": config.get("uses_real_llm"),
            "diagnostic_only": config.get("diagnostic_only"),
            "warning_if_pilot_only": config.get("warning_if_pilot_only"),
            "case_metadata": case_metadata or {},
        },
    )
    _write_json(
        run_dir / "retrieval_diagnostics.json",
        {
            "retrieval_status": "planning_only",
            "ranked_papers_based_on_real_retrieval": False,
            "raw_retrieved_paper_count": 0,
            "merged_paper_count": 0,
        },
    )
    _write_json(
        run_dir / "evaluation.json",
        {
            "retrieval_status": "planning_only",
            "ranked_papers_based_on_real_retrieval": False,
        },
    )
    if intent_trace is not None:
        _write_json(run_dir / "llm_intent_enhancement_trace.json", intent_trace)
    if critic_trace is not None:
        _write_json(run_dir / "llm_query_critic_trace.json", critic_trace)
    if repair_artifact is not None:
        _write_json(run_dir / "query_repair_after_llm_critic.json", repair_artifact)
    if before_queries is not None:
        _write_json(
            run_dir / "query_plan_before_llm_critic.json",
            {"queries": before_queries},
        )
    if after_queries is not None:
        _write_json(
            run_dir / "query_plan_after_llm_critic.json",
            {"queries": after_queries},
        )
    return run_dir


def test_llm_plan_ablation_configs_exist():
    for config_name in [
        "full_system",
        "llm_intent_frame_only",
        "llm_query_critic_diagnostic_only",
        "llm_query_critic_repair_applied",
        "llm_intent_plus_query_critic_repair",
    ]:
        assert config_name in ABLATION_CONFIGS


def test_llm_plan_ablation_configs_mark_pilot_or_diagnostic():
    assert ABLATION_CONFIGS["full_system"]["support_status"]["full_system"] == "baseline"
    for config_name in [
        "llm_intent_frame_only",
        "llm_query_critic_diagnostic_only",
        "llm_query_critic_repair_applied",
        "llm_intent_plus_query_critic_repair",
    ]:
        config = ABLATION_CONFIGS[config_name]
        assert config["mode_support"] == "plan_level"
        assert config["uses_real_llm"] is False
        assert config["warning_if_pilot_only"]


def test_weak_plan_cases_are_diagnostic_stress_cases():
    benchmark = load_benchmark_cases()
    weak_cases = {
        case["id"]: case
        for case in benchmark["cases"]
        if str(case.get("llm_plan_diagnostic_group") or "")
        == "weak_plan_positive_control"
    }

    assert {
        "weak_sei_acronym_query",
        "weak_oer_acronym_query",
        "weak_mof_short_query",
    }.issubset(weak_cases)
    for case in weak_cases.values():
        assert case["diagnostic_only"] is True
        assert case["stress_case"] is True
        assert case["formal_benchmark_inclusion"] == "not part of formal benchmark conclusion"


def test_run_ablations_default_plan_group_excludes_weak_controls():
    benchmark = load_benchmark_cases()

    clean_cases = select_cases(benchmark, None, "plan")
    weak_cases = select_cases(benchmark, None, "plan", "weak_plan_positive_controls")

    assert all(not case.get("stress_case") for case in clean_cases)
    assert {case["id"] for case in clean_cases} == {
        "ai_literature_screening",
        "mof_co2_capture",
        "thin_film_deposition",
        "sei_lithium_battery",
        "oer_spin_state",
    }
    assert {case["id"] for case in weak_cases} >= {
        "weak_sei_acronym_query",
        "weak_oer_acronym_query",
        "weak_mof_short_query",
    }


def test_llm_query_critic_diagnostic_config_does_not_apply_repairs():
    config = ABLATION_CONFIGS["llm_query_critic_diagnostic_only"]
    assert "--enable-llm-query-critic" in config["flags"]
    assert "--apply-llm-query-critic-repairs" not in config["flags"]
    assert config["applies_query_changes"] is False
    assert config["diagnostic_only"] is True


def test_llm_query_critic_repair_config_applies_only_with_apply_flag():
    diagnostic = ABLATION_CONFIGS["llm_query_critic_diagnostic_only"]
    repair = ABLATION_CONFIGS["llm_query_critic_repair_applied"]
    assert "--apply-llm-query-critic-repairs" not in diagnostic["flags"]
    assert "--apply-llm-query-critic-repairs" in repair["flags"]
    assert repair["applies_query_changes"] is True


def test_llm_intent_plus_query_critic_config_enables_both_modules():
    config = ABLATION_CONFIGS["llm_intent_plus_query_critic_repair"]
    assert "--enable-llm-intent-enhancer" in config["flags"]
    assert "--enable-llm-query-critic" in config["flags"]
    assert "--apply-llm-query-critic-repairs" in config["flags"]
    assert "llm_intent_frame_enhancer" in config["enabled_modules_extra"]
    assert "llm_query_plan_critic" in config["enabled_modules_extra"]


def test_llm_intent_frame_fake_positive_called_when_configured(tmp_path):
    root = tmp_path / "ablations"
    exit_code = run_ablations_main(
        [
            "--mode",
            "plan",
            "--configs",
            "llm_intent_frame_only",
            "--case-id",
            "sei_lithium",
            "--output-root",
            str(root),
        ]
    )

    assert exit_code == 0
    trace = json.loads(
        (
            root
            / "sei_lithium_battery"
            / "llm_intent_frame_only"
            / "llm_intent_enhancement_trace.json"
        ).read_text(encoding="utf-8")
    )
    assert trace["llm_enabled"] is True
    assert trace["llm_called"] is True
    assert trace["fallback_used"] is False
    assert trace["verified_candidate_count"] > 0
    assert trace["applied_suggestion_count"] > 0


def test_compare_ablations_includes_llm_intent_metrics(tmp_path):
    root = tmp_path / "ablations"
    _write_llm_plan_run(
        root,
        config_name="llm_intent_frame_only",
        intent_trace={
            "llm_enabled": True,
            "llm_called": True,
            "fallback_used": False,
            "verified_candidate_count": 2,
            "applied_suggestion_count": 1,
            "rejected_suggestion_count": 1,
            "unsupported_suggestion_count": 0,
        },
    )
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"

    rows = compare_ablations(root, csv_path=csv_path, markdown_path=md_path)

    row = rows[0]
    assert row["llm_intent_enabled"] is True
    assert row["llm_intent_called"] is True
    assert row["llm_intent_verified_candidate_count"] == 2
    assert row["llm_intent_applied_suggestion_count"] == 1
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert "llm_intent_verified_candidate_count" in reader.fieldnames


def test_compare_ablations_includes_llm_query_critic_metrics(tmp_path):
    root = tmp_path / "ablations"
    _write_llm_plan_run(
        root,
        config_name="llm_query_critic_repair_applied",
        critic_trace={
            "llm_query_critic_enabled": True,
            "llm_called": True,
            "fallback_used": False,
            "verified_issue_count": 1,
            "rejected_issue_count": 0,
            "unsupported_issue_count": 0,
            "applied_issue_count": 1,
            "rejected_for_application_count": 0,
            "query_added_count": 0,
            "query_dropped_count": 0,
            "query_modified_count": 1,
        },
        repair_artifact={
            "apply_enabled": True,
            "applied_issue_count": 1,
            "applied_issue_records": [
                {
                    "applied_terms": ["solid electrolyte interphase"],
                    "rejected_terms": [{"term": "lithium battery"}],
                }
            ],
        },
    )
    csv_path = tmp_path / "reports" / "ablation_summary.csv"
    md_path = tmp_path / "reports" / "ablation_summary.md"

    rows = compare_ablations(root, csv_path=csv_path, markdown_path=md_path)

    row = rows[0]
    assert row["llm_query_critic_enabled"] is True
    assert row["verified_issue_count"] == 1
    assert row["applied_issue_count"] == 1
    assert row["query_modified_count"] == 1
    assert row["repair_grounded_term_count"] == 1
    assert row["repair_rejected_term_count"] == 1
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert "verified_issue_count" in reader.fieldnames
        assert "repair_grounded_term_count" in reader.fieldnames


def test_plan_level_llm_ablation_does_not_generate_fake_gaps(tmp_path):
    root = tmp_path / "ablations"
    _write_llm_plan_run(root, config_name="llm_query_critic_diagnostic_only")
    rows = compare_ablations(
        root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=tmp_path / "reports" / "ablation_summary.md",
    )

    assert rows[0]["retrieval_performed"] is False
    assert rows[0]["ranked_papers_based_on_real_retrieval"] is False
    assert rows[0]["research_gap_generation_status"] == "skipped"


def test_plan_level_llm_ablation_does_not_mutate_paper_decisions(tmp_path):
    root = tmp_path / "ablations"
    _write_llm_plan_run(root, config_name="llm_query_critic_repair_applied")
    rows = compare_ablations(
        root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=tmp_path / "reports" / "ablation_summary.md",
    )

    assert rows[0]["paper_decision_mutation_count"] == 0


def test_llm_plan_ablation_clean_case_no_verified_opportunity_is_not_failure(tmp_path):
    root = tmp_path / "ablations"
    _write_llm_plan_run(
        root,
        config_name="llm_query_critic_diagnostic_only",
        critic_trace={
            "llm_query_critic_enabled": True,
            "llm_called": True,
            "fallback_used": False,
            "verified_issue_count": 0,
            "rejected_issue_count": 1,
            "applied_issue_count": 0,
        },
    )

    rows = compare_ablations(
        root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=tmp_path / "reports" / "ablation_summary.md",
    )

    assert "No verified repair opportunity; clean plan remained unchanged." in rows[0]["notes"]


def test_llm_plan_ablation_repair_enabled_but_no_verified_issue_notes_no_opportunity(tmp_path):
    root = tmp_path / "ablations"
    _write_llm_plan_run(
        root,
        config_name="llm_query_critic_repair_applied",
        critic_trace={
            "llm_query_critic_enabled": True,
            "llm_called": True,
            "fallback_used": False,
            "verified_issue_count": 0,
            "rejected_issue_count": 1,
            "applied_issue_count": 0,
        },
    )

    rows = compare_ablations(
        root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=tmp_path / "reports" / "ablation_summary.md",
    )

    assert "Repair flag enabled, but no verified grounded issue was available." in rows[0]["notes"]
    assert "No verified repair opportunity; clean plan remained unchanged." in rows[0]["notes"]


def test_llm_intent_frame_fallback_summary_not_counted_as_positive_llm_effect(tmp_path):
    root = tmp_path / "ablations"
    _write_llm_plan_run(
        root,
        config_name="llm_intent_frame_only",
        intent_trace={
            "llm_enabled": True,
            "llm_called": False,
            "fallback_used": True,
            "reason_if_no_change": "llm_provider_unavailable",
        },
    )

    rows = compare_ablations(
        root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=tmp_path / "reports" / "ablation_summary.md",
    )

    assert "Intent enhancer fallback only; not a positive LLM intent test." in rows[0]["notes"]


def test_summary_distinguishes_clean_diagnostic_from_positive_control(tmp_path):
    root = tmp_path / "ablations"
    _write_llm_plan_run(
        root,
        config_name="llm_query_critic_diagnostic_only",
        critic_trace={
            "llm_query_critic_enabled": True,
            "llm_called": True,
            "verified_issue_count": 0,
            "applied_issue_count": 0,
        },
    )
    _write_llm_plan_run(
        root,
        case_id="weak_sei_acronym_query",
        config_name="llm_query_critic_repair_applied",
        critic_trace={
            "llm_query_critic_enabled": True,
            "llm_called": True,
            "verified_issue_count": 1,
            "applied_issue_count": 1,
            "query_modified_count": 1,
        },
        repair_artifact={
            "apply_enabled": True,
            "applied_issue_count": 1,
            "applied_issue_records": [
                {
                    "applied_terms": ["solid electrolyte interphase"],
                    "rejected_terms": [{"term": "lithium battery"}],
                }
            ],
        },
        case_metadata={
            "diagnostic_group": "weak_plan_positive_control",
            "diagnostic_only": True,
            "stress_case": True,
            "formal_benchmark_inclusion": "not part of formal benchmark conclusion",
        },
        before_queries=["SEI"],
        after_queries=['SEI "solid electrolyte interphase"'],
    )
    md_path = tmp_path / "reports" / "ablation_summary.md"

    compare_ablations(
        root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=md_path,
    )

    text = md_path.read_text(encoding="utf-8")
    assert "No verified repair opportunity; clean plan remained unchanged." in text
    assert "Weak-plan positive controls:" in text
    assert "weak_sei_acronym_query / llm_query_critic_repair_applied" in text
    assert 'before=SEI, after=SEI "solid electrolyte interphase"' in text


def test_llm_query_critic_repair_updates_final_queries_after_repair(weak_sei_repair_root):
    run_dir = _weak_sei_repair_run_dir(weak_sei_repair_root)
    final_queries = json.loads(
        (run_dir / "final_queries_after_repair.json").read_text(encoding="utf-8")
    )

    assert 'sei SEI "solid electrolyte interphase"' in final_queries["queries"]
    assert "sei SEI" not in final_queries["queries"]
    assert (
        'sei SEI "solid electrolyte interphase"'
        in final_queries["queries_by_provider"]["openalex"]
    )
    assert "sei SEI" not in final_queries["queries_by_provider"]["openalex"]


def test_llm_query_critic_repair_updates_query_provenance(weak_sei_repair_root):
    run_dir = _weak_sei_repair_run_dir(weak_sei_repair_root)
    provenance = json.loads(
        (run_dir / "query_provenance.json").read_text(encoding="utf-8")
    )
    record = provenance["llm_query_critic_repairs"][0]

    assert (
        'sei SEI "solid electrolyte interphase"'
        in provenance["final_provider_queries"]["openalex"]
    )
    assert "sei SEI" not in provenance["final_provider_queries"]["openalex"]
    assert record["source"] == "llm_query_critic_suggested_rule_applied"
    assert record["issue_type"] == "single_acronym_query"
    assert record["suggested_action"] == "strengthen_query"
    assert record["original_query"] == "sei SEI"
    assert record["new_query"] == 'sei SEI "solid electrolyte interphase"'
    assert record["applied_terms"] == ["solid electrolyte interphase"]
    assert record["rejected_terms"][0]["term"] == "lithium battery"
    assert record["term_grounding"][0]["grounding_source"]
    assert record["term_grounding"][0]["grounding_reason"]
    assert record["verifier_reason"]


def test_llm_query_critic_repair_updates_query_repair_stage_status(weak_sei_repair_root):
    run_dir = _weak_sei_repair_run_dir(weak_sei_repair_root)
    status = json.loads(
        (run_dir / "query_repair_stage_status.json").read_text(encoding="utf-8")
    )

    assert status["deterministic_query_repair_enabled"] is False
    assert status["llm_query_critic_repair_enabled"] is True
    assert status["llm_query_critic_repair_applied"] is True
    assert status["repair_applied"] is True
    assert status["query_modified_count"] > 0
    assert (
        status["reason_if_no_difference"]
        != "repair stage disabled but upstream sanitizer still applied"
    )


def test_llm_query_critic_repair_updates_evaluation_query_repair_section(weak_sei_repair_root):
    run_dir = _weak_sei_repair_run_dir(weak_sei_repair_root)
    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    query_repair = evaluation["query_repair"]

    assert query_repair["deterministic_query_repair_enabled"] is False
    assert query_repair["llm_query_critic_repair_enabled"] is True
    assert query_repair["llm_query_critic_repair_applied"] is True
    assert query_repair["llm_query_critic_applied_issue_count"] == 1
    assert query_repair["query_modified_count"] == 1
    assert query_repair["repair_grounded_term_count"] == 1
    assert query_repair["repair_rejected_term_count"] == 1


def test_llm_query_critic_repair_artifacts_are_consistent(weak_sei_repair_root):
    run_dir = _weak_sei_repair_run_dir(weak_sei_repair_root)
    metrics = evaluate_run(run_dir, case_id="weak_sei_acronym_query")

    assert metrics["llm_query_critic_repair_artifact_consistent"] is True
    assert metrics["final_query_artifact_consistent"] is True
    assert metrics["llm_query_critic_repair_provenance_count"] == 1


def test_compare_ablations_reports_llm_repair_examples(weak_sei_repair_root, tmp_path):
    md_path = tmp_path / "reports" / "ablation_summary.md"

    rows = compare_ablations(
        weak_sei_repair_root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=md_path,
    )

    repair_row = next(
        row for row in rows if row["config_name"] == "llm_query_critic_repair_applied"
    )
    assert repair_row["llm_query_critic_original_query_example"] == "sei SEI"
    assert (
        repair_row["llm_query_critic_repaired_query_example"]
        == 'sei SEI "solid electrolyte interphase"'
    )
    text = md_path.read_text(encoding="utf-8")
    assert "applied_terms=solid electrolyte interphase" in text
    assert "artifact_consistent=true" in text


def test_summary_does_not_claim_quality_improvement_when_score_drops(
    weak_sei_repair_root,
    tmp_path,
):
    md_path = tmp_path / "reports" / "ablation_summary.md"

    compare_ablations(
        weak_sei_repair_root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=md_path,
    )

    text = md_path.read_text(encoding="utf-8")
    assert (
        "Repair was applied and provenance is valid, but query_quality_score did not improve "
        "under current heuristic; this should be interpreted as repair-mechanism validation, "
        "not quality improvement."
    ) in text


def test_final_query_artifact_consistent_metric(tmp_path):
    run_dir = tmp_path / "weak_sei_acronym_query" / "llm_query_critic_repair_applied"
    _write_query_run(
        run_dir,
        "weak_sei_acronym_query",
        "llm_query_critic_repair_applied",
        ['sei SEI "solid electrolyte interphase"'],
    )
    _write_json(
        run_dir / "final_queries_after_repair.json",
        {"queries_by_provider": {"openalex": ["sei SEI"]}, "queries": ["sei SEI"]},
    )
    _write_json(
        run_dir / "query_provenance.json",
        {
            "final_provider_queries": {
                "openalex": ['sei SEI "solid electrolyte interphase"']
            },
            "llm_query_critic_repairs": [{"source": "llm_query_critic_suggested_rule_applied"}],
        },
    )

    metrics = evaluate_run(run_dir, case_id="weak_sei_acronym_query")

    assert metrics["final_query_artifact_consistent"] is False


def test_weak_sei_acronym_positive_control_applies_repair(tmp_path):
    root = tmp_path / "ablations"
    exit_code = run_ablations_main(
        [
            "--mode",
            "plan",
            "--configs",
            "llm_query_critic_repair_applied",
            "--case-id",
            "weak_sei_acronym_query",
            "--output-root",
            str(root),
        ]
    )

    assert exit_code == 0
    run_dir = root / "weak_sei_acronym_query" / "llm_query_critic_repair_applied"
    trace = json.loads((run_dir / "llm_query_critic_trace.json").read_text(encoding="utf-8"))
    repair = json.loads((run_dir / "query_repair_after_llm_critic.json").read_text(encoding="utf-8"))

    assert trace["verified_issue_count"] > 0
    assert trace["applied_issue_count"] > 0
    assert trace["query_modified_count"] + trace["query_added_count"] > 0
    assert repair["applied_issue_records"][0]["source"] == "llm_query_critic_suggested_rule_applied"
    assert repair["applied_issue_records"][0]["applied_terms"]


def test_weak_oer_acronym_positive_control_applies_repair(tmp_path):
    root = tmp_path / "ablations"
    exit_code = run_ablations_main(
        [
            "--mode",
            "plan",
            "--configs",
            "llm_query_critic_repair_applied",
            "--case-id",
            "weak_oer_acronym_query",
            "--output-root",
            str(root),
        ]
    )

    assert exit_code == 0
    run_dir = root / "weak_oer_acronym_query" / "llm_query_critic_repair_applied"
    trace = json.loads((run_dir / "llm_query_critic_trace.json").read_text(encoding="utf-8"))
    repair = json.loads((run_dir / "query_repair_after_llm_critic.json").read_text(encoding="utf-8"))

    assert trace["verified_issue_count"] > 0
    assert trace["applied_issue_count"] > 0
    assert trace["query_modified_count"] + trace["query_added_count"] > 0
    record = repair["applied_issue_records"][0]
    assert record["source"] == "llm_query_critic_suggested_rule_applied"
    assert "oxygen evolution reaction" in record["applied_terms"]


def test_llm_plan_ablation_summary_warns_not_formal_full_ablation(tmp_path):
    root = tmp_path / "ablations"
    _write_llm_plan_run(
        root,
        config_name="llm_intent_plus_query_critic_repair",
        intent_trace={"llm_enabled": True, "llm_called": True},
        critic_trace={"llm_query_critic_enabled": True, "llm_called": True},
    )
    md_path = tmp_path / "reports" / "ablation_summary.md"

    compare_ablations(
        root,
        csv_path=tmp_path / "reports" / "ablation_summary.csv",
        markdown_path=md_path,
    )

    text = md_path.read_text(encoding="utf-8")
    assert "LLM plan-level diagnostic summary" in text
    assert "does not prove full retrieval improvement" in text
    assert "LLM modules do not make paper-level decisions" in text
