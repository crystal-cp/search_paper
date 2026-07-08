import csv
import json

import pytest

from tools.compare_ablations import SUMMARY_COLUMNS, compare_ablations
from tools.evaluate_run import evaluate_run
from tools.run_ablations import ABLATION_CONFIGS, main as run_ablations_main


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
