import json

from tools.evaluate_run import evaluate_run, load_benchmark_cases


def test_benchmark_cases_yaml_is_readable():
    benchmark = load_benchmark_cases()

    assert benchmark["cases"]
    assert any(case["id"] == "sei_lithium_battery" for case in benchmark["cases"])
    sei = next(case for case in benchmark["cases"] if case["id"] == "sei_lithium_battery")
    assert sei["expected_query_anchors"]
    assert sei["forbidden_top10_patterns"]


def test_evaluate_run_handles_existing_run_directory(tmp_path):
    run_dir = tmp_path / "outputs" / "validation_v8" / "sei_full"
    run_dir.mkdir(parents=True)
    (run_dir / "planned_queries.json").write_text(
        json.dumps(
            {
                "query_family_applied": True,
                "final_provider_queries": {
                    "openalex": [
                        '"solid electrolyte interphase" "lithium metal anode"',
                        "SEI",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "query_provenance.json").write_text(
        json.dumps({"applied": True, "final_openalex_queries": ["artificial SEI lithium battery"]}),
        encoding="utf-8",
    )
    (run_dir / "provider_status.json").write_text(
        json.dumps({"openalex": {"status": "success", "returned_paper_count": 2}}),
        encoding="utf-8",
    )
    (run_dir / "retrieval_diagnostics.json").write_text(
        json.dumps(
            {
                "raw_retrieved_paper_count": 3,
                "merged_paper_count": 2,
                "duplicate_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "ranked_papers.csv").write_text(
        "\n".join(
            [
                "rank,title,abstract,reading_priority",
                "1,Artificial SEI for lithium metal batteries,solid electrolyte interphase suppresses dendrites,must_read",
                "2,Software Engineering Institute report,software process maturity,read_later",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "exploration_quality.json").write_text("{}", encoding="utf-8")
    (run_dir / "user_report.md").write_text(
        "\n".join(
            [
                "# User-Friendly Literature Screening Summary",
                "## Retrieval status",
                "- provider_summary: openalex:success(2 papers)",
                "## How the system interpreted the task",
                "The system interpreted the user intent as lithium battery SEI screening.",
            ]
        ),
        encoding="utf-8",
    )

    metrics = evaluate_run(run_dir, case_id="sei_lithium_battery")

    assert metrics["query_family_applied"] is True
    assert metrics["final_provider_query_count"] == 3
    assert metrics["single_acronym_query_count"] == 1
    assert metrics["provider_success_rate"] == 1.0
    assert metrics["merged_count"] == 2
    assert metrics["duplicate_ratio"] == 1 / 3
    assert metrics["top20_false_positive_count"] == 1
    assert metrics["must_read_count"] == 1
    assert metrics["must_read_precision_heuristic"] == 1.0
    assert metrics["report_has_provider_status"] is True
    assert metrics["report_has_user_intent_summary"] is True
