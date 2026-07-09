"""Compare pilot ablation output directories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.evaluate_run import DEFAULT_BENCHMARK_PATH, evaluate_run
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_run import DEFAULT_BENCHMARK_PATH, evaluate_run


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = REPO_ROOT / "reports" / "ablation_summary.csv"
DEFAULT_MD_PATH = REPO_ROOT / "reports" / "ablation_summary.md"

SUMMARY_COLUMNS = [
    "case_id",
    "config_name",
    "mode",
    "diagnostic_case_group",
    "stress_case",
    "query_family_applied",
    "final_provider_query_count",
    "anchor_coverage",
    "query_family_coverage",
    "expected_anchor_coverage",
    "overbroad_query_count",
    "repeated_phrase_query_count",
    "single_axis_query_count",
    "single_acronym_query_count",
    "weak_query_count",
    "duplicate_query_count",
    "missing_expected_anchor_count",
    "cross_domain_injection_count",
    "query_quality_score",
    "query_repair_enabled",
    "query_repair_applied",
    "dropped_query_count",
    "repaired_query_count",
    "raw_to_final_query_change_count",
    "repair_disabled_but_sanitizer_active",
    "no_query_repair_conclusive",
    "raw_retrieved_count",
    "merged_count",
    "retrieval_status",
    "duplicate_ratio",
    "provider_success_rate",
    "provider_error_count",
    "must_read_count",
    "include_count",
    "optional_count",
    "exclude_count",
    "forbidden_pattern_top10_count",
    "forbidden_pattern_top20_count",
    "forbidden_pattern_must_read_count",
    "forbidden_pattern_include_count",
    "negative_context_top10",
    "negative_context_top20",
    "negative_context_must_read",
    "negative_context_include",
    "required_group_coverage_top10",
    "intent_centrality_mean_top10",
    "reading_path_paper_count",
    "reading_path_exclude_count",
    "reading_path_out_of_scope_count",
    "reading_path_duplicate_count",
    "reading_path_negative_context_count",
    "target_context_required_for_priority",
    "llm_intent_enabled",
    "llm_intent_called",
    "llm_intent_fallback_used",
    "llm_intent_verified_candidate_count",
    "llm_intent_applied_suggestion_count",
    "llm_intent_rejected_suggestion_count",
    "llm_intent_unsupported_suggestion_count",
    "llm_intent_reason_if_no_change",
    "llm_query_critic_enabled",
    "llm_query_critic_called",
    "llm_query_critic_fallback_used",
    "verified_issue_count",
    "rejected_issue_count",
    "unsupported_issue_count",
    "applied_issue_count",
    "rejected_for_application_count",
    "query_added_count",
    "query_dropped_count",
    "query_modified_count",
    "repair_applied",
    "repair_grounded_term_count",
    "repair_rejected_term_count",
    "llm_query_critic_repair_enabled",
    "llm_query_critic_repair_applied",
    "llm_query_critic_verified_issue_count",
    "llm_query_critic_applied_issue_count",
    "llm_query_critic_repaired_query_count",
    "llm_query_critic_original_query_example",
    "llm_query_critic_repaired_query_example",
    "llm_query_critic_applied_terms",
    "llm_query_critic_rejected_terms",
    "llm_query_critic_repair_provenance_count",
    "llm_query_critic_repair_artifact_consistent",
    "final_query_artifact_consistent",
    "llm_query_critic_reason_if_no_change",
    "llm_query_before_example",
    "llm_query_after_example",
    "llm_direct_paper_decision_mutation_count",
    "llm_direct_evidence_validation_mutation_count",
    "llm_direct_ranking_mutation_count",
    "paper_decision_mutation_count",
    "retrieval_performed",
    "research_gap_generation_status",
    "ranked_papers_based_on_real_retrieval",
    "notes",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def discover_run_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    for path in sorted(root.glob("*/*")):
        if not path.is_dir():
            continue
        if (path / "ablation_config.json").exists() or (path / "planned_queries.json").exists():
            dirs.append(path)
    return dirs


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def md_cell(value: Any) -> str:
    return format_value(value).replace("|", "/")


def support_level_for_config(config: dict[str, Any]) -> tuple[str, list[str]]:
    disabled_modules = [
        str(module)
        for module in config.get("disabled_modules", []) or []
        if str(module).strip()
    ]
    support_status = config.get("support_status", {})
    if not isinstance(support_status, dict):
        support_status = {}
    details = [
        f"{module}:{support_status.get(module, 'supported')}"
        for module in disabled_modules
    ]
    if not details and support_status:
        details = [
            f"{module}:{status}"
            for module, status in sorted(support_status.items())
            if str(module).strip()
        ]
    values = [str(value) for value in support_status.values()]
    if any(value == "unsupported" for value in values):
        return "unsupported", details
    diagnostic_values = {
        "partially_supported",
        "diagnostic_artifacts_only",
        "diagnostic_controlled",
        "controlled_repair_pilot",
        "supported_diagnostic_only",
    }
    if (
        any(value in diagnostic_values for value in values)
        or bool(config.get("diagnostic_only"))
        or bool(config.get("warning_if_pilot_only"))
    ):
        return "partially_supported", details
    return "fully_supported", details or ["full_system:supported"]


def ablation_config_name(config: dict[str, Any], run_dir: Path) -> str:
    return str(
        config.get("ablation_config_name")
        or config.get("config_name")
        or run_dir.name
    )


def row_for_run(run_dir: Path, benchmark: Path) -> dict[str, Any]:
    config = read_json(run_dir / "ablation_config.json")
    command_record = read_json(run_dir / "ablation_command.json")
    case_id = run_dir.parent.name
    config_name = ablation_config_name(config, run_dir)
    metrics = evaluate_run(run_dir, benchmark_path=benchmark, case_id=case_id)
    support_level, support_details = support_level_for_config(config)
    case_metadata = config.get("case_metadata", {})
    if not isinstance(case_metadata, dict):
        case_metadata = {}
    diagnostic_case_group = str(case_metadata.get("diagnostic_group") or "").strip()
    stress_case = bool(case_metadata.get("stress_case", False))
    if config_name == "no_query_repair" and metrics.get("no_query_repair_conclusive") is False:
        support_level = "partially_supported"
        support_details = [
            *support_details,
            "query_repair:diagnostic_non_conclusive",
        ]
    run_mode = str(config.get("mode") or command_record.get("mode", "")).strip()

    notes: list[str] = []
    if support_level == "partially_supported":
        notes.append("partially_supported ablation; diagnostic only")
    elif support_level == "unsupported":
        notes.append("unsupported ablation; do not interpret as strict ablation")
    if metrics.get("query_repair_disabled_by_ablation"):
        reason = str(metrics.get("query_repair_reason_if_no_difference") or "").strip()
        label = (
            "deterministic query repair disabled"
            if metrics.get("llm_query_critic_repair_applied")
            else "query repair disabled"
        )
        notes.append(
            label
            + (f"; no-difference reason: {reason}" if reason else "")
        )
    if metrics.get("repair_disabled_but_sanitizer_active"):
        notes.append(
            "query repair ablation is non-conclusive because upstream sanitizer remains active"
        )
    if config_name == "no_query_repair" and metrics.get("no_query_repair_conclusive") is False:
        notes.append(
            "no_query_repair is diagnostic/non-conclusive in this pilot because upstream sanitizer remains active"
        )
    if metrics.get("weak_query_count"):
        notes.append("weak query heuristic hit")
    if metrics.get("forbidden_pattern_top20_count"):
        notes.append("forbidden-pattern hit in top20")
    if metrics.get("forbidden_pattern_must_read_count"):
        notes.append("forbidden-pattern hit among must-read papers")
    if metrics.get("forbidden_pattern_include_count"):
        notes.append("forbidden-pattern hit among included papers")
    if run_mode == "full" and str(config_name).startswith("llm_"):
        notes.append(
            "Small full-retrieval LLM safety pilot; not a formal LLM ablation conclusion"
        )
    elif str(config.get("mode_support") or "") == "plan_level" and str(
        config_name
    ).startswith("llm_"):
        notes.append("LLM plan-level diagnostic; not a formal full retrieval ablation")
    if stress_case:
        notes.append("weak-plan positive control; not part of formal benchmark conclusion")
    if (
        metrics.get("llm_intent_enabled")
        and not metrics.get("llm_intent_called")
        and metrics.get("llm_intent_fallback_used")
        and metrics.get("llm_intent_reason_if_no_change") == "llm_provider_unavailable"
    ):
        notes.append("Intent enhancer fallback only; not a positive LLM intent test.")
    if metrics.get("llm_query_critic_enabled") and not metrics.get("applied_issue_count"):
        if metrics.get("llm_query_critic_repair_enabled"):
            notes.append(
                "LLM repair apply flag enabled, but no verified grounded issue was available; clean plan remained unchanged."
            )
        elif bool(config.get("applies_query_changes", False)):
            notes.append("Repair flag configured, but status did not report enabled.")
        else:
            notes.append("LLM repair disabled; critic diagnostics only.")
        if not stress_case and not metrics.get("verified_issue_count"):
            notes.append("No verified repair opportunity; clean plan remained unchanged.")
        if not metrics.get("llm_query_critic_repair_enabled") and (
            stress_case or metrics.get("verified_issue_count")
        ):
            notes.append("LLM query critic did not mutate query plan")
    if metrics.get("applied_issue_count"):
        notes.append("LLM critique issue applied by deterministic rule applier")
    if metrics.get("llm_query_critic_repair_applied") and not metrics.get(
        "llm_query_critic_repair_artifact_consistent"
    ):
        notes.append("LLM repair artifact consistency failed; inspect final query artifacts")
    if config_name == "full_system" and metrics.get("forbidden_pattern_must_read_count", 0):
        notes.append(
            "full_system still has forbidden_pattern_must_read_count > 0; future benchmark constraints or guardrail improvement needed"
        )

    return {
        "case_id": case_id,
        "config_name": config_name,
        "mode": run_mode,
        "diagnostic_case_group": diagnostic_case_group,
        "stress_case": stress_case,
        "query_family_applied": metrics.get("query_family_applied"),
        "final_provider_query_count": metrics.get("final_provider_query_count"),
        "anchor_coverage": metrics.get("anchor_coverage"),
        "query_family_coverage": metrics.get("query_family_coverage"),
        "expected_anchor_coverage": metrics.get("expected_anchor_coverage"),
        "overbroad_query_count": metrics.get("overbroad_query_count"),
        "repeated_phrase_query_count": metrics.get("repeated_phrase_query_count"),
        "single_axis_query_count": metrics.get("single_axis_query_count"),
        "single_acronym_query_count": metrics.get("single_acronym_query_count"),
        "weak_query_count": metrics.get("weak_query_count"),
        "duplicate_query_count": metrics.get("duplicate_query_count"),
        "missing_expected_anchor_count": metrics.get("missing_expected_anchor_count"),
        "cross_domain_injection_count": metrics.get("cross_domain_injection_count"),
        "query_quality_score": metrics.get("query_quality_score"),
        "query_repair_enabled": metrics.get("query_repair_enabled"),
        "query_repair_applied": metrics.get("query_repair_applied"),
        "dropped_query_count": metrics.get("dropped_query_count"),
        "repaired_query_count": metrics.get("repaired_query_count"),
        "raw_to_final_query_change_count": metrics.get("raw_to_final_query_change_count"),
        "repair_disabled_but_sanitizer_active": metrics.get(
            "repair_disabled_but_sanitizer_active"
        ),
        "no_query_repair_conclusive": metrics.get("no_query_repair_conclusive"),
        "raw_retrieved_count": metrics.get("raw_retrieved_count"),
        "merged_count": metrics.get("merged_count"),
        "retrieval_status": metrics.get("retrieval_status"),
        "duplicate_ratio": metrics.get("duplicate_ratio"),
        "provider_success_rate": metrics.get("provider_success_rate"),
        "provider_error_count": metrics.get("provider_error_count"),
        "must_read_count": metrics.get("must_read_count"),
        "include_count": metrics.get("include_count"),
        "optional_count": metrics.get("optional_count"),
        "exclude_count": metrics.get("exclude_count"),
        "forbidden_pattern_top10_count": metrics.get("forbidden_pattern_top10_count"),
        "forbidden_pattern_top20_count": metrics.get("forbidden_pattern_top20_count"),
        "forbidden_pattern_must_read_count": metrics.get(
            "forbidden_pattern_must_read_count"
        ),
        "forbidden_pattern_include_count": metrics.get("forbidden_pattern_include_count"),
        "negative_context_top10": metrics.get("negative_context_top10"),
        "negative_context_top20": metrics.get("negative_context_top20"),
        "negative_context_must_read": metrics.get("negative_context_must_read"),
        "negative_context_include": metrics.get("negative_context_include"),
        "required_group_coverage_top10": metrics.get("required_group_coverage_top10"),
        "intent_centrality_mean_top10": metrics.get("intent_centrality_mean_top10"),
        "reading_path_paper_count": metrics.get("reading_path_paper_count"),
        "reading_path_exclude_count": metrics.get("reading_path_exclude_count"),
        "reading_path_out_of_scope_count": metrics.get("reading_path_out_of_scope_count"),
        "reading_path_duplicate_count": metrics.get("reading_path_duplicate_count"),
        "reading_path_negative_context_count": metrics.get(
            "reading_path_negative_context_count"
        ),
        "target_context_required_for_priority": metrics.get(
            "target_context_required_for_priority"
        ),
        "llm_intent_enabled": metrics.get("llm_intent_enabled"),
        "llm_intent_called": metrics.get("llm_intent_called"),
        "llm_intent_fallback_used": metrics.get("llm_intent_fallback_used"),
        "llm_intent_verified_candidate_count": metrics.get(
            "llm_intent_verified_candidate_count"
        ),
        "llm_intent_applied_suggestion_count": metrics.get(
            "llm_intent_applied_suggestion_count"
        ),
        "llm_intent_rejected_suggestion_count": metrics.get(
            "llm_intent_rejected_suggestion_count"
        ),
        "llm_intent_unsupported_suggestion_count": metrics.get(
            "llm_intent_unsupported_suggestion_count"
        ),
        "llm_intent_reason_if_no_change": metrics.get(
            "llm_intent_reason_if_no_change"
        ),
        "llm_query_critic_enabled": metrics.get("llm_query_critic_enabled"),
        "llm_query_critic_called": metrics.get("llm_query_critic_called"),
        "llm_query_critic_fallback_used": metrics.get(
            "llm_query_critic_fallback_used"
        ),
        "verified_issue_count": metrics.get("verified_issue_count"),
        "rejected_issue_count": metrics.get("rejected_issue_count"),
        "unsupported_issue_count": metrics.get("unsupported_issue_count"),
        "applied_issue_count": metrics.get("applied_issue_count"),
        "rejected_for_application_count": metrics.get(
            "rejected_for_application_count"
        ),
        "query_added_count": metrics.get("query_added_count"),
        "query_dropped_count": metrics.get("query_dropped_count"),
        "query_modified_count": metrics.get("query_modified_count"),
        "repair_applied": metrics.get("repair_applied"),
        "repair_grounded_term_count": metrics.get("repair_grounded_term_count"),
        "repair_rejected_term_count": metrics.get("repair_rejected_term_count"),
        "llm_query_critic_repair_enabled": metrics.get(
            "llm_query_critic_repair_enabled"
        ),
        "llm_query_critic_repair_applied": metrics.get(
            "llm_query_critic_repair_applied"
        ),
        "llm_query_critic_verified_issue_count": metrics.get(
            "llm_query_critic_verified_issue_count"
        ),
        "llm_query_critic_applied_issue_count": metrics.get(
            "llm_query_critic_applied_issue_count"
        ),
        "llm_query_critic_repaired_query_count": metrics.get(
            "llm_query_critic_repaired_query_count"
        ),
        "llm_query_critic_original_query_example": metrics.get(
            "llm_query_critic_original_query_example"
        ),
        "llm_query_critic_repaired_query_example": metrics.get(
            "llm_query_critic_repaired_query_example"
        ),
        "llm_query_critic_applied_terms": metrics.get(
            "llm_query_critic_applied_terms"
        ),
        "llm_query_critic_rejected_terms": metrics.get(
            "llm_query_critic_rejected_terms"
        ),
        "llm_query_critic_repair_provenance_count": metrics.get(
            "llm_query_critic_repair_provenance_count"
        ),
        "llm_query_critic_repair_artifact_consistent": metrics.get(
            "llm_query_critic_repair_artifact_consistent"
        ),
        "final_query_artifact_consistent": metrics.get(
            "final_query_artifact_consistent"
        ),
        "llm_query_critic_reason_if_no_change": metrics.get(
            "llm_query_critic_reason_if_no_change"
        ),
        "llm_query_before_example": metrics.get("llm_query_before_example"),
        "llm_query_after_example": metrics.get("llm_query_after_example"),
        "llm_direct_paper_decision_mutation_count": metrics.get(
            "llm_direct_paper_decision_mutation_count"
        ),
        "llm_direct_evidence_validation_mutation_count": metrics.get(
            "llm_direct_evidence_validation_mutation_count"
        ),
        "llm_direct_ranking_mutation_count": metrics.get(
            "llm_direct_ranking_mutation_count"
        ),
        "paper_decision_mutation_count": metrics.get("paper_decision_mutation_count"),
        "retrieval_performed": metrics.get("retrieval_performed"),
        "research_gap_generation_status": metrics.get("research_gap_generation_status"),
        "ranked_papers_based_on_real_retrieval": metrics.get(
            "ranked_papers_based_on_real_retrieval"
        ),
        "notes": "; ".join(notes),
        "_support_level": support_level,
        "_support_details": ", ".join(support_details),
        "_query_repair_reason_if_no_difference": metrics.get(
            "query_repair_reason_if_no_difference",
            "",
        ),
        "_single_acronym_query_count": metrics.get("single_acronym_query_count"),
        "_anchor_coverage": metrics.get("anchor_coverage"),
        "_expected_anchor_coverage": metrics.get("expected_anchor_coverage"),
        "_overbroad_query_count": metrics.get("overbroad_query_count"),
        "_top20_false_positive_count": metrics.get("top20_false_positive_count"),
    }


def write_csv_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in SUMMARY_COLUMNS})


def metric_delta(row: dict[str, Any], baseline: dict[str, Any], metric: str) -> float | None:
    try:
        value = row.get(metric)
        base = baseline.get(metric)
        if value is None or base is None or value == "" or base == "":
            return None
        return float(value) - float(base)
    except (TypeError, ValueError):
        return None


def delta_text(row: dict[str, Any], baseline: dict[str, Any], metric: str) -> str:
    delta = metric_delta(row, baseline, metric)
    if delta is None:
        return "n/a"
    return f"{delta:+.2f}"


def comparison_notes(rows: list[dict[str, Any]]) -> list[str]:
    baseline = next((row for row in rows if row["config_name"] == "full_system"), None)
    notes: list[str] = []
    if not baseline:
        notes.append("No `full_system` baseline found for this case.")
    elif baseline.get("forbidden_pattern_must_read_count"):
        notes.append(
            "`full_system` still has `forbidden_pattern_must_read_count > 0`; this indicates the benchmark case needs either stronger expected constraints or future guardrail improvement."
        )
    if baseline and (
        baseline.get("forbidden_pattern_top10_count")
        or baseline.get("forbidden_pattern_top20_count")
    ):
        notes.append(
            "`full_system` still has forbidden-pattern papers in the top-ranked set for this benchmark. Do not treat `full_system` as a perfect baseline; for SEI this surfaces sodium/potassium/zinc or other non-target battery context when present."
        )
    diagnostic_query_critic = next(
        (
            row
            for row in rows
            if row.get("config_name") == "llm_query_critic_diagnostic_only"
        ),
        None,
    )
    for row in rows:
        if row is baseline:
            continue
        config_name = str(row.get("config_name", ""))
        if baseline is None and not config_name.startswith("llm_"):
            notes.append(
                f"`{config_name}` has no full_system baseline in this case; treat observed metrics as diagnostics only."
            )
            continue
        if config_name == "no_query_family":
            quality_delta = metric_delta(row, baseline, "query_quality_score")
            anchor_delta = metric_delta(row, baseline, "expected_anchor_coverage")
            overbroad_delta = metric_delta(row, baseline, "_overbroad_query_count")
            weak_delta = metric_delta(row, baseline, "weak_query_count")
            repeated_delta = metric_delta(row, baseline, "repeated_phrase_query_count")
            single_axis_delta = metric_delta(row, baseline, "single_axis_query_count")
            if (
                (quality_delta is not None and quality_delta < -0.05)
                or (anchor_delta is not None and anchor_delta < -0.05)
                or (overbroad_delta is not None and overbroad_delta > 0)
                or (weak_delta is not None and weak_delta > 0)
                or (repeated_delta is not None and repeated_delta > 0)
                or (single_axis_delta is not None and single_axis_delta > 0)
            ):
                notes.append(
                    "`no_query_family` degraded query quality in this pilot: "
                    f"query_quality_score {delta_text(row, baseline, 'query_quality_score')}, "
                    f"expected_anchor_coverage {delta_text(row, baseline, 'expected_anchor_coverage')}, "
                    f"weak_query_count {delta_text(row, baseline, 'weak_query_count')}, "
                    f"overbroad_query_count {delta_text(row, baseline, '_overbroad_query_count')}, "
                    f"repeated_phrase_query_count {delta_text(row, baseline, 'repeated_phrase_query_count')}, "
                    f"single_axis_query_count {delta_text(row, baseline, 'single_axis_query_count')}."
                )
            else:
                notes.append(
                    "`no_query_family` produced no strong query-quality degradation signal in this pilot; inspect query-family artifacts before interpreting."
                )
        elif config_name == "no_query_repair":
            changed = any(
                (row.get(metric) or 0) != (baseline.get(metric) or 0)
                for metric in [
                    "final_provider_query_count",
                    "dropped_query_count",
                    "repaired_query_count",
                ]
            )
            if row.get("repair_disabled_but_sanitizer_active"):
                notes.append(
                    "`no_query_repair` is diagnostic only because upstream sanitizer remains active. If final queries are unchanged, do not conclude QueryRepair has no value. "
                    f"`full_system` raw_to_final_query_change_count={format_value(baseline.get('raw_to_final_query_change_count'))}; "
                    f"`no_query_repair` raw_to_final_query_change_count={format_value(row.get('raw_to_final_query_change_count'))}."
                )
            elif changed:
                notes.append(
                    "`no_query_repair` changed query-repair diagnostics; inspect `query_repair_stage_status.json` for exact dropped/repaired counts."
                )
            else:
                reason = str(row.get("_query_repair_reason_if_no_difference") or "").strip()
                notes.append(
                    "`no_query_repair` produced no measurable degradation in this pilot; do not treat this as proof that QueryRepair has no value without inspecting raw-to-final query artifacts"
                    + (f" ({reason})." if reason else ".")
                )
        elif config_name == "no_domain_guardrail":
            forbidden_delta = metric_delta(
                row,
                baseline,
                "forbidden_pattern_top20_count",
            )
            must_read_delta = metric_delta(
                row,
                baseline,
                "forbidden_pattern_must_read_count",
            )
            if (
                (forbidden_delta is not None and forbidden_delta > 0)
                or (must_read_delta is not None and must_read_delta > 0)
            ):
                notes.append(
                    "`no_domain_guardrail` increased broad/peripheral forbidden-pattern hits in this pilot: "
                    f"top20 {delta_text(row, baseline, 'forbidden_pattern_top20_count')}, "
                    f"must_read {delta_text(row, baseline, 'forbidden_pattern_must_read_count')}."
                )
            else:
                notes.append(
                    "`no_domain_guardrail` did not increase forbidden-pattern counts in this pilot; this is diagnostic, not a final conclusion."
                )
        elif config_name == "no_intent_centrality":
            notes.append(
                "`no_intent_centrality` tests whether broad review papers displace intersection papers; interpret with top10 context, not as a final ranking claim."
            )
        elif config_name == "no_group_coverage_ranking":
            notes.append(
                "`no_group_coverage_ranking` is partially supported and not conclusive because group coverage also affects upstream assessment artifacts."
            )
        elif config_name.startswith("llm_"):
            is_full_llm_pilot = str(row.get("mode") or "") == "full"
            if is_full_llm_pilot:
                notes.append(
                    "Small full-retrieval LLM safety pilot; interpret retrieval and ranking deltas cautiously because provider ranking and retrieval stochasticity may affect results."
                )
            if row.get("llm_query_critic_repair_applied"):
                quality_delta = (
                    metric_delta(row, diagnostic_query_critic, "query_quality_score")
                    if diagnostic_query_critic is not None
                    and row is not diagnostic_query_critic
                    else None
                )
                if quality_delta is not None and quality_delta > 0:
                    notes.append(
                        "Repair was applied and query_quality_score improved under current heuristic."
                    )
                else:
                    notes.append(
                        "Repair was applied and provenance is valid, but query_quality_score did not improve under current heuristic; this should be interpreted as repair-mechanism validation, not quality improvement."
                    )
                if not row.get("llm_query_critic_repair_artifact_consistent"):
                    notes.append(
                        "LLM repair artifact consistency failed; inspect final query artifacts before interpreting this run."
                    )
            if (
                row.get("llm_intent_enabled")
                and not row.get("llm_intent_called")
                and row.get("llm_intent_fallback_used")
            ):
                notes.append(
                    "Intent enhancer fallback only; not a positive LLM intent test."
                )
            if (
                row.get("llm_query_critic_enabled")
                and not row.get("verified_issue_count")
                and not row.get("stress_case")
            ):
                notes.append(
                    "No verified repair opportunity; clean plan remained unchanged."
                )
            if (
                row.get("llm_query_critic_enabled")
                and bool(row.get("applied_issue_count"))
                and row.get("stress_case")
            ):
                notes.append(
                    f"Weak-plan positive control applied repair: verified_issue_count={format_value(row.get('verified_issue_count'))}, "
                    f"applied_issue_count={format_value(row.get('applied_issue_count'))}, "
                    f"query_modified_count={format_value(row.get('query_modified_count'))}, "
                    f"repair_grounded_term_count={format_value(row.get('repair_grounded_term_count'))}, "
                    f"repair_rejected_term_count={format_value(row.get('repair_rejected_term_count'))}, "
                    f"before={format_value(row.get('llm_query_before_example'))}, "
                    f"after={format_value(row.get('llm_query_after_example'))}."
                )
            elif row.get("llm_query_critic_enabled") and bool(row.get("applied_issue_count")):
                notes.append(
                    "LLM critic repair was applied by the deterministic rule applier; this is a safety-pilot mechanism signal, not proof of retrieval quality improvement."
                )
            if (
                row.get("llm_query_critic_enabled")
                and not row.get("applied_issue_count")
                and str(row.get("config_name", "")) in {
                    "llm_query_critic_repair_applied",
                    "llm_intent_plus_query_critic_repair",
                }
            ):
                if row.get("llm_query_critic_repair_enabled"):
                    notes.append(
                        "LLM repair apply flag enabled, but no verified grounded issue was available; clean plan remained unchanged."
                    )
                else:
                    notes.append(
                        "LLM repair disabled; critic diagnostics only."
                    )
            if config_name == "llm_query_critic_diagnostic_only" or config_name == "llm_query_critic_only":
                notes.append(
                    "`llm_query_critic_diagnostic_only` is plan-level diagnostic only: it records verified/rejected critique issues but must not mutate final provider queries."
                )
            elif config_name == "llm_query_critic_repair_applied":
                notes.append(
                    "`llm_query_critic_repair_applied` is a controlled repair pilot: query changes only count when the deterministic rule applier accepts verified, grounded issues."
                )
            elif config_name == "llm_intent_frame_only":
                notes.append(
                    "`llm_intent_frame_only` tests controlled intent-frame suggestions entering SearchContract provenance after deterministic verification."
                )
            elif config_name == "llm_intent_plus_query_critic_repair":
                if is_full_llm_pilot:
                    notes.append(
                        "`llm_intent_plus_query_critic_repair` is a small full-retrieval safety pilot config; it does not prove retrieval precision, ranking, or screening improvement."
                    )
                else:
                    notes.append(
                        "`llm_intent_plus_query_critic_repair` is still plan-level only; it does not prove full retrieval or ranking improvement."
                    )
            else:
                notes.append(
                    f"`{config_name}` is an LLM plan-level pilot signal, not a formal full ablation conclusion."
                )
        else:
            notes.append(
                f"`{config_name}` has no tailored interpretation rule; treat observed deltas as pilot diagnostics only."
            )
    return notes


def support_summary(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets = {
        "fully_supported": [],
        "partially_supported": [],
        "unsupported": [],
    }
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("config_name", "")), str(row.get("_support_level", "")))
        if key in seen:
            continue
        seen.add(key)
        level = key[1] if key[1] in buckets else "fully_supported"
        detail = str(row.get("_support_details") or "").strip()
        label = key[0] + (f" ({detail})" if detail else "")
        buckets[level].append(label)
    return buckets


def write_markdown_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)

    lines = [
        "# Pilot Ablation Summary",
        "",
        "This is a pilot / diagnostic ablation summary generated from artifact-level heuristics.",
        "It is not a final experimental conclusion and should not be over-interpreted without larger benchmarks, provider-stability checks, and human labels.",
        "",
        "## Ablation support status",
        "",
    ]
    buckets = support_summary(rows)
    for title, key in [
        ("Fully supported ablations", "fully_supported"),
        ("Partially supported ablations (diagnostic / non-conclusive)", "partially_supported"),
        ("Unsupported ablations", "unsupported"),
    ]:
        values = buckets[key]
        lines.extend([f"### {title}", ""])
        if values:
            lines.extend([f"- {value}" for value in values])
        else:
            lines.append("- None observed in this summary.")
        lines.append("")

    full_pilot_rows = [
        row
        for row in rows
        if str(row.get("mode") or "") == "full"
        and str(row.get("case_id") or "")
        in {"sei_lithium_battery", "oer_spin_state"}
        and str(row.get("config_name") or "")
        in {
            "full_system",
            "llm_intent_plus_query_critic_repair",
            "llm_query_critic_repair_applied",
        }
    ]
    if full_pilot_rows:
        lines.extend(
            [
                "## Small full-retrieval LLM pilot",
                "",
                "This is a safety pilot, not a formal LLM ablation conclusion.",
                "It checks whether LLM-enhanced planning can run through full retrieval without breaking guardrails, ranking, reading_path, or reports.",
                "Any quality improvement claims are tentative.",
                "If no verified LLM repair was available, this is not a failure; it means the query plan may already have been strong.",
                "If full retrieval changes, interpret cautiously because provider ranking and retrieval stochasticity may affect results.",
                "LLM modules do not make paper-level decisions.",
                "",
                "| Case | Config | Retrieval status | Retrieval performed | Real retrieval | Raw | Merged | Provider success | Provider errors | Must-read | Include | Optional | Exclude | Forbidden must-read | Negative must-read | Reading leaks | Intent called | Critic called | Repair enabled | Verified issues | Applied issues | Query modified | Direct paper/evidence/ranking mutations |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in sorted(
            full_pilot_rows,
            key=lambda item: (str(item.get("case_id", "")), str(item.get("config_name", ""))),
        ):
            reading_leaks = (
                f"exclude={format_value(row.get('reading_path_exclude_count'))}; "
                f"out_of_scope={format_value(row.get('reading_path_out_of_scope_count'))}; "
                f"duplicate={format_value(row.get('reading_path_duplicate_count'))}; "
                f"negative={format_value(row.get('reading_path_negative_context_count'))}"
            )
            direct_mutations = (
                f"paper={format_value(row.get('llm_direct_paper_decision_mutation_count'))}; "
                f"evidence={format_value(row.get('llm_direct_evidence_validation_mutation_count'))}; "
                f"ranking={format_value(row.get('llm_direct_ranking_mutation_count'))}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(row["case_id"]),
                        md_cell(row["config_name"]),
                        md_cell(row["retrieval_status"]),
                        md_cell(row["retrieval_performed"]),
                        md_cell(row["ranked_papers_based_on_real_retrieval"]),
                        md_cell(row["raw_retrieved_count"]),
                        md_cell(row["merged_count"]),
                        md_cell(row["provider_success_rate"]),
                        md_cell(row["provider_error_count"]),
                        md_cell(row["must_read_count"]),
                        md_cell(row["include_count"]),
                        md_cell(row["optional_count"]),
                        md_cell(row["exclude_count"]),
                        md_cell(row["forbidden_pattern_must_read_count"]),
                        md_cell(row["negative_context_must_read"]),
                        md_cell(reading_leaks),
                        md_cell(row["llm_intent_called"]),
                        md_cell(row["llm_query_critic_called"]),
                        md_cell(row["llm_query_critic_repair_enabled"]),
                        md_cell(row["verified_issue_count"]),
                        md_cell(row["applied_issue_count"]),
                        md_cell(row["query_modified_count"]),
                        md_cell(direct_mutations),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "- Do not claim LLM improves retrieval precision from this section unless independent metrics clearly support it.",
                "- Use this section to check full-pipeline safety signals: retrieval completed, reading-path leaks stayed bounded, negative-context papers were not promoted, and LLM direct mutation counters stayed zero.",
                "",
            ]
        )

    llm_rows = [
        row
        for row in rows
        if str(row.get("mode") or "") != "full"
        and (
            str(row.get("config_name", "")).startswith("llm_")
            or row.get("llm_intent_enabled")
            or row.get("llm_query_critic_enabled")
        )
    ]
    if llm_rows:
        lines.extend(
            [
                "## LLM plan-level diagnostic summary",
                "",
                "This is plan-level diagnostic only; it does not prove full retrieval improvement.",
                "Query changes only occur when the deterministic rule applier accepts verified and grounded critique issues.",
                "LLM modules do not make paper-level decisions such as include/exclude, must_read, domain_decision, final_score, evidence validity, or reading_priority.",
                "",
                "| Case | Group | Stress | Config | Intent called | Intent fallback | Intent verified | Intent applied | Query critic called | Verified issues | Applied issues | Query +/-/~ | Grounded terms | Rejected terms | Before query | After query | Applied terms | Rejected term names | Provenance | Artifacts consistent | Final query consistent | Retrieval performed | Gaps | Paper decision mutation |",
                "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: |",
            ]
        )
        for row in sorted(
            llm_rows,
            key=lambda item: (str(item.get("case_id", "")), str(item.get("config_name", ""))),
        ):
            query_delta = (
                f"+{format_value(row.get('query_added_count'))}/"
                f"-{format_value(row.get('query_dropped_count'))}/"
                f"~{format_value(row.get('query_modified_count'))}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(row["case_id"]),
                        md_cell(row["diagnostic_case_group"]),
                        md_cell(row["stress_case"]),
                        md_cell(row["config_name"]),
                        md_cell(row["llm_intent_called"]),
                        md_cell(row["llm_intent_fallback_used"]),
                        md_cell(row["llm_intent_verified_candidate_count"]),
                        md_cell(row["llm_intent_applied_suggestion_count"]),
                        md_cell(row["llm_query_critic_called"]),
                        md_cell(row["verified_issue_count"]),
                        md_cell(row["applied_issue_count"]),
                        md_cell(query_delta),
                        md_cell(row["repair_grounded_term_count"]),
                        md_cell(row["repair_rejected_term_count"]),
                        md_cell(row["llm_query_critic_original_query_example"]),
                        md_cell(row["llm_query_critic_repaired_query_example"]),
                        md_cell(row["llm_query_critic_applied_terms"]),
                        md_cell(row["llm_query_critic_rejected_terms"]),
                        md_cell(row["llm_query_critic_repair_provenance_count"]),
                        md_cell(row["llm_query_critic_repair_artifact_consistent"]),
                        md_cell(row["final_query_artifact_consistent"]),
                        md_cell(row["retrieval_performed"]),
                        md_cell(row["research_gap_generation_status"]),
                        md_cell(row["paper_decision_mutation_count"]),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "- Treat these rows as pilot diagnostics for controlled LLM planning behavior, not as a formal LLM ablation study.",
                "",
            ]
        )
        weak_rows = [row for row in llm_rows if row.get("stress_case")]
        if weak_rows:
            lines.extend(["Weak-plan positive controls:", ""])
            for row in sorted(
                weak_rows,
                key=lambda item: (str(item.get("case_id", "")), str(item.get("config_name", ""))),
            ):
                lines.append(
                    "- "
                    f"{format_value(row.get('case_id'))} / {format_value(row.get('config_name'))}: "
                    f"verified_issue_count={format_value(row.get('verified_issue_count'))}, "
                    f"applied_issue_count={format_value(row.get('applied_issue_count'))}, "
                    f"query_modified_count={format_value(row.get('query_modified_count'))}, "
                    f"repair_grounded_term_count={format_value(row.get('repair_grounded_term_count'))}, "
                    f"repair_rejected_term_count={format_value(row.get('repair_rejected_term_count'))}, "
                    f"before={format_value(row.get('llm_query_critic_original_query_example'))}, "
                    f"after={format_value(row.get('llm_query_critic_repaired_query_example'))}, "
                    f"applied_terms={format_value(row.get('llm_query_critic_applied_terms'))}, "
                    f"rejected_terms={format_value(row.get('llm_query_critic_rejected_terms'))}, "
                    f"artifact_consistent={format_value(row.get('llm_query_critic_repair_artifact_consistent'))}."
                )
            lines.append("")

    for case_id, case_rows in sorted(by_case.items()):
        lines.extend(
            [
                f"## {case_id}",
                "",
                "| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in sorted(case_rows, key=lambda item: item["config_name"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(row["config_name"]),
                        md_cell(row["mode"]),
                        md_cell(row["query_family_applied"]),
                        md_cell(row["final_provider_query_count"]),
                        md_cell(row["query_quality_score"]),
                        md_cell(row["weak_query_count"]),
                        md_cell(row["overbroad_query_count"]),
                        md_cell(row["repeated_phrase_query_count"]),
                        md_cell(row["single_axis_query_count"]),
                        md_cell(row["query_repair_enabled"]),
                        md_cell(row["query_repair_applied"]),
                        md_cell(row["raw_to_final_query_change_count"]),
                        md_cell(row["repair_disabled_but_sanitizer_active"]),
                        md_cell(row["no_query_repair_conclusive"]),
                        md_cell(row["merged_count"]),
                        md_cell(row["duplicate_ratio"]),
                        md_cell(row["provider_success_rate"]),
                        md_cell(row["forbidden_pattern_top10_count"]),
                        md_cell(row["forbidden_pattern_top20_count"]),
                        md_cell(row["forbidden_pattern_must_read_count"]),
                        md_cell(row["forbidden_pattern_include_count"]),
                        md_cell(row["must_read_count"]),
                        md_cell(row["include_count"]),
                        md_cell(row["required_group_coverage_top10"]),
                        md_cell(row["intent_centrality_mean_top10"]),
                        md_cell(row["notes"]),
                    ]
                )
                + " |"
            )
        lines.extend(["", "Pilot interpretation:", ""])
        lines.extend([f"- {note}" for note in comparison_notes(case_rows)])
        lines.extend(["", "Reading-path diagnostics:", ""])
        lines.extend(
            [
                "| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in sorted(case_rows, key=lambda item: item["config_name"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(row["config_name"]),
                        md_cell(row["reading_path_paper_count"]),
                        md_cell(row["reading_path_exclude_count"]),
                        md_cell(row["reading_path_out_of_scope_count"]),
                        md_cell(row["reading_path_duplicate_count"]),
                        md_cell(row["reading_path_negative_context_count"]),
                        md_cell(row["target_context_required_for_priority"]),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "- These signals are pilot diagnostics only, not final ablation evidence.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_ablations(
    root: Path,
    *,
    benchmark: Path = DEFAULT_BENCHMARK_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    markdown_path: Path = DEFAULT_MD_PATH,
) -> list[dict[str, Any]]:
    rows = [row_for_run(run_dir, benchmark) for run_dir in discover_run_dirs(root)]
    write_csv_summary(rows, csv_path)
    write_markdown_summary(rows, markdown_path)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare pilot ablation outputs.")
    parser.add_argument("root", help="Ablation output root, e.g. outputs/ablations_pilot")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--markdown", default=str(DEFAULT_MD_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compare_ablations(
        Path(args.root),
        benchmark=Path(args.benchmark),
        csv_path=Path(args.csv),
        markdown_path=Path(args.markdown),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
