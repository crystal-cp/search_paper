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
    "query_family_applied",
    "final_provider_query_count",
    "expected_anchor_coverage",
    "overbroad_query_count",
    "repeated_phrase_query_count",
    "single_axis_query_count",
    "weak_query_count",
    "query_quality_score",
    "query_repair_enabled",
    "query_repair_applied",
    "dropped_query_count",
    "repaired_query_count",
    "raw_to_final_query_change_count",
    "repair_disabled_but_sanitizer_active",
    "no_query_repair_conclusive",
    "merged_count",
    "duplicate_ratio",
    "provider_success_rate",
    "must_read_count",
    "include_count",
    "forbidden_pattern_top10_count",
    "forbidden_pattern_top20_count",
    "forbidden_pattern_must_read_count",
    "forbidden_pattern_include_count",
    "required_group_coverage_top10",
    "intent_centrality_mean_top10",
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
    values = [str(support_status.get(module, "supported")) for module in disabled_modules]
    if any(value == "unsupported" for value in values):
        return "unsupported", details
    if any(value == "partially_supported" for value in values):
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

    notes: list[str] = []
    if support_level == "partially_supported":
        notes.append("partially_supported ablation; diagnostic only")
    elif support_level == "unsupported":
        notes.append("unsupported ablation; do not interpret as strict ablation")
    if metrics.get("query_repair_disabled_by_ablation"):
        reason = str(metrics.get("query_repair_reason_if_no_difference") or "").strip()
        notes.append(
            "query repair disabled"
            + (f"; no-difference reason: {reason}" if reason else "")
        )
    if metrics.get("repair_disabled_but_sanitizer_active"):
        notes.append(
            "query repair ablation is non-conclusive because upstream sanitizer remains active"
        )
    if metrics.get("weak_query_count"):
        notes.append("weak query heuristic hit")
    if metrics.get("forbidden_pattern_top20_count"):
        notes.append("forbidden-pattern hit in top20")
    if metrics.get("forbidden_pattern_must_read_count"):
        notes.append("forbidden-pattern hit among must-read papers")
    if metrics.get("forbidden_pattern_include_count"):
        notes.append("forbidden-pattern hit among included papers")
    if config_name == "full_system" and metrics.get("forbidden_pattern_must_read_count", 0):
        notes.append(
            "full_system still has forbidden_pattern_must_read_count > 0; future benchmark constraints or guardrail improvement needed"
        )

    return {
        "case_id": case_id,
        "config_name": config_name,
        "mode": config.get("mode") or command_record.get("mode", ""),
        "query_family_applied": metrics.get("query_family_applied"),
        "final_provider_query_count": metrics.get("final_provider_query_count"),
        "expected_anchor_coverage": metrics.get("expected_anchor_coverage"),
        "overbroad_query_count": metrics.get("overbroad_query_count"),
        "repeated_phrase_query_count": metrics.get("repeated_phrase_query_count"),
        "single_axis_query_count": metrics.get("single_axis_query_count"),
        "weak_query_count": metrics.get("weak_query_count"),
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
        "merged_count": metrics.get("merged_count"),
        "duplicate_ratio": metrics.get("duplicate_ratio"),
        "provider_success_rate": metrics.get("provider_success_rate"),
        "must_read_count": metrics.get("must_read_count"),
        "include_count": metrics.get("include_count"),
        "forbidden_pattern_top10_count": metrics.get("forbidden_pattern_top10_count"),
        "forbidden_pattern_top20_count": metrics.get("forbidden_pattern_top20_count"),
        "forbidden_pattern_must_read_count": metrics.get(
            "forbidden_pattern_must_read_count"
        ),
        "forbidden_pattern_include_count": metrics.get("forbidden_pattern_include_count"),
        "required_group_coverage_top10": metrics.get("required_group_coverage_top10"),
        "intent_centrality_mean_top10": metrics.get("intent_centrality_mean_top10"),
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
    if not baseline:
        return ["No `full_system` baseline found for this case."]
    notes: list[str] = []
    if baseline.get("forbidden_pattern_must_read_count"):
        notes.append(
            "`full_system` still has `forbidden_pattern_must_read_count > 0`; this indicates the benchmark case needs either stronger expected constraints or future guardrail improvement."
        )
    if baseline.get("forbidden_pattern_top10_count") or baseline.get("forbidden_pattern_top20_count"):
        notes.append(
            "`full_system` still has forbidden-pattern papers in the top-ranked set for this benchmark. Do not treat `full_system` as a perfect baseline; for SEI this surfaces sodium/potassium/zinc or other non-target battery context when present."
        )
    for row in rows:
        if row is baseline:
            continue
        config_name = str(row.get("config_name", ""))
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
        ("Partially supported ablations", "partially_supported"),
        ("Unsupported ablations", "unsupported"),
    ]:
        values = buckets[key]
        lines.extend([f"### {title}", ""])
        if values:
            lines.extend([f"- {value}" for value in values])
        else:
            lines.append("- None observed in this summary.")
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
