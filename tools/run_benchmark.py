"""Run benchmark cases and summarize artifact-level evaluation metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.evaluate_run import DEFAULT_BENCHMARK_PATH, evaluate_run, load_benchmark_cases
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_run import DEFAULT_BENCHMARK_PATH, evaluate_run, load_benchmark_cases


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = REPO_ROOT / "reports" / "benchmark_summary.md"


def provider_key_name(provider: str) -> str | None:
    normalized = provider.strip().lower()
    if normalized == "openalex":
        return "OPENALEX_API_KEY"
    if normalized in {"semantic_scholar", "semanticscholar", "s2"}:
        return "S2_API_KEY"
    return None


def selected_cases(
    benchmark: dict[str, Any],
    case_ids: list[str] | None,
    limit: int | None,
    mode: str,
) -> list[dict[str, Any]]:
    cases = list(benchmark.get("cases") or [])
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case.get("id") in wanted]
    if mode == "plan":
        cases = [case for case in cases if case.get("plan_only_eligible", True)]
    else:
        cases = [case for case in cases if case.get("full_retrieval_required", True)]
    if limit is not None:
        cases = cases[:limit]
    return cases


def case_question(case: dict[str, Any]) -> str:
    for key in ("novice_question_zh", "novice_question", "question", "canonical_intent"):
        value = str(case.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"Benchmark case {case.get('id', '<unknown>')} has no question.")


def output_dir_for_case(output_root: Path, case: dict[str, Any], mode: str) -> Path:
    return output_root / str(case["id"]) / mode


def build_pipeline_command(
    case: dict[str, Any],
    mode: str,
    providers: list[str],
    output_dir: Path,
    max_per_query: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lit_screening.pipeline",
        "run",
        "--question",
        case_question(case),
        "--providers",
        *providers,
        "--max-per-query",
        str(0 if mode == "plan" else max_per_query),
        "--strictness",
        "balanced",
        "--output-dir",
        str(output_dir),
    ]
    if mode == "full":
        command.extend(
            [
                "--openalex-mode",
                "keyword+semantic",
                "--sort-preference",
                "relevance",
                "--ranking-profile",
                "balanced",
            ]
        )
    return command


def synthetic_provider_status(providers: list[str]) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for provider in providers:
        key_name = provider_key_name(provider)
        if key_name and not os.environ.get(key_name):
            state = "missing_api_key"
            reason = f"{key_name} is not set for benchmark full mode."
        else:
            state = "not_attempted"
            reason = "Pipeline did not write provider_status.json."
        status[provider] = {
            "status": state,
            "returned_paper_count": 0,
            "error_type": state,
            "message": reason,
        }
    return status


def ensure_provider_status(output_dir: Path, providers: list[str], mode: str) -> None:
    provider_status_path = output_dir / "provider_status.json"
    if provider_status_path.exists():
        return
    if mode != "full":
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_status_path.write_text(
        json.dumps(synthetic_provider_status(providers), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def run_case(
    case: dict[str, Any],
    *,
    mode: str,
    providers: list[str],
    output_root: Path,
    benchmark_path: Path,
    max_per_query: int,
) -> dict[str, Any]:
    output_dir = output_dir_for_case(output_root, case, mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_pipeline_command(case, mode, providers, output_dir, max_per_query)
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    command_record = {
        "case_id": case.get("id"),
        "mode": mode,
        "providers": providers,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    (output_dir / "benchmark_command.json").write_text(
        json.dumps(command_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ensure_provider_status(output_dir, providers, mode)
    metrics = evaluate_run(
        output_dir,
        benchmark_path=benchmark_path,
        case_id=str(case.get("id")),
    )
    metrics["pipeline_returncode"] = completed.returncode
    metrics_path = output_dir / "benchmark_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "case": case,
        "output_dir": output_dir,
        "metrics": metrics,
        "returncode": completed.returncode,
    }


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_summary(
    results: list[dict[str, Any]],
    *,
    mode: str,
    providers: list[str],
    output_root: Path,
    summary_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Mode: `{mode}`",
        f"- Providers: `{', '.join(providers)}`",
        f"- Output root: `{output_root}`",
        "",
        "| Case | Return code | Provider success | Queries | Single acronym | Anchor coverage | Merged | Must-read | Report provider status | Report intent summary | Metrics |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in results:
        case = result["case"]
        metrics = result["metrics"]
        metrics_path = Path(result["output_dir"]) / "benchmark_metrics.json"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("id")),
                    str(result["returncode"]),
                    format_metric(metrics.get("provider_success_rate")),
                    format_metric(metrics.get("final_provider_query_count")),
                    format_metric(metrics.get("single_acronym_query_count")),
                    format_metric(metrics.get("anchor_coverage")),
                    format_metric(metrics.get("merged_count")),
                    format_metric(metrics.get("must_read_count")),
                    "yes" if metrics.get("report_has_provider_status") else "no",
                    "yes" if metrics.get("report_has_user_intent_summary") else "no",
                    f"`{metrics_path}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "This summary is generated by `tools/run_benchmark.py` from existing",
            "pipeline artifacts and `tools/evaluate_run.py` metrics. It does not",
            "modify the core screening pipeline.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run search_paper benchmark cases.")
    parser.add_argument("--mode", choices=["plan", "full"], required=True)
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openalex"],
        help="Providers passed to the pipeline.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/benchmark",
        help="Benchmark output root.",
    )
    parser.add_argument(
        "--benchmark",
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Benchmark YAML path.",
    )
    parser.add_argument("--case", action="append", default=None, help="Case ID. Repeatable.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-per-query", type=int, default=10)
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Markdown summary path.",
    )
    parser.add_argument(
        "--fail-on-pipeline-error",
        action="store_true",
        help="Return non-zero if any pipeline subprocess returns non-zero.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark_path = Path(args.benchmark)
    benchmark = load_benchmark_cases(benchmark_path)
    cases = selected_cases(benchmark, args.case, args.limit, args.mode)
    output_root = Path(args.output_root)

    results = [
        run_case(
            case,
            mode=args.mode,
            providers=args.providers,
            output_root=output_root,
            benchmark_path=benchmark_path,
            max_per_query=args.max_per_query,
        )
        for case in cases
    ]
    write_summary(
        results,
        mode=args.mode,
        providers=args.providers,
        output_root=output_root,
        summary_path=Path(args.summary_path),
    )
    if args.fail_on_pipeline_error:
        return 0 if all(result["returncode"] == 0 for result in results) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
