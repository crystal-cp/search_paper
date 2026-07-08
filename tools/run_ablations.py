"""Run pilot ablation configs over benchmark cases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tools.evaluate_run import CASE_ID_ALIASES, DEFAULT_BENCHMARK_PATH, load_benchmark_cases
    from tools.run_benchmark import case_question, provider_key_name
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_run import CASE_ID_ALIASES, DEFAULT_BENCHMARK_PATH, load_benchmark_cases
    from run_benchmark import case_question, provider_key_name


REPO_ROOT = Path(__file__).resolve().parents[1]

ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "full_system": {
        "flags": [],
        "disabled_modules": [],
    },
    "legacy_query_planning": {
        "flags": ["--legacy-query-planning"],
        "disabled_modules": ["intent_repair", "query_family"],
    },
    "no_query_family": {
        "flags": ["--skip-query-families"],
        "disabled_modules": ["query_family"],
    },
    "no_intent_repair": {
        "flags": ["--disable-intent-repair"],
        "disabled_modules": ["intent_repair"],
    },
    "no_query_repair": {
        "flags": ["--disable-query-repair"],
        "disabled_modules": ["query_repair"],
    },
    "no_domain_guardrail": {
        "flags": ["--disable-domain-guardrail"],
        "disabled_modules": ["domain_guardrail"],
    },
    "no_intent_centrality": {
        "flags": ["--disable-intent-centrality"],
        "disabled_modules": ["intent_centrality"],
    },
    "no_group_coverage_ranking": {
        "flags": ["--disable-group-coverage-ranking"],
        "disabled_modules": ["group_coverage_ranking"],
        "support_status": {"group_coverage_ranking": "partially_supported"},
    },
}


def normalize_case_id(case_id: str) -> str:
    return CASE_ID_ALIASES.get(case_id, case_id)


def parse_config_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in ABLATION_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown ablation config(s): {', '.join(unknown)}")
    return names


def select_cases(benchmark: dict[str, Any], case_id: str | None, mode: str) -> list[dict[str, Any]]:
    cases = list(benchmark.get("cases") or [])
    if case_id:
        normalized = normalize_case_id(case_id)
        cases = [case for case in cases if case.get("id") == normalized]
    if mode == "plan":
        cases = [case for case in cases if case.get("plan_only_eligible", True)]
    else:
        cases = [case for case in cases if case.get("full_retrieval_required", True)]
    return cases


def build_pipeline_command(
    case: dict[str, Any],
    *,
    config_name: str,
    mode: str,
    providers: list[str],
    output_dir: Path,
    max_per_query: int,
    query_family_provider_cap: int,
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
        "--query-family-provider-cap",
        str(query_family_provider_cap),
        "--ablation-config-name",
        config_name,
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
    command.extend(ABLATION_CONFIGS[config_name]["flags"])
    return command


def synthetic_provider_status(providers: list[str]) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for provider in providers:
        key_name = provider_key_name(provider)
        if key_name and not os.environ.get(key_name):
            state = "missing_api_key"
            message = f"{key_name} is not set for full ablation run."
        else:
            state = "not_attempted"
            message = "Pipeline did not write provider_status.json."
        status[provider] = {
            "status": state,
            "returned_paper_count": 0,
            "error_type": state,
            "message": message,
        }
    return status


def write_fallback_ablation_config(
    output_dir: Path,
    *,
    config_name: str,
    mode: str,
    command: list[str],
) -> None:
    config = ABLATION_CONFIGS[config_name]
    payload = {
        "ablation_config_name": config_name,
        "mode": mode,
        "disabled_modules": config.get("disabled_modules", []),
        "enabled_modules": [
            module
            for module in [
                "query_family",
                "query_repair",
                "domain_guardrail",
                "intent_repair",
                "intent_centrality",
                "group_coverage_ranking",
            ]
            if module not in set(config.get("disabled_modules", []))
        ],
        "cli_flags_used": config.get("flags", []),
        "support_status": config.get("support_status", {}),
        "pipeline_command": command,
        "written_by": "tools/run_ablations.py",
    }
    path = output_dir / "ablation_config.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload = {
                    **payload,
                    **existing,
                    "mode": existing.get("mode") or mode,
                    "pipeline_command": existing.get("pipeline_command") or command,
                }
        except json.JSONDecodeError:
            pass
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_one(
    case: dict[str, Any],
    *,
    config_name: str,
    mode: str,
    providers: list[str],
    output_root: Path,
    max_per_query: int,
    query_family_provider_cap: int,
) -> dict[str, Any]:
    output_dir = output_root / str(case["id"]) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_pipeline_command(
        case,
        config_name=config_name,
        mode=mode,
        providers=providers,
        output_dir=output_dir,
        max_per_query=max_per_query,
        query_family_provider_cap=query_family_provider_cap,
    )
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    record = {
        "case_id": case["id"],
        "config_name": config_name,
        "mode": mode,
        "providers": providers,
        "output_dir": str(output_dir),
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    (output_dir / "ablation_command.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_fallback_ablation_config(
        output_dir,
        config_name=config_name,
        mode=mode,
        command=command,
    )
    if mode == "full" and not (output_dir / "provider_status.json").exists():
        (output_dir / "provider_status.json").write_text(
            json.dumps(synthetic_provider_status(providers), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return record


def load_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def load_existing_run_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for command_path in sorted(output_root.glob("*/*/ablation_command.json")):
        try:
            data = json.loads(command_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data.setdefault("output_dir", str(command_path.parent))
            records.append(data)
    return records


def merge_run_records(
    existing: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge ablation run records without discarding earlier invocations."""

    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str, str]] = []
    for record in [*existing, *new_records]:
        key = (
            str(record.get("case_id", "")),
            str(record.get("config_name", "")),
            str(record.get("mode", "")),
            str(record.get("output_dir", "")),
        )
        if key not in merged:
            order.append(key)
        merged[key] = record
    return [merged[key] for key in order]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pilot ablation configs.")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--mode", choices=["plan", "full"], required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--providers", nargs="+", default=["openalex"])
    parser.add_argument("--output-root", default="outputs/ablations")
    parser.add_argument("--query-family-provider-cap", type=int, default=12)
    parser.add_argument("--max-per-query", type=int, default=10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear the output root before running. By default, manifests are appended/merged.",
    )
    parser.add_argument(
        "--fail-on-pipeline-error",
        action="store_true",
        help="Return non-zero if any ablation subprocess returns non-zero.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_names = parse_config_names(args.configs)
    benchmark = load_benchmark_cases(Path(args.benchmark))
    cases = select_cases(benchmark, args.case_id, args.mode)
    output_root = Path(args.output_root)
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    records = [
        run_one(
            case,
            config_name=config_name,
            mode=args.mode,
            providers=args.providers,
            output_root=output_root,
            max_per_query=args.max_per_query,
            query_family_provider_cap=args.query_family_provider_cap,
        )
        for case in cases
        for config_name in config_names
    ]
    manifest_path = output_root / "ablation_runs.json"
    output_root.mkdir(parents=True, exist_ok=True)
    merged_records = merge_run_records(
        [
            *load_existing_manifest(manifest_path),
            *load_existing_run_records(output_root),
        ],
        records,
    )
    manifest_path.write_text(
        json.dumps(merged_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.fail_on_pipeline_error:
        return 0 if all(record["returncode"] == 0 for record in records) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
