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

CLEAN_LLM_PLAN_BENCHMARK_CASE_IDS = {
    "ai_literature_screening",
    "mof_co2_capture",
    "thin_film_deposition",
    "sei_lithium_battery",
    "oer_spin_state",
}

SMALL_FULL_LLM_PILOT_CASE_IDS = {
    "sei_lithium_battery",
    "oer_spin_state",
}

ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "deterministic_baseline": {
        "flags": [],
        "disabled_modules": [],
        "mode_support": "plan_level",
        "support_status": {"full_system": "baseline"},
        "applies_query_changes": False,
        "uses_real_llm": False,
        "diagnostic_only": False,
        "warning_if_pilot_only": "",
    },
    "full_system": {
        "flags": [],
        "disabled_modules": [],
        "mode_support": "plan_level",
        "full_mode_support": "small_full_retrieval_safety_pilot",
        "support_status": {"full_system": "baseline"},
        "applies_query_changes": False,
        "uses_real_llm": False,
        "diagnostic_only": False,
        "warning_if_pilot_only": "",
    },
    "llm_intent_frame_only": {
        "flags": [
            "--enable-llm-intent-enhancer",
            "--llm-intent-provider",
            "fake",
            "--fake-llm-intent-mode",
            "valid",
        ],
        "disabled_modules": [],
        "enabled_modules_extra": ["llm_intent_frame_enhancer"],
        "mode_support": "plan_level",
        "support_status": {"llm_intent_frame_enhancer": "diagnostic_controlled"},
        "applies_query_changes": False,
        "uses_real_llm": False,
        "diagnostic_only": True,
        "warning_if_pilot_only": "Plan-level diagnostic only; not a formal LLM ablation conclusion.",
    },
    "llm_query_critic_diagnostic_only": {
        "flags": [
            "--enable-llm-query-critic",
            "--llm-query-critic-provider",
            "fake",
            "--fake-llm-query-critic-mode",
            "case_aware_weak_query",
        ],
        "disabled_modules": [],
        "enabled_modules_extra": ["llm_query_plan_critic"],
        "mode_support": "plan_level",
        "support_status": {"llm_query_plan_critic": "diagnostic_artifacts_only"},
        "applies_query_changes": False,
        "uses_real_llm": False,
        "diagnostic_only": True,
        "warning_if_pilot_only": "Critique artifacts only; query plan must remain unchanged.",
    },
    "llm_query_critic_repair_applied": {
        "flags": [
            "--enable-llm-query-critic",
            "--apply-llm-query-critic-repairs",
            "--llm-query-critic-provider",
            "fake",
            "--fake-llm-query-critic-mode",
            "case_aware_weak_query",
        ],
        "disabled_modules": [],
        "enabled_modules_extra": ["llm_query_plan_critic"],
        "mode_support": "plan_level",
        "full_mode_support": "small_full_retrieval_safety_pilot",
        "support_status": {"llm_query_plan_critic_repairs": "controlled_repair_pilot"},
        "applies_query_changes": True,
        "uses_real_llm": False,
        "diagnostic_only": False,
        "warning_if_pilot_only": "Controlled repair pilot; only deterministic rule-applied query changes are allowed.",
    },
    "llm_intent_plus_query_critic_repair": {
        "flags": [
            "--enable-llm-intent-enhancer",
            "--llm-intent-provider",
            "fake",
            "--fake-llm-intent-mode",
            "valid",
            "--enable-llm-query-critic",
            "--apply-llm-query-critic-repairs",
            "--llm-query-critic-provider",
            "fake",
            "--fake-llm-query-critic-mode",
            "case_aware_weak_query",
        ],
        "disabled_modules": [],
        "enabled_modules_extra": [
            "llm_intent_frame_enhancer",
            "llm_query_plan_critic",
        ],
        "mode_support": "plan_level",
        "full_mode_support": "small_full_retrieval_safety_pilot",
        "support_status": {
            "llm_intent_frame_enhancer": "diagnostic_controlled",
            "llm_query_plan_critic_repairs": "controlled_repair_pilot",
        },
        "applies_query_changes": True,
        "uses_real_llm": False,
        "diagnostic_only": False,
        "warning_if_pilot_only": "Controlled combined LLM plan-level pilot; not a full retrieval ablation.",
    },
    "llm_query_critic_only": {
        "flags": [
            "--enable-llm-query-critic",
            "--llm-query-critic-provider",
            "fake",
            "--fake-llm-query-critic-mode",
            "case_aware_weak_query",
        ],
        "disabled_modules": [],
        "enabled_modules_extra": ["llm_query_plan_critic"],
        "mode_support": "plan_level",
        "support_status": {"llm_query_plan_critic": "diagnostic_artifacts_only"},
        "applies_query_changes": False,
        "uses_real_llm": False,
        "diagnostic_only": True,
        "warning_if_pilot_only": "Backward-compatible alias for llm_query_critic_diagnostic_only.",
        "alias_for": "llm_query_critic_diagnostic_only",
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


def select_cases(
    benchmark: dict[str, Any],
    case_id: str | None,
    mode: str,
    case_group: str = "clean_benchmark",
) -> list[dict[str, Any]]:
    cases = list(benchmark.get("cases") or [])
    if case_id:
        normalized = normalize_case_id(case_id)
        cases = [case for case in cases if case.get("id") == normalized]
    if mode == "plan":
        cases = [case for case in cases if case.get("plan_only_eligible", True)]
    else:
        cases = [case for case in cases if case.get("full_retrieval_required", True)]
        if not case_id and case_group == "small_full_llm_pilot":
            cases = [
                case
                for case in cases
                if case.get("id") in SMALL_FULL_LLM_PILOT_CASE_IDS
            ]
    if not case_id and mode == "plan":
        if case_group == "clean_benchmark":
            cases = [
                case
                for case in cases
                if case.get("id") in CLEAN_LLM_PLAN_BENCHMARK_CASE_IDS
                and not case.get("diagnostic_only")
                and not case.get("stress_case")
            ]
        elif case_group == "weak_plan_positive_controls":
            cases = [
                case
                for case in cases
                if str(case.get("llm_plan_diagnostic_group") or "")
                == "weak_plan_positive_control"
                or bool(case.get("stress_case"))
            ]
    return cases


def llm_query_critic_config_enabled(config_name: str) -> bool:
    config = ABLATION_CONFIGS[config_name]
    modules = set(config.get("enabled_modules_extra", []) or [])
    return "llm_query_plan_critic" in modules or config_name in {
        "llm_query_critic_diagnostic_only",
        "llm_query_critic_repair_applied",
        "llm_query_critic_only",
        "llm_intent_plus_query_critic_repair",
    }


def case_extra_flags(case: dict[str, Any], *, config_name: str, mode: str) -> list[str]:
    if mode != "plan" or not llm_query_critic_config_enabled(config_name):
        return []
    if str(case.get("llm_plan_diagnostic_group") or "") != "weak_plan_positive_control":
        return []
    configured = case.get("llm_plan_extra_flags")
    if isinstance(configured, list) and configured:
        return [str(flag) for flag in configured]
    return ["--legacy-query-planning", "--skip-query-families", "--disable-query-repair"]


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
    extra_flags = case_extra_flags(case, config_name=config_name, mode=mode)
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
    command.extend(extra_flags)
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
    case: dict[str, Any],
    config_name: str,
    mode: str,
    command: list[str],
    extra_flags: list[str] | None = None,
) -> None:
    config = ABLATION_CONFIGS[config_name]
    case_metadata = {
        "case_id": str(case.get("id") or ""),
        "diagnostic_group": str(
            case.get("llm_plan_diagnostic_group")
            or ("weak_plan_positive_control" if case.get("stress_case") else "clean_benchmark_diagnostic")
        ),
        "diagnostic_only": bool(case.get("diagnostic_only", False)),
        "stress_case": bool(case.get("stress_case", False)),
        "formal_benchmark_inclusion": str(
            case.get("formal_benchmark_inclusion")
            or case.get("benchmark_inclusion")
            or ""
        ),
        "case_extra_flags_used": extra_flags or [],
    }
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
        ]
        + list(config.get("enabled_modules_extra", [])),
        "cli_flags_used": config.get("flags", []),
        "support_status": config.get("support_status", {}),
        "mode_support": config.get("mode_support", ""),
        "full_mode_support": config.get("full_mode_support", ""),
        "applies_query_changes": bool(config.get("applies_query_changes", False)),
        "uses_real_llm": bool(config.get("uses_real_llm", False)),
        "diagnostic_only": bool(config.get("diagnostic_only", False)),
        "warning_if_pilot_only": config.get("warning_if_pilot_only", ""),
        "alias_for": config.get("alias_for", ""),
        "case_metadata": case_metadata,
        "pipeline_command": command,
        "written_by": "tools/run_ablations.py",
    }
    path = output_dir / "ablation_config.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                merged_support_status = {
                    **dict(existing.get("support_status", {}) or {}),
                    **dict(config.get("support_status", {}) or {}),
                }
                payload = {
                    **payload,
                    **existing,
                    "mode": existing.get("mode") or mode,
                    "support_status": merged_support_status,
                    "case_metadata": {
                        **case_metadata,
                        **dict(existing.get("case_metadata", {}) or {}),
                    },
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
    extra_flags = case_extra_flags(case, config_name=config_name, mode=mode)
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
        case=case,
        config_name=config_name,
        mode=mode,
        command=command,
        extra_flags=extra_flags,
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
    parser.add_argument(
        "--case-group",
        choices=[
            "clean_benchmark",
            "weak_plan_positive_controls",
            "small_full_llm_pilot",
            "all",
        ],
        default="clean_benchmark",
        help="Case family to run when --case-id is not provided.",
    )
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
    cases = select_cases(benchmark, args.case_id, args.mode, args.case_group)
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
