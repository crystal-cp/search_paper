"""Evaluate an existing search_paper output directory.

This tool is intentionally artifact-only: it reads files produced by the
pipeline and computes benchmark-facing metrics without importing or changing the
core pipeline.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "data" / "benchmark_cases.yaml"

SINGLE_ACRONYM_QUERY = re.compile(
    r'^\+?"?(AI|LLM|MOF|OER|PFM|SEI|SHG)"?$',
    re.IGNORECASE,
)

SUCCESS_PROVIDER_STATES = {
    "success",
    "success_no_results",
    "partial_success",
    "partial_success_rate_limited",
}

FATAL_PROVIDER_STATES = {
    "api_key_missing",
    "error",
    "failed",
    "missing_api_key",
    "not_attempted",
    "rate_limited",
}

CASE_ID_ALIASES = {
    "ai_screening": "ai_literature_screening",
    "mof_co2": "mof_co2_capture",
    "thin_film_methods": "thin_film_deposition",
    "sei_lithium": "sei_lithium_battery",
    "sei_full": "sei_lithium_battery",
    "oer": "oer_spin_state",
}

HTML_TAG_RE = re.compile(r"<[^>]+>")
UNICODE_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"

SEI_MIXED_CHEMISTRY_TERMS = [
    "sodium",
    "sodium ion",
    "sodium-ion",
    "potassium",
    "potassium ion",
    "potassium-ion",
    "zinc",
    "zinc-ion",
    "zn",
    "azib",
    "magnesium",
    "beyond lithium",
]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def flatten_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(flatten_strings(item))
    elif isinstance(value, list):
        for item in value:
            strings.extend(flatten_strings(item))
    return strings


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def normalized_query_key(query: str) -> str:
    return re.sub(r"\s+", " ", str(query).strip().lower())


def duplicate_query_count(queries: list[str]) -> int:
    keys = [normalized_query_key(query) for query in queries if str(query).strip()]
    return max(len(keys) - len(set(keys)), 0)


def queries_from_query_artifact(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []
    if "queries" in data:
        return unique_strings(flatten_strings(data.get("queries")))
    return unique_strings(flatten_strings(data.get("queries_by_provider", data)))


def collect_final_provider_queries(
    planned_queries: dict[str, Any],
    query_provenance: dict[str, Any],
    retrieval_diagnostics: dict[str, Any],
) -> list[str]:
    candidates: list[str] = []

    for source in (query_provenance, planned_queries, retrieval_diagnostics):
        for key in (
            "final_provider_queries",
            "queries_by_provider",
            "final_openalex_queries",
            "final_semantic_scholar_queries",
        ):
            if key in source:
                candidates.extend(flatten_strings(source.get(key)))

    query_plan = planned_queries.get("query_plan", {})
    if isinstance(query_plan, dict):
        for key in ("openalex_queries", "semantic_scholar_queries", "queries"):
            candidates.extend(flatten_strings(query_plan.get(key)))

    if not candidates:
        for key in ("openalex_queries", "semantic_scholar_queries", "queries"):
            candidates.extend(flatten_strings(planned_queries.get(key)))

    return unique_strings(candidates)


def collect_final_provider_query_candidates(
    planned_queries: dict[str, Any],
    query_provenance: dict[str, Any],
    retrieval_diagnostics: dict[str, Any],
) -> list[str]:
    candidates: list[str] = []
    for source in (query_provenance, planned_queries, retrieval_diagnostics):
        for key in (
            "final_provider_queries",
            "queries_by_provider",
            "final_openalex_queries",
            "final_semantic_scholar_queries",
        ):
            if key in source:
                candidates.extend(flatten_strings(source.get(key)))
    query_plan = planned_queries.get("query_plan", {})
    if isinstance(query_plan, dict):
        for key in ("openalex_queries", "semantic_scholar_queries", "queries"):
            candidates.extend(flatten_strings(query_plan.get(key)))
    if not candidates:
        for key in ("openalex_queries", "semantic_scholar_queries", "queries"):
            candidates.extend(flatten_strings(planned_queries.get(key)))
    return [str(query).strip() for query in candidates if str(query).strip()]


def llm_trace_bool(trace: dict[str, Any], key: str, default: bool = False) -> bool:
    if not isinstance(trace, dict):
        return default
    return bool(trace.get(key, default))


def llm_trace_count(trace: dict[str, Any], key: str) -> int:
    if not isinstance(trace, dict):
        return 0
    value = first_number(trace.get(key))
    return int(value or 0)


def repair_term_counts(repair_artifact: dict[str, Any]) -> tuple[int, int]:
    if not isinstance(repair_artifact, dict):
        return (0, 0)
    grounded = 0
    rejected = 0
    for record in repair_artifact.get("applied_issue_records", []) or []:
        if not isinstance(record, dict):
            continue
        grounded += len(record.get("applied_terms", []) or [])
        rejected += len(record.get("rejected_terms", []) or [])
    return (grounded, rejected)


def repair_term_lists(repair_artifact: dict[str, Any]) -> tuple[list[str], list[str]]:
    if not isinstance(repair_artifact, dict):
        return ([], [])
    applied_terms: list[str] = []
    rejected_terms: list[str] = []
    for record in repair_artifact.get("applied_issue_records", []) or []:
        if not isinstance(record, dict):
            continue
        applied_terms.extend(
            str(term)
            for term in (record.get("applied_terms", []) or [])
            if str(term).strip()
        )
        rejected_terms.extend(
            str(term.get("term") if isinstance(term, dict) else term)
            for term in (record.get("rejected_terms", []) or [])
            if str(term.get("term") if isinstance(term, dict) else term).strip()
        )
    return (unique_strings(applied_terms), unique_strings(rejected_terms))


def normalized_query_signature(queries: list[str]) -> tuple[str, ...]:
    return tuple(sorted({normalized_query_key(query) for query in queries if str(query).strip()}))


def direct_final_queries_from_artifact(data: dict[str, Any]) -> list[str]:
    if not isinstance(data, dict):
        return []
    candidates: list[str] = []
    for key in (
        "final_provider_queries",
        "queries_by_provider",
        "final_openalex_queries",
        "final_semantic_scholar_queries",
        "queries",
    ):
        if key in data:
            candidates.extend(flatten_strings(data.get(key)))
    query_plan = data.get("query_plan", {})
    if isinstance(query_plan, dict):
        for key in ("openalex_queries", "semantic_scholar_queries", "queries"):
            candidates.extend(flatten_strings(query_plan.get(key)))
    return unique_strings(candidates)


def query_artifacts_consistent(query_lists: list[list[str]]) -> bool:
    signatures = [
        normalized_query_signature(queries)
        for queries in query_lists
        if queries
    ]
    if len(signatures) <= 1:
        return True
    first = signatures[0]
    return all(signature == first for signature in signatures[1:])


def llm_query_before_after_example(
    before_queries: list[str],
    after_queries: list[str],
) -> tuple[str, str]:
    if not before_queries and not after_queries:
        return ("", "")
    max_len = max(len(before_queries), len(after_queries))
    for index in range(max_len):
        before = before_queries[index] if index < len(before_queries) else ""
        after = after_queries[index] if index < len(after_queries) else ""
        if normalized_query_key(before) != normalized_query_key(after):
            return (before, after)
    return (
        before_queries[0] if before_queries else "",
        after_queries[0] if after_queries else "",
    )


def research_gap_generation_status(
    run: Path,
    evaluation: dict[str, Any],
    *,
    retrieval_performed: bool,
) -> str:
    if isinstance(evaluation, dict):
        for key in ("research_gap_generation_status", "gap_generation_status"):
            value = str(evaluation.get(key) or "").strip()
            if value:
                return value
    for path in (run / "research_gap_matrix.csv", run / "research_gap_matrix.tsv"):
        rows = read_csv_rows(path)
        if rows:
            value = str(rows[0].get("gap_generation_status") or "").strip()
            if value:
                return value
    if not retrieval_performed:
        return "skipped"
    return ""


def load_benchmark_cases(path: Path = DEFAULT_BENCHMARK_PATH) -> dict[str, Any]:
    """Load benchmark YAML, with a small fallback parser for minimal envs."""

    text = read_text(path)
    if not text:
        return {"cases": []}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return parse_benchmark_cases_fallback(text)


def parse_benchmark_cases_fallback(text: str) -> dict[str, Any]:
    """Parse the fields needed by the benchmark tools if PyYAML is absent.

    This is not a general YAML parser. It supports this repository's benchmark
    schema well enough to read IDs, questions, anchors, and forbidden patterns.
    """

    cases: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section = ""
    current_anchor: dict[str, Any] | None = None
    in_cases = False

    for raw_line in text.splitlines():
        if raw_line.strip() == "cases:":
            in_cases = True
            continue
        if not in_cases:
            continue

        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        case_match = re.match(r"^  - id:\s*(.+)$", line)
        if case_match:
            current = {
                "id": clean_yaml_scalar(case_match.group(1)),
                "expected_query_anchors": [],
                "forbidden_top10_patterns": [],
            }
            cases.append(current)
            section = ""
            current_anchor = None
            continue

        if current is None:
            continue

        key_match = re.match(r"^    ([A-Za-z0-9_]+):\s*(.*)$", line)
        if key_match:
            key, value = key_match.groups()
            section = key
            current_anchor = None
            if key in {"expected_query_anchors", "forbidden_top10_patterns"}:
                continue
            if value and value != ">":
                current[key] = clean_yaml_scalar(value)
            continue

        if section == "expected_query_anchors":
            anchor_match = re.match(r"^      - id:\s*(.+)$", line)
            if anchor_match:
                current_anchor = {
                    "id": clean_yaml_scalar(anchor_match.group(1)),
                    "any_of": [],
                }
                current["expected_query_anchors"].append(current_anchor)
                continue
            term_match = re.match(r"^          -\s*(.+)$", line)
            if term_match and current_anchor is not None:
                current_anchor["any_of"].append(clean_yaml_scalar(term_match.group(1)))
                continue

        if section == "forbidden_top10_patterns":
            pattern_match = re.match(r"^      -\s*(.+)$", line)
            if pattern_match:
                current["forbidden_top10_patterns"].append(
                    clean_yaml_scalar(pattern_match.group(1))
                )

    return {"cases": cases}


def clean_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {'"', "'"}
    ):
        return value[1:-1]
    return value


def find_benchmark_case(
    benchmark: dict[str, Any],
    run_dir: Path,
    case_id: str | None = None,
) -> dict[str, Any] | None:
    cases = benchmark.get("cases") or []
    if case_id:
        case_id = CASE_ID_ALIASES.get(case_id, case_id)
        for case in cases:
            if case.get("id") == case_id:
                return case
        return None

    run_key = re.sub(r"[^a-z0-9]+", "_", str(run_dir).lower())
    exact_matches = [case for case in cases if str(case.get("id", "")).lower() in run_key]
    if len(exact_matches) == 1:
        return exact_matches[0]

    display_matches = []
    for case in cases:
        display = re.sub(r"[^a-z0-9]+", "_", str(case.get("display_name", "")).lower())
        if display and display in run_key:
            display_matches.append(case)
    if len(display_matches) == 1:
        return display_matches[0]

    prefix_matches = []
    for case in cases:
        prefix = str(case.get("id", "")).split("_", maxsplit=1)[0].lower()
        if prefix and re.search(rf"(^|_){re.escape(prefix)}(_|$)", run_key):
            prefix_matches.append(case)
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def calculate_anchor_coverage(case: dict[str, Any] | None, text_blob: str) -> dict[str, Any]:
    anchors = (case or {}).get("expected_query_anchors") or []
    if not anchors:
        return {
            "score": None,
            "covered": 0,
            "total": 0,
            "covered_anchor_ids": [],
            "missing_anchor_ids": [],
        }

    lowered = text_blob.lower()
    covered_ids: list[str] = []
    missing_ids: list[str] = []
    for anchor in anchors:
        anchor_id = str(anchor.get("id") or "unnamed_anchor")
        terms = [str(term).lower() for term in anchor.get("any_of", []) if str(term).strip()]
        if terms and any(term in lowered for term in terms):
            covered_ids.append(anchor_id)
        else:
            missing_ids.append(anchor_id)

    total = len(anchors)
    return {
        "score": len(covered_ids) / total if total else None,
        "covered": len(covered_ids),
        "total": total,
        "covered_anchor_ids": covered_ids,
        "missing_anchor_ids": missing_ids,
    }


def provider_success_rate(provider_status: dict[str, Any]) -> float | None:
    statuses = [
        str(status.get("status") or "").strip().lower()
        for status in provider_status.values()
        if isinstance(status, dict)
    ]
    if not statuses:
        return None
    successes = 0
    for status in statuses:
        if status in SUCCESS_PROVIDER_STATES:
            successes += 1
        elif status and status not in FATAL_PROVIDER_STATES:
            successes += 1
    return successes / len(statuses)


def first_number(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def normalize_match_text(text: str) -> str:
    """Normalize artifact text for heuristic pattern matching."""

    value = html.unescape(str(text or ""))
    value = HTML_TAG_RE.sub(" ", value)
    for dash in UNICODE_DASHES:
        value = value.replace(dash, "-")
    value = value.lower()
    value = re.sub(r"[\u00a0\t\r\n]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def match_text_forms(text: str) -> list[str]:
    normalized = normalize_match_text(text)
    return [normalized, normalized.replace("-", " ")]


def pattern_forms(pattern: str) -> list[str]:
    normalized = normalize_match_text(pattern)
    return [normalized, normalized.replace("-", " ")]


def form_contains_pattern(text_form: str, pattern_form: str) -> bool:
    if not pattern_form:
        return False
    if re.fullmatch(r"[a-z0-9]{1,3}", pattern_form):
        return re.search(rf"(?<![a-z0-9]){re.escape(pattern_form)}(?![a-z0-9])", text_form) is not None
    return pattern_form in text_form


def text_matches_any_pattern(text: str, patterns: list[str]) -> bool:
    text_forms = match_text_forms(text)
    for pattern in patterns:
        for pattern_form in pattern_forms(pattern):
            if any(form_contains_pattern(text_form, pattern_form) for text_form in text_forms):
                return True
    return False


def matched_patterns(text: str, patterns: list[str]) -> list[str]:
    matches: list[str] = []
    text_forms = match_text_forms(text)
    for pattern in patterns:
        pattern_matched = False
        for pattern_form in pattern_forms(pattern):
            if any(form_contains_pattern(text_form, pattern_form) for text_form in text_forms):
                pattern_matched = True
                break
        if pattern_matched:
            matches.append(pattern)
    return unique_strings(matches)


def allows_comparative_battery_systems(case: dict[str, Any] | None) -> bool:
    return bool(
        (case or {}).get("allow_comparative_battery_systems")
        or (case or {}).get("allows_comparative_battery_systems")
    )


def has_lithium_specific_mixed_chemistry(text: str, case: dict[str, Any] | None) -> bool:
    if allows_comparative_battery_systems(case):
        return False
    if str((case or {}).get("id", "")) != "sei_lithium_battery":
        return False
    forms = match_text_forms(text)
    has_lithium = any(
        form_contains_pattern(text_form, "lithium") or form_contains_pattern(text_form, "li")
        for text_form in forms
    )
    has_non_lithium = text_matches_any_pattern(text, SEI_MIXED_CHEMISTRY_TERMS)
    return has_lithium and has_non_lithium


def row_has_forbidden_pattern(
    row: dict[str, str],
    patterns: list[str],
    case: dict[str, Any] | None,
) -> bool:
    text = ranked_row_text(row)
    return text_matches_any_pattern(text, patterns) or has_lithium_specific_mixed_chemistry(
        text,
        case,
    )


def is_overbroad_query(query: str) -> bool:
    cleaned = query.strip().strip('"')
    if not cleaned:
        return False
    if SINGLE_ACRONYM_QUERY.match(cleaned):
        return True
    lowered = cleaned.lower()
    if lowered in {
        "comparison",
        "failure mechanism",
        "human-in-the-loop",
        "in situ",
        "ex situ",
        "spin state",
        "thin film",
        "lithium-ion battery",
        "experimental characterization",
        "theoretical mechanism",
    }:
        return True
    word_count = len(re.findall(r"[A-Za-z0-9]+", cleaned))
    has_anchor_signal = any(
        signal in lowered
        for signal in (
            "battery",
            "catalyst",
            "capture",
            "deposition",
            "literature",
            "screening",
            "interphase",
            "oxygen evolution",
            "mof",
            "co2",
        )
    )
    return word_count <= 2 and not has_anchor_signal


def query_words(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_match_text(query))


def has_any(text: str, terms: list[str]) -> bool:
    return text_matches_any_pattern(text, terms)


def repeated_phrase_query(query: str) -> bool:
    words = query_words(query)
    if not words:
        return False
    for left, right in zip(words, words[1:]):
        if left == right:
            return True
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    if any(count >= 3 for count in counts.values()):
        return True
    for size in (2, 3):
        seen: set[tuple[str, ...]] = set()
        for index in range(0, len(words) - size + 1):
            ngram = tuple(words[index : index + size])
            if ngram in seen:
                return True
            seen.add(ngram)
    return False


def query_quality_flags(query: str, case: dict[str, Any] | None) -> dict[str, bool]:
    case_id = str((case or {}).get("id") or "")
    normalized = normalize_match_text(query)
    words = query_words(query)
    word_count = len(words)
    repeated = repeated_phrase_query(query)
    overbroad = is_overbroad_query(query)
    weak = overbroad or repeated
    single_axis = False

    if case_id == "mof_co2_capture":
        has_mof = has_any(query, ["mof", "metal organic framework", "metal-organic framework"])
        has_co2 = has_any(query, ["co2", "carbon dioxide", "adsorption", "capture"])
        has_aspect = has_any(
            query,
            [
                "pore size",
                "pore",
                "functional group",
                "functionalized",
                "water stability",
                "stability",
                "humidity",
                "moisture",
                "adsorption performance",
                "selectivity",
            ],
        )
        weak = weak or not (has_mof and has_co2) or (
            has_mof and has_co2 and not has_aspect and word_count <= 5
        )
        overbroad = overbroad or (
            has_mof and has_co2 and not has_aspect and word_count <= 5
        )
        single_axis = has_mof and has_co2 and not has_aspect

    elif case_id == "thin_film_deposition":
        has_thin_film = has_any(query, ["thin film", "deposition", "film"])
        methods = [
            term
            for term in ["ald", "atomic layer deposition", "pld", "sputter", "sputtering", "cvd"]
            if has_any(query, [term])
        ]
        has_comparison = has_any(
            query,
            ["compare", "comparison", "versus", "vs", "tradeoff", "method selection"],
        ) or len(methods) >= 2
        weak = weak or (
            has_thin_film
            and not has_comparison
            and (
                word_count <= 4
                or normalized in {"deposition film", "thin film deposition"}
            )
        )
        overbroad = overbroad or normalized in {"deposition film", "thin film deposition"}
        single_axis = len(methods) == 1 and not has_comparison

    elif case_id == "ai_literature_screening":
        has_ai = has_any(
            query,
            ["ai", "llm", "large language model", "artificial intelligence"],
        )
        has_screening = has_any(
            query,
            ["systematic review", "literature screening", "study selection", "screening"],
        )
        has_eval_or_human = has_any(
            query,
            [
                "human feedback",
                "human-in-the-loop",
                "evidence validation",
                "recall",
                "precision",
                "accuracy",
                "evaluation",
            ],
        )
        weak = weak or not (has_ai and has_screening) or (
            not has_eval_or_human and word_count <= 4
        )
        overbroad = overbroad or normalized in {"literature screening", "human-in-the-loop"}
        single_axis = (has_ai != has_screening) or (
            has_ai and has_screening and not has_eval_or_human and word_count <= 4
        )

    return {
        "overbroad": overbroad,
        "repeated_phrase": repeated,
        "single_axis": single_axis,
        "weak": weak,
    }


def query_quality_metrics(
    case: dict[str, Any] | None,
    final_queries: list[str],
) -> dict[str, Any]:
    anchor = calculate_anchor_coverage(case, " ".join(final_queries))
    details = [
        {"query": query, **query_quality_flags(query, case)}
        for query in final_queries
    ]
    total = len(details)
    overbroad_queries = [item["query"] for item in details if item["overbroad"]]
    repeated_queries = [item["query"] for item in details if item["repeated_phrase"]]
    single_axis_queries = [item["query"] for item in details if item["single_axis"]]
    weak_queries = [item["query"] for item in details if item["weak"]]
    bad_queries = {
        item["query"]
        for item in details
        if item["overbroad"]
        or item["repeated_phrase"]
        or item["single_axis"]
        or item["weak"]
    }
    expected_anchor_coverage = anchor["score"]
    quality_component = 1.0 - (len(bad_queries) / total if total else 0.0)
    anchor_component = expected_anchor_coverage if expected_anchor_coverage is not None else 0.0
    query_quality_score = max(
        0.0,
        min(1.0, 0.6 * anchor_component + 0.4 * quality_component),
    )
    return {
        "expected_anchor_coverage": expected_anchor_coverage,
        "expected_anchor_coverage_detail": anchor,
        "overbroad_query_count": len(overbroad_queries),
        "overbroad_queries": overbroad_queries,
        "repeated_phrase_query_count": len(repeated_queries),
        "repeated_phrase_queries": repeated_queries,
        "single_axis_query_count": len(single_axis_queries),
        "single_axis_queries": single_axis_queries,
        "weak_query_count": len(weak_queries),
        "weak_queries": weak_queries,
        "query_quality_score": round(query_quality_score, 4),
        "query_quality_details": details,
    }


def average_float(rows: list[dict[str, str]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            text = str(row.get(field, "")).strip()
            if text:
                values.append(float(text))
        except ValueError:
            continue
    if not values:
        return None
    return sum(values) / len(values)


def ranked_row_text(row: dict[str, str]) -> str:
    fields = [
        "title",
        "abstract",
        "claim",
        "evidence",
        "evidence_sentence",
        "matched_evidence_sentence",
        "supporting_sentence",
        "matched_sentence",
        "abstract_snippet",
        "relevance_reason",
        "off_topic_reason",
        "primary_reason",
    ]
    return " ".join(str(row.get(field, "")) for field in fields)


def evaluate_run(
    run_dir: Path | str,
    *,
    benchmark_path: Path | str = DEFAULT_BENCHMARK_PATH,
    case_id: str | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    benchmark = load_benchmark_cases(Path(benchmark_path))
    case = find_benchmark_case(benchmark, run, case_id)

    planned_queries = read_json(run / "planned_queries.json", {})
    query_provenance = read_json(run / "query_provenance.json", {})
    provider_status = read_json(run / "provider_status.json", {})
    retrieval_diagnostics = read_json(run / "retrieval_diagnostics.json", {})
    query_repair_stage_status = read_json(run / "query_repair_stage_status.json", {})
    query_repair_suggestions = read_json(run / "query_repair_suggestions.json", {})
    evaluation = read_json(run / "evaluation.json", {})
    raw_candidate_queries_artifact = read_json(
        run / "raw_candidate_queries_before_repair.json",
        {},
    )
    final_queries_after_repair_artifact = read_json(
        run / "final_queries_after_repair.json",
        {},
    )
    ranked_rows = read_csv_rows(run / "ranked_papers.csv")
    merged_rows = read_csv_rows(run / "merged_papers.csv")
    exploration_quality = read_json(run / "exploration_quality.json", {})
    user_report = read_text(run / "user_report.md")
    llm_intent_trace = read_json(run / "llm_intent_enhancement_trace.json", {})
    llm_query_critic_trace = read_json(run / "llm_query_critic_trace.json", {})
    query_repair_after_llm_critic = read_json(
        run / "query_repair_after_llm_critic.json",
        {},
    )
    query_plan_before_llm_critic = read_json(
        run / "query_plan_before_llm_critic.json",
        {},
    )
    query_plan_after_llm_critic = read_json(
        run / "query_plan_after_llm_critic.json",
        {},
    )

    if not provider_status and isinstance(retrieval_diagnostics, dict):
        provider_status = retrieval_diagnostics.get("provider_status") or {}

    final_queries = collect_final_provider_queries(
        planned_queries,
        query_provenance,
        retrieval_diagnostics,
    )
    final_query_candidates = collect_final_provider_query_candidates(
        planned_queries,
        query_provenance,
        retrieval_diagnostics,
    )
    raw_candidate_queries = queries_from_query_artifact(raw_candidate_queries_artifact)
    final_queries_after_repair = queries_from_query_artifact(
        final_queries_after_repair_artifact
    )
    raw_set = {query.lower() for query in raw_candidate_queries}
    final_after_repair_set = {query.lower() for query in final_queries_after_repair}
    raw_to_final_query_change_count = len(raw_set.symmetric_difference(final_after_repair_set))
    repair_disabled = bool(
        query_repair_stage_status.get(
            "disabled_by_ablation",
            query_repair_suggestions.get("disabled_by_ablation", False),
        )
    )
    repair_reason = str(
        query_repair_stage_status.get("reason_if_no_difference", "")
    ).lower()
    llm_stage_repair_applied = bool(
        query_repair_stage_status.get("llm_query_critic_repair_applied")
    )
    repair_disabled_but_sanitizer_active = repair_disabled and (
        not llm_stage_repair_applied
    ) and (
        "upstream sanitizer" in repair_reason or raw_to_final_query_change_count > 0
    )
    no_query_repair_conclusive = not (
        repair_disabled
        and (
            repair_disabled_but_sanitizer_active
            or raw_set == final_after_repair_set
        )
    )
    single_acronym_queries = [
        query for query in final_queries if SINGLE_ACRONYM_QUERY.match(query.strip())
    ]
    quality_metrics = query_quality_metrics(case, final_queries)
    overbroad_queries = quality_metrics["overbroad_queries"]

    text_blob = json.dumps(
        {
            "planned_queries": planned_queries,
            "query_provenance": query_provenance,
            "exploration_quality": exploration_quality,
            "user_report": user_report,
            "final_queries": final_queries,
        },
        ensure_ascii=False,
    )
    anchor_coverage = calculate_anchor_coverage(case, text_blob)
    query_family_coverage = (
        exploration_quality.get("query_family_coverage")
        if isinstance(exploration_quality, dict)
        and "query_family_coverage" in exploration_quality
        else exploration_quality.get("query_family_coverage_score")
        if isinstance(exploration_quality, dict)
        and "query_family_coverage_score" in exploration_quality
        else anchor_coverage["score"]
    )
    missing_expected_anchor_count = (
        len(anchor_coverage.get("missing_anchor_ids", []))
        if anchor_coverage.get("total")
        else None
    )
    cross_domain_injection_count = first_number(
        exploration_quality.get("cross_domain_injection_count")
        if isinstance(exploration_quality, dict)
        else None,
        retrieval_diagnostics.get("cross_domain_injection_count"),
    )

    raw_count = first_number(
        retrieval_diagnostics.get("raw_retrieved_paper_count"),
        retrieval_diagnostics.get("raw_retrieved_count"),
        retrieval_diagnostics.get("original_paper_count"),
        retrieval_diagnostics.get("raw_paper_count"),
        exploration_quality.get("raw_retrieved_paper_count")
        if isinstance(exploration_quality, dict)
        else None,
    )
    merged_count = first_number(
        retrieval_diagnostics.get("merged_paper_count"),
        retrieval_diagnostics.get("merged_count"),
        exploration_quality.get("merged_paper_count")
        if isinstance(exploration_quality, dict)
        else None,
    )
    if merged_count is None:
        merged_count = len(merged_rows) if merged_rows else len(ranked_rows)

    duplicate_count = first_number(
        retrieval_diagnostics.get("duplicate_count"),
        retrieval_diagnostics.get("duplicate_records_removed"),
    )
    if duplicate_count is None and raw_count is not None and merged_count is not None:
        duplicate_count = max(raw_count - merged_count, 0)
    duplicate_ratio = (
        duplicate_count / raw_count
        if raw_count and duplicate_count is not None
        else None
    )

    forbidden_patterns = [
        str(pattern)
        for pattern in ((case or {}).get("forbidden_top10_patterns") or [])
        if str(pattern).strip()
    ]
    top20_false_positive_rows = [
        index + 1
        for index, row in enumerate(ranked_rows[:20])
        if row_has_forbidden_pattern(row, forbidden_patterns, case)
    ]
    forbidden_top10_rows = [
        index + 1
        for index, row in enumerate(ranked_rows[:10])
        if row_has_forbidden_pattern(row, forbidden_patterns, case)
    ]
    forbidden_top20_rows = [
        index + 1
        for index, row in enumerate(ranked_rows[:20])
        if row_has_forbidden_pattern(row, forbidden_patterns, case)
    ]
    top20_false_positive_count: int | None
    top20_false_positive_count = (
        len(top20_false_positive_rows) if forbidden_patterns else None
    )

    must_read_rows = [
        row
        for row in ranked_rows
        if str(row.get("reading_priority", "")).strip().lower() == "must_read"
    ]
    must_read_false_positive_count = (
        sum(
            1
            for row in must_read_rows
            if row_has_forbidden_pattern(row, forbidden_patterns, case)
        )
        if forbidden_patterns
        else None
    )
    include_rows = [
        row
        for row in ranked_rows
        if str(row.get("decision", "")).strip().lower() == "include"
    ]
    include_false_positive_count = (
        sum(
            1
            for row in include_rows
            if row_has_forbidden_pattern(row, forbidden_patterns, case)
        )
        if forbidden_patterns
        else None
    )
    must_read_precision = (
        (len(must_read_rows) - must_read_false_positive_count) / len(must_read_rows)
        if forbidden_patterns and must_read_rows and must_read_false_positive_count is not None
        else None
    )

    report_lower = user_report.lower()
    reading_path_diagnostics = (
        evaluation.get("reading_path_diagnostics", {})
        if isinstance(evaluation, dict)
        else {}
    )
    reading_priority_policy = (
        evaluation.get("reading_priority_policy", {})
        if isinstance(evaluation, dict)
        else {}
    )
    retrieval_status = str(
        evaluation.get(
            "retrieval_status",
            retrieval_diagnostics.get(
                "retrieval_status",
                exploration_quality.get("retrieval_status")
                if isinstance(exploration_quality, dict)
                else "",
            ),
        )
        or ""
    ).strip()
    ranked_papers_based_on_real_retrieval = evaluation.get(
        "ranked_papers_based_on_real_retrieval"
    )
    if ranked_papers_based_on_real_retrieval is None:
        ranked_papers_based_on_real_retrieval = bool(
            (raw_count or 0) > 0
            and retrieval_status not in {"planning_only", "retrieval_not_performed"}
        )
    retrieval_performed = bool(
        ranked_papers_based_on_real_retrieval
        or (raw_count or 0) > 0
        or retrieval_status in {"success", "partial_success"}
    )
    repair_grounded_term_count, repair_rejected_term_count = repair_term_counts(
        query_repair_after_llm_critic
    )
    llm_applied_terms, llm_rejected_terms = repair_term_lists(
        query_repair_after_llm_critic
    )
    llm_query_before_example, llm_query_after_example = llm_query_before_after_example(
        queries_from_query_artifact(query_plan_before_llm_critic),
        queries_from_query_artifact(query_plan_after_llm_critic),
    )
    llm_query_repair_applied = bool(
        llm_trace_count(llm_query_critic_trace, "applied_issue_count")
        or llm_trace_count(llm_query_critic_trace, "query_added_count")
        or llm_trace_count(llm_query_critic_trace, "query_dropped_count")
        or llm_trace_count(llm_query_critic_trace, "query_modified_count")
        or query_repair_after_llm_critic.get("applied_issue_count")
    )
    planned_final_queries = direct_final_queries_from_artifact(planned_queries)
    provenance_final_queries = direct_final_queries_from_artifact(query_provenance)
    critic_after_queries = direct_final_queries_from_artifact(query_plan_after_llm_critic)
    final_query_artifact_consistent = query_artifacts_consistent(
        [
            planned_final_queries,
            final_queries_after_repair,
            provenance_final_queries,
        ]
    )
    repair_provenance_count = len(query_provenance.get("llm_query_critic_repairs", []) or [])
    if not repair_provenance_count:
        repair_provenance_count = len(
            query_repair_after_llm_critic.get("applied_issue_records", []) or []
        )
    llm_query_critic_repair_artifact_consistent = (
        query_artifacts_consistent(
            [
                planned_final_queries,
                final_queries_after_repair,
                provenance_final_queries,
                critic_after_queries,
            ]
        )
        and (not llm_query_repair_applied or repair_provenance_count > 0)
    )
    metrics = {
        "run_dir": str(run),
        "case_id": (case or {}).get("id") or case_id,
        "artifact_presence": {
            "planned_queries_json": (run / "planned_queries.json").exists(),
            "query_provenance_json": (run / "query_provenance.json").exists(),
            "provider_status_json": (run / "provider_status.json").exists(),
            "retrieval_diagnostics_json": (run / "retrieval_diagnostics.json").exists(),
            "ranked_papers_csv": (run / "ranked_papers.csv").exists(),
            "exploration_quality_json": (run / "exploration_quality.json").exists(),
            "user_report_md": (run / "user_report.md").exists(),
        },
        "query_family_applied": bool(
            planned_queries.get("query_family_applied")
            or query_provenance.get("applied")
            or retrieval_diagnostics.get("query_family_applied")
        ),
        "final_provider_query_count": len(final_queries),
        "query_family_coverage": query_family_coverage,
        "single_acronym_query_count": len(single_acronym_queries),
        "single_acronym_queries": single_acronym_queries,
        "duplicate_query_count": duplicate_query_count(final_query_candidates),
        "missing_expected_anchor_count": missing_expected_anchor_count,
        "cross_domain_injection_count": cross_domain_injection_count or 0,
        "overbroad_query_count": len(overbroad_queries),
        "overbroad_queries": overbroad_queries,
        "anchor_coverage": anchor_coverage["score"],
        "anchor_coverage_detail": anchor_coverage,
        "provider_success_rate": provider_success_rate(provider_status),
        "provider_status": provider_status,
        "query_repair_enabled": bool(
            query_repair_stage_status.get(
                "repair_enabled",
                query_repair_suggestions.get("enabled", False),
            )
        ),
        "query_repair_applied": bool(
            query_repair_stage_status.get(
                "repair_applied",
                query_repair_suggestions.get("applied", False),
            )
        ),
        "query_repair_disabled_by_ablation": bool(
            query_repair_stage_status.get(
                "disabled_by_ablation",
                query_repair_suggestions.get("disabled_by_ablation", False),
            )
        ),
        "dropped_query_count": first_number(
            query_repair_stage_status.get("dropped_query_count"),
            len(query_repair_suggestions.get("dropped_queries", []) or []),
        ),
        "repaired_query_count": first_number(
            query_repair_stage_status.get("repaired_query_count"),
            len(query_repair_suggestions.get("removed_queries", []) or []),
        ),
        "added_query_count": first_number(
            query_repair_stage_status.get("added_query_count"),
            len(query_repair_suggestions.get("added_queries", []) or []),
        ),
        "query_repair_reason_if_no_difference": query_repair_stage_status.get(
            "reason_if_no_difference",
            "",
        ),
        "raw_to_final_query_change_count": raw_to_final_query_change_count,
        "repair_disabled_but_sanitizer_active": repair_disabled_but_sanitizer_active,
        "no_query_repair_conclusive": no_query_repair_conclusive,
        "merged_count": merged_count,
        "raw_retrieved_count": raw_count,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": duplicate_ratio,
        "top20_false_positive_count": top20_false_positive_count,
        "top20_false_positive_rows": top20_false_positive_rows,
        "forbidden_pattern_top10_count": (
            len(forbidden_top10_rows) if forbidden_patterns else None
        ),
        "forbidden_pattern_top10_rows": forbidden_top10_rows,
        "forbidden_pattern_top20_count": (
            len(forbidden_top20_rows) if forbidden_patterns else None
        ),
        "forbidden_pattern_top20_rows": forbidden_top20_rows,
        "must_read_count": len(must_read_rows),
        "include_count": len(include_rows),
        "forbidden_pattern_must_read_count": must_read_false_positive_count,
        "forbidden_pattern_include_count": include_false_positive_count,
        "must_read_false_positive_count": must_read_false_positive_count,
        "must_read_precision_heuristic": must_read_precision,
        **quality_metrics,
        "required_group_coverage_top10": average_float(
            ranked_rows[:10],
            "required_group_coverage_score",
        ),
        "intent_centrality_mean_top10": average_float(
            ranked_rows[:10],
            "intent_centrality_score",
        ),
        "reading_path_paper_count": metric_from_evaluation(
            evaluation,
            reading_path_diagnostics,
            "reading_path_paper_count",
        ),
        "reading_path_exclude_count": metric_from_evaluation(
            evaluation,
            reading_path_diagnostics,
            "reading_path_exclude_count",
        ),
        "reading_path_out_of_scope_count": metric_from_evaluation(
            evaluation,
            reading_path_diagnostics,
            "reading_path_out_of_scope_count",
        ),
        "reading_path_duplicate_count": metric_from_evaluation(
            evaluation,
            reading_path_diagnostics,
            "reading_path_duplicate_count",
        ),
        "reading_path_negative_context_count": metric_from_evaluation(
            evaluation,
            reading_path_diagnostics,
            "reading_path_negative_context_count",
        ),
        "target_context_required_for_priority": metric_from_evaluation(
            evaluation,
            reading_priority_policy,
            "target_context_required_for_priority",
        ),
        "llm_intent_enabled": llm_trace_bool(llm_intent_trace, "llm_enabled"),
        "llm_intent_called": llm_trace_bool(llm_intent_trace, "llm_called"),
        "llm_intent_fallback_used": llm_trace_bool(
            llm_intent_trace,
            "fallback_used",
        ),
        "llm_intent_verified_candidate_count": llm_trace_count(
            llm_intent_trace,
            "verified_candidate_count",
        ),
        "llm_intent_applied_suggestion_count": llm_trace_count(
            llm_intent_trace,
            "applied_suggestion_count",
        ),
        "llm_intent_rejected_suggestion_count": llm_trace_count(
            llm_intent_trace,
            "rejected_suggestion_count",
        ),
        "llm_intent_unsupported_suggestion_count": llm_trace_count(
            llm_intent_trace,
            "unsupported_suggestion_count",
        ),
        "llm_intent_reason_if_no_change": str(
            llm_intent_trace.get("reason_if_no_change", "")
        )
        if isinstance(llm_intent_trace, dict)
        else "",
        "llm_query_critic_enabled": llm_trace_bool(
            llm_query_critic_trace,
            "llm_query_critic_enabled",
        ),
        "llm_query_critic_called": llm_trace_bool(
            llm_query_critic_trace,
            "llm_called",
        ),
        "llm_query_critic_fallback_used": llm_trace_bool(
            llm_query_critic_trace,
            "fallback_used",
        ),
        "verified_issue_count": llm_trace_count(
            llm_query_critic_trace,
            "verified_issue_count",
        ),
        "rejected_issue_count": llm_trace_count(
            llm_query_critic_trace,
            "rejected_issue_count",
        ),
        "unsupported_issue_count": llm_trace_count(
            llm_query_critic_trace,
            "unsupported_issue_count",
        ),
        "applied_issue_count": llm_trace_count(
            llm_query_critic_trace,
            "applied_issue_count",
        ),
        "rejected_for_application_count": llm_trace_count(
            llm_query_critic_trace,
            "rejected_for_application_count",
        ),
        "query_added_count": llm_trace_count(
            llm_query_critic_trace,
            "query_added_count",
        ),
        "query_dropped_count": llm_trace_count(
            llm_query_critic_trace,
            "query_dropped_count",
        ),
        "query_modified_count": llm_trace_count(
            llm_query_critic_trace,
            "query_modified_count",
        ),
        "repair_applied": llm_query_repair_applied,
        "repair_grounded_term_count": repair_grounded_term_count,
        "repair_rejected_term_count": repair_rejected_term_count,
        "llm_query_critic_repair_applied": llm_query_repair_applied,
        "llm_query_critic_repaired_query_count": (
            llm_trace_count(llm_query_critic_trace, "query_added_count")
            + llm_trace_count(llm_query_critic_trace, "query_dropped_count")
            + llm_trace_count(llm_query_critic_trace, "query_modified_count")
        ),
        "llm_query_critic_original_query_example": llm_query_before_example,
        "llm_query_critic_repaired_query_example": llm_query_after_example,
        "llm_query_critic_applied_terms": "; ".join(llm_applied_terms),
        "llm_query_critic_rejected_terms": "; ".join(llm_rejected_terms),
        "llm_query_critic_repair_provenance_count": repair_provenance_count,
        "llm_query_critic_repair_artifact_consistent": (
            llm_query_critic_repair_artifact_consistent
        ),
        "final_query_artifact_consistent": final_query_artifact_consistent,
        "llm_query_critic_reason_if_no_change": str(
            llm_query_critic_trace.get("reason_if_no_change", "")
        )
        if isinstance(llm_query_critic_trace, dict)
        else "",
        "llm_query_before_example": llm_query_before_example,
        "llm_query_after_example": llm_query_after_example,
        "paper_decision_mutation_count": 0 if not retrieval_performed else None,
        "retrieval_performed": retrieval_performed,
        "research_gap_generation_status": research_gap_generation_status(
            run,
            evaluation,
            retrieval_performed=retrieval_performed,
        ),
        "ranked_papers_based_on_real_retrieval": ranked_papers_based_on_real_retrieval,
        "report_has_provider_status": (
            "provider status" in report_lower
            or "provider_summary" in report_lower
            or "retrieval_status" in report_lower
        ),
        "report_has_user_intent_summary": (
            "how the system interpreted" in report_lower
            or "user intent" in report_lower
            or "research intent" in report_lower
            or "what the system thinks" in report_lower
        ),
    }
    return metrics


def metric_from_evaluation(
    evaluation: dict[str, Any],
    nested: dict[str, Any],
    key: str,
) -> Any:
    """Read a metric from evaluation.json without inventing missing values."""

    if isinstance(evaluation, dict) and key in evaluation:
        return evaluation.get(key)
    if isinstance(nested, dict) and key in nested:
        return nested.get(key)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing search_paper output directory."
    )
    parser.add_argument("run_dir", help="Existing output directory to evaluate.")
    parser.add_argument(
        "--benchmark",
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Benchmark YAML path.",
    )
    parser.add_argument("--case-id", default=None, help="Benchmark case ID.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path. If omitted, metrics are printed to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = evaluate_run(
        Path(args.run_dir),
        benchmark_path=Path(args.benchmark),
        case_id=args.case_id,
    )
    payload = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
