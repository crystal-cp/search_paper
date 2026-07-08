# Evaluation Protocol for `search_paper`

## Purpose

`search_paper` is evaluated as a human-centric multi-agent LLM system for
evidence-grounded scientific literature screening, not as a generic paper search
tool. The evaluation asks whether the system repairs novice intent, plans useful
query families, retrieves or imports candidate papers, grounds evidence in
abstracts, applies domain guardrails, ranks a useful reading path, and explains
its reasoning.

The v9 deterministic baseline is the current frozen rule-controlled baseline.
LLM behavior is optional or planned for controlled future evaluation and must
not directly decide `include` / `exclude`, `must_read`, `out_of_scope`,
`final_score`, `domain_decision`, or evidence validity.

## Evaluation Modes

Plan-only mode evaluates intent repair and query planning without provider
retrieval.

Expected plan-only behavior:

- `retrieval_status = planning_only`
- `ranked_papers_based_on_real_retrieval = false`
- `research_gap_matrix` generation is skipped
- `suggested_next_searches = []`

Full-run mode evaluates retrieval, deduplication, evidence extraction, span
validation, domain guardrails, ranking, reading priority, reports, and feedback
artifacts.

LLMIntentFrameEnhancer phase 1 evaluation is optional and should be compared
against the frozen v9 deterministic baseline. It evaluates whether a controlled
LLM advisory layer improves novice intent understanding before SearchContract
finalization. It does not evaluate LLM ranking, LLM evidence validation,
LLMQueryPlanCritic, or paper-level LLM decisions.

## Benchmark Cases

The v9 smoke baseline uses these benchmark cases:

| Mode | Case IDs |
| --- | --- |
| Plan-only | `ai_literature_screening`, `mof_co2_capture`, `thin_film_deposition` |
| Full OpenAlex-only | `sei_lithium_battery`, `oer_spin_state` |

Benchmark definitions live in `data/benchmark_cases.yaml`.

## Reproduction Commands

Plan-only cases:

```bash
PYTHONPATH=. python tools/run_ablations.py \
  --mode plan \
  --configs full_system,no_query_family,no_query_repair \
  --case-id ai_literature_screening \
  --output-root outputs/baseline_v9_smoke

PYTHONPATH=. python tools/run_ablations.py \
  --mode plan \
  --configs full_system,no_query_family,no_query_repair \
  --case-id mof_co2_capture \
  --output-root outputs/baseline_v9_smoke

PYTHONPATH=. python tools/run_ablations.py \
  --mode plan \
  --configs full_system,no_query_family,no_query_repair \
  --case-id thin_film_deposition \
  --output-root outputs/baseline_v9_smoke
```

Full OpenAlex-only cases:

```bash
PYTHONPATH=. python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system \
  --case-id sei_lithium_battery \
  --output-root outputs/baseline_v9_smoke

PYTHONPATH=. python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system \
  --case-id oer_spin_state \
  --output-root outputs/baseline_v9_smoke

PYTHONPATH=. python tools/compare_ablations.py outputs/baseline_v9_smoke
```

For full pilot ablation diagnostics beyond the frozen 11-run smoke baseline,
use a separate output root:

```bash
PYTHONPATH=. python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system,no_domain_guardrail,no_intent_centrality,no_group_coverage_ranking \
  --case-id sei_lithium_battery \
  --output-root outputs/ablations_pilot

PYTHONPATH=. python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system,no_domain_guardrail,no_intent_centrality,no_group_coverage_ranking \
  --case-id oer_spin_state \
  --output-root outputs/ablations_pilot

PYTHONPATH=. python tools/compare_ablations.py outputs/ablations_pilot
```

Test command from the project root:

```bash
PYTHONPATH=. pytest -q
```

Equivalent module form:

```bash
python -m pytest -q
```

## Required Artifacts

Plan-only runs should produce at least:

- `search_brief.json`
- `search_contract.json`
- `ambiguity_analysis.json`
- `question_refinement.json`
- `planned_queries.json`
- `query_families.json`
- `query_provenance.json`
- `query_repair_stage_status.json`
- `exploration_quality.json`
- `provider_status.json`
- `retrieval_diagnostics.json`
- `report.md`
- `agent_trace.json`
- `run_events.jsonl`

Full runs should additionally produce:

- raw provider results, when providers are enabled
- `merged_papers.csv`
- `evidence_table.csv`
- `evidence_functions.json`
- `domain_assessments.json`
- `aspect_coverage.csv`
- `paper_roles.json`
- `screening_decisions.csv`
- `screening_decisions.json`
- `ranked_papers.csv`
- `paper_cards.md`
- `reading_path.md`
- `evaluation.json`

Ablation comparison should produce:

- `reports/ablation_summary.csv`
- `reports/ablation_summary.md`

## v9 Metrics

The evaluator should report these metrics when available.

### Query And Planning Metrics

| Metric | Meaning |
| --- | --- |
| `query_family_applied` | Whether QueryFamily planning affected the run. In v9 full_system this should be true unless disabled by ablation. |
| `final_provider_query_count` | Final provider-facing query count. |
| `single_acronym_query_count` | Count of weak acronym-only queries. |
| `anchor_coverage` | Coverage of expected benchmark anchors in plan artifacts. |
| `expected_anchor_coverage` | Expected anchor coverage computed by benchmark-aware evaluation. |
| `query_quality_score` | Heuristic score combining anchor coverage and penalties for weak or overbroad queries. |
| `overbroad_query_count` | Queries too broad to express the benchmark intent. |
| `repeated_phrase_query_count` | Queries with repeated phrases or malformed duplication. |
| `single_axis_query_count` | Queries that cover only one axis of a multi-axis task. |
| `weak_query_count` | Queries missing enough anchors to be useful. |
| `query_repair_enabled` | Whether query repair was enabled. |
| `query_repair_applied` | Whether query repair changed or accepted a repair plan. |
| `raw_to_final_query_change_count` | Difference count between raw candidate queries and final queries. |
| `repair_disabled_but_sanitizer_active` | Whether no-query-repair remains diagnostic because upstream sanitization still applies. |
| `no_query_repair_conclusive` | Whether a no-query-repair ablation supports a conclusion. |

### Retrieval Metrics

| Metric | Meaning |
| --- | --- |
| `raw_retrieved_count` / `raw_retrieved_paper_count` | Raw provider results before deduplication. |
| `merged_count` / `merged_paper_count` | Deduplicated candidate papers. |
| `duplicate_ratio` | Duplicates divided by raw retrieved count, when raw count is available. |
| `duplicate_count` | Raw minus merged count, when available. |
| `provider_success_rate` | Successful providers divided by enabled providers. |
| `retrieval_status` | `planning_only`, `success`, `partial_success`, or `failed`. |
| `ranked_papers_based_on_real_retrieval` | Whether ranked papers came from real retrieval rather than plan-only artifacts. |

### Screening And Domain Metrics

| Metric | Meaning |
| --- | --- |
| `must_read_count` | Count of first-read papers. This should be bounded and smaller than include count. |
| `include_count` | Count of papers kept as relevant. |
| `forbidden_pattern_top10_count` | Forbidden benchmark patterns in top 10. |
| `forbidden_pattern_top20_count` | Forbidden benchmark patterns in top 20. |
| `forbidden_pattern_must_read_count` | Forbidden benchmark patterns among must-read papers. |
| `forbidden_pattern_include_count` | Forbidden benchmark patterns among included papers. |
| `required_group_coverage_top10` | Mean or aggregate required group coverage in top 10. |
| `intent_centrality_mean_top10` | Mean intent-centrality score in top 10. |
| `target_context_required_for_priority` | Whether `must_read` requires explicit target context for this task. |
| `must_read_policy_reason` | Explanation of the reading-priority policy selected for the run. |

### Reading Path Metrics

| Metric | Meaning |
| --- | --- |
| `reading_path_paper_count` | Number of papers recommended in the reading path. |
| `reading_path_exclude_count` | Excluded papers that leaked into reading path. Target is `0`. |
| `reading_path_out_of_scope_count` | Out-of-scope papers that leaked into reading path. Target is `0`. |
| `reading_path_duplicate_count` | Duplicate papers across reading-path sections. Target is `0`. |
| `reading_path_negative_context_count` | Negative-context papers recommended in reading path. Target is `0` for constrained tasks. |

### Report Metrics

| Metric | Meaning |
| --- | --- |
| `report_has_provider_status` | Whether report documents provider/retrieval status. |
| `report_has_user_intent_summary` | Whether report explains the interpreted user intent. |
| `report_has_coverage_summary` | Whether report explains coverage and remaining gaps or skipped gap status. |

### LLMIntentFrameEnhancer Phase 1 Metrics

Compare these configurations when explicitly evaluating the LLM intent layer:

- `full_system` / v9 deterministic baseline.
- `llm_intent_frame_only`, which enables only LLMIntentFrameEnhancer.

`llm_intent_frame_only` does not enable LLMQueryPlanCritic, LLM ranking, LLM
report adaptation, or LLM paper-level decisions. It is a phase 1 diagnostic for
intent-frame suggestions before SearchContract finalization.

| Metric | Meaning |
| --- | --- |
| `verified_candidate_count` | LLM suggestions that passed deterministic verification before any fallback/application decision. |
| `applied_suggestion_count` | Verified suggestions finally written into active SearchContract provenance or constraint groups. Must be `0` when `fallback_used = true`. |
| `accepted_suggestion_count` | Backward-compatible alias for suggestions applied into the contract, not merely verifier-approved candidates. |
| `rejected_suggestion_count` | LLM suggestions rejected by deterministic verification. |
| `unsupported_suggestion_count` | Suggestions rejected as unsupported by the user question or deterministic intent. |
| `malformed_output` | Whether the LLM output failed JSON/schema parsing. |
| `fallback_used` | Whether the deterministic intent frame was kept unchanged because of malformed, unsafe, or unavailable LLM output. |
| `intent_slot_accuracy` | Human- or fixture-scored accuracy of suggested intent slots when labels are available. |
| `missing_user_aspect_count` | User-stated aspects missed by the final verified intent frame. |
| `cross_domain_injection_count` | Unsupported cross-domain additions proposed by the LLM. |
| `query_quality_score` | Downstream query-quality score after verified intent suggestions, when query planning is run. |
| `anchor_coverage` | Coverage of benchmark anchors in final planning artifacts. |
| `overbroad_query_count` | Count of weak or overly broad final queries. |

These metrics are diagnostic. They do not constitute a completed formal LLM
ablation study.

## v9 Smoke Baseline Checks

The `baseline_v9_smoke` review completed 11 runs with `returncode=0`.

SEI full_system expected smoke metrics:

| Metric | Value |
| --- | ---: |
| `raw_retrieved_paper_count` | 220 |
| `merged_paper_count` | 130 |
| `duplicate_count` | 90 |
| `must_read_count` | 12 |
| `reading_path_exclude_count` | 0 |
| `reading_path_out_of_scope_count` | 0 |
| `reading_path_duplicate_count` | 0 |
| `reading_path_negative_context_count` | 0 |
| `target_context_required_for_priority` | true |

OER full_system expected smoke metrics:

| Metric | Value |
| --- | ---: |
| `raw_retrieved_paper_count` | 110 |
| `merged_paper_count` | 66 |
| `duplicate_count` | 44 |
| `must_read_count` | 12 |
| `reading_path_exclude_count` | 0 |
| `reading_path_out_of_scope_count` | 0 |
| `reading_path_duplicate_count` | 0 |
| `reading_path_negative_context_count` | 0 |
| `target_context_required_for_priority` | false |

## Interpretation Rules

Pilot summaries must be diagnostic and cautious.

- If no-query-family increases weak, overbroad, repeated, or single-axis queries, report query-quality degradation even if query count is similar.
- If no-query-repair final queries are unchanged while upstream sanitization remains active, mark the result diagnostic/non-conclusive rather than claiming QueryRepair has no value.
- If a partially supported ablation is run, label it as partially supported in CSV and Markdown summaries.
- If full_system has forbidden must-read papers, report this as a benchmark/guardrail backlog and do not treat full_system as perfect.
- Plan-only research-gap conclusions must be skipped because no screened corpus exists.

## Future LLM Evaluation

LLMIntentFrameEnhancer phase 1 should be measured against this deterministic
baseline before any stronger LLM components are introduced. The first question is
whether the LLM helps recover novice intent slots, especially for multilingual
questions, abbreviations, aliases, target context, and negative context, without
injecting unsupported domains. LLMQueryPlanCritic remains future work and should
not be treated as part of the phase 1 intent-frame evaluation. LLM variants must
not bypass rule-owned screening, evidence, domain, ranking, or reading-priority
decisions.
