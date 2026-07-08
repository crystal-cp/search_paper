# v9 Deterministic Baseline And Pilot Ablation Summary

## Status

This document summarizes the v9 deterministic baseline smoke review for
`search_paper`. It is a pilot diagnostic and smoke-regression baseline, not a
final formal ablation study and not a paper-level experimental conclusion.

The baseline reviewed here is `baseline_v9_smoke` from
`baseline_v9_smoke_review.zip`.

## What v9 Includes

The v9 deterministic baseline includes:

- v8 pipeline behavior.
- Pilot ablation framework v1.
- QueryFamily enabled by default in the deterministic full-system path.
- SEI target chemistry fix for lithium-specific battery questions.
- `must_read` calibration so `include` does not imply `must_read`.
- Reading-path filtering so exclude/out-of-scope/duplicate/negative-context papers do not appear in recommended reading sections.
- Context-aware `must_read` policy so tasks with explicit target context require target matches, while tasks without target context can still promote high-coverage core papers.

The baseline remains deterministic and rule-controlled. LLM components are
optional or planned controlled enhancements. LLM output must not directly decide
include/exclude, must-read status, out-of-scope status, final score, domain
decision, or evidence validity.

## Smoke Run Coverage

`baseline_v9_smoke` completed 11 ablation runs, all with `returncode=0`.

Plan-only cases:

- `ai_literature_screening`
- `mof_co2_capture`
- `thin_film_deposition`

Full OpenAlex-only cases:

- `sei_lithium_battery`
- `oer_spin_state`

Plan-only runs have the expected planning-only semantics:

- `retrieval_status = planning_only`
- `ranked_papers_based_on_real_retrieval = false`
- research-gap generation is skipped
- `suggested_next_searches = []`

## SEI Full-System Smoke Metrics

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

Interpretation:

The lithium-specific SEI target chemistry guardrail is now effective in this
smoke baseline. Negative-context sodium, potassium, zinc, magnesium, AZIB, and
mixed non-target battery papers should not appear as must-read/core reading-path
items for the shorter novice lithium SEI question. `must_read_count = 12` keeps
the first-read set bounded instead of treating every included paper as a core
first-read paper.

## OER Full-System Smoke Metrics

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

Interpretation:

The context-aware reading-priority policy now generalizes beyond lithium-style
target-context tasks. OER spin-state papers with in-scope domain decisions,
strict support, full group coverage, high topic focus, and high intent
centrality can become `must_read` even when no explicit target chemistry group is
required.

## QueryFamily Diagnostic

QueryFamily remains diagnostically strong in the pilot suite. The
`no_query_family` ablation degrades query quality for AI literature screening,
MOF CO2 capture, and thin-film deposition cases. The degradation should be read
through query-quality metrics, not query count alone:

- `query_quality_score`
- `expected_anchor_coverage`
- `overbroad_query_count`
- `repeated_phrase_query_count`
- `single_axis_query_count`
- `weak_query_count`

The pilot interpretation is that QueryFamily contributes useful structure for
multi-axis novice questions, especially when a task is not reducible to a single
keyword.

## QueryRepair Diagnostic

The `no_query_repair` ablation remains diagnostic/non-conclusive when upstream
sanitization is still active. If final queries are unchanged, the correct
interpretation is not "QueryRepair has no value." The summary should report:

- whether repair was enabled;
- whether repair was applied;
- whether raw candidate queries differed from final queries in full_system;
- whether upstream sanitization remained active;
- whether the no-query-repair run is conclusive.

## Partially Supported Ablations

Partially supported ablations must be marked clearly in summaries. In particular,
`no_group_coverage_ranking` is not a strict isolated ablation because group
coverage can influence both upstream domain assessment and downstream ranking or
reading-priority decisions.

Pilot summaries should separate:

- fully supported ablations;
- partially supported ablations;
- unsupported ablations.

## Known Backlog

Known backlog after the v9 smoke baseline:

- PaperRoleClassifier may still mark some top OER `must_read` papers as `role=unclassified`.
- The benchmark set remains small and diagnostic.
- Several quality metrics are heuristic.
- Full retrieval remains sensitive to provider availability, provider metadata, API keys, and rate limits.
- Future LLMIntentFrameEnhancer and LLMQueryPlanCritic experiments should be evaluated against this deterministic baseline, not against an unstable moving target.

## How To Read This Baseline

This baseline freezes deterministic behavior for regression control. It should
be used to answer whether future changes preserve core guarantees:

- plan-only runs do not fake research gaps;
- QueryFamily is active in the full-system baseline;
- lithium-specific SEI tasks suppress non-target battery chemistry from must-read recommendations;
- `must_read` is a bounded first-read set;
- OER can promote high-coverage core papers without a target-context group;
- reading paths do not recommend excluded, out-of-scope, duplicate, or negative-context papers.
