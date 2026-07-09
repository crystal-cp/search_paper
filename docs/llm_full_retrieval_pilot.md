# Small Full-Retrieval LLM Safety Pilot

## Purpose

This document summarizes the small full-retrieval LLM safety pilot for
`search_paper`. It is a full-pipeline safety diagnostic, not a formal full LLM
ablation and not a performance-improvement conclusion.

The pilot checks whether the optional LLM-enhanced planning configuration can
run through retrieval, guardrails, ranking, reading-path generation, reports,
and audit artifacts without breaking the deterministic v9 baseline safety
properties.

## Compared Configs

The pilot compares:

- `full_system`
- `llm_intent_plus_query_critic_repair`

`full_system` is the deterministic v9 baseline. The LLM-enhanced configuration
uses fake/mock LLM providers in the pilot path unless a real provider is
explicitly configured elsewhere.

## Cases

The pilot uses two full-retrieval benchmark cases:

- `sei_lithium`
- `oer_spin_state`

## What Was Tested

This pilot tested whether:

- LLM-enhanced planning can pass through the complete retrieval pipeline.
- Retrieval still succeeds.
- DomainGuardrail, ranking, reading_path, and reports are not broken.
- No paper-level decision mutation appears.
- Negative-context papers are not reintroduced into `must_read` or `include`.
- Final query and LLM repair artifacts remain consistent.

The pilot did not change retrieval, ranking, DomainGuardrail, reading priority,
evidence validation, or paper-level decision logic.

## Key Results

### SEI lithium

| Metric | Value |
| --- | ---: |
| `raw_retrieved_count` | 220 |
| `merged_count` | 130 |
| `must_read_count` | 12 |
| `include_count` | 59 |
| `forbidden_pattern_must_read_count` | 0 |
| `negative_context_must_read` | 0 |
| `negative_context_include` | 0 |
| `reading_path_exclude_count` | 0 |
| `reading_path_out_of_scope_count` | 0 |
| `reading_path_duplicate_count` | 0 |
| `reading_path_negative_context_count` | 0 |

### OER spin state

| Metric | Value |
| --- | ---: |
| `raw_retrieved_count` | 110 |
| `merged_count` | 66 |
| `must_read_count` | 12 |
| `include_count` | 25 |
| `reading_path_exclude_count` | 0 |
| `reading_path_out_of_scope_count` | 0 |
| `reading_path_duplicate_count` | 0 |
| `reading_path_negative_context_count` | 0 |

### Safety

| Metric | Value |
| --- | --- |
| `llm_direct_paper_decision_mutation_count` | `0` |
| `llm_direct_evidence_validation_mutation_count` | `0` |
| `llm_direct_ranking_mutation_count` | `0` |
| `final_query_artifact_consistent` | `true` |
| `llm_query_critic_repair_artifact_consistent` | `true` |

## LLM Interpretation

In `llm_intent_plus_query_critic_repair`:

- `LLMIntentFrameEnhancer` was enabled.
- `LLMQueryPlanCritic` was enabled.
- The apply-LLM-query-critic-repairs flag was enabled.

However, the clean SEI and OER full QueryFamily plans had no verified grounded
repair opportunity. Therefore:

- `llm_query_critic_repair_enabled=true`
- `llm_query_critic_repair_applied=false`

This is not a failure. It means the clean plans remained unchanged because the
deterministic verifier did not find a grounded repair opportunity that should be
applied.

## What This Does And Does Not Prove

This pilot supports the following limited claims:

- The LLM-enhanced config can run through full retrieval safely on the SEI and OER pilot cases.
- It did not break guardrails, ranking, reading_path, reports, or artifact consistency.
- It did not introduce paper-level decision mutation.

This pilot does not support the following claims:

- LLM improves retrieval precision.
- LLM improves `Precision@10`.
- LLM improves ranking quality.
- A formal full LLM ablation is complete.

Any quality-improvement interpretation would require a larger, controlled
full-retrieval study with stable provider conditions and human-validated labels.

## Next Step

Possible next steps:

- Run targeted full weak-retrieval stress cases if needed.
- Expand the full safety pilot to 2-3 additional benchmark cases.
- Only then consider a formal LLM ablation.
