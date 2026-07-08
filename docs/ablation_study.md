# Pilot Ablation Study Framework

This document describes a pilot ablation framework for `search_paper`. It is an
infrastructure plan for reproducible debugging and early evaluation, not a final
paper-level ablation result. Any numbers produced by these tools should be read
as pilot evidence until the benchmark set, provider conditions, and human labels
are expanded.

## Why Ablation Is Needed

The system has several coordinated agents and ranking modules. Ablation is
needed to check whether those modules actually improve result quality instead
of merely adding complexity.

The pilot framework is meant to:

- Test whether each agent or module contributes measurable signal.
- Avoid turning the system into a stack of features with no proof burden.
- Identify fragile modules before formal evaluation.
- Support later controlled comparisons against keyword search, legacy planning,
  and stronger gold-label benchmarks.

## Pilot Ablation Configs

The first supported configs are:

- `full_system` / deterministic baseline
- `legacy_query_planning`
- `no_query_family`
- `no_intent_repair`
- `no_query_repair`
- `no_domain_guardrail`
- `no_intent_centrality`
- `no_group_coverage_ranking`
- `llm_intent_frame_only`

All configs are debug/evaluation-only. They should be invoked through
`tools/run_ablations.py` or explicit CLI flags. Default pipeline behavior should
not change.

## Config Hypotheses

`full_system`

This is the current stable baseline for pilot comparison. It uses the normal
rule-based pipeline with intent repair, QueryFamily retrieval, query repair,
domain guardrails, intent centrality, and group coverage ranking enabled unless
the user explicitly chooses otherwise.

`legacy_query_planning`

This disables the novice intent repair and QueryFamily path and uses the older
planning behavior. It is useful as a coarse historical baseline.

`no_query_family`

Hypothesis: `final_provider_query_count`, `anchor_coverage`, and
`query_family_coverage` should decline. Non-single-keyword tasks such as
SEI/OER/MOF/thin-film deposition should degrade more clearly because they need
multiple concept routes rather than one broad query.

`no_intent_repair`

Hypothesis: novice wording will remain more literal, anchor coverage may fall,
and ambiguous terms may be less cleanly operationalized. This is mainly visible
in plan-only runs.

`no_query_repair`

Hypothesis: `overbroad_query_count` should rise. MOF and thin-film cases may
show short, biased, or under-anchored queries that query repair would otherwise
flag or repair.

`no_domain_guardrail`

Hypothesis: false positives should increase. In SEI, non-lithium-battery papers
should more easily enter top-10 results. In OER, OER-only or spin-only papers
should rank higher even when they miss the intended intersection.

`no_intent_centrality`

Hypothesis: broad review papers may rank above intersection papers. For OER,
the top-10 share of papers connecting spin state with oxygen evolution should
decline.

`no_group_coverage_ranking`

Hypothesis: papers missing required groups may still become `include` or
`must_read`. This ablation is currently partially supported because required
group coverage also influences upstream domain assessment, not only final
ranking and reading-priority decisions.

`llm_intent_frame_only`

Hypothesis: a controlled LLM intent-frame advisory layer may improve novice
intent understanding, especially for Chinese novice questions, abbreviations,
aliases, and target/negative context recovery. This config enables only
LLMIntentFrameEnhancer. It does not enable LLMQueryPlanCritic, LLM ranking, LLM
report adaptation, or LLM paper-level decisions.

## Plan-Only Vs Full-Run

Plan-only runs are best for query-layer questions:

- QueryFamily coverage.
- Query repair behavior.
- Final provider query count.
- Single-acronym and overbroad query checks.
- Anchor coverage against benchmark cases.

Full retrieval runs are best for result-quality questions:

- Domain guardrail contribution.
- Intent centrality ranking effects.
- Group coverage ranking effects.
- False positives in top results.
- Must-read precision heuristics.
- Provider and retrieval diagnostics.

## Current Flag Support

The following debug flags are supported:

- `--skip-query-families`
- `--legacy-query-planning`
- `--disable-intent-repair`
- `--disable-query-repair`
- `--disable-domain-guardrail`
- `--disable-intent-centrality`
- `--disable-group-coverage-ranking`

Each ablation run should write `ablation_config.json` and also record the same
configuration under `agent_trace.json` and `evaluation.json`.

Support status:

- `query_family`: supported.
- `intent_repair`: supported.
- `query_repair`: supported for pilot query repair artifacts and auto-repair.
- `domain_guardrail`: supported through neutral pass-through domain assessments.
- `intent_centrality`: supported for ranking score blending.
- `group_coverage_ranking`: partially supported because group coverage is used
  in both domain assessment and downstream ranking/decision logic.
- `llm_intent_frame_only`: phase 1 diagnostic support for the optional
  LLMIntentFrameEnhancer only; not a formal LLM ablation conclusion.

## Pilot Limitations

- The benchmark set is small.
- Several metrics are heuristic, especially false-positive and overbroad-query
  counts.
- Full retrieval depends on provider availability, API keys, rate limits, and
  metadata quality.
- Plan-only metrics cannot validate final paper quality.
- LLMIntentFrameEnhancer phase 1 is available only as a controlled diagnostic
  config (`llm_intent_frame_only`). Formal LLM ablations and LLMQueryPlanCritic
  evaluation remain future work.

## Example Pilot Commands

Plan-only:

```bash
python tools/run_ablations.py \
  --mode plan \
  --configs full_system,no_query_family,no_query_repair \
  --case-id ai_screening \
  --output-root outputs/ablations_pilot

python tools/run_ablations.py \
  --mode plan \
  --configs full_system,no_query_family,no_query_repair \
  --case-id mof_co2 \
  --output-root outputs/ablations_pilot

python tools/run_ablations.py \
  --mode plan \
  --configs full_system,no_query_family,no_query_repair \
  --case-id thin_film_methods \
  --output-root outputs/ablations_pilot
```

Full retrieval, OpenAlex-only:

```bash
python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system,no_domain_guardrail,no_intent_centrality,no_group_coverage_ranking \
  --case-id sei_lithium \
  --output-root outputs/ablations_pilot

python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system,no_domain_guardrail,no_intent_centrality,no_group_coverage_ranking \
  --case-id oer_spin_state \
  --output-root outputs/ablations_pilot
```

Then compare:

```bash
python tools/compare_ablations.py outputs/ablations_pilot
```
