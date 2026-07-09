# LLM Plan-Level Pilot Diagnostic

## Purpose

This document summarizes the LLM plan-level pilot diagnostic for
`search_paper`. It is a plan-only mechanism and artifact-consistency check. It
is not a full retrieval ablation, not a paper-level evaluation, and not a formal
LLM ablation conclusion.

The pilot asks whether optional LLM planning components can run safely around
the deterministic v9 baseline without changing retrieval, ranking,
DomainGuardrail behavior, reading priority, evidence validation, or paper-level
decisions. It also checks whether weak query plans can trigger a verified,
grounded, deterministic repair path when the repair flag is explicitly enabled.

The generated comparison outputs in `reports/ablation_summary.csv` and
`reports/ablation_summary.md` are produced by `tools/compare_ablations.py` and
should be treated as generated artifacts, not hand-edited source documents.

## Compared Configs

The pilot compares these plan-level configurations:

- `full_system`
- `llm_intent_frame_only`
- `llm_query_critic_diagnostic_only`
- `llm_query_critic_repair_applied`
- `llm_intent_plus_query_critic_repair`

The LLM configurations remain explicit opt-in paths. They do not replace the
deterministic baseline and do not authorize paper-level decisions.

## Clean Benchmark Diagnostic

The clean benchmark diagnostic uses the existing benchmark cases:

- `ai_screening`
- `mof_co2`
- `thin_film_methods`
- `sei_lithium`
- `oer_spin_state`

Observed interpretation:

- Clean QueryFamily plans mostly remained unchanged.
- `LLMQueryPlanCritic` did not force unnecessary repairs on already strong plans.
- No paper-level decisions were mutated.
- Plan-only fake research gaps did not appear.

If a clean run has `verified_issue_count = 0` or `applied_issue_count = 0`, that
should be read as "no verified repair opportunity in a clean plan", not as a
failure of the critic.

## Weak-Plan Positive Controls

The weak-plan positive controls are diagnostic stress cases. They are designed
to create weak plan-only queries so that the critic, verifier, and deterministic
rule applier can be exercised. They are not formal benchmark evidence.

The positive-control cases are:

- `weak_sei_acronym_query`
- `weak_oer_acronym_query`
- `weak_mof_short_query`

Repair examples:

| Case | Before | After | Applied term | Rejected term |
| --- | --- | --- | --- | --- |
| `weak_sei_acronym_query` | `sei SEI` | `sei SEI "solid electrolyte interphase"` | `solid electrolyte interphase` | `lithium battery` |
| `weak_oer_acronym_query` | `oer OER` | `oer OER "oxygen evolution reaction"` | `oxygen evolution reaction` | `spin state` |
| `weak_mof_short_query` | `MOF CO2` | `MOF CO2 "metal-organic framework"` | `metal-organic framework` | `CO2 capture` |

These examples validate the repair mechanism and provenance path. They should
not be overread as proof that retrieval quality or downstream ranking improves.
If a repair is applied but the heuristic `query_quality_score` does not improve
uniformly, the correct interpretation is mechanism validation rather than query
quality improvement.

## Safety Interpretation

The safety contract for this pilot is:

- The LLM critic only proposes query-plan issues.
- A deterministic verifier checks whether proposed issues are supported.
- The deterministic rule applier only applies verified and grounded terms.
- Ungrounded target or context anchors are rejected.
- The LLM does not decide `include`, `exclude`, `must_read`, `domain_decision`, `final_score`, evidence validity, or `reading_priority`.

When `--apply-llm-query-critic-repairs` is not enabled, verified critic findings
remain diagnostic artifacts and should not change final queries. When the flag
is enabled, only verified and grounded issues can produce deterministic query
mutations, with provenance recorded in the query repair artifacts.

## Limitations

This pilot has important limits:

- This does not prove full retrieval improvement.
- Query quality score did not uniformly improve.
- Weak controls are diagnostic stress cases, not formal benchmark evidence.
- `LLMIntentFrameEnhancer` positive path is only partially covered in this pilot.
- Full retrieval LLM ablation still remains future work.

Because all runs are plan-level, retrieval-dependent outcomes such as provider
recall, paper ranking, evidence coverage, reading-path usefulness, and final
screening quality remain outside the scope of this pilot.

## Next Step

Reasonable next steps are:

- Expand case-aware fake `LLMIntentFrameEnhancer` tests.
- Run a small full-retrieval pilot only after plan-level evidence is stable.
- Then compare `full_system` with LLM-enhanced configs on SEI, OER, or selected cases.
