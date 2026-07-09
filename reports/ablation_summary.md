# Pilot Ablation Summary

This is a pilot / diagnostic ablation summary generated from artifact-level heuristics.
It is not a final experimental conclusion and should not be over-interpreted without larger benchmarks, provider-stability checks, and human labels.

## Ablation support status

### Fully supported ablations

- full_system (full_system:baseline)

### Partially supported ablations (diagnostic / non-conclusive)

- llm_intent_frame_only (llm_intent_frame_enhancer:diagnostic_controlled)
- llm_intent_plus_query_critic_repair (llm_intent_frame_enhancer:diagnostic_controlled, llm_query_plan_critic:diagnostic_artifacts_only, llm_query_plan_critic_repairs:controlled_repair_pilot)
- llm_query_critic_diagnostic_only (llm_query_plan_critic:diagnostic_artifacts_only)
- llm_query_critic_repair_applied (llm_query_plan_critic:diagnostic_artifacts_only, llm_query_plan_critic_repairs:controlled_repair_pilot)

### Unsupported ablations

- None observed in this summary.

## LLM plan-level diagnostic summary

This is plan-level diagnostic only; it does not prove full retrieval improvement.
Query changes only occur when the deterministic rule applier accepts verified and grounded critique issues.
LLM modules do not make paper-level decisions such as include/exclude, must_read, domain_decision, final_score, evidence validity, or reading_priority.

| Case | Group | Stress | Config | Intent called | Intent fallback | Intent verified | Intent applied | Query critic called | Verified issues | Applied issues | Query +/-/~ | Grounded terms | Rejected terms | Before query | After query | Applied terms | Rejected term names | Provenance | Artifacts consistent | Final query consistent | Retrieval performed | Gaps | Paper decision mutation |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: |
| ai_literature_screening | clean_benchmark_diagnostic | false | llm_intent_frame_only | true | true | 0 | 0 | false | 0 | 0 | +0/-0/~0 | 0 | 0 |  |  |  |  | 0 | true | true | false | skipped | 0 |
| ai_literature_screening | clean_benchmark_diagnostic | false | llm_intent_plus_query_critic_repair | true | true | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | LLM "systematic review screening" | LLM "systematic review screening" |  |  | 0 | true | true | false | skipped | 0 |
| ai_literature_screening | clean_benchmark_diagnostic | false | llm_query_critic_diagnostic_only | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | LLM "systematic review screening" | LLM "systematic review screening" |  |  | 0 | true | true | false | skipped | 0 |
| ai_literature_screening | clean_benchmark_diagnostic | false | llm_query_critic_repair_applied | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | LLM "systematic review screening" | LLM "systematic review screening" |  |  | 0 | true | true | false | skipped | 0 |
| mof_co2_capture | clean_benchmark_diagnostic | false | llm_intent_frame_only | true | true | 0 | 0 | false | 0 | 0 | +0/-0/~0 | 0 | 0 |  |  |  |  | 0 | true | true | false | skipped | 0 |
| mof_co2_capture | clean_benchmark_diagnostic | false | llm_intent_plus_query_critic_repair | true | true | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | MOF "CO2 adsorption" "pore size" | MOF "CO2 adsorption" "pore size" |  |  | 0 | true | true | false | skipped | 0 |
| mof_co2_capture | clean_benchmark_diagnostic | false | llm_query_critic_diagnostic_only | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | MOF "CO2 adsorption" "pore size" | MOF "CO2 adsorption" "pore size" |  |  | 0 | true | true | false | skipped | 0 |
| mof_co2_capture | clean_benchmark_diagnostic | false | llm_query_critic_repair_applied | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | MOF "CO2 adsorption" "pore size" | MOF "CO2 adsorption" "pore size" |  |  | 0 | true | true | false | skipped | 0 |
| oer_spin_state | clean_benchmark_diagnostic | false | llm_intent_frame_only | true | true | 0 | 0 | false | 0 | 0 | +0/-0/~0 | 0 | 0 |  |  |  |  | 0 | true | true | false | skipped | 0 |
| oer_spin_state | clean_benchmark_diagnostic | false | llm_intent_plus_query_critic_repair | true | true | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | "oxygen evolution reaction" "spin state" catalyst | "oxygen evolution reaction" "spin state" catalyst |  |  | 0 | true | true | false | skipped | 0 |
| oer_spin_state | clean_benchmark_diagnostic | false | llm_query_critic_diagnostic_only | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | "oxygen evolution reaction" "spin state" catalyst | "oxygen evolution reaction" "spin state" catalyst |  |  | 0 | true | true | false | skipped | 0 |
| oer_spin_state | clean_benchmark_diagnostic | false | llm_query_critic_repair_applied | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | "oxygen evolution reaction" "spin state" catalyst | "oxygen evolution reaction" "spin state" catalyst |  |  | 0 | true | true | false | skipped | 0 |
| sei_lithium_battery | clean_benchmark_diagnostic | false | llm_intent_frame_only | true | false | 12 | 7 | false | 0 | 0 | +0/-0/~0 | 0 | 0 |  |  |  |  | 0 | true | true | false | skipped | 0 |
| sei_lithium_battery | clean_benchmark_diagnostic | false | llm_intent_plus_query_critic_repair | true | false | 12 | 7 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | "solid electrolyte interphase" "lithium battery" | "solid electrolyte interphase" "lithium battery" |  |  | 0 | true | true | false | skipped | 0 |
| sei_lithium_battery | clean_benchmark_diagnostic | false | llm_query_critic_diagnostic_only | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | "solid electrolyte interphase" "lithium battery" | "solid electrolyte interphase" "lithium battery" |  |  | 0 | true | true | false | skipped | 0 |
| sei_lithium_battery | clean_benchmark_diagnostic | false | llm_query_critic_repair_applied | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | "solid electrolyte interphase" "lithium battery" | "solid electrolyte interphase" "lithium battery" |  |  | 0 | true | true | false | skipped | 0 |
| thin_film_deposition | clean_benchmark_diagnostic | false | llm_intent_frame_only | true | true | 0 | 0 | false | 0 | 0 | +0/-0/~0 | 0 | 0 |  |  |  |  | 0 | true | true | false | skipped | 0 |
| thin_film_deposition | clean_benchmark_diagnostic | false | llm_intent_plus_query_critic_repair | true | true | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | "thin film deposition" ALD PLD sputtering CVD comparison | "thin film deposition" ALD PLD sputtering CVD comparison |  |  | 0 | true | true | false | skipped | 0 |
| thin_film_deposition | clean_benchmark_diagnostic | false | llm_query_critic_diagnostic_only | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | "thin film deposition" ALD PLD sputtering CVD comparison | "thin film deposition" ALD PLD sputtering CVD comparison |  |  | 0 | true | true | false | skipped | 0 |
| thin_film_deposition | clean_benchmark_diagnostic | false | llm_query_critic_repair_applied | false | false | 0 | 0 | true | 0 | 0 | +0/-0/~0 | 0 | 0 | "thin film deposition" ALD PLD sputtering CVD comparison | "thin film deposition" ALD PLD sputtering CVD comparison |  |  | 0 | true | true | false | skipped | 0 |
| weak_mof_short_query | weak_plan_positive_control | true | llm_query_critic_diagnostic_only | false | false | 0 | 0 | true | 1 | 0 | +0/-0/~0 | 0 | 0 | MOF CO2 | MOF CO2 |  |  | 0 | true | true | false | skipped | 0 |
| weak_mof_short_query | weak_plan_positive_control | true | llm_query_critic_repair_applied | false | false | 0 | 0 | true | 1 | 1 | +0/-0/~1 | 1 | 1 | MOF CO2 | MOF CO2 "metal-organic framework" | metal-organic framework | CO2 capture | 1 | true | true | false | skipped | 0 |
| weak_oer_acronym_query | weak_plan_positive_control | true | llm_query_critic_diagnostic_only | false | false | 0 | 0 | true | 1 | 0 | +0/-0/~0 | 0 | 0 | oer OER | oer OER |  |  | 0 | true | true | false | skipped | 0 |
| weak_oer_acronym_query | weak_plan_positive_control | true | llm_query_critic_repair_applied | false | false | 0 | 0 | true | 1 | 1 | +0/-0/~1 | 1 | 1 | oer OER | oer OER "oxygen evolution reaction" | oxygen evolution reaction | spin state | 1 | true | true | false | skipped | 0 |
| weak_sei_acronym_query | weak_plan_positive_control | true | llm_query_critic_diagnostic_only | false | false | 0 | 0 | true | 1 | 0 | +0/-0/~0 | 0 | 0 | sei SEI | sei SEI |  |  | 0 | true | true | false | skipped | 0 |
| weak_sei_acronym_query | weak_plan_positive_control | true | llm_query_critic_repair_applied | false | false | 0 | 0 | true | 1 | 1 | +0/-0/~1 | 1 | 1 | sei SEI | sei SEI "solid electrolyte interphase" | solid electrolyte interphase | lithium battery | 1 | true | true | false | skipped | 0 |

- Treat these rows as pilot diagnostics for controlled LLM planning behavior, not as a formal LLM ablation study.

Weak-plan positive controls:

- weak_mof_short_query / llm_query_critic_diagnostic_only: verified_issue_count=1, applied_issue_count=0, query_modified_count=0, repair_grounded_term_count=0, repair_rejected_term_count=0, before=MOF CO2, after=MOF CO2, applied_terms=, rejected_terms=, artifact_consistent=true.
- weak_mof_short_query / llm_query_critic_repair_applied: verified_issue_count=1, applied_issue_count=1, query_modified_count=1, repair_grounded_term_count=1, repair_rejected_term_count=1, before=MOF CO2, after=MOF CO2 "metal-organic framework", applied_terms=metal-organic framework, rejected_terms=CO2 capture, artifact_consistent=true.
- weak_oer_acronym_query / llm_query_critic_diagnostic_only: verified_issue_count=1, applied_issue_count=0, query_modified_count=0, repair_grounded_term_count=0, repair_rejected_term_count=0, before=oer OER, after=oer OER, applied_terms=, rejected_terms=, artifact_consistent=true.
- weak_oer_acronym_query / llm_query_critic_repair_applied: verified_issue_count=1, applied_issue_count=1, query_modified_count=1, repair_grounded_term_count=1, repair_rejected_term_count=1, before=oer OER, after=oer OER "oxygen evolution reaction", applied_terms=oxygen evolution reaction, rejected_terms=spin state, artifact_consistent=true.
- weak_sei_acronym_query / llm_query_critic_diagnostic_only: verified_issue_count=1, applied_issue_count=0, query_modified_count=0, repair_grounded_term_count=0, repair_rejected_term_count=0, before=sei SEI, after=sei SEI, applied_terms=, rejected_terms=, artifact_consistent=true.
- weak_sei_acronym_query / llm_query_critic_repair_applied: verified_issue_count=1, applied_issue_count=1, query_modified_count=1, repair_grounded_term_count=1, repair_rejected_term_count=1, before=sei SEI, after=sei SEI "solid electrolyte interphase", applied_terms=solid electrolyte interphase, rejected_terms=lithium battery, artifact_consistent=true.

## ai_literature_screening

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 10 | 0.9600 | 1 | 0 | 0 | 1 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| llm_intent_frame_only | plan | true | 10 | 0.9600 | 1 | 0 | 0 | 1 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation |
| llm_intent_plus_query_critic_repair | plan | true | 10 | 0.9600 | 1 | 0 | 0 | 1 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_diagnostic_only | plan | true | 10 | 0.9600 | 1 | 0 | 0 | 1 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_repair_applied | plan | true | 10 | 0.9600 | 1 | 0 | 0 | 1 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |

Pilot interpretation:

- `llm_intent_frame_only` tests controlled intent-frame suggestions entering SearchContract provenance after deterministic verification.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_intent_plus_query_critic_repair` is still plan-level only; it does not prove full retrieval or ranking improvement.
- No verified repair opportunity; clean plan remained unchanged.
- `llm_query_critic_diagnostic_only` is plan-level diagnostic only: it records verified/rejected critique issues but must not mutate final provider queries.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_query_critic_repair_applied` is a controlled repair pilot: query changes only count when the deterministic rule applier accepts verified, grounded issues.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_frame_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_plus_query_critic_repair | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_diagnostic_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_repair_applied | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## mof_co2_capture

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 8 | 0.5000 | 2 | 2 | 0 | 4 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| llm_intent_frame_only | plan | true | 8 | 0.5000 | 2 | 2 | 0 | 4 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation |
| llm_intent_plus_query_critic_repair | plan | true | 8 | 0.5000 | 2 | 2 | 0 | 4 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_diagnostic_only | plan | true | 8 | 0.5000 | 2 | 2 | 0 | 4 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_repair_applied | plan | true | 8 | 0.5000 | 2 | 2 | 0 | 4 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |

Pilot interpretation:

- `llm_intent_frame_only` tests controlled intent-frame suggestions entering SearchContract provenance after deterministic verification.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_intent_plus_query_critic_repair` is still plan-level only; it does not prove full retrieval or ranking improvement.
- No verified repair opportunity; clean plan remained unchanged.
- `llm_query_critic_diagnostic_only` is plan-level diagnostic only: it records verified/rejected critique issues but must not mutate final provider queries.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_query_critic_repair_applied` is a controlled repair pilot: query changes only count when the deterministic rule applier accepts verified, grounded issues.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_frame_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_plus_query_critic_repair | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_diagnostic_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_repair_applied | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## oer_spin_state

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 6 | 0.6000 | 0 | 0 | 0 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |
| llm_intent_frame_only | plan | true | 6 | 0.6000 | 0 | 0 | 0 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; LLM plan-level diagnostic; not a formal full retrieval ablation |
| llm_intent_plus_query_critic_repair | plan | true | 6 | 0.6000 | 0 | 0 | 0 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_diagnostic_only | plan | true | 6 | 0.6000 | 0 | 0 | 0 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; LLM plan-level diagnostic; not a formal full retrieval ablation; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_repair_applied | plan | true | 6 | 0.6000 | 0 | 0 | 0 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |

Pilot interpretation:

- `llm_intent_frame_only` tests controlled intent-frame suggestions entering SearchContract provenance after deterministic verification.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_intent_plus_query_critic_repair` is still plan-level only; it does not prove full retrieval or ranking improvement.
- No verified repair opportunity; clean plan remained unchanged.
- `llm_query_critic_diagnostic_only` is plan-level diagnostic only: it records verified/rejected critique issues but must not mutate final provider queries.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_query_critic_repair_applied` is a controlled repair pilot: query changes only count when the deterministic rule applier accepts verified, grounded issues.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_frame_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_plus_query_critic_repair | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_diagnostic_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_repair_applied | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## sei_lithium_battery

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 11 | 0.6636 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| llm_intent_frame_only | plan | true | 11 | 0.6636 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation |
| llm_intent_plus_query_critic_repair | plan | true | 11 | 0.6636 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_diagnostic_only | plan | true | 11 | 0.6636 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_repair_applied | plan | true | 11 | 0.6636 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |

Pilot interpretation:

- `llm_intent_frame_only` tests controlled intent-frame suggestions entering SearchContract provenance after deterministic verification.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_intent_plus_query_critic_repair` is still plan-level only; it does not prove full retrieval or ranking improvement.
- No verified repair opportunity; clean plan remained unchanged.
- `llm_query_critic_diagnostic_only` is plan-level diagnostic only: it records verified/rejected critique issues but must not mutate final provider queries.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_query_critic_repair_applied` is a controlled repair pilot: query changes only count when the deterministic rule applier accepts verified, grounded issues.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_frame_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_plus_query_critic_repair | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_diagnostic_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_repair_applied | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## thin_film_deposition

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 5 | 0.6200 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| llm_intent_frame_only | plan | true | 5 | 0.6200 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation |
| llm_intent_plus_query_critic_repair | plan | true | 5 | 0.6200 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_diagnostic_only | plan | true | 5 | 0.6200 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; No verified repair opportunity; clean plan remained unchanged. |
| llm_query_critic_repair_applied | plan | true | 5 | 0.6200 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; Repair flag enabled, but no verified grounded issue was available.; No verified repair opportunity; clean plan remained unchanged. |

Pilot interpretation:

- `llm_intent_frame_only` tests controlled intent-frame suggestions entering SearchContract provenance after deterministic verification.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_intent_plus_query_critic_repair` is still plan-level only; it does not prove full retrieval or ranking improvement.
- No verified repair opportunity; clean plan remained unchanged.
- `llm_query_critic_diagnostic_only` is plan-level diagnostic only: it records verified/rejected critique issues but must not mutate final provider queries.
- No verified repair opportunity; clean plan remained unchanged.
- Repair flag enabled, but no verified grounded issue was available.
- `llm_query_critic_repair_applied` is a controlled repair pilot: query changes only count when the deterministic rule applier accepts verified, grounded issues.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_frame_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_intent_plus_query_critic_repair | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_diagnostic_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_repair_applied | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## weak_mof_short_query

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 8 | 1.0000 | 0 | 0 | 0 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 |  |  |  |  | 0 | 0 |  |  | weak-plan positive control; not part of formal benchmark conclusion |
| llm_query_critic_diagnostic_only | plan | false | 4 | 0.8000 | 2 | 0 | 2 | 0 | false | false | 0 | true | false | 0 |  | 0.0000 |  |  |  |  | 0 | 0 |  |  | partially_supported ablation; diagnostic only; query repair disabled; no-difference reason: repair stage disabled but upstream sanitizer still applied; query repair ablation is non-conclusive because upstream sanitizer remains active; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; weak-plan positive control; not part of formal benchmark conclusion; LLM query critic did not mutate query plan |
| llm_query_critic_repair_applied | plan | false | 5 | 0.8400 | 2 | 0 | 2 | 0 | false | true | 1 | false | true | 0 |  | 0.0000 |  |  |  |  | 0 | 0 |  |  | partially_supported ablation; diagnostic only; deterministic query repair disabled; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; weak-plan positive control; not part of formal benchmark conclusion; LLM critique issue applied by deterministic rule applier |

Pilot interpretation:

- `llm_query_critic_diagnostic_only` is plan-level diagnostic only: it records verified/rejected critique issues but must not mutate final provider queries.
- Repair was applied and query_quality_score improved under current heuristic.
- Weak-plan positive control applied repair: verified_issue_count=1, applied_issue_count=1, query_modified_count=1, repair_grounded_term_count=1, repair_rejected_term_count=1, before=MOF CO2, after=MOF CO2 "metal-organic framework".
- `llm_query_critic_repair_applied` is a controlled repair pilot: query changes only count when the deterministic rule applier accepts verified, grounded issues.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_diagnostic_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_repair_applied | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## weak_oer_acronym_query

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 4 | 0.9000 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 |  |  |  |  | 0 | 0 |  |  | weak query heuristic hit; weak-plan positive control; not part of formal benchmark conclusion |
| llm_query_critic_diagnostic_only | plan | false | 4 | 0.7000 | 3 | 1 | 3 | 0 | false | false | 0 | true | false | 0 |  | 0.0000 |  |  |  |  | 0 | 0 |  |  | partially_supported ablation; diagnostic only; query repair disabled; no-difference reason: repair stage disabled but upstream sanitizer still applied; query repair ablation is non-conclusive because upstream sanitizer remains active; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; weak-plan positive control; not part of formal benchmark conclusion; LLM query critic did not mutate query plan |
| llm_query_critic_repair_applied | plan | false | 4 | 0.7000 | 3 | 0 | 3 | 0 | false | true | 2 | false | true | 0 |  | 0.0000 |  |  |  |  | 0 | 0 |  |  | partially_supported ablation; diagnostic only; deterministic query repair disabled; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; weak-plan positive control; not part of formal benchmark conclusion; LLM critique issue applied by deterministic rule applier |

Pilot interpretation:

- `llm_query_critic_diagnostic_only` is plan-level diagnostic only: it records verified/rejected critique issues but must not mutate final provider queries.
- Repair was applied and provenance is valid, but query_quality_score did not improve under current heuristic; this should be interpreted as repair-mechanism validation, not quality improvement.
- Weak-plan positive control applied repair: verified_issue_count=1, applied_issue_count=1, query_modified_count=1, repair_grounded_term_count=1, repair_rejected_term_count=1, before=oer OER, after=oer OER "oxygen evolution reaction".
- `llm_query_critic_repair_applied` is a controlled repair pilot: query changes only count when the deterministic rule applier accepts verified, grounded issues.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_diagnostic_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_repair_applied | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## weak_sei_acronym_query

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 7 | 1.0000 | 0 | 0 | 0 | 0 | true | false | 0 | false | true | 0 |  | 0.0000 |  |  |  |  | 0 | 0 |  |  | weak-plan positive control; not part of formal benchmark conclusion |
| llm_query_critic_diagnostic_only | plan | false | 4 | 0.7000 | 3 | 1 | 3 | 0 | false | false | 0 | true | false | 0 |  | 0.0000 |  |  |  |  | 0 | 0 |  |  | partially_supported ablation; diagnostic only; query repair disabled; no-difference reason: repair stage disabled but upstream sanitizer still applied; query repair ablation is non-conclusive because upstream sanitizer remains active; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; weak-plan positive control; not part of formal benchmark conclusion; LLM query critic did not mutate query plan |
| llm_query_critic_repair_applied | plan | false | 4 | 0.7000 | 3 | 0 | 3 | 0 | false | true | 2 | false | true | 0 |  | 0.0000 |  |  |  |  | 0 | 0 |  |  | partially_supported ablation; diagnostic only; deterministic query repair disabled; weak query heuristic hit; LLM plan-level diagnostic; not a formal full retrieval ablation; weak-plan positive control; not part of formal benchmark conclusion; LLM critique issue applied by deterministic rule applier |

Pilot interpretation:

- `llm_query_critic_diagnostic_only` is plan-level diagnostic only: it records verified/rejected critique issues but must not mutate final provider queries.
- Repair was applied and provenance is valid, but query_quality_score did not improve under current heuristic; this should be interpreted as repair-mechanism validation, not quality improvement.
- Weak-plan positive control applied repair: verified_issue_count=1, applied_issue_count=1, query_modified_count=1, repair_grounded_term_count=1, repair_rejected_term_count=1, before=sei SEI, after=sei SEI "solid electrolyte interphase".
- `llm_query_critic_repair_applied` is a controlled repair pilot: query changes only count when the deterministic rule applier accepts verified, grounded issues.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_diagnostic_only | 0 | 0 | 0 | 0 | 0 | false |
| llm_query_critic_repair_applied | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.
