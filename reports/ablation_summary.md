# Pilot Ablation Summary

This is a pilot / diagnostic ablation summary generated from artifact-level heuristics.
It is not a final experimental conclusion and should not be over-interpreted without larger benchmarks, provider-stability checks, and human labels.

## Ablation support status

### Fully supported ablations

- full_system (full_system:baseline)

### Partially supported ablations (diagnostic / non-conclusive)

- llm_intent_plus_query_critic_repair (llm_intent_frame_enhancer:diagnostic_controlled, llm_query_plan_critic:diagnostic_artifacts_only, llm_query_plan_critic_repairs:controlled_repair_pilot)

### Unsupported ablations

- None observed in this summary.

## Small full-retrieval LLM pilot

This is a safety pilot, not a formal LLM ablation conclusion.
It checks whether LLM-enhanced planning can run through full retrieval without breaking guardrails, ranking, reading_path, or reports.
Any quality improvement claims are tentative.
If no verified LLM repair was available, this is not a failure; it means the query plan may already have been strong.
If full retrieval changes, interpret cautiously because provider ranking and retrieval stochasticity may affect results.
LLM modules do not make paper-level decisions.

| Case | Config | Retrieval status | Retrieval performed | Real retrieval | Raw | Merged | Provider success | Provider errors | Must-read | Include | Optional | Exclude | Forbidden must-read | Negative must-read | Reading leaks | Intent called | Critic called | Repair enabled | Verified issues | Applied issues | Query modified | Direct paper/evidence/ranking mutations |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| oer_spin_state | full_system | success | true | true | 110 | 66 | 1.0000 | 0 | 12 | 25 | 12 | 27 | 0 | 0 | exclude=0; out_of_scope=0; duplicate=0; negative=0 | false | false | false | 0 | 0 | 0 | paper=0; evidence=0; ranking=0 |
| oer_spin_state | llm_intent_plus_query_critic_repair | success | true | true | 110 | 66 | 1.0000 | 0 | 12 | 25 | 12 | 27 | 0 | 0 | exclude=0; out_of_scope=0; duplicate=0; negative=0 | true | true | true | 0 | 0 | 0 | paper=0; evidence=0; ranking=0 |
| sei_lithium_battery | full_system | success | true | true | 220 | 130 | 1.0000 | 0 | 12 | 59 | 39 | 14 | 0 | 0 | exclude=0; out_of_scope=0; duplicate=0; negative=0 | false | false | false | 0 | 0 | 0 | paper=0; evidence=0; ranking=0 |
| sei_lithium_battery | llm_intent_plus_query_critic_repair | success | true | true | 220 | 130 | 1.0000 | 0 | 12 | 59 | 39 | 14 | 0 | 0 | exclude=0; out_of_scope=0; duplicate=0; negative=0 | true | true | true | 0 | 0 | 0 | paper=0; evidence=0; ranking=0 |

- Do not claim LLM improves retrieval precision from this section unless independent metrics clearly support it.
- Use this section to check full-pipeline safety signals: retrieval completed, reading-path leaks stayed bounded, negative-context papers were not promoted, and LLM direct mutation counters stayed zero.

## oer_spin_state

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | full | true | 6 | 0.6000 | 0 | 0 | 0 | 0 | true | true | 0 | false | true | 66 | 0.4000 | 1.0000 | 0 | 0 | 0 | 0 | 12 | 25 | 1.0000 | 0.9927 |  |
| llm_intent_plus_query_critic_repair | full | true | 6 | 0.6000 | 0 | 0 | 0 | 0 | true | true | 0 | false | true | 66 | 0.4000 | 1.0000 | 0 | 0 | 0 | 0 | 12 | 25 | 1.0000 | 0.9927 | partially_supported ablation; diagnostic only; Small full-retrieval LLM safety pilot; not a formal LLM ablation conclusion; LLM repair apply flag enabled, but no verified grounded issue was available; clean plan remained unchanged.; No verified repair opportunity; clean plan remained unchanged. |

Pilot interpretation:

- Small full-retrieval LLM safety pilot; interpret retrieval and ranking deltas cautiously because provider ranking and retrieval stochasticity may affect results.
- No verified repair opportunity; clean plan remained unchanged.
- LLM repair apply flag enabled, but no verified grounded issue was available; clean plan remained unchanged.
- `llm_intent_plus_query_critic_repair` is a small full-retrieval safety pilot config; it does not prove retrieval precision, ranking, or screening improvement.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 11 | 0 | 0 | 0 | 0 | false |
| llm_intent_plus_query_critic_repair | 11 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## sei_lithium_battery

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | full | true | 11 | 0.6636 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 130 | 0.4091 | 1.0000 | 0 | 0 | 0 | 0 | 12 | 59 | 1.0000 | 0.9985 | weak query heuristic hit |
| llm_intent_plus_query_critic_repair | full | true | 11 | 0.6636 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 130 | 0.4091 | 1.0000 | 0 | 0 | 0 | 0 | 12 | 59 | 1.0000 | 0.9512 | partially_supported ablation; diagnostic only; weak query heuristic hit; Small full-retrieval LLM safety pilot; not a formal LLM ablation conclusion; LLM repair apply flag enabled, but no verified grounded issue was available; clean plan remained unchanged.; No verified repair opportunity; clean plan remained unchanged. |

Pilot interpretation:

- Small full-retrieval LLM safety pilot; interpret retrieval and ranking deltas cautiously because provider ranking and retrieval stochasticity may affect results.
- No verified repair opportunity; clean plan remained unchanged.
- LLM repair apply flag enabled, but no verified grounded issue was available; clean plan remained unchanged.
- `llm_intent_plus_query_critic_repair` is a small full-retrieval safety pilot config; it does not prove retrieval precision, ranking, or screening improvement.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 12 | 0 | 0 | 0 | 0 | true |
| llm_intent_plus_query_critic_repair | 12 | 0 | 0 | 0 | 0 | true |
- These signals are pilot diagnostics only, not final ablation evidence.
