# Pilot Ablation Summary

This is a pilot / diagnostic ablation summary generated from artifact-level heuristics.
It is not a final experimental conclusion and should not be over-interpreted without larger benchmarks, provider-stability checks, and human labels.

## Ablation support status

### Fully supported ablations

- full_system (full_system:supported)
- no_query_family (query_family:supported)

### Partially supported ablations (diagnostic / non-conclusive)

- no_query_repair (query_repair:supported, query_repair:diagnostic_non_conclusive)

### Unsupported ablations

- None observed in this summary.

## ai_literature_screening

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 10 | 0.9600 | 1 | 0 | 0 | 1 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| no_query_family | plan | false | 5 | 0.3800 | 4 | 2 | 0 | 3 | true | false | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| no_query_repair | plan | true | 10 | 0.9600 | 1 | 0 | 0 | 1 | false | false | 0 | true | false | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; query repair disabled; no-difference reason: repair stage disabled but upstream sanitizer still applied; query repair ablation is non-conclusive because upstream sanitizer remains active; no_query_repair is diagnostic/non-conclusive in this pilot because upstream sanitizer remains active; weak query heuristic hit |

Pilot interpretation:

- `no_query_family` degraded query quality in this pilot: query_quality_score -0.58, expected_anchor_coverage -0.50, weak_query_count +3.00, overbroad_query_count +2.00, repeated_phrase_query_count +0.00, single_axis_query_count +2.00.
- `no_query_repair` is diagnostic only because upstream sanitizer remains active. If final queries are unchanged, do not conclude QueryRepair has no value. `full_system` raw_to_final_query_change_count=0; `no_query_repair` raw_to_final_query_change_count=0.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| no_query_family | 0 | 0 | 0 | 0 | 0 | false |
| no_query_repair | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## mof_co2_capture

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 8 | 0.5000 | 2 | 2 | 0 | 4 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| no_query_family | plan | false | 12 | 0.2000 | 12 | 9 | 5 | 11 | true | false | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| no_query_repair | plan | true | 8 | 0.5000 | 2 | 2 | 0 | 4 | false | false | 0 | true | false | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; query repair disabled; no-difference reason: repair stage disabled but upstream sanitizer still applied; query repair ablation is non-conclusive because upstream sanitizer remains active; no_query_repair is diagnostic/non-conclusive in this pilot because upstream sanitizer remains active; weak query heuristic hit |

Pilot interpretation:

- `no_query_family` degraded query quality in this pilot: query_quality_score -0.30, expected_anchor_coverage -0.17, weak_query_count +10.00, overbroad_query_count +7.00, repeated_phrase_query_count +5.00, single_axis_query_count +7.00.
- `no_query_repair` is diagnostic only because upstream sanitizer remains active. If final queries are unchanged, do not conclude QueryRepair has no value. `full_system` raw_to_final_query_change_count=0; `no_query_repair` raw_to_final_query_change_count=0.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| no_query_family | 0 | 0 | 0 | 0 | 0 | false |
| no_query_repair | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## oer_spin_state

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | full | true | 6 | 0.6000 | 0 | 0 | 0 | 0 | true | true | 0 | false | true | 66 |  | 1.0000 | 0 | 0 | 0 | 0 | 12 | 25 | 1.0000 | 0.9927 |  |

Pilot interpretation:


Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 11 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.

## sei_lithium_battery

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | full | true | 11 | 0.6636 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 130 |  | 1.0000 | 0 | 0 | 0 | 0 | 12 | 59 | 1.0000 | 0.9985 | weak query heuristic hit |

Pilot interpretation:


Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 12 | 0 | 0 | 0 | 0 | true |
- These signals are pilot diagnostics only, not final ablation evidence.

## thin_film_deposition

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | plan | true | 5 | 0.6200 | 1 | 0 | 1 | 0 | true | true | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| no_query_family | plan | false | 7 | 0.1000 | 7 | 1 | 1 | 0 | true | false | 0 | false | true | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | weak query heuristic hit |
| no_query_repair | plan | true | 5 | 0.6200 | 1 | 0 | 1 | 0 | false | false | 0 | true | false | 0 |  | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0 |  |  | partially_supported ablation; diagnostic only; query repair disabled; no-difference reason: repair stage disabled but upstream sanitizer still applied; query repair ablation is non-conclusive because upstream sanitizer remains active; no_query_repair is diagnostic/non-conclusive in this pilot because upstream sanitizer remains active; weak query heuristic hit |

Pilot interpretation:

- `no_query_family` degraded query quality in this pilot: query_quality_score -0.52, expected_anchor_coverage -0.33, weak_query_count +6.00, overbroad_query_count +1.00, repeated_phrase_query_count +0.00, single_axis_query_count +0.00.
- `no_query_repair` is diagnostic only because upstream sanitizer remains active. If final queries are unchanged, do not conclude QueryRepair has no value. `full_system` raw_to_final_query_change_count=0; `no_query_repair` raw_to_final_query_change_count=0.

Reading-path diagnostics:

| Config | Reading path papers | Exclude leaks | Out-of-scope leaks | Duplicate leaks | Negative-context leaks | Target context required |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | 0 | 0 | 0 | 0 | 0 | false |
| no_query_family | 0 | 0 | 0 | 0 | 0 | false |
| no_query_repair | 0 | 0 | 0 | 0 | 0 | false |
- These signals are pilot diagnostics only, not final ablation evidence.
