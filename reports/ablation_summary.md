# Pilot Ablation Summary

This is a pilot / diagnostic ablation summary generated from artifact-level heuristics.
It is not a final experimental conclusion and should not be over-interpreted without larger benchmarks, provider-stability checks, and human labels.

## Ablation support status

### Fully supported ablations

- full_system (full_system:supported)

### Partially supported ablations

- None observed in this summary.

### Unsupported ablations

- None observed in this summary.

## oer_spin_state

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | full | True | 6 | 0.6000 | 0 | 0 | 0 | 0 | True | True | 0 | False | True | 66 |  | 1.0000 | 0 | 0 | 0 | 0 | 12 | 25 | 1.0000 | 0.9927 |  |

Pilot interpretation:

- These signals are pilot diagnostics only, not final ablation evidence.

## sei_lithium_battery

| Config | Mode | QF applied | Queries | Quality | Weak | Overbroad | Repeated | Single-axis | Repair enabled | Repair applied | Raw-final change | Sanitizer active | Repair conclusive | Merged | Duplicate | Provider success | Forbidden top10 | Forbidden top20 | Forbidden must-read | Forbidden include | Must-read | Include | Req group top10 | Intent centrality top10 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_system | full | True | 11 | 0.6636 | 1 | 0 | 1 | 0 | True | True | 0 | False | True | 130 |  | 1.0000 | 0 | 0 | 0 | 0 | 12 | 59 | 1.0000 | 0.9985 | weak query heuristic hit |

Pilot interpretation:

- These signals are pilot diagnostics only, not final ablation evidence.
