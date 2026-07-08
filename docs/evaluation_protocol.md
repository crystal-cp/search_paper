# Evaluation Protocol for `search_paper`

## 1. Purpose

`search_paper` is evaluated as a **human-centric multi-agent LLM scientific literature screening system**, not as a simple paper search tool.

The evaluation therefore asks five questions:

1. Did the system correctly repair and structure the novice user's research intent?
2. Did the system generate domain-aware query families rather than a single shallow keyword query?
3. Did provider retrieval return a sufficiently broad and relevant candidate pool?
4. Did evidence validation, domain guardrails, ranking, and paper roles produce a useful screening result?
5. Did the user report and feedback mechanism help the user understand and refine the literature search?

The protocol evaluates both:

* **plan-only mode**, where no provider retrieval is performed;
* **full retrieval mode**, where provider retrieval, evidence validation, ranking, report generation, and feedback adaptation are evaluated.

It also separates:

* **rule-mode**, where the pipeline uses rule-based agents;
* **LLM-enhanced mode**, where planner, extractor, and/or verifier use an LLM backend.

---

## 2. Evaluation Units

Each benchmark case is one imperfect novice research question.

For each case, the evaluator should run four configurations when possible:

| Run ID      | Mode         | Retrieval | LLM usage                  | Purpose                                                     |
| ----------- | ------------ | --------: | -------------------------- | ----------------------------------------------------------- |
| `rule_plan` | rule         |        no | no                         | Tests deterministic novice intent repair and query planning |
| `llm_plan`  | LLM-enhanced |        no | planner only               | Tests whether LLM improves intent repair/query planning     |
| `rule_full` | rule         |       yes | no                         | Tests full rule-based screening pipeline                    |
| `llm_full`  | LLM-enhanced |       yes | planner/extractor/verifier | Tests whether LLM improves screening without hallucination  |

Recommended output layout:

```text
outputs/eval/
  sei_lithium_battery/
    rule_plan/
    llm_plan/
    rule_full/
    llm_full/
  oer_spin_state/
  ai_literature_screening/
  ...
```

---

## 3. Recommended Commands

### 3.1 Plan-only rule-mode

```bash
python -m lit_screening.pipeline run \
  --question "$QUESTION" \
  --providers openalex semantic_scholar \
  --max-per-query 0 \
  --strictness balanced \
  --output-dir "outputs/eval/$CASE_ID/rule_plan"
```

### 3.2 Plan-only LLM-enhanced mode

```bash
python -m lit_screening.pipeline run \
  --question "$QUESTION" \
  --providers openalex semantic_scholar \
  --max-per-query 0 \
  --strictness balanced \
  --llm-backend deepseek \
  --planner-mode llm \
  --output-dir "outputs/eval/$CASE_ID/llm_plan"
```

### 3.3 Full rule-mode

```bash
python -m lit_screening.pipeline run \
  --question "$QUESTION" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --strictness balanced \
  --openalex-mode keyword+semantic \
  --sort-preference relevance \
  --ranking-profile balanced \
  --output-dir "outputs/eval/$CASE_ID/rule_full"
```

### 3.4 Full LLM-enhanced mode

```bash
python -m lit_screening.pipeline run \
  --question "$QUESTION" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --strictness balanced \
  --openalex-mode keyword+semantic \
  --sort-preference relevance \
  --ranking-profile balanced \
  --llm-backend deepseek \
  --planner-mode llm \
  --extractor-mode llm \
  --verifier-mode llm \
  --output-dir "outputs/eval/$CASE_ID/llm_full"
```

If `--use-query-families` is available and stable in the current branch, run an additional retrieval condition:

```bash
python -m lit_screening.pipeline run \
  --question "$QUESTION" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --use-query-families \
  --output-dir "outputs/eval/$CASE_ID/rule_full_query_families"
```

This isolates whether `QueryFamily` planning only improves explanation or also improves actual retrieval.

---

## 4. Required Artifacts

The evaluator should treat missing artifacts as failures unless the artifact is not expected for the run type.

| Artifact                            |    Plan-only |                              Full retrieval | Purpose                                    |
| ----------------------------------- | -----------: | ------------------------------------------: | ------------------------------------------ |
| `search_brief.json`                 |     required |                                    required | Repaired user intent                       |
| `search_contract.json`              |     required |                                    required | Domain boundary and exclusion criteria     |
| `ambiguity_analysis.json`           |     required |                                    required | Ambiguous terms and selected meanings      |
| `question_refinement.json`          |     required |                                    required | Novice intent repair trace                 |
| `planned_queries.json`              |     required |                                    required | Provider-aware query plan                  |
| `concept_map.json`                  |     required |                                    required | Concept decomposition                      |
| `query_families.json`               |     required |                                    required | QueryFamily planning                       |
| `seed_hints.json`                   |     optional |                                    optional | Seed title/DOI/author extraction           |
| `query_provenance.json`             |     optional | required if query families affect retrieval | Query-to-provider trace                    |
| `raw_openalex_results.json`         | not required |                required if provider enabled | Raw provider result audit                  |
| `raw_semantic_scholar_results.json` | not required |                required if provider enabled | Raw provider result audit                  |
| `retrieval_diagnostics.json`        |     optional |                                    required | Provider/query success and error audit     |
| `merged_papers.csv`                 | not required |                                    required | Deduplicated candidate pool                |
| `evidence_table.csv`                | not required |                                    required | Evidence claims and span validation fields |
| `domain_assessments.json`           | not required |                                    required | In-scope/borderline/out-of-scope labels    |
| `aspect_coverage.csv`               | not required |                                    required | Required aspect coverage by paper          |
| `paper_roles.json`                  |     optional |                                    required | Research role classification               |
| `screening_decisions.csv`           | not required |                                    required | Include/maybe/exclude recommendations      |
| `ranked_papers.csv`                 | not required |                                    required | Final ranking                              |
| `ranked_papers_before_feedback.csv` |     optional |          required when feedback is provided | Feedback comparison                        |
| `ranked_papers_after_feedback.csv`  |     optional |          required when feedback is provided | Feedback comparison                        |
| `preference_learning.json`          |     optional |          required when feedback is provided | Learned preference terms                   |
| `feedback_query_refinement.json`    |     optional |          required when feedback is provided | Feedback-informed next-query suggestions   |
| `evaluation.json`                   |     optional |                                    required | Built-in ranking/evidence metrics          |
| `exploration_quality.json`          |     required |                                    required | Sensemaking quality metrics                |
| `paper_cards.md`                    | not required |                                    required | Paper-level explanations                   |
| `reading_path.md`                   | not required |                                    required | Novice reading plan                        |
| `research_gap_matrix.csv`           |     optional |                                    required | Remaining gap analysis                     |
| `suggested_next_searches.json`      |     optional |                                    required | Next search directions                     |
| `report.md`                         |     required |                                    required | User-facing research report                |
| `agent_trace.json`                  |     required |                                    required | Multi-agent decision trace                 |
| `run_events.jsonl`                  |     required |                                    required | Runtime stage/error trace                  |

---

# 5. Metric Layers

## Layer A: Intent Understanding

This layer evaluates whether the system repaired the novice question into a usable research intent without over-repairing it.

| Metric                            | Calculation                                                                                                                                                                                                      | Artifact                                                                                             | Pass Standard                                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `search_brief_completeness`       | Required non-empty fields / required fields. Required fields: `refined_question`, `search_intent`, `inclusion_criteria`, `exclusion_criteria`, `required_aspects`, `preferred_paper_types`, `success_definition` | `search_brief.json`                                                                                  | `>= 0.85`                                    |
| `intent_anchor_coverage`          | Number of benchmark expected anchor groups mentioned in `search_brief`, `question_refinement`, or `search_contract` / total expected anchor groups                                                               | `search_brief.json`, `question_refinement.json`, `search_contract.json`, `data/benchmark_cases.yaml` | `>= 0.70`                                    |
| `required_aspect_recall_in_brief` | Number of expected aspects represented in `search_brief.required_aspects` / total expected aspects                                                                                                               | `search_brief.json`, `data/benchmark_cases.yaml`                                                     | `>= 0.70`                                    |
| `ambiguity_resolution_score`      | Number of ambiguous or overloaded terms correctly identified with selected meaning and excluded meanings / number of benchmark ambiguous terms                                                                   | `ambiguity_analysis.json`                                                                            | `>= 0.60`; `>= 0.80` for acronym-heavy cases |
| `domain_boundary_completeness`    | Presence of required concepts, excluded concepts, field boundaries, and false-positive terms                                                                                                                     | `search_contract.json`                                                                               | `>= 0.75`                                    |
| `over_repair_failure_count`       | Manual or rule-based count of cases where the system imposes a narrower domain not supported by the user query                                                                                                   | `search_brief.json`, `agent_trace.json`                                                              | `0 critical failures`                        |
| `novice_readability_score`        | Human check: can a novice understand the repaired question, inclusion/exclusion criteria, and success definition? 0–2 scale per field                                                                            | `search_brief.json`, `report.md`                                                                     | average `>= 1.5 / 2`                         |

### Hard failures for Layer A

A case fails the intent layer if:

* the system selects the wrong meaning of a central ambiguous term;
* the repaired intent excludes the actual target domain;
* the generated intent is generic and does not contain domain-specific anchors;
* Chinese novice input is not converted into usable English scientific search terms.

---

## Layer B: Query Planning

This layer evaluates whether the system created a structured query plan and multiple query families.

| Metric                            | Calculation                                                                                                                            | Artifact                                                                   | Pass Standard                                          |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------ |
| `planned_query_anchor_coverage`   | Expected anchor groups matched in `planned_queries` and `query_families` / total expected anchor groups                                | `planned_queries.json`, `query_families.json`, `data/benchmark_cases.yaml` | `>= 0.75`                                              |
| `provider_query_completeness`     | Number of enabled providers with at least one non-empty query / number of enabled providers                                            | `planned_queries.json`                                                     | `1.00` unless provider disabled                        |
| `query_family_coverage`           | Lenses with at least one query family / total lenses                                                                                   | `concept_map.json`, `query_families.json`, `exploration_quality.json`      | `>= 0.70`                                              |
| `lens_diversity`                  | Number of distinct research lenses                                                                                                     | `concept_map.json`                                                         | `>= 3`; `>= 4` for broad review-like cases             |
| `query_specificity_score`         | Non-generic query count / total query count. Generic queries are those with only one acronym, only one broad noun, or no domain anchor | `planned_queries.json`, `query_families.json`, `exploration_quality.json`  | `>= 0.80`                                              |
| `single_acronym_query_count`      | Count of queries such as only `SEI`, `OER`, `MOF`, `AI`, `PFM`                                                                         | `exploration_quality.json`, `planned_queries.json`                         | `<= 1`; ideally `0`                                    |
| `forbidden_query_leakage_rate`    | Queries containing benchmark forbidden patterns / total queries                                                                        | `planned_queries.json`, `query_families.json`, `data/benchmark_cases.yaml` | `0` for severe forbidden patterns; `<= 0.05` otherwise |
| `provider_query_adaptation_score` | Queries use provider-appropriate syntax, such as phrase queries, required terms, OR alternatives, and exclusions                       | `planned_queries.json`                                                     | `>= 0.75`                                              |
| `query_provenance_completeness`   | Query provenance records / provider query count                                                                                        | `query_provenance.json`, `retrieval_diagnostics.json`                      | `>= 0.90` when query provenance is expected            |

### Hard failures for Layer B

A case fails the query-planning layer if:

* all provider queries collapse into one generic query;
* the plan uses only the novice's raw words without expert anchor expansion;
* forbidden meanings appear in planned queries;
* QueryFamily planning is empty for a non-trivial case;
* the query plan does not contain provider-specific queries.

---

## Layer C: Retrieval

This layer requires full retrieval.

| Metric                      | Calculation                                                                                                     | Artifact                                                                   | Pass Standard                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `provider_success_rate`     | Providers without fatal error / enabled providers                                                               | `retrieval_diagnostics.json`, `run_events.jsonl`                           | `>= 0.50`; provider API outage should be marked infrastructure failure, not algorithm failure |
| `raw_retrieved_count`       | Total raw provider results before deduplication                                                                 | `evaluation.json`, `retrieval_diagnostics.json`                            | `>= case.min_raw_results`, default `>= 20`                                                    |
| `merged_paper_count`        | Number of deduplicated papers                                                                                   | `merged_papers.csv`, `evaluation.json`                                     | `>= case.min_merged_results`, default `>= 15`                                                 |
| `zero_result_failure`       | Whether all providers return zero usable papers                                                                 | `retrieval_diagnostics.json`                                               | must be `false`                                                                               |
| `retrieval_anchor_recall`   | Expected anchor groups appearing in title/abstract/keywords across merged papers / total expected anchor groups | `merged_papers.csv`, `data/benchmark_cases.yaml`                           | `>= 0.60`                                                                                     |
| `top_query_yield_balance`   | Number of query families contributing at least one merged paper / total query families used for retrieval       | `query_provenance.json`, `retrieval_diagnostics.json`, `merged_papers.csv` | `>= 0.40`; used only if QueryFamily retrieval is enabled                                      |
| `dedup_sanity_ratio`        | `merged_paper_count / raw_retrieved_count`                                                                      | `evaluation.json`                                                          | between `0.25` and `1.00`; outside range requires audit                                       |
| `missing_abstract_ratio`    | Papers without abstracts / merged papers                                                                        | `evaluation.json`                                                          | `<= 0.50`; stricter `<= 0.30` for evidence-heavy cases                                        |
| `provider_error_visibility` | Provider errors are recorded with provider, query, and error reason                                             | `retrieval_diagnostics.json`, `run_events.jsonl`                           | all fatal errors must be visible                                                              |

### Hard failures for Layer C

A case fails retrieval if:

* no usable papers are retrieved and no provider error is reported;
* all top results come from an unintended meaning of the query;
* provider errors are silently treated as valid zero-result retrieval;
* retrieval ignores the planned provider queries.

---

## Layer D: Screening, Evidence, Domain Guardrail, Ranking, and Paper Role

This layer evaluates whether the system turns retrieved candidates into reliable screening decisions.

| Metric                            | Calculation                                                                                                                                   | Artifact                                                              | Pass Standard                                                                      |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `top10_domain_in_scope_rate`      | Top-10 papers labeled `in_scope` or acceptable `borderline` / 10                                                                              | `ranked_papers.csv`, `domain_assessments.json`                        | `>= 0.80`                                                                          |
| `top10_out_of_scope_rate`         | Top-10 papers labeled `out_of_scope` / 10                                                                                                     | `ranked_papers.csv`, `domain_assessments.json`                        | `<= 0.10`                                                                          |
| `top10_forbidden_pattern_rate`    | Top-10 papers matching benchmark forbidden patterns / 10                                                                                      | `ranked_papers.csv`, `merged_papers.csv`, `data/benchmark_cases.yaml` | `<= 0.10`; severe cases require `0`                                                |
| `top10_anchor_coverage`           | Expected anchor groups represented by top-10 title/abstract/evidence / total expected anchor groups                                           | `ranked_papers.csv`, `evidence_table.csv`                             | `>= 0.60`                                                                          |
| `aspect_coverage_at_10`           | Expected aspects covered by at least one top-10 paper / total expected aspects                                                                | `aspect_coverage.csv`, `ranked_papers.csv`                            | `>= 0.60`; `>= 0.70` for mature domains                                            |
| `screening_decision_completeness` | Papers with include/maybe/exclude decision / merged papers                                                                                    | `screening_decisions.csv`                                             | `>= 0.95`                                                                          |
| `evidence_record_coverage`        | Papers with at least one evidence record / merged papers with abstracts                                                                       | `evidence_table.csv`, `merged_papers.csv`                             | `>= 0.70`                                                                          |
| `strict_support_rate`             | Strictly supported evidence records / all verification records                                                                                | `evaluation.json`, `evidence_table.csv`                               | `>= 0.45` rule-mode; `>= 0.55` LLM-enhanced mode                                   |
| `llm_invalid_evidence_rate`       | LLM-invalid evidence records / all verification records                                                                                       | `evaluation.json`, `evidence_table.csv`                               | `<= 0.05`; hard fail if `> 0.10`                                                   |
| `span_audit_completeness`         | Evidence records with `support_level`, `span_match_type`, `span_match_confidence`, `matched_text`, `strict_span_validated` / evidence records | `evidence_table.csv`                                                  | `>= 0.95`                                                                          |
| `ranking_score_transparency`      | Ranked papers with score breakdown fields / ranked papers                                                                                     | `ranked_papers.csv`                                                   | `>= 0.95`                                                                          |
| `precision_at_10`                 | Included gold-label papers in top 10 / judged top-10 papers                                                                                   | `evaluation.json`, `data/gold_labels/*.csv`                           | `>= 0.60` when gold labels exist                                                   |
| `ndcg_at_10`                      | Binary nDCG@10 using include labels as relevant                                                                                               | `evaluation.json`, `data/gold_labels/*.csv`                           | `>= 0.70` when gold labels exist                                                   |
| `recall_at_10`                    | Included gold-label papers in top 10 / all included gold-label papers                                                                         | `evaluation.json`, `data/gold_labels/*.csv`                           | case-dependent; default `>= 0.40`                                                  |
| `paper_role_assignment_rate`      | Papers with at least one non-empty role / top-20 papers                                                                                       | `paper_roles.json`                                                    | `>= 0.70`                                                                          |
| `paper_role_diversity`            | Distinct useful roles / expected role families                                                                                                | `paper_roles.json`, `exploration_quality.json`                        | `>= 0.30` current minimum; increase after domain-general role taxonomy is improved |

### Hard failures for Layer D

A case fails the screening/ranking layer if:

* more than two top-10 papers are clear false positives;
* LLM-generated evidence is not grounded and still presented as reliable;
* out-of-domain papers are not demoted;
* ranking is dominated by citation count or recency while ignoring intent centrality;
* evidence span fields are missing from the evidence table.

---

## Layer E: Report Quality and User Feedback Adaptation

This layer evaluates whether the system helps a novice user understand and refine the literature.

| Metric                                | Calculation                                                                                                 | Artifact                                                                                            | Pass Standard                     |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------- |
| `report_section_completeness`         | Required report sections present / required sections                                                        | `report.md`                                                                                         | `>= 0.85`                         |
| `paper_card_coverage`                 | Top-10 papers with paper cards / 10                                                                         | `paper_cards.md`                                                                                    | `>= 0.90`                         |
| `reading_path_presence`               | Reading path exists and includes grouped next-reading recommendations                                       | `reading_path.md`                                                                                   | must be present in full retrieval |
| `gap_matrix_specificity`              | Specific gap rows with reason and suggested next search / all gap rows                                      | `research_gap_matrix.csv`, `exploration_quality.json`                                               | `>= 0.60`                         |
| `suggested_next_search_specificity`   | Suggested next searches containing domain anchors and method/material constraints / all suggested searches  | `suggested_next_searches.json`                                                                      | `>= 0.70`                         |
| `uncertainty_visibility`              | Report explicitly marks uncertain, weakly supported, or borderline findings                                 | `report.md`, `evidence_table.csv`, `domain_assessments.json`                                        | required                          |
| `unsupported_citation_relation_count` | Claims of citation lineage not supported by seed/citation artifacts                                         | `report.md`, `retrieval_paths.csv`, `citation_expansion.csv`                                        | `0`                               |
| `feedback_directional_accuracy`       | User-included papers move up or stay top; user-excluded papers move down or leave top-10 / feedback actions | `ranked_papers_before_feedback.csv`, `ranked_papers_after_feedback.csv`, `preference_learning.json` | `>= 0.80`                         |
| `feedback_query_refinement_quality`   | Learned include/exclude terms align with feedback and do not overfit to one title                           | `feedback_query_refinement.json`, `preference_learning.json`                                        | manual score `>= 1.5 / 2`         |
| `feedback_overfit_failure_count`      | Number of feedback actions causing unrelated relevant papers to be severely demoted                         | before/after rankings                                                                               | `0 critical failures`             |

Required report sections:

```text
Research question interpretation
Search strategy / query families
Screening and inclusion criteria
Ranked papers
Evidence grounding
Domain guardrail / uncertainty
Paper roles / reading path
Research gaps
Suggested next searches
Feedback effects, if feedback is provided
```

### Hard failures for Layer E

A case fails the report/adaptation layer if:

* the report reads like a generic summary rather than a screening explanation;
* top-paper claims are not tied to evidence;
* feedback changes are not visible;
* the report hides uncertainty;
* suggested next searches are generic and do not contain domain anchors.

---

# 6. Plan-Only Metrics

The following metrics are valid without provider retrieval:

| Layer                | Metrics                                                                                                                                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Intent understanding | `search_brief_completeness`, `intent_anchor_coverage`, `required_aspect_recall_in_brief`, `ambiguity_resolution_score`, `domain_boundary_completeness`, `over_repair_failure_count`, `novice_readability_score`                       |
| Query planning       | `planned_query_anchor_coverage`, `provider_query_completeness`, `query_family_coverage`, `lens_diversity`, `query_specificity_score`, `single_acronym_query_count`, `forbidden_query_leakage_rate`, `provider_query_adaptation_score` |
| Report structure     | Report existence, Research Process section existence, explanation of query families, explicit statement that paper-dependent artifacts are not generated                                                                              |

Plan-only mode is especially useful for testing:

* Chinese-to-English planning;
* novice intent repair;
* domain pack activation;
* ambiguous acronym handling;
* QueryFamily generation;
* provider query syntax;
* false-positive prevention before spending provider API calls.

Plan-only mode must **not** evaluate:

* retrieval recall;
* top-10 relevance;
* evidence grounding;
* paper roles based on retrieved papers;
* ranking quality;
* feedback ranking delta.

---

# 7. Full Retrieval Metrics

The following metrics require provider retrieval:

| Layer            | Metrics                                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Retrieval        | `provider_success_rate`, `raw_retrieved_count`, `merged_paper_count`, `retrieval_anchor_recall`, `dedup_sanity_ratio`, `missing_abstract_ratio` |
| Evidence         | `evidence_record_coverage`, `strict_support_rate`, `llm_invalid_evidence_rate`, `span_audit_completeness`                                       |
| Domain guardrail | `top10_domain_in_scope_rate`, `top10_out_of_scope_rate`, `top10_forbidden_pattern_rate`                                                         |
| Ranking          | `top10_anchor_coverage`, `aspect_coverage_at_10`, `precision_at_10`, `ndcg_at_10`, `recall_at_10`, `ranking_score_transparency`                 |
| Paper role       | `paper_role_assignment_rate`, `paper_role_diversity`                                                                                            |
| User adaptation  | `feedback_directional_accuracy`, `feedback_query_refinement_quality`, `feedback_overfit_failure_count`                                          |
| Report           | `paper_card_coverage`, `reading_path_presence`, `gap_matrix_specificity`, `suggested_next_search_specificity`, `uncertainty_visibility`         |

Full retrieval mode should be run only when provider API keys and network conditions are stable. Provider outages should be recorded as infrastructure failures rather than silently treated as model failures.

---

# 8. Rule-Mode vs LLM-Enhanced Mode

## 8.1 Rule-mode definition

A run is considered **rule-mode** when:

```text
planner_mode = rule
extractor_mode = rule
verifier_mode = rule
LLM backend inactive
```

Rule-mode is the deterministic baseline. It should be judged on:

* robustness;
* reproducibility;
* low hallucination risk;
* conservative evidence validation;
* transparent failure cases.

## 8.2 LLM-enhanced definition

A run is considered **LLM-enhanced** only when:

```text
LLM backend requested
LLM backend actually active
At least one of planner/extractor/verifier uses llm mode
```

If an LLM key is missing and the system falls back to rule-mode, the run must be labeled:

```text
llm_requested_but_inactive
```

Such a run must not be counted as LLM-enhanced.

## 8.3 Comparison rules

LLM-enhanced mode should be considered better only if it improves intent/query/report quality **without damaging grounding or domain precision**.

Recommended comparison metrics:

| Comparison                                                                         | Desired result   |
| ---------------------------------------------------------------------------------- | ---------------- |
| `llm_plan.intent_anchor_coverage - rule_plan.intent_anchor_coverage`               | `>= 0`           |
| `llm_plan.planned_query_anchor_coverage - rule_plan.planned_query_anchor_coverage` | `>= 0`           |
| `llm_full.top10_forbidden_pattern_rate - rule_full.top10_forbidden_pattern_rate`   | `<= 0`           |
| `llm_full.llm_invalid_evidence_rate`                                               | `<= 0.05`        |
| `llm_full.strict_support_rate - rule_full.strict_support_rate`                     | `>= -0.05`       |
| `llm_full.aspect_coverage_at_10 - rule_full.aspect_coverage_at_10`                 | `>= 0` preferred |
| `llm_full.report_section_completeness - rule_full.report_section_completeness`     | `>= 0`           |
| `llm_full.feedback_directional_accuracy - rule_full.feedback_directional_accuracy` | `>= 0`           |

A useful LLM-enhanced run may still be rejected if it improves recall but introduces unsupported evidence claims or domain drift.

## 8.4 LLM-specific failure checks

LLM-enhanced runs must additionally check:

| Failure                       | Detection                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Query hallucination           | Planned queries include invented materials, methods, or papers not implied by user intent                  |
| Evidence hallucination        | `llm_invalid_evidence_rate > 0.05`                                                                         |
| Over-repair                   | LLM narrows the domain beyond user intent                                                                  |
| False authority               | Report claims a paper is foundational without evidence, citation path, or clear reason                     |
| Unsupported citation relation | Report states that paper A cites paper B without `citation_expansion.csv` or `retrieval_paths.csv` support |
| English translation drift     | Chinese question translated into a different scientific domain                                             |

---

# 9. Benchmark Case Set

The benchmark currently contains eight cases:

| Case ID                               | Domain                    | Main ambiguity                                                         |
| ------------------------------------- | ------------------------- | ---------------------------------------------------------------------- |
| `sei_lithium_battery`                 | Battery materials         | `SEI` acronym ambiguity; interphase vs solid electrolyte               |
| `oer_spin_state`                      | Electrocatalysis          | `OER` as oxygen evolution vs open educational resources                |
| `ai_literature_screening`             | AI for evidence synthesis | Screening as literature screening vs medical/material screening        |
| `mof_co2_capture`                     | Porous materials          | CO2 capture vs catalysis/reduction/geological storage                  |
| `thin_film_deposition`                | Materials processing      | Deposition as materials growth vs legal/finance/geology                |
| `ferroelectric_surface_polarization`  | Ferroelectric thin films  | Polarization as ferroelectric vs optical/political polarization        |
| `surface_magnetization`               | Magnetic materials        | Surface magnetization vs Microsoft Surface/ferromagnetic nanoparticles |
| `perovskite_solar_defect_passivation` | Photovoltaics             | Halide perovskite solar cells vs oxide perovskites/catalysis           |

Each case defines:

```text
expected_query_anchors
forbidden_top10_patterns
expected_aspects
false_positive_types
optional feedback scenario
case-specific pass thresholds
```

The benchmark file should live at:

```text
data/benchmark_cases.yaml
```

---

# 10. Overall Pass/Fail Criteria

## 10.1 Case-level pass

A benchmark case passes if:

1. no hard failure occurs;
2. all required artifacts for the run type are present;
3. at least 80% of applicable metrics meet the pass standard;
4. `top10_forbidden_pattern_rate <= 0.10` in full retrieval;
5. `llm_invalid_evidence_rate <= 0.05` in LLM-enhanced full retrieval;
6. report and uncertainty fields are present.

## 10.2 Project-level pass

The project passes the evaluation suite if:

| Condition            | Requirement                                                                           |
| -------------------- | ------------------------------------------------------------------------------------- |
| Plan-only cases      | At least 8/8 pass in rule-mode                                                        |
| Full retrieval cases | At least 6/8 pass in rule-mode                                                        |
| LLM-enhanced plan    | At least 6/8 improve or match rule-mode on intent/query metrics                       |
| LLM-enhanced full    | At least 5/8 improve or match rule-mode without increasing hallucination/domain drift |
| Safety/grounding     | 0 cases with severe unsupported evidence presented as reliable                        |
| Domain guardrail     | 0 cases where a dominant forbidden meaning controls the top-10                        |

## 10.3 Failure severity

| Severity | Meaning                            | Example                                                               |
| -------- | ---------------------------------- | --------------------------------------------------------------------- |
| Minor    | Artifact exists but weak quality   | Query family names are generic                                        |
| Moderate | Useful run but clear weakness      | Missing one expected aspect                                           |
| Major    | Screening result unreliable        | Top-10 has several false positives                                    |
| Critical | System violates core research goal | Wrong domain, unsupported LLM evidence, or invisible provider failure |

Critical failures must be fixed before claiming improvement.

---

# 11. Implementation Notes for Evaluator

## 11.1 Text normalization

For anchor and forbidden-pattern matching:

```text
lowercase
remove repeated whitespace
normalize Unicode punctuation
treat hyphen and space variants as equivalent
match both title and abstract
match query strings, family purposes, and required aspects
```

## 11.2 Anchor group matching

An anchor group is satisfied if **any** term in its `any_of` list appears in the relevant artifact.

Example:

```yaml
- id: artificial_sei
  any_of:
    - artificial SEI
    - artificial solid electrolyte interphase
    - protective interphase
```

If one of these terms appears in the query plan, the group is counted as covered.

## 11.3 Forbidden top-10 matching

A top-10 paper is a likely false positive if:

1. its title or abstract matches a forbidden pattern; and
2. it does not match enough expected domain anchors; or
3. `domain_assessments.json` labels it `out_of_scope`.

Forbidden matching should be conservative. Do not penalize a paper simply because it contains a shared word such as `screening` or `surface`; penalize it when the context is clearly the wrong domain.

## 11.4 Gold labels

For more formal evaluation, create one file per benchmark case:

```text
data/gold_labels/{case_id}.csv
```

Recommended schema:

```csv
paper_id,title,label,reason
```

Allowed labels:

```text
include
maybe
exclude
```

Gold labels should be produced by human review of retrieved papers. They should not be generated solely by the system being evaluated.

## 11.5 Human review protocol

For each full retrieval case, a reviewer should inspect:

1. top-10 ranked papers;
2. bottom-10 excluded papers;
3. all papers marked `borderline`;
4. all papers with `weak_support`, `unverified`, or `llm_invalid_evidence`;
5. before/after feedback ranking changes.

The reviewer should answer:

```text
Would a novice user be helped or misled by this output?
Are the top papers central to the repaired intent?
Are evidence claims traceable to abstract spans?
Are off-domain papers clearly demoted or labeled?
Does the report explain what to read next?
```

---

# 12. Recommended Ablations

Use the same benchmark cases for ablation tests.

| Ablation              | Disabled module                          | Expected degradation                                       |
| --------------------- | ---------------------------------------- | ---------------------------------------------------------- |
| `no_intent_repair`    | novice intent repair                     | lower intent anchor coverage; more acronym false positives |
| `no_query_families`   | QueryFamily planning                     | lower aspect coverage and lens diversity                   |
| `single_provider`     | one provider only                        | lower retrieval recall and source diversity                |
| `no_domain_guardrail` | domain assessment/demotion               | higher top-10 forbidden pattern rate                       |
| `no_span_validation`  | strict evidence verification             | higher unsupported evidence rate                           |
| `no_paper_roles`      | role classifier                          | weaker reading path and report usability                   |
| `no_feedback`         | feedback adaptation                      | no before/after improvement                                |
| `llm_no_verifier`     | LLM evidence extraction without verifier | higher hallucination risk                                  |

Ablation success criterion:

```text
The full system should outperform each ablated version on the metric targeted by the removed module, without causing new critical failures.
```
