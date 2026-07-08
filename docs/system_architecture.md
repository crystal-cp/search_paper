# System Architecture

## Scope

This document describes the current `search_paper` pipeline as of the v9
deterministic baseline. It is documentation only. It does not define a new
pipeline, provider, ranker, UI framework, vector database, PDF parser, domain
pack, or mandatory LLM dependency.

The system is a human-centric multi-agent LLM literature-screening prototype,
but the current baseline is deterministic and rule-controlled. The word
"agent" means a bounded pipeline role with explicit inputs, outputs, and
auditable artifacts. It does not mean every stage is an autonomous LLM agent.

Core boundaries:

- The default path runs without an LLM key.
- QueryFamily planning is enabled by default in the v9 deterministic baseline.
- LLMs are optional controlled enhancements and must not directly decide final screening labels, evidence validity, reading priority, domain decisions, or scores.
- OpenAlex and Semantic Scholar retrieval remain modular.
- Snowballing remains optional and disabled unless requested.
- The Streamlit UI remains a thin wrapper over core pipeline functions.

## Current Agent Flow

```text
User question
  -> NoviceIntentInterpreter / Intent Repair
  -> ExpertResearchIntent / structured concepts
  -> optional LLMIntentFrameEnhancer
  -> deterministic verifier
  -> DomainRouter
  -> SearchContract
  -> QueryFamilyPlanner
  -> QueryRepair / QueryCritic diagnostics
  -> OpenAlex / Semantic Scholar retrieval
  -> evidence extraction
  -> span validation
  -> DomainGuardrail
  -> PaperRoleClassifier
  -> intent_centrality ranking
  -> context-aware reading_priority
  -> reading_path / user_report
  -> user feedback
```

The orchestration entrypoint is `lit_screening/pipeline.py`. The same core path
is used by the CLI and Streamlit UI.

## Stage Details And Artifacts

### 1. User Question To Novice Intent

The raw user question is treated as noisy evidence of intent, not as a final
search query. Intent repair converts novice phrasing, multilingual wording, and
implicit domain context into structured search intent.

Representative modules:

- `lit_screening/agents/research_intent.py`
- `lit_screening/agents/intent_repair.py`
- `lit_screening/agents/question_refiner.py`

Artifacts:

- `search_brief.json`
- `question_refinement.json`
- `ambiguity_analysis.json`

### 2. Expert Research Intent And Domain Routing

The system maps repaired intent into expert concepts, domain hints, required
concept groups, and exclusion boundaries. Domain packs stay lightweight and
should describe terms and guardrails rather than hide business logic.

Representative modules:

- `lit_screening/agents/concept_mapper.py`
- `lit_screening/domain_packs/`

Artifacts:

- `concept_map.json`
- `search_contract.json`

### Optional LLMIntentFrameEnhancer Phase 1

`LLMIntentFrameEnhancer` is an optional advisory layer between structured
concept interpretation and SearchContract finalization. The v9 baseline remains
deterministic: when the enhancer is disabled, no LLM is called and the
deterministic intent frame and SearchContract remain unchanged.

When explicitly enabled, the enhancer may propose intent-frame suggestions such
as normalized topic wording, target-context candidates, negative-context
candidates, aliases, abbreviations, method needs, mechanism needs, application
needs, and ambiguity notes. A deterministic verifier checks whether each
suggestion is grounded in the user question, strongly supported by the
deterministic intent, or a safe alias/normalization. Accepted suggestions can be
added to the SearchContract with provenance
`source = llm_suggested_rule_verified`. Rejected suggestions are recorded for
auditability but are not used as active contract constraints.

The LLM enhancer is advisory only. It must not directly decide paper-level
labels, evidence validity, reading priority, domain decisions, or scores.

Optional artifacts:

- `intent_frame_before_llm.json`
- `llm_intent_frame_raw.json`
- `llm_intent_frame_verified.json`
- `search_contract_before_llm.json`
- `search_contract_after_llm.json`
- `llm_intent_enhancement_trace.json`

### 3. Search Contract

The SearchContract records what the run is trying to retrieve and what it should
avoid. It may include required groups, target context groups, target chemistry
groups, negative context groups, ambiguous terms, and provider/query budget
constraints.

For example, a lithium-specific SEI question should create target lithium
context and negative non-target battery chemistry context. OER spin-state tasks
do not require a lithium-style target context for `must_read` eligibility.

Artifact:

- `search_contract.json`

### 4. QueryFamilyPlanner

QueryFamilyPlanner decomposes the repaired intent into multiple query routes.
In the v9 deterministic baseline, QueryFamily is part of the default planning
path. Ablation flags can disable it for debug/pilot evaluation, but the default
baseline is QueryFamily-on.

Artifacts:

- `query_families.json`
- `query_provenance.json`
- `planned_queries.json`

### 5. QueryRepair And QueryCritic Diagnostics

Query repair and critic artifacts make query quality auditable. They record raw
candidate queries, repaired/final queries, whether repair was enabled, whether
it changed final queries, and whether a no-query-repair ablation is conclusive.

Artifacts:

- `raw_candidate_queries_before_repair.json`
- `final_queries_after_repair.json`
- `query_repair_stage_status.json`
- `query_repair_suggestions.json`
- `llm_query_critic.json`, when available

### 6. Retrieval And Import

Retrieval is provider-modular. OpenAlex and Semantic Scholar clients normalize
provider metadata into shared `Paper` objects. Local BibTeX, RIS, and CSV import
records pass through the same downstream screening path.

Representative modules:

- `lit_screening/agents/retriever.py`
- `lit_screening/retrieval/openalex_client.py`
- `lit_screening/retrieval/semantic_scholar_client.py`
- `lit_screening/importers/`
- `lit_screening/agents/snowball.py`, only when snowballing is enabled

Artifacts:

- `raw_openalex_results.json`
- `raw_semantic_scholar_results.json`
- `retrieval_diagnostics.json`
- `provider_status.json`
- `merged_papers.csv`
- `imported_papers.csv`, when import is used
- `seed_papers.json`, `citation_expansion.csv`, and `retrieval_paths.csv`, when seed expansion is used

### 7. Evidence Extraction And Span Validation

Evidence extraction creates candidate claims from title and abstract. Span
validation checks whether evidence is exactly or fuzzily grounded in the
abstract. Weak keyword overlap must not be treated as strict support.

Representative modules:

- `lit_screening/agents/extractor.py`
- `lit_screening/agents/verifier.py`
- `lit_screening/span_validation.py`
- `lit_screening/agents/evidence_function_classifier.py`

Artifacts:

- `evidence_table.csv`
- `evidence_functions.json`

Important fields:

- `support_level`
- `matched_text`
- `span_match_type`
- `span_match_confidence`
- `strict_span_validated`
- `missing_abstract`

### 8. DomainGuardrail

DomainGuardrail uses the search contract to decide whether a paper is in scope,
borderline/peripheral, or out of scope. It also records target context matches,
negative context matches, and peripheral context reasons.

For lithium-specific SEI tasks, sodium-ion, potassium-ion, zinc-ion, magnesium,
AZIB, and broad beyond-lithium papers should not become `include` or `must_read`
unless explicitly allowed by the benchmark/task. Mixed Li/Na/K reviews may be
kept as peripheral background, but not as first-read core papers.

Artifact:

- `domain_assessments.json`

Diagnostics in `ranked_papers.csv` include:

- `target_context_match`
- `negative_context_match`
- `peripheral_context_reason`
- `topic_focus_score`
- `intent_centrality_score`
- `required_group_coverage_score`
- `missing_required_group_count`

### 9. PaperRoleClassifier And Research Sensemaking

PaperRoleClassifier and sensemaking writers explain how papers function in the
reading landscape. Roles are useful for reports and reading paths, but they must
not bypass screening decisions. Excluded or out-of-scope papers must never appear
in recommended reading sections just because they have a role.

Representative modules:

- `lit_screening/agents/paper_role_classifier.py`
- `lit_screening/agents/controversy_boundary.py`
- `lit_screening/paper_cards.py`
- `lit_screening/reading_path.py`

Artifacts:

- `paper_roles.json`
- `research_tensions.json`
- `paper_cards.md`
- `reading_path.md`

### 10. Intent-Centrality Ranking

Ranking organizes the candidate set by usefulness for the repaired intent. It
combines provider/relevance signals, evidence support, recency, quality,
diversity, group coverage, intent centrality, and optional feedback.

Representative modules:

- `lit_screening/agents/ranker.py`
- `lit_screening/scoring.py`
- `lit_screening/reranking.py`
- `lit_screening/agents/screening_decision.py`

Artifacts:

- `ranked_papers_before_feedback.csv`
- `ranked_papers_after_feedback.csv`, when feedback is provided
- `ranked_papers.csv`
- `screening_decisions.csv`
- `screening_decisions.json`

### 11. Context-Aware Reading Priority

Reading priority is not the same as inclusion. `include` means the paper is
relevant enough to keep. `must_read` means it belongs in the first reading set.

For tasks with explicit target context groups, such as lithium-specific SEI,
`must_read` requires a target context match and no strong negative context. For
tasks without explicit target context groups, such as OER spin state, strong
in-scope papers with full group coverage, high intent centrality, strict support,
and high topic focus can become `must_read` without a target-context field.

Diagnostics:

- `must_read_count`
- `target_context_required_for_priority`
- `must_read_policy_reason`

### 12. Reading Path, User Report, And Feedback

Reading path and user report generation must filter through screening decisions.
Excluded, out-of-scope, reading-priority-exclude, duplicate, or negative-context
papers must not appear in recommended reading sections.

Artifacts:

- `reading_path.md`
- `report.md`
- `user_report.md`, when generated
- `evaluation.json`
- `exploration_quality.json`
- `agent_trace.json`
- `run_events.jsonl`

Reading-path diagnostics:

- `reading_path_paper_count`
- `reading_path_exclude_count`
- `reading_path_out_of_scope_count`
- `reading_path_duplicate_count`
- `reading_path_negative_context_count`

Human feedback can be imported and applied after ranking. Feedback should adjust
preferences and future directions without hiding the original retrieval and
screening trace.

Artifacts:

- `preference_learning.json`
- `feedback_query_refinement.json`

## Plan-Only Behavior

Plan-only runs are first-class evaluation artifacts. They should not pretend
retrieval happened.

Expected plan-only diagnostics:

- `retrieval_status = planning_only`
- `ranked_papers_based_on_real_retrieval = false`
- `research_gap_matrix` contains a skipped/status row rather than generated gaps
- `suggested_next_searches = []` when no real screened corpus exists

## UI Boundary

`app.py` is a thin Streamlit wrapper. It should call `plan_screening_queries()`,
`run_pipeline()`, and feedback helpers from the core package. It should not
duplicate planner, retrieval, scoring, verification, import, evaluation, or
report logic.

## Extension Boundaries

Future work may add stronger benchmarks, controlled LLM enhancement layers,
additional providers, or improved role taxonomy. Those extensions should remain
artifact-first, optional when risky, and testable without real API keys.
