# Baseline Audit: search_paper

Date: 2026-07-07

Scope: baseline audit only. No production code was changed. This document summarizes the current pipeline, query-planning behavior, data flow, tests, and safe insertion points for later small-step changes.

## 1. Current Module Responsibilities

### Intent, Contract, And Query Planning

- `lit_screening/agents/research_intent.py`
  - Builds `SearchBrief` from a user question.
  - Rule mode detects broad user intent such as `overview`, `frontier`, `implementation`, `evidence_verification`, `proposal`, and `systematic_review`.
  - For Chinese input, it uses `fallback_translate_chinese_question()` from `planner.py` to create an English `refined_question`.
  - Optional LLM mode can enrich the brief, but invalid or unavailable LLM output falls back to rule mode.

- `lit_screening/agents/search_contract.py`
  - Builds a `SearchContract` after intent analysis and ambiguity detection.
  - Infers coarse `DomainProfile` values such as `ai_literature_screening`, `materials_magnetism`, `biomedical_screening`, and `general_science`.
  - Converts domain boundaries into `must_include_concepts`, `must_exclude_concepts`, `required_aspects`, field whitelists/blacklists, and terminology maps.
  - The current materials-magnetism profile recognizes terms such as `surface magnetization`, `antiferromagnetism`, `surface spin`, `magnetization`, and excludes LLM / clinical-screening drift.

- `lit_screening/agents/planner.py`
  - Builds `QueryPlan`, including `core_terms`, `must_terms`, `optional_terms`, `exclude_terms`, `required_aspects`, `openalex_queries`, `semantic_scholar_queries`, and `filters`.
  - Provides rule-based Chinese glossary translation and optional LLM-enhanced query planning.
  - Builds provider-specific queries through `build_openalex_queries()` and `build_semantic_scholar_queries()`.
  - Avoids injecting LLM / human-feedback / literature-screening terms when the user question is a plain science or materials question.
  - Current limits: the rule planner extracts local n-grams and known phrases, but it does not yet model cited papers, authors, seed-paper intent, measurement lens, or domain-specific query families.

- `lit_screening/agents/query_pilot.py`
  - Runs low-volume retrieval before full retrieval.
  - Uses `DomainGuardrailAgent` on pilot results to estimate off-topic rate.
  - Emits drift categories such as healthcare screening drift, biological agent drift, materials screening drift, or LLM-agent drift.
  - Returns keep / repair / drop recommendations per provider query.

- `lit_screening/agents/query_repair.py`
  - Converts pilot diagnostics plus `SearchContract` constraints into repaired provider queries.
  - Replaces ambiguous terms like `screening` with `literature screening` when appropriate.
  - Adds provider-specific required phrases and exclusions.
  - Returns suggestions and a repaired `QueryPlan`; application is controlled by pipeline/UI settings.

### Evidence, Verification, Ranking

- `lit_screening/agents/extractor.py`
  - Extracts one abstract-grounded evidence sentence per paper.
  - Rule mode selects the abstract sentence with highest keyword overlap with the planning question.
  - LLM mode is optional and must return JSON; missing or unsafe LLM output falls back to rule extraction.
  - Missing abstracts produce empty claim/evidence records rather than invented evidence.

- `lit_screening/agents/verifier.py`
  - Validates whether extracted evidence is grounded in the paper abstract.
  - Strict support requires exact or high-confidence fuzzy span validation through `span_validation.py`.
  - Keyword overlap without a valid span is downgraded to `weak_support`; LLM evidence that cannot match the abstract becomes `llm_invalid_evidence`.
  - This is one of the most important correctness gates and should remain conservative.

- `lit_screening/agents/ranker.py`
  - Joins papers with evidence, verification, aspect coverage, and domain assessments.
  - Calls `compute_score_breakdown()` in `lit_screening/scoring.py`.
  - Applies two-pass ranking: first with neutral diversity, then with venue-diversity adjustment.
  - Domain penalties and aspect coverage are applied through scoring inputs, not hidden in the UI.

### Pipeline, Report, Export, Sensemaking

- `lit_screening/pipeline.py`
  - Main orchestration entrypoint for CLI and UI.
  - Builds `SearchBrief`, `question_refinement`, `ambiguity_analysis`, `SearchContract`, and `QueryPlan`.
  - Optionally runs pilot search and query repair.
  - Retrieves provider metadata, imports external libraries, filters by year, deduplicates, extracts evidence, verifies spans, computes aspect coverage, applies domain guardrails, ranks, optionally snowballs from seed papers, applies feedback, computes metrics, and writes artifacts.
  - Also contains backward-compatible helpers for UI edited query plans and user-confirmed query overrides.

- `lit_screening/report.py`
  - Generates `report.md` as a Markdown decision report.
  - Summarizes question preprocessing, user intent, Search Contract, ambiguity handling, query strategy, planned queries, domain guardrails, pilot diagnostics, PRISMA-like flow, reading path, ranked decisions, decision artifacts, preference learning, seed expansion, limitations, and trace summary.

- `lit_screening/decision_artifacts.py`
  - Note: this file is at `lit_screening/decision_artifacts.py`, not `lit_screening/agents/decision_artifacts.py`.
  - Writes `method_comparison_matrix.csv/md`, `research_gap_matrix.csv/md`, and `suggested_next_searches.json/md`.
  - Uses rule-based inference from ranked paper text, aspect coverage, Search Contract, query plan, pilot diagnostics, and PRISMA-like flow.

- `lit_screening/paper_cards.py`
  - Writes `paper_cards.md` with top-paper evidence cards, decision, reading priority, domain decision, claim, evidence sentence, verification result, and suggested action.

- `lit_screening/reading_path.py`
  - Writes `reading_path.md`, grouped by overview/background, method papers, frontier papers, evaluation papers, and optional peripheral papers.

- `lit_screening/result_groups.py`
  - Groups ranked papers into `must_read`, `recent_frontier`, `implementation_relevant`, `evaluation_relevant`, `background_or_survey`, `peripheral`, and `excluded_or_low_confidence`.

- `lit_screening/screening_flow.py`
  - Builds a PRISMA-like count summary from retrieval counts, duplicate counts, missing abstracts, verification results, ranked papers, and screening decisions.

### UI Entry

- There is no `streamlit_app.py` in the current tree. The Streamlit entrypoint is `app.py`.
- `app.py` is a thin but large UI layer:
  - sidebar settings for providers, year filter, cache, LLM backend, query mode, ranking profile, seed-paper mode, API keys, agent modes, and scoring weights;
  - query preview checkpoint using `plan_screening_queries()`;
  - editable `SearchBrief` and `QueryPlan`;
  - optional `run_query_pilot_workflow()`;
  - full screening through `run_pipeline()`;
  - feedback reranking through `apply_feedback_to_pipeline_result()`;
  - project history and artifact rendering.
- The UI does not duplicate extraction, verification, ranking, import, report, or core scoring logic.

## 2. Current Data Flow

High-level flow:

```text
user question
  -> ResearchIntentAgent.analyze()
  -> SearchBrief
  -> QuestionRefinementAgent.refine()
  -> AmbiguityDetectorAgent.analyze()
  -> SearchContractAgent.build()
  -> SearchContract / DomainProfile
  -> PlannerAgent.plan_structured()
  -> QueryPlan
  -> provider-specific queries
  -> optional QueryPilotAgent / QueryRepairAgent
  -> RetrieverAgent.retrieve()
  -> raw provider bundles + Paper objects
  -> optional imported library papers
  -> local year filter
  -> deduplicate_with_stats()
  -> merged Paper list
  -> ExtractorAgent.extract_many()
  -> EvidenceRecord list
  -> VerifierAgent.verify_many()
  -> VerificationResult list
  -> AspectCoverageAgent.classify_many()
  -> DomainGuardrailAgent.assess_many()
  -> RankerAgent.rank()
  -> RankedPaper list
  -> optional CitationSnowballAgent expansion and rerank
  -> ScreeningDecisionAgent.decide_many()
  -> optional HumanFeedbackAgent / PreferenceLearningAgent rerank
  -> evaluation metrics, trace, CSV/JSON/Markdown outputs, report
```

The main in-memory objects are dataclasses in `lit_screening/models.py`:

- `SearchBrief`: user intent and inclusion/exclusion logic.
- `DomainProfile` / `SearchContract`: domain boundary and retrieval contract.
- `QueryPlan`: provider-aware planning artifact.
- `Paper`: normalized provider/import metadata plus provenance.
- `EvidenceRecord` and `VerificationResult`: evidence and grounding audit.
- `AspectCoverageRecord`, `DomainAssessment`, `ScoreBreakdown`, `ScreeningDecision`, `RankedPaper`: ranking and decision surface.
- `PipelineResult`: the object returned to CLI/UI and feedback reranking.

## 3. Most Relevant Existing Tests

- Planner:
  - `tests/test_agents.py`
    - Checks no unrelated LLM terms are injected into materials questions.
    - Checks structured planner behavior, provider query builders, SearchContract use, and rule glossary translation.
  - `tests/test_llm_agents.py`
    - Checks mocked LLM planner translation/query behavior and fallback expectations.
  - `tests/test_sensemaking.py`
    - Checks planner use of `SearchBrief` inclusion/exclusion terms.

- Search Contract:
  - `tests/test_agents.py`
    - Checks materials-magnetism contract and AI literature-screening contract.
  - `tests/test_domain_guardrail.py`
    - Checks SearchContract-driven domain assessment and score penalties.
  - `tests/test_pipeline.py`
    - Checks `search_contract.json`, ambiguity outputs, and report sections.

- Query Pilot / Query Repair:
  - `tests/test_query_pilot_repair.py`
    - Checks fake pilot retrieval, drift detection, repair suggestions, and pipeline-written pilot/repair outputs.

- Pipeline:
  - `tests/test_pipeline.py`
    - Most important integration suite for output compatibility, year filtering, imports, Chinese planning question, query overrides, progress events, retrieval diagnostics, OpenAlex mode stages, error writing, and sensemaking outputs.

- Report / Export:
  - `tests/test_pipeline.py`
    - Verifies report sections and required output files.
  - `tests/test_sensemaking.py`
    - Verifies reading path and PRISMA-like flow helpers.
  - There is no dedicated `tests/test_report.py` at this baseline.

- Snowball:
  - `tests/test_snowball.py`
    - Checks seed parsing, snowball disabled-by-default behavior, fake Semantic Scholar expansion, and pipeline outputs/provenance.

- Sensemaking:
  - `tests/test_sensemaking.py`
    - Checks research intent modes, aspect coverage, result grouping, reading path generation, and PRISMA-like flow counts.

- Scoring:
  - `tests/test_scoring_feedback.py`
    - Checks formula, custom weights, feedback adjustment, hybrid relevance behavior, evidence score, and ranking profile behavior.
  - `tests/test_preference_learning.py`
    - Checks preference learning and feedback-derived scoring/refinement behavior.

- Verifier:
  - `tests/test_agents.py`
    - Checks missing evidence, strict span support, weak-support downgrade, and LLM-invalid evidence behavior.
  - `tests/test_llm_agents.py`
    - Checks LLM verifier fallback and invalid-output flags.

## 4. Planner-Only Dry Run: Materials Magnetism Question

Question used:

```text
有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets 和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order 相关的探测表面磁化和自旋极化重要性的文章
```

Command shape:

```bash
conda run -n a-share-monitor python -c 'from lit_screening.pipeline import plan_screening_queries; ...'
```

No external provider API was called. The check used `plan_screening_queries()` with `llm_backend="none"` and `planner_mode="rule"`.

Observed baseline:

- `planning_question`:
  - `surface magnetization spin importance nicola a spaldin surface magnetization in antiferromagnets local magnetoelectric effects as predictors of surface magnetic order`
- `translation_mode`:
  - `rule_glossary`
- `translation_warning`:
  - `rule_glossary_translation_is_approximate`
- `SearchBrief.search_intent`:
  - `overview`
- `SearchContract.domain_profile.domain_name`:
  - `materials_magnetism`
- `must_include_concepts`:
  - `surface magnetization`
  - `magnetization`
  - `antiferromagnetism`
- `must_exclude_concepts`:
  - `LLM agent`
  - `large language model`
  - `human feedback`
  - `literature screening`
  - `patient screening`
  - `drug screening`
- Representative `core_terms`:
  - `surface magnetization`
  - `magnetization`
  - `antiferromagnetism`
  - `surface magnetization spin`
  - `nicola spaldin surface`
  - `spaldin surface magnetization`
  - `antiferromagnets local magnetoelectric`
  - `surface magnetic order`
- Representative provider queries:
  - OpenAlex includes broad and focused forms like `surface magnetization`, `surface magnetization magnetization antiferromagnetism`, and `"surface magnetization" AND magnetization AND antiferromagnetism`, with negative filters for LLM/clinical drift.
  - Semantic Scholar includes `+"surface magnetization" +magnetization +antiferromagnetism ...` forms and negative filters.

## 5. Current Query Planning Strengths

- The rule planner is provider-aware and creates separate OpenAlex and Semantic Scholar query lists.
- Chinese input is converted into an English planning question without requiring an LLM key.
- The Search Contract correctly identifies this dry-run as `materials_magnetism`, not AI literature screening.
- The materials question avoids unrelated LLM, human-feedback, and literature-screening query injection.
- Multi-word terms such as `surface magnetization` are quoted or required in provider-specific ways.
- OpenAlex `keyword+semantic` mode is represented as a downstream retrieval-stage setting.
- The pipeline preserves an editable checkpoint: UI users can inspect `SearchBrief`, terms, and provider queries before retrieval.
- Query pilot and repair are already separated from planning, which is good for auditable intervention.

## 6. Query Planning Gaps For Materials Magnetism

- The planner treats named reference papers as ordinary text fragments rather than as seed-paper anchors or related-work constraints.
  - `Surface Magnetization in Antiferromagnets`
  - `Local Magnetoelectric Effects as Predictors of Surface Magnetic Order`
- Author names are tokenized into weak phrase fragments such as `nicola spaldin surface` and `spaldin surface magnetization`.
- The planner captures `surface magnetization` and `antiferromagnetism`, but it does not explicitly add materials-magnetism lens terms such as:
  - `surface magnetic order`
  - `surface spin polarization`
  - `local magnetoelectric effect`
  - `boundary magnetization`
  - `probe`
  - `detection`
  - `spin-polarized surface states`
  - `magnetoelectric response`
- The Chinese word `探测` is not currently in the glossary, so the detection/probe intent is lost.
- `自旋极化` is not currently translated as `spin polarization`; only the broader `spin` term is captured.
- The query set mixes review/background terms with a user intent that may actually be "find related papers for a PRL proof chain"; this suggests a need for a domain-specific research lens rather than only generic overview/frontier/proposal modes.
- Required aspects are generic (`surface`, `magnetization`, `spin`, `background`, `significance`) and do not yet reflect the proof/evidence axes a materials-magnetism user would expect.

## 7. Best Insertion Points For ResearchLens / QueryFamily / DomainPack

Recommended minimal insertion order:

1. `DomainPack`
   - Best location: near `SearchContractAgent` and `infer_domain_profile()`.
   - Purpose: move hard-coded domain profiles and terminology maps into a small, testable domain package.
   - For materials magnetism, this should own synonyms, exclusions, preferred fields/venues, author/title handling rules, and required aspect templates.
   - It should feed `SearchContract`, not replace it.

2. `ResearchLens`
   - Best location: between `SearchBrief` and `QueryPlan`, probably as an added field or companion object produced after `SearchContract`.
   - Purpose: interpret the user's task shape within a domain.
   - Example lenses for this project:
     - `background_overview`
     - `related_to_seed_papers`
     - `experimental_detection`
     - `surface_spin_polarization`
     - `proof_chain_for_surface_magnetization`
   - It should influence required aspects, optional terms, and ranking emphasis without changing retrieval clients.

3. `QueryFamily`
   - Best location: inside or directly beside `PlannerAgent.plan_structured()`, after terms are extracted and before provider-specific query builders run.
   - Purpose: create multiple intent-specific query families rather than a single mixed list.
   - Possible families for the dry-run:
     - seed-paper related queries
     - surface magnetization mechanism queries
     - experimental detection/probe queries
     - spin polarization / surface order queries
     - review/background queries
   - It should compile into the existing `QueryPlan.openalex_queries` and `QueryPlan.semantic_scholar_queries` to preserve backward compatibility.

Important compatibility point: do not replace `QueryPlan` immediately. Add fields only if needed, and keep existing `openalex_queries`, `semantic_scholar_queries`, `core_terms`, `must_terms`, `optional_terms`, and `exclude_terms` stable for CLI/UI/tests.

## 8. Stable Modules Not Recommended For Early Refactor

- `lit_screening/agents/extractor.py`
  - Stable evidence extraction fallback and no-hallucination behavior.
  - Only extend after query/domain changes are proven.

- `lit_screening/agents/verifier.py` and `lit_screening/span_validation.py`
  - Critical correctness gate. Avoid loosening strict span validation.

- `lit_screening/scoring.py` and `lit_screening/agents/ranker.py`
  - The scoring/ranking surface is already covered by tests and report outputs.
  - Future lens/ranking changes should pass through existing parameters first.

- `lit_screening/retrieval/*`
  - Provider clients and retrieval provenance are modular.
  - Query-planning changes should not require retrieval client rewrites.

- `lit_screening/importers/*`
  - Existing library import is modular and dependency-light.
  - Not related to the immediate materials query-planning gap.

- `app.py`
  - Large but mostly thin UI wrapper.
  - Avoid moving business logic into it; expose future ResearchLens/QueryFamily only after the core pipeline supports them.

- Output writer compatibility in `lit_screening/pipeline.py`
  - Existing CSV/JSON/Markdown artifact names are part of the user-facing contract.
  - Prefer additive fields/artifacts over renames or removals.

## 9. Suggested Next Small-Step Refactor Sequence

Do not implement these during this baseline audit. Suggested sequence for later prompts:

1. Add focused tests that capture the dry-run materials-magnetism failure:
   - `探测` should create `detection/probe` terms.
   - `自旋极化` should create `spin polarization`.
   - Spaldin paper titles should be treated as seed/related-paper signals, not noisy n-grams.

2. Add a tiny materials-magnetism terminology expansion in one place.
   - Lowest-risk first step: extend the Chinese glossary and known materials phrases.
   - Avoid introducing new abstractions until tests show the minimal change is insufficient.

3. Introduce a lightweight `DomainPack` only if terminology starts spreading across modules.

4. Introduce `QueryFamily` after the planner needs to distinguish seed-related, detection, mechanism, and review queries.

5. Introduce `ResearchLens` only when user intent requires a separate domain-level lens beyond current `search_intent`.

## 10. Audit Risks And Notes

- This audit did not call OpenAlex or Semantic Scholar.
- The dry-run is based on rule mode only; LLM mode could produce better translations but must remain optional.
- There is no dedicated report test file; report coverage is mostly via pipeline integration tests.
- `app.py` is functionally thin but large; future UI changes should be conservative and core-first.
- `decision_artifacts.py` is not under `lit_screening/agents/`, despite the requested path name.
