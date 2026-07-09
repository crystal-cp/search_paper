# Human-centric Multi-agent LLM System for Evidence-grounded Scientific Literature Screening

[中文说明](README.zh-CN.md)

This repository is a lightweight, reproducible research prototype for
human-centric, evidence-grounded scientific literature screening under imperfect
research intent. It studies how a multi-agent AI system can turn incomplete,
ambiguous, multilingual, or partially incorrect novice research questions into
auditable screening decisions, ranked papers, evidence spans, reading roles, and
next-search suggestions.

The project is not a generic keyword-search wrapper, a paper summarizer, or a
commercial literature-management product. Its research focus is novice intent
repair, provider-aware query planning, abstract-grounded evidence validation,
domain guardrails, role-aware ranking, human feedback, and transparent reporting.
The current frozen baseline is **v9 deterministic baseline**: a rule-controlled,
reproducible pipeline where QueryFamily planning is enabled by default and LLM
behavior is not required.

LLM is a controlled enhancement layer, not the decision maker. LLM-enhanced
behavior is optional or planned for controlled evaluation; the rule-based core
must continue to run without an LLM key. An LLM must not directly decide
`include` / `exclude`, `must_read`, `out_of_scope`, `final_score`,
`domain_decision`, or evidence validity. Those decisions are owned by explicit
rules, span validation, domain guardrails, ranking diagnostics, and auditable
artifacts.

For the detailed research framing and current system design, see
[`docs/research_problem.md`](docs/research_problem.md) and
[`docs/system_architecture.md`](docs/system_architecture.md).
For the plan-level LLM pilot diagnostic, see
[`docs/llm_plan_level_pilot.md`](docs/llm_plan_level_pilot.md); it is not a
formal full-retrieval LLM ablation conclusion.
For the small full-retrieval LLM safety pilot, see
[`docs/llm_full_retrieval_pilot.md`](docs/llm_full_retrieval_pilot.md); it is a
safety check, not a formal performance-ablation conclusion.

The v9 deterministic baseline pipeline:

1. interprets the user's search intent as a `SearchBrief`,
2. refines broad questions into optional subquestions,
3. builds QueryFamily routes and a structured provider-aware query plan,
4. retrieves metadata from OpenAlex and Semantic Scholar using provider-specific queries,
5. optionally imports existing BibTeX, RIS, or CSV literature-library exports,
6. optionally expands from seed papers through references, citations, and recommendations,
7. normalizes and deduplicates papers,
8. extracts claim-level evidence from abstracts,
9. verifies whether evidence is grounded in the abstract with strict span validation,
10. ranks papers with hybrid TF-IDF/API/field relevance, aspect coverage, intent centrality, and transparent scoring,
11. groups papers into reading roles and generates paper evidence cards,
12. assigns context-aware reading priorities and reading paths,
13. optionally applies human feedback,
14. writes CSV, JSON, Markdown outputs, and an agent decision trace.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the optional Streamlit UI:

```bash
pip install -r requirements-ui.txt
```

API keys can be supplied through environment variables:

```bash
cp .env.example .env
```

The package reads environment variables from the process environment. If you use
`.env`, load it with your shell or preferred environment manager before running
the CLI.

The rule-based core pipeline works without `DEEPSEEK_API_KEY`. The current
OpenAlex API requires a free `OPENALEX_API_KEY`; free keys have a daily usage
budget. Semantic Scholar can be queried without `S2_API_KEY`, but a key is
recommended to reduce rate-limit failures.

## Optional LLMIntentFrameEnhancer

`LLMIntentFrameEnhancer` is an optional controlled enhancement for intent
understanding. It is disabled by default, so the v9 deterministic baseline does
not require an LLM key and does not call an LLM unless the user explicitly
enables this phase.

When enabled, the enhancer can only propose intent-frame suggestions such as
normalized topic wording, target-context candidates, negative-context candidates,
aliases, abbreviations, method needs, mechanism needs, application needs, or
ambiguity notes. A deterministic verifier decides which suggestions, if any,
are allowed to enter the `SearchContract`. Rejected LLM suggestions are recorded
for auditability but are not used as active contract constraints.

The enhancer writes auditable artifacts when requested:

- `intent_frame_before_llm.json`
- `llm_intent_frame_raw.json`
- `llm_intent_frame_verified.json`
- `search_contract_before_llm.json`
- `search_contract_after_llm.json`
- `llm_intent_enhancement_trace.json`

The LLMIntentFrameEnhancer must not directly decide `include` / `exclude`,
`must_read`, `out_of_scope`, `domain_decision`, `final_score`, evidence
validity, or reading priority. Those remain deterministic, rule-controlled,
paper-level decisions.

## Optional DeepSeek LLM Enhancement

The default v9 baseline is deterministic and rule-controlled. You can optionally
use an OpenAI-compatible LLM backend for controlled query-planning, evidence
extraction, and/or verification experiments. DeepSeek support is built in
through `DEEPSEEK_API_KEY`.

LLM output is advisory. It can propose translations, query ideas, extraction
candidates, or critic diagnostics, but it must not directly decide
`include` / `exclude`, `must_read`, `out_of_scope`, `final_score`,
`domain_decision`, or evidence validity.
When the research question is written in Chinese, the planner prepares an
English `planning_question` before retrieval. With `--planner-mode llm`, the
LLM is asked to translate the question and generate English scholarly queries.
Without an available LLM key, the rule-based planner falls back to a small
scientific glossary so the pipeline still runs.

```bash
export DEEPSEEK_API_KEY="your-key"

python -m lit_screening.pipeline run \
  --question "How can human-in-the-loop multi-agent LLM systems improve scientific literature screening and evidence verification?" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --from-year 2020 \
  --llm-backend deepseek \
  --planner-mode llm \
  --extractor-mode llm \
  --verifier-mode llm \
  --output-dir outputs
```

You can also screen an existing literature library exported from Zotero, Web of
Science, Scopus, Google Scholar, or a curated spreadsheet:

```bash
python -m lit_screening.pipeline run \
  --question "surface magnetization boundary spin signals" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --input-file examples/my_library.bib \
  --input-format auto \
  --output-dir outputs
```

Supported import formats are BibTeX (`.bib` / `.bibtex`), RIS (`.ris`), and CSV
(`.csv`). CSV columns can include `title`, `abstract`, `authors`, `year`,
`venue`, `doi`, `url`, and `citation_count`.

You can also start from known seed papers and optionally expand through
Semantic Scholar references, citations, and recommendations. Snowballing is off
by default and requires `--enable-snowballing`. If no seed papers are provided,
the pipeline uses the top high-confidence ranked papers as seeds when possible.
Functional snowballing uses Semantic Scholar paper lookup, reference, citation,
and recommendation endpoints. If `S2_API_KEY` is missing, the pipeline keeps the
seed records for auditability and safely skips expansion instead of failing.

```bash
python -m lit_screening.pipeline run \
  --question "surface magnetization boundary spin signals" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --seed-paper "10.48550/arXiv.2301.10140" \
  --seed-file examples/seed_papers.csv \
  --enable-snowballing \
  --snowball-top-n 3 \
  --output-dir outputs
```

Seed CSV columns are `seed_id`, `seed_type`, `title`, `doi`, and `note`.
Supported seed types include `doi`, `semantic_scholar`, `openalex`, and `title`.

The DeepSeek base URL and model live in `lit_screening/config.py`:

- `deepseek_base_url = "https://api.deepseek.com"`
- `deepseek_model = "deepseek-chat"`

If `DEEPSEEK_API_KEY` is missing, the pipeline does not fail. It records the
LLM backend as inactive and falls back to the rule-based agents.

## Run

```bash
python -m lit_screening.pipeline run \
  --question "How can human-in-the-loop multi-agent LLM systems improve scientific literature screening and evidence verification?" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --from-year 2020 \
  --strictness balanced \
  --openalex-mode keyword+semantic \
  --sort-preference relevance \
  --ranking-profile balanced \
  --weight-relevance 0.40 \
  --weight-evidence 0.25 \
  --weight-recency 0.15 \
  --weight-quality 0.15 \
  --weight-diversity 0.05 \
  --feedback examples/human_feedback.csv \
  --gold-labels examples/gold_labels.csv \
  --output-dir outputs
```

Optional pilot search runs a small pre-retrieval sample and can apply repaired
queries before full retrieval:

```bash
python -m lit_screening.pipeline run \
  --question "How can LLM agents improve literature screening?" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --pilot-search \
  --pilot-max-per-query 5 \
  --auto-repair-queries \
  --output-dir outputs
```

## v9 Deterministic Baseline Smoke Reproduction

The frozen v9 smoke baseline can be reproduced with the deterministic ablation
runner. Plan-only runs do not require provider API keys and should record
`retrieval_status = planning_only`,
`ranked_papers_based_on_real_retrieval = false`, skipped research-gap generation,
and empty `suggested_next_searches`.

Plan-only cases:

```bash
PYTHONPATH=. python tools/run_ablations.py \
  --mode plan \
  --configs full_system,no_query_family,no_query_repair \
  --case-id ai_literature_screening \
  --output-root outputs/baseline_v9_smoke

PYTHONPATH=. python tools/run_ablations.py \
  --mode plan \
  --configs full_system,no_query_family,no_query_repair \
  --case-id mof_co2_capture \
  --output-root outputs/baseline_v9_smoke

PYTHONPATH=. python tools/run_ablations.py \
  --mode plan \
  --configs full_system,no_query_family,no_query_repair \
  --case-id thin_film_deposition \
  --output-root outputs/baseline_v9_smoke
```

Full OpenAlex-only smoke cases:

```bash
PYTHONPATH=. python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system \
  --case-id sei_lithium_battery \
  --output-root outputs/baseline_v9_smoke

PYTHONPATH=. python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system \
  --case-id oer_spin_state \
  --output-root outputs/baseline_v9_smoke

PYTHONPATH=. python tools/compare_ablations.py outputs/baseline_v9_smoke
```

The `baseline_v9_smoke` review passed 11 runs with `returncode=0`. In that
baseline, SEI and OER full-system runs both produced bounded first-read sets
(`must_read_count = 12`) and reading paths with zero exclude, out-of-scope,
duplicate, or negative-context recommendations.

For broader full-run pilot ablation diagnostics beyond the frozen v9 smoke
baseline, run the exploratory configs under a separate output root:

```bash
PYTHONPATH=. python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system,no_domain_guardrail,no_intent_centrality,no_group_coverage_ranking \
  --case-id sei_lithium_battery \
  --output-root outputs/ablations_pilot

PYTHONPATH=. python tools/run_ablations.py \
  --mode full \
  --providers openalex \
  --configs full_system,no_domain_guardrail,no_intent_centrality,no_group_coverage_ranking \
  --case-id oer_spin_state \
  --output-root outputs/ablations_pilot

PYTHONPATH=. python tools/compare_ablations.py outputs/ablations_pilot
```

## Tests

Run tests from the project root with the package on `PYTHONPATH`:

```bash
PYTHONPATH=. pytest -q
```

Equivalent module form:

```bash
python -m pytest -q
```

## Run The UI

```bash
streamlit run app.py
```

The UI calls the same core pipeline functions as the CLI. Feedback changes are
applied to the in-memory ranking and do not rerun provider API calls.
Each UI run is saved as a screening project under `outputs/projects/`, with
run history, feedback CSV import/export, and an inspectable agent trace.

The UI also supports:

- English / 中文 interface mode. Core workflow labels such as `Planned Queries`,
  `Ranked Papers`, `Evidence`, and `Trace` remain in English for research-demo clarity.
- Chinese research questions are converted into an English planning question for
  retrieval, evidence extraction, and ranking. The Queries tab shows the original
  question, translated/planning question, and translation mode.
- A structured query checkpoint: click `Step 2: Generate Query Plan`, inspect
  or edit the SearchBrief, `core_terms`, `must_terms`, `optional_terms`,
  `exclude_terms`, `required_aspects`, OpenAlex queries, and Semantic Scholar
  queries, then click `Step 4: Run Retrieval`.
  The preview step does not call literature provider APIs, which helps avoid
  spending requests on a search direction that does not match the user's intent.
  You can also click `Run Pilot Search` to retrieve a small sample, diagnose
  off-topic drift, and accept repaired queries before full retrieval.
  Step 3 is a collapsible checkpoint with a field guide, research-intent editor,
  and provider-query editor. After Step 4 completes, the Step 3 panel collapses
  by default but remains available for auditing the exact query plan used.
- Project-style tabs for `Research Intent`, `Search Strategy`, `Results Map`,
  `Paper Cards`, `Feedback`, `Report & Export`, `Trace`, and `Metrics`.
- The `Research Whiteboard` tab summarizes generated sensemaking artifacts before
  the user dives into ranked papers: concept map, query families, seed hints,
  query provenance, paper role distribution, evidence functions, research
  tensions, and suggested next searches.
- Aspect coverage tables, grouped result lists, a PRISMA-like screening flow,
  recommended reading path, and top-paper evidence cards.
- Existing-library import from BibTeX, RIS, or CSV. Imported records are merged
  with provider retrieval results, deduplicated, screened, and ranked through
  the same core pipeline.
- Seed Paper Mode: enter seed DOIs, titles, Semantic Scholar IDs, or OpenAlex IDs,
  upload `seed_papers.csv`, enable citation snowballing, and inspect retrieval
  paths showing whether an expanded paper came from a reference, citation, or
  recommendation.
- Search mode controls for strictness, OpenAlex mode, sort preference, and ranking
  profile (`relevance_first`, `balanced`, `high_quality_review`).
  OpenAlex `keyword`, `exact`, and `semantic` modes map to separate OpenAlex
  request parameters: `search`, `search.exact`, and `search.semantic`.
- A collapsible run-status panel shows what the pipeline is doing during
  screening: retrieval by provider/query, deduplication, evidence extraction,
  grounding verification, ranking, evaluation, and artifact writing.
- If no papers are retrieved, the UI now separates zero-result searches from
  provider API errors such as HTTP failures or rate limits.
- Runtime API key entry for `OPENALEX_API_KEY`, `S2_API_KEY`, and `DEEPSEEK_API_KEY`.
  `OPENALEX_API_KEY` is required for current OpenAlex API access.
  Keys are applied to the current Streamlit process and are not written into project files.
- Adjustable scoring weights for relevance, evidence, recency, quality, and diversity,
  with tooltip explanations for how each weight affects ranking.
- Optional year filtering through the `Apply year filter` and `From year`
  settings. The UI leaves this off by default so broad background searches are
  not accidentally restricted to recent papers only. When enabled, the core
  pipeline enforces a local hard year filter after provider retrieval, so older
  papers cannot enter deduplication, evidence extraction, or ranking.

For an offline smoke run that avoids provider calls:

```bash
python -m lit_screening.pipeline run \
  --question "How can human-in-the-loop systems improve literature screening?" \
  --providers openalex semantic_scholar \
  --max-per-query 0 \
  --output-dir outputs
```

## Research Sensemaking Mode

Research Sensemaking Mode is an explanatory layer around the existing screening
pipeline. It helps the system show how it understood the research question,
which concepts it split the question into, why different search routes exist,
and which parts of the resulting literature map are verified or still uncertain.
In the v9 deterministic baseline, QueryFamily planning is enabled by default as
part of the normal rule-controlled path. The legacy `QueryPlan` and
`planned_queries.json` remain auditable compatibility artifacts, while
QueryFamily-related ablation flags are reserved for debug and pilot evaluation.

### Domain Packs

Domain packs are small JSON knowledge files under `lit_screening/domain_packs/`.
They hold domain terms, synonyms, materials, methods, applications, and
false-positive terms without adding YAML or heavy ontology dependencies. The
first pack is `materials_magnetism.json`, covering concepts such as surface
magnetization, boundary magnetization, surface spin polarization, local
magnetoelectric response, Cr2O3/chromia, FeF2, NiO, CuMnAs, SPLEEM, XMCD-PEEM,
SP-STM, NV magnetometry, and common off-topic phrases.

Use `load_domain_pack("materials_magnetism")` and `list_domain_packs()` from
`lit_screening.domain_packs.loader` when adding or testing domain-aware logic.
Keep domain packs lightweight, fixture-backed, and safe to load without network
or API keys.

### ResearchLens and QueryFamily

`ResearchLens` represents a researcher-style angle on the question, for example
`theory_origin`, `spaldin_framework`, `direct_surface_detection`,
`nanoscale_readout`, `local_magnetoelectric_predictor`, `applications`,
`limitations`, or `frontier`. `ConceptMapper` writes these lenses to
`concept_map.json`.

`QueryFamily` explains why a set of provider queries exists for a lens. For
example, a `direct_surface_detection` family may search SPLEEM, XMCD-PEEM,
spin-resolved photoemission, and SP-STM routes, while `nanoscale_readout` may
search NV and scanning diamond magnetometry routes. `QueryFamilyPlanner` writes
these route explanations to `query_families.json`.

In the v9 deterministic baseline, QueryFamily planning is enabled by default and
contributes to the provider-facing query plan. The pipeline writes
`query_provenance.json` so each query records its provider, source, family name,
lens name, and purpose. QueryFamily ablation/debug flags can disable this layer
for pilot evaluation, but the full-system baseline is QueryFamily-on.

### Seed hints

`SeedExtractionAgent` extracts lightweight seed hints from the user question:
explicit titles, DOI strings, arXiv IDs, and author mentions such as Nicola A.
Spaldin. These hints are written to `seed_hints.json`. Seed hints can strengthen
research lenses and query families, but they do not trigger citation expansion
unless the existing Seed Paper Mode and snowballing flags are explicitly used.

### Paper roles

`PaperRoleClassifier` assigns rule-based research roles from title, abstract,
venue, year, and query provenance. Roles include `theory_origin`,
`conceptual_framework`, `experimental_proof`, `surface_probe_method`,
`nanoscale_readout`, `material_case`, `application_bridge`,
`frontier_extension`, `limitation_or_challenge`, and `review_background`.
The role records are written to `paper_roles.json` and used by the report and
Research Whiteboard to organize papers by research lineage rather than only by
final score.

### Research Process Report

The Research Process Report is currently the `# Research Process` section inside
`report.md`; the pipeline does not write a separate `research_process_report.md`
file. This section summarizes:

- research question interpretation,
- concept decomposition,
- search lenses and query families,
- screening and inclusion criteria,
- paper roles and why they matter,
- research lineage,
- controversies, limitations, and gaps,
- missing keywords, methods, authors, or schools,
- suggested next searches,
- verified vs uncertain findings.

If an artifact was not generated, the report says `Not generated in this run.`
It must not invent citation relations. Unless citation or snowballing artifacts
verify a link, the report states that the citation relation is not verified.

### Exploration quality metrics

`lit_screening/evaluation/exploration_quality.py` adds an exploration-quality
summary without replacing ranking metrics such as precision@k or nDCG. It writes
`exploration_quality.json` with first-pass measures for concept coverage, query
family coverage, paper role diversity, seed hint utilization, evidence function
diversity, gap specificity, and research tension count.

### Materials Magnetism Example

For an offline check of the Spaldin surface-magnetization case, run:

```bash
python -m lit_screening.pipeline run \
  --question "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets 和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order 相关的探测表面磁化和自旋极化重要性的文章" \
  --providers openalex semantic_scholar \
  --max-per-query 0 \
  --output-dir outputs/spaldin_surface_demo
```

Expected sensemaking outputs include:

- `outputs/spaldin_surface_demo/concept_map.json`
- `outputs/spaldin_surface_demo/query_families.json`
- `outputs/spaldin_surface_demo/seed_hints.json`
- `outputs/spaldin_surface_demo/paper_roles.json`
- `outputs/spaldin_surface_demo/exploration_quality.json`
- `outputs/spaldin_surface_demo/report.md`, including the Research Process section

With `--max-per-query 0`, paper-dependent artifacts such as `paper_roles.json`
may be empty, but the concept map, seed hints, query-family rationale, and report
structure should still be available for review.

## Outputs

The pipeline writes:

- `outputs/planned_queries.json`
- `outputs/search_brief.json`
- `outputs/search_contract.json`
- `outputs/ambiguity_analysis.json`
- `outputs/question_refinement.json`
- `outputs/concept_map.json`
- `outputs/query_families.json`
- `outputs/seed_hints.json`
- `outputs/query_provenance.json`
- `outputs/raw_openalex_results.json`
- `outputs/raw_semantic_scholar_results.json`
- `outputs/merged_papers.csv`
- `outputs/evidence_table.csv`
- `outputs/evidence_functions.json`
- `outputs/aspect_coverage.csv`
- `outputs/domain_assessments.json`
- `outputs/paper_roles.json`
- `outputs/research_tensions.json`
- `outputs/screening_decisions.csv`
- `outputs/screening_decisions.json`
- `outputs/preference_learning.json`
- `outputs/feedback_query_refinement.json`
- `outputs/seed_papers.json`
- `outputs/citation_expansion.csv`
- `outputs/retrieval_paths.csv`
- `outputs/method_comparison_matrix.csv`
- `outputs/method_comparison_matrix.md`
- `outputs/research_gap_matrix.csv`
- `outputs/research_gap_matrix.md`
- `outputs/suggested_next_searches.json`
- `outputs/suggested_next_searches.md`
- `outputs/ranked_papers_before_feedback.csv`
- `outputs/ranked_papers_after_feedback.csv`, when feedback is provided
- `outputs/ranked_papers.csv`
- `outputs/evaluation.json`
- `outputs/agent_trace.json`
- `outputs/run_events.jsonl`
- `outputs/imported_papers.csv`, when an external library is imported
- `outputs/import_diagnostics.json`, when an external library is imported
- `outputs/retrieval_diagnostics.json`
- `outputs/query_pilot_diagnostics.json`
- `outputs/query_repair_suggestions.json`
- `outputs/result_groups.json`
- `outputs/prisma_like_flow.json`
- `outputs/paper_cards.md`
- `outputs/reading_path.md`
- `outputs/report.md`
- `outputs/exploration_quality.json`

Raw cache files are stored under `data/cache/` and ignored by git.

`run_events.jsonl` is written incrementally while the screening run is executing.
It records stage transitions, provider errors, and fatal exceptions so failed
runs can be diagnosed even when later artifacts were not produced.

`retrieval_diagnostics.json` records the query plan, per-provider queries,
per-query raw counts, provider errors, imported-library counts, top titles per
query, top score breakdowns, and year-filter audit information. When
`from_year` is set, papers published before that year, or papers with missing
year metadata, are filtered locally before deduplication and ranking.

`search_contract.json` records the intended domain, required concepts, excluded
concepts, and field-of-study boundaries used before query planning.
`ambiguity_analysis.json` records ambiguous terms such as `screening`, `agent`,
`evidence`, `optimization`, and `ranking` with the selected meaning and
recommended exclude terms.
`domain_assessments.json` records whether each merged paper is `in_scope`,
`borderline`, or `out_of_scope` for the SearchContract. Ranking applies a
transparent hard-demotion multiplier: `in_scope` x1.0, `borderline` x0.7,
and `out_of_scope` x0.3.
`screening_decisions.csv` and `screening_decisions.json` record per-paper
include / maybe / exclude recommendations, decision confidence, reading
priority, suggested action, and normalized exclusion reasons.
`preference_learning.json` and `feedback_query_refinement.json` record terms
learned from human include/exclude feedback and suggested terms for the next
retrieval run.
`concept_map.json`, `query_families.json`, `seed_hints.json`,
`query_provenance.json`, `paper_roles.json`, `evidence_functions.json`, and
`research_tensions.json` are the Research Sensemaking artifacts used by the
Research Whiteboard and the Research Process section of `report.md`.
`seed_papers.json`, `citation_expansion.csv`, and `retrieval_paths.csv` record
Seed Paper Mode inputs, expanded candidate papers, and why each expanded paper
entered the candidate set.
`method_comparison_matrix.*`, `research_gap_matrix.*`, and
`suggested_next_searches.*` turn the ranking into a decision memo: what methods
are represented, what gaps remain, and what to search next.
`query_pilot_diagnostics.json` and `query_repair_suggestions.json` are written
for the optional pilot-search workflow. Pilot search is off by default and runs
only when requested through the UI or CLI flags such as `--pilot-search`.
`exploration_quality.json` summarizes concept coverage, query-family coverage,
paper-role diversity, seed-hint utilization, evidence-function diversity, gap
specificity, and research tension count.

`ranked_papers.csv` includes provenance columns such as `retrieval_provider`,
`retrieval_stage`, `retrieval_query`, `source_stage`, `seed_paper_id`,
`seed_title`, and `seed_reason`, so a ranked paper can be traced back to keyword
search, semantic search, imported-library input, reference expansion, citation
expansion, or recommendation expansion.

## Query Planning, Sensemaking, And Scoring

The intent agent writes a `SearchBrief` with the refined question, search intent,
inclusion/exclusion criteria, required aspects, preferred paper types, and a
success definition. The planner then writes a structured `QueryPlan` into
`planned_queries.json`, including topic terms, provider-specific queries, and
search controls. OpenAlex queries use quoted multi-word core terms plus
boolean-style `AND` / `OR` / `NOT` where useful. Semantic Scholar queries use
quoted phrases, `+required` terms, `-excluded` terms, and OR alternatives.

After evidence verification, the aspect classifier checks which required aspects
each paper covers. The report and UI expose:

- aspect coverage records,
- grouped result lists such as `must_read`, `recent_frontier`, and `background_or_survey`,
- a recommended reading path,
- paper evidence cards with suggested include/exclude/uncertain actions,
- a PRISMA-like screening-flow summary.

Hybrid relevance combines TF-IDF similarities with provider metadata:

```text
hybrid_relevance_score =
0.30 * title_similarity
+ 0.25 * abstract_similarity
+ 0.15 * evidence_similarity
+ 0.10 * api_relevance_score
+ 0.10 * must_term_coverage
+ 0.10 * field_match_score
```

Code: `lit_screening/reranking.py::compute_hybrid_relevance_features`.

Evidence score combines grounding and relevance:

```text
evidence_score =
0.60 * verifier_confidence
+ 0.40 * evidence_question_relevance
```

Code: `lit_screening/scoring.py::score_evidence`.

The final ranking score remains transparent and profile-driven:

```text
final_score =
0.40 * relevance_score
+ 0.25 * evidence_score
+ 0.15 * recency_score
+ 0.15 * quality_score
+ 0.05 * diversity_score
+ human_feedback_adjustment
```

Code: `lit_screening/scoring.py::compute_score_breakdown` is the main scoring
entrypoint used by `RankerAgent`; `compute_final_score` is the low-level formula
helper for combining already-computed score components. `score_paper` remains
only as a backward-compatible alias.

Base scores are clamped to `[0, 1]`. Human feedback is an explicit additive adjustment.
Ranking profiles can change the base weights before any user-provided weight
overrides are applied.

## Evidence Validation

The verifier treats evidence as strict only when the evidence sentence can be
matched back to the abstract by either:

- exact span match, or
- high-confidence fuzzy span match.

Keyword overlap alone is no longer counted as strict support. It is marked as
`weak_support`. Evidence that cannot be matched is marked as `unverified`; LLM
evidence that cannot be matched is marked as `llm_invalid_evidence`.

Evidence audit fields are written to CSV outputs, the Streamlit UI, the report,
and `agent_trace.json`:

- `support_level`
- `span_match_type`
- `span_match_confidence`
- `matched_text`
- `strict_span_validated`
- `llm_invalid_evidence`
- `missing_abstract`

Evaluation includes grounding-oriented and ranking-oriented metrics:

- `grounding_accuracy`
- `strict_support_rate`
- `weak_support_rate`
- `llm_invalid_evidence_rate`
- `precision_at_10`
- `ndcg_at_10`
- `map`
- `recall_at_10`
- `feedback_before_after_ranking_delta`

## Test

```bash
pytest
```

The tests use fake retrievers for pipeline behavior, so they do not require internet access or API keys.
They also cover seed-file parsing, citation snowballing with fake Semantic
Scholar responses, and pipeline output generation for Seed Paper Mode.
