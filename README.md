# Human-in-the-loop Multi-agent LLM System for Scientific Literature Screening

[中文说明](README.zh-CN.md)

This repository is a lightweight, reproducible research prototype for evidence-grounded scientific literature screening. It is intentionally simple: the Streamlit UI is a thin wrapper around the core pipeline, with no PDF parsing, no vector database, and no mandatory LLM key.

The MVP pipeline:

1. interprets the user's search intent as a `SearchBrief`,
2. refines broad questions into optional subquestions,
3. builds a structured, provider-aware query plan from the brief,
4. retrieves metadata from OpenAlex and Semantic Scholar using provider-specific queries,
5. optionally imports existing BibTeX, RIS, or CSV literature-library exports,
6. optionally expands from seed papers through references, citations, and recommendations,
7. normalizes and deduplicates papers,
8. extracts claim-level evidence from abstracts,
9. verifies whether evidence is grounded in the abstract with strict span validation,
10. ranks papers with hybrid TF-IDF/API/field relevance, aspect coverage, and transparent scoring,
11. groups papers into reading roles and generates paper evidence cards,
12. optionally applies human feedback,
13. writes CSV, JSON, Markdown outputs, and an agent decision trace.

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

## Optional DeepSeek LLM Enhancement

The default pipeline is fully rule-based. You can optionally use an
OpenAI-compatible LLM backend for query planning, evidence extraction, and/or
verification. DeepSeek support is built in through `DEEPSEEK_API_KEY`.
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

## Outputs

The pipeline writes:

- `outputs/planned_queries.json`
- `outputs/search_brief.json`
- `outputs/search_contract.json`
- `outputs/ambiguity_analysis.json`
- `outputs/question_refinement.json`
- `outputs/raw_openalex_results.json`
- `outputs/raw_semantic_scholar_results.json`
- `outputs/merged_papers.csv`
- `outputs/evidence_table.csv`
- `outputs/aspect_coverage.csv`
- `outputs/domain_assessments.json`
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
`seed_papers.json`, `citation_expansion.csv`, and `retrieval_paths.csv` record
Seed Paper Mode inputs, expanded candidate papers, and why each expanded paper
entered the candidate set.
`method_comparison_matrix.*`, `research_gap_matrix.*`, and
`suggested_next_searches.*` turn the ranking into a decision memo: what methods
are represented, what gaps remain, and what to search next.
`query_pilot_diagnostics.json` and `query_repair_suggestions.json` are written
for the optional pilot-search workflow. Pilot search is off by default and runs
only when requested through the UI or CLI flags such as `--pilot-search`.

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
