# AGENTS.md

This repository is `crystal-cp/search_paper`, a research prototype for
human-in-the-loop, intent-aware, multi-agent scientific literature screening.

The project should behave like a reproducible PhD-application demo, not a
commercial product and not a generic keyword-search wrapper. Given a research
question, the system should understand the user's research intent, retrieve or
import candidate papers, ground evidence in abstracts, rank papers transparently,
accept human feedback, and write auditable outputs.

## Development Rules

- Keep the MVP simple, readable, and reproducible.
- Preserve backward compatibility for the CLI and Streamlit UI.
- Add new outputs when needed; do not remove or rename existing outputs unless
  the user explicitly asks.
- Do not turn small feature requests into one large pipeline rewrite. Add
  optional artifacts and feature flags in small, reviewable steps, preserving
  existing retrieval and ranking behavior unless the user explicitly asks for a
  replacement.
- Do not implement PDF parsing, vector databases, or a new UI framework unless
  explicitly requested.
- Keep OpenAlex and Semantic Scholar retrieval modular.
- Keep external literature-library import modular and dependency-light.
- Keep domain packs lightweight, JSON/dataclass based, dependency-light, and
  covered by fixture tests. They should describe terms and guardrails, not hide
  business logic.
- Do not hard-code API keys.
- Do not print API keys or write them into repo files.
- The core pipeline must run without any LLM key by falling back to rule-based
  agents.
- LLM-enhanced behavior must remain optional.
- The Streamlit UI must stay a thin wrapper around core pipeline functions.
  Do not move business logic into `app.py`.
- Use dataclasses or similarly simple structured models.
- Prefer focused tests with fake retrievers, mocked LLM responses, and local
  fixtures. Tests must not require internet access or real API keys.
- Any new inferred artifact must distinguish verified/span-grounded findings
  from uncertain hints derived only from titles, query matches, roles, or rules.
- Do not invent citation relations. Unless a relation is supported by real
  citation/snowballing artifacts, mark it as not verified.
- Do not commit `.env`, raw cache files, generated outputs, or local runtime
  artifacts except placeholder `.gitkeep` files.

## API Keys

Use environment variables only:

- `OPENALEX_API_KEY`
- `S2_API_KEY`
- `DEEPSEEK_API_KEY`

OpenAlex and Semantic Scholar clients may run with different key availability,
but missing keys should produce clear user-facing diagnostics instead of secret
leaks. DeepSeek is optional; if `DEEPSEEK_API_KEY` is missing, rule-based
planning, extraction, and verification must continue to work.

## Architecture Overview

The main orchestration entrypoint is `lit_screening/pipeline.py`.

The pipeline flow is:

1. interpret the user's intent as a `SearchBrief`,
2. optionally refine broad or mixed questions,
3. build a provider-aware `QueryPlan`,
4. retrieve papers from selected providers,
5. optionally import BibTeX, RIS, or CSV literature-library exports,
6. optionally expand from seed papers through citation snowballing,
7. normalize and deduplicate papers,
8. extract claim-level evidence from title and abstract,
9. verify whether extracted evidence is grounded in the abstract,
10. compute aspect coverage and transparent ranking scores,
11. optionally apply human feedback,
12. write CSV, JSON, Markdown, diagnostics, and trace outputs.

### Core Models

`lit_screening/models.py` defines the shared dataclasses:

- `SearchBrief`: interpreted research intent and inclusion logic.
- `DomainPack` / `DomainConcept`: lightweight JSON-backed domain terms,
  synonyms, methods, materials, applications, and false-positive terms.
- `ResearchLens`, `QueryFamily`, `ResearchLensPlan`, and `QueryFamilyPlan`:
  optional research-sensemaking structures that explain concept decomposition
  and query-route purpose without replacing the legacy `QueryPlan` by default.
- `QueryPlan`: topic terms, provider-specific queries, and search filters.
- `Paper`: normalized metadata from providers or imported libraries.
- `EvidenceRecord`: extracted claim and evidence sentence.
- `VerificationResult`: grounding decision, span validation details, and LLM
  validity flags.
- `ScoreBreakdown`: relevance, evidence, recency, quality, diversity, feedback,
  and final score.
- `AspectCoverageRecord`: required-aspect coverage per paper.
- `FeedbackRecord`: human include/exclude/uncertain signal.
- `PaperRoleRecord` and `ResearchTension`: optional sensemaking records for
  research roles, limitations, controversies, and boundary conditions.
- `RankedPaper`: paper plus evidence, verification, score, and feedback.
- `PipelineResult`: in-memory summary returned by the pipeline and UI.

### Agents

Agents live under `lit_screening/agents/`:

- `research_intent.py`: builds the `SearchBrief`.
- `question_refiner.py`: splits or clarifies broad research questions.
- `concept_mapper.py`: maps a question and optional seed hints to research
  lenses using rule-based domain-pack knowledge.
- `query_family_planner.py`: turns research lenses into provider-aware query
  families for explanation and optional feature-flagged retrieval.
- `seed_extraction.py`: extracts title, DOI, arXiv, and author seed hints from
  the user question without triggering citation expansion.
- `planner.py`: builds topic-aware structured query plans and provider-specific
  OpenAlex / Semantic Scholar queries.
- `retriever.py`: runs provider clients and records raw retrieval outputs.
- `extractor.py`: extracts abstract-grounded claim-level evidence.
- `verifier.py`: validates whether evidence is grounded in the abstract.
- `aspect_classifier.py`: evaluates coverage of required research aspects.
- `paper_role_classifier.py`: assigns rule-based research roles without
  changing ranking.
- `evidence_function_classifier.py`: labels evidence function while preserving
  strict span validation.
- `controversy_boundary.py`: identifies rule-based tensions, limitations, and
  boundary conditions.
- `ranker.py`: ranks papers using the central scoring entrypoint.
- `human_feedback.py`: reads and applies human feedback CSV records.
- `snowball.py`: optional Seed Paper Mode and citation snowballing through
  Semantic Scholar references, citations, and recommendations.

Rule-based agents are the required default. LLM modes may enhance planner,
extractor, and verifier behavior, but must never become mandatory.

### Retrieval And Import

Retrieval clients live under `lit_screening/retrieval/`:

- `base.py`: shared retrieval protocol/result structure.
- `openalex_client.py`: OpenAlex search, normalization, cache use, timeouts, and
  request-error handling.
- `semantic_scholar_client.py`: Semantic Scholar search, normalization,
  optional API-key header, retry/rate-limit handling, and timeouts.

External library import lives under `lit_screening/importers/`:

- `base.py`: format detection and common `Paper` normalization helpers.
- `bibtex.py`: BibTeX import.
- `ris.py`: RIS import.
- `csv_importer.py`: CSV import.

Imported records must become normal `Paper` objects and pass through the same
deduplication, evidence extraction, verification, scoring, and reporting path as
retrieved provider records.

### Evidence, Ranking, And Evaluation

- `lit_screening/span_validation.py` performs exact and high-confidence fuzzy
  evidence-span validation against abstracts.
- `lit_screening/scoring.py` contains the main scoring entrypoint:
  `compute_score_breakdown()`.
- `lit_screening/reranking.py` computes hybrid TF-IDF/API/field relevance
  features used by ranking.
- `lit_screening/evaluation.py` computes retrieval, grounding, feedback, and
  ranking metrics.
- `lit_screening/report.py` writes the Markdown report.
- `lit_screening/paper_cards.py`, `reading_path.py`, `result_groups.py`, and
  `screening_flow.py` generate sensemaking artifacts for the UI and outputs.

Keep README scoring formulas aligned with `scoring.py` and `reranking.py`.
`compute_final_score()` is a low-level formula helper; `score_paper()` is kept
for backward compatibility.

Evidence must not be treated as strict support unless the evidence sentence can
be matched back to the abstract by exact match or high-confidence fuzzy match.
Keyword-overlap support should remain a weaker category such as `weak_support`,
not strict support.

### UI

`app.py` is the Streamlit UI. It should:

- call `plan_screening_queries()`, `run_pipeline()`, and feedback helpers from
  the core package,
- use `st.session_state` to avoid rerunning provider retrieval when only
  feedback changes,
- expose project history, editable query plans, trace/diagnostic views,
  ranked-paper tables, evidence chains, feedback import/export, and downloads,
- show clear messages for missing keys, no retrieved papers, provider errors,
  and empty imported libraries,
- never print or persist API keys.

Do not duplicate planner, retrieval, scoring, verification, import, evaluation,
or report logic in the UI.

## Required Commands

The main CLI command must keep working:

```bash
python -m lit_screening.pipeline run \
  --question "How can human-in-the-loop multi-agent LLM systems improve scientific literature screening and evidence verification?" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --from-year 2020 \
  --output-dir outputs
```

Optional external-library import must keep working:

```bash
python -m lit_screening.pipeline run \
  --question "surface magnetization boundary spin signals" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --input-file path/to/library.bib \
  --input-format auto \
  --output-dir outputs
```

Optional Seed Paper Mode must remain disabled unless requested:

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

The Streamlit UI must keep running:

```bash
streamlit run app.py
```

Run tests with:

```bash
pytest
```

## Required Outputs

Keep these output files compatible:

- `planned_queries.json`
- `search_brief.json`
- `search_contract.json`
- `ambiguity_analysis.json`
- `question_refinement.json`
- `concept_map.json`
- `query_families.json`
- `seed_hints.json`
- `query_provenance.json`
- `raw_openalex_results.json`
- `raw_semantic_scholar_results.json`
- `merged_papers.csv`
- `evidence_table.csv`
- `evidence_functions.json`
- `aspect_coverage.csv`
- `domain_assessments.json`
- `paper_roles.json`
- `research_tensions.json`
- `screening_decisions.csv`
- `screening_decisions.json`
- `preference_learning.json`
- `feedback_query_refinement.json`
- `seed_papers.json`
- `citation_expansion.csv`
- `retrieval_paths.csv`
- `method_comparison_matrix.csv`
- `method_comparison_matrix.md`
- `research_gap_matrix.csv`
- `research_gap_matrix.md`
- `suggested_next_searches.json`
- `suggested_next_searches.md`
- `ranked_papers_before_feedback.csv`
- `ranked_papers_after_feedback.csv`, if feedback is provided
- `ranked_papers.csv`
- `evaluation.json`
- `agent_trace.json`
- `run_events.jsonl`
- `imported_papers.csv`, if an external library is imported
- `import_diagnostics.json`, if an external library is imported
- `retrieval_diagnostics.json`
- `query_pilot_diagnostics.json`
- `query_repair_suggestions.json`
- `result_groups.json`
- `prisma_like_flow.json`
- `paper_cards.md`
- `reading_path.md`
- `report.md`
- `exploration_quality.json`

Raw API cache files belong under `data/cache/` and should remain ignored by git.
Generated output files belong under `outputs/` and should remain ignored by git
except for placeholder files.

## Testing Expectations

Before finishing a coding task:

- run `pytest` when practical,
- verify the CLI parser still accepts existing flags,
- verify the Streamlit app still imports or starts when UI code changed,
- add or update tests for each new module or behavior,
- use mocked LLM responses and fake retrievers for tests,
- confirm no API keys are printed, stored, or committed,
- check `git status --short` so generated outputs and caches are not staged.

For documentation-only changes, do not change functional behavior. A lightweight
syntax/import check is enough unless the documentation update accompanies code
changes.
