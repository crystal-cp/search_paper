# System Architecture

## Scope

This document describes the current `search_paper` pipeline as implemented in
the repository. It is a documentation artifact only. It does not introduce a new
pipeline, provider, ranking model, UI framework, vector database, PDF parser, or
mandatory LLM dependency.

The architecture is deliberately conservative:

- The core pipeline runs with rule-based agents when no LLM key is available.
- DeepSeek/OpenAI-compatible LLM enhancement is optional.
- Seed-paper snowballing is optional and disabled unless requested.
- Query-family retrieval is optional; the legacy `QueryPlan` remains the default
  retrieval path.
- The Streamlit UI is a thin wrapper around core pipeline functions.

## High-Level Flow

The main orchestration entrypoint is `lit_screening/pipeline.py`. A run converts
a user question and optional inputs into a set of auditable artifacts.

```text
User question
  -> intent repair and search contract
  -> concept map, seed hints, and query plans
  -> provider retrieval / library import / optional snowballing
  -> normalization and deduplication
  -> evidence extraction and span verification
  -> aspect coverage, domain assessment, and paper roles
  -> ranking and optional feedback adjustment
  -> decision artifacts, diagnostics, reports, and trace
```

## Agent Flow

### 1. Intent Interpretation

`lit_screening/agents/research_intent.py` converts the raw question into a
`SearchBrief` with refined question text, inclusion logic, exclusion logic,
required aspects, preferred paper types, and a success definition.

`lit_screening/agents/question_refiner.py` can split broad or mixed questions
into optional subquestions. The output is explanatory and auditable; it should
not silently replace the user's research direction.

Primary artifacts:

- `search_brief.json`
- `question_refinement.json`

### 2. Contract, Ambiguity, And Domain Boundaries

The pipeline builds a search contract that records intended domain boundaries,
required concepts, excluded concepts, and ambiguous terms. Domain packs remain
lightweight JSON/dataclass resources and should describe terms and guardrails,
not hide business logic.

Primary artifacts:

- `search_contract.json`
- `ambiguity_analysis.json`
- `domain_assessments.json`, after papers are available

### 3. Concept Mapping And Query Families

`lit_screening/agents/concept_mapper.py` decomposes the repaired intent into
research lenses and concept groups.

`lit_screening/agents/query_family_planner.py` explains why different search
routes exist for different lenses. Query families are useful for audit and
sensemaking, but they do not replace default retrieval unless query-family
retrieval is explicitly enabled.

`lit_screening/agents/seed_extraction.py` extracts DOI, title, arXiv, author,
Semantic Scholar, or OpenAlex hints from the question. Seed hints do not trigger
snowballing by themselves.

Primary artifacts:

- `concept_map.json`
- `query_families.json`
- `seed_hints.json`
- `query_provenance.json`, when query provenance is available

### 4. Provider-Aware Query Planning

`lit_screening/agents/planner.py` builds the default `QueryPlan`, including
topic terms, provider-specific OpenAlex queries, provider-specific Semantic
Scholar queries, filters, and search controls.

The planner may be rule-based or optionally LLM-enhanced. Missing LLM keys must
fall back to rule-based behavior.

Primary artifact:

- `planned_queries.json`

### 5. Retrieval And Import

`lit_screening/agents/retriever.py` executes provider retrieval through modular
clients:

- `lit_screening/retrieval/openalex_client.py`
- `lit_screening/retrieval/semantic_scholar_client.py`

External-library import is handled by:

- `lit_screening/importers/bibtex.py`
- `lit_screening/importers/ris.py`
- `lit_screening/importers/csv_importer.py`

Seed Paper Mode and citation snowballing are handled by
`lit_screening/agents/snowball.py` and remain disabled unless explicitly
enabled. If Semantic Scholar credentials or endpoints are unavailable, the
pipeline should preserve seed records for auditability and skip expansion
safely.

Primary artifacts:

- `raw_openalex_results.json`
- `raw_semantic_scholar_results.json`
- `imported_papers.csv`, when an external library is imported
- `import_diagnostics.json`, when an external library is imported
- `seed_papers.json`, when seed inputs exist
- `citation_expansion.csv`, when snowballing is enabled
- `retrieval_paths.csv`, when seed expansion paths exist
- `retrieval_diagnostics.json`
- `run_events.jsonl`

### 6. Normalization And Deduplication

Provider, import, and seed-expansion records are normalized into shared `Paper`
dataclasses from `lit_screening/models.py`. Deduplication merges candidate
records while preserving provenance fields such as provider, query, retrieval
stage, seed paper ID, and source stage.

Primary artifact:

- `merged_papers.csv`

### 7. Evidence Extraction And Verification

`lit_screening/agents/extractor.py` extracts claim-level evidence from title and
abstract.

`lit_screening/agents/verifier.py` verifies whether extracted evidence is
grounded in the abstract. `lit_screening/span_validation.py` performs exact and
high-confidence fuzzy matching. Keyword overlap alone is weaker support and must
not be treated as strict evidence.

`lit_screening/agents/evidence_function_classifier.py` labels what role an
evidence span plays while preserving strict span validation.

Primary artifacts:

- `evidence_table.csv`
- `evidence_functions.json`

Important fields include:

- `support_level`
- `span_match_type`
- `span_match_confidence`
- `matched_text`
- `strict_span_validated`
- `llm_invalid_evidence`
- `missing_abstract`

### 8. Coverage, Roles, Tensions, And Decisions

`lit_screening/agents/aspect_classifier.py` checks required-aspect coverage.

`lit_screening/agents/paper_role_classifier.py` assigns research roles such as
background review, method paper, theory origin, experimental proof, material
case, application bridge, frontier extension, or limitation.

`lit_screening/agents/controversy_boundary.py` identifies rule-based tensions,
limitations, and boundary conditions.

Decision-artifact generation turns the ranked set into method comparisons,
coverage summaries, research-gap rows, suggested next searches, result groups,
screening decisions, paper cards, and reading paths. These artifacts must
distinguish verified/span-grounded findings from uncertain hints.

Primary artifacts:

- `aspect_coverage.csv`
- `paper_roles.json`
- `research_tensions.json`
- `screening_decisions.csv`
- `screening_decisions.json`
- `method_comparison_matrix.csv`
- `method_comparison_matrix.md`
- `research_gap_matrix.csv`
- `research_gap_matrix.md`
- `suggested_next_searches.json`
- `suggested_next_searches.md`
- `result_groups.json`
- `prisma_like_flow.json`
- `paper_cards.md`
- `reading_path.md`

### 9. Ranking And Feedback

`lit_screening/agents/ranker.py` uses the central scoring entrypoint in
`lit_screening/scoring.py`. Ranking combines relevance, evidence, recency,
quality, diversity, and optional human feedback adjustment.

`lit_screening/reranking.py` computes hybrid TF-IDF/API/field relevance
features. `lit_screening/agents/human_feedback.py` reads and applies feedback
CSV records.

Primary artifacts:

- `ranked_papers_before_feedback.csv`
- `ranked_papers_after_feedback.csv`, when feedback is provided
- `ranked_papers.csv`
- `preference_learning.json`
- `feedback_query_refinement.json`
- `evaluation.json`

### 10. Reporting And Trace

The report layer writes both research-facing and user-facing summaries from the
same core artifacts. Reports should explain provider status, retrieval status,
evidence caveats, reading priorities, coverage, remaining gaps, and uncertainty.

Planning-only runs must not invent research gaps. If retrieval is not performed
or no screened papers are available, research-gap artifacts should explicitly
record a skipped status rather than pretending that gaps were inferred.

Primary artifacts:

- `report.md`
- `user_report.md`, when the user-report path is enabled
- `agent_trace.json`
- `run_events.jsonl`
- `exploration_quality.json`

## Artifact Map

The architecture is artifact-first: each agent writes inspectable files that can
be reviewed without rerunning the full pipeline.

| Stage | Main artifacts |
| --- | --- |
| Intent and contract | `search_brief.json`, `search_contract.json`, `ambiguity_analysis.json`, `question_refinement.json` |
| Query planning | `planned_queries.json`, `concept_map.json`, `query_families.json`, `seed_hints.json`, `query_provenance.json` |
| Retrieval and import | `raw_openalex_results.json`, `raw_semantic_scholar_results.json`, `imported_papers.csv`, `import_diagnostics.json`, `retrieval_diagnostics.json` |
| Seed mode | `seed_papers.json`, `citation_expansion.csv`, `retrieval_paths.csv` |
| Candidate set | `merged_papers.csv` |
| Evidence | `evidence_table.csv`, `evidence_functions.json` |
| Screening | `domain_assessments.json`, `aspect_coverage.csv`, `screening_decisions.csv`, `screening_decisions.json` |
| Sensemaking | `paper_roles.json`, `research_tensions.json`, `method_comparison_matrix.*`, `research_gap_matrix.*`, `suggested_next_searches.*` |
| Ranking and feedback | `ranked_papers_before_feedback.csv`, `ranked_papers_after_feedback.csv`, `ranked_papers.csv`, `preference_learning.json`, `feedback_query_refinement.json` |
| Reports and diagnostics | `paper_cards.md`, `reading_path.md`, `report.md`, `evaluation.json`, `agent_trace.json`, `run_events.jsonl`, `exploration_quality.json` |

## UI Boundary

`app.py` is the Streamlit UI. It should call core pipeline functions such as
`plan_screening_queries()` and `run_pipeline()` and should not duplicate planner,
retrieval, scoring, verification, import, evaluation, or report logic. The UI is
responsible for checkpointing user intent, collecting configuration, preserving
session state, displaying artifacts, applying feedback, and offering downloads.

## Extension Boundaries

Future work can add better agents, providers, evaluation sets, or visualizations,
but the current architecture depends on a few boundaries:

- Do not hard-code or print API keys.
- Keep OpenAlex and Semantic Scholar clients modular.
- Keep external-library import modular and dependency-light.
- Keep domain packs lightweight and fixture-testable.
- Keep LLM use optional.
- Keep snowballing optional.
- Keep QueryFamily retrieval optional.
- Keep generated outputs and raw caches out of git.
- Keep every inferred artifact explicit about what is verified, weakly
  supported, uncertain, skipped, or not generated.
