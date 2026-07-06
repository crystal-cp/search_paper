# AGENTS.md

## Project goal

Build a lightweight, reproducible, human-in-the-loop multi-agent system for evidence-grounded scientific literature screening.

Given a research question, the system should:
1. plan scholarly search queries,
2. retrieve papers from OpenAlex and Semantic Scholar,
3. optionally import papers from BibTeX, RIS, or CSV literature-library exports,
4. normalize and deduplicate metadata,
5. extract claim-level evidence from title and abstract,
6. verify whether the claim is grounded in the abstract,
7. rank papers with a transparent multi-objective scoring function,
8. refine ranking with human feedback,
9. output CSV files, evaluation metrics, and a Markdown report.

## Development rules

- Keep the MVP simple and reliable.
- Do not implement PDF parsing or vector databases unless explicitly requested.
- The Streamlit UI already exists; keep it as a thin wrapper around core pipeline functions.
- Do not hard-code API keys.
- Use environment variables:
  - OPENALEX_API_KEY
  - S2_API_KEY
  - DEEPSEEK_API_KEY
- The core pipeline must run without an LLM key using rule-based fallback agents.
- Existing-library import should stay dependency-light and normalize into `Paper`.
- `compute_score_breakdown()` is the main ranking score entrypoint; keep README formulas aligned with scoring.py/reranking.py.
- All external API calls must have timeout, retry, and basic rate-limit handling.
- Cache API responses locally under data/cache/.
- Do not commit .env, cache files, or outputs except .gitkeep.
- Use dataclasses or Pydantic-style structured models.
- Write tests for normalization, deduplication, scoring, feedback, and pipeline behavior with fake data.
- The UI layer must not contain business logic; it should call core pipeline functions only.

## Required command

The main command should work like:

python -m lit_screening.pipeline run \
  --question "How can human-in-the-loop multi-agent LLM systems improve scientific literature screening and evidence verification?" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --from-year 2020 \
  --output-dir outputs

Optional external-library import should work like:

python -m lit_screening.pipeline run \
  --question "surface magnetization boundary spin signals" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --input-file path/to/library.bib \
  --input-format auto \
  --output-dir outputs

## Required outputs

outputs/
- planned_queries.json
- search_brief.json
- question_refinement.json
- raw_openalex_results.json
- raw_semantic_scholar_results.json
- merged_papers.csv
- evidence_table.csv
- aspect_coverage.csv
- ranked_papers_before_feedback.csv
- ranked_papers_after_feedback.csv, if feedback is provided
- ranked_papers.csv
- evaluation.json
- agent_trace.json
- run_events.jsonl
- imported_papers.csv, if an external library is imported
- import_diagnostics.json, if an external library is imported
- retrieval_diagnostics.json
- result_groups.json
- prisma_like_flow.json
- paper_cards.md
- reading_path.md
- report.md

## Testing

Run:

pytest

Before finishing a task, ensure:
- tests pass,
- CLI still works,
- no API keys are printed,
- no raw cache or output files are committed accidentally.
