# AGENTS.md

## Project goal

Build a lightweight, reproducible, human-in-the-loop multi-agent system for evidence-grounded scientific literature screening.

Given a research question, the system should:
1. plan scholarly search queries,
2. retrieve papers from OpenAlex and Semantic Scholar,
3. normalize and deduplicate metadata,
4. extract claim-level evidence from title and abstract,
5. verify whether the claim is grounded in the abstract,
6. rank papers with a transparent multi-objective scoring function,
7. refine ranking with human feedback,
8. output CSV files, evaluation metrics, and a Markdown report.

## Development rules

- Keep the MVP simple and reliable.
- Do not implement PDF parsing, vector databases, or web UI unless explicitly requested.
- Do not hard-code API keys.
- Use environment variables:
  - OPENALEX_API_KEY
  - S2_API_KEY
  - DEEPSEEK_API_KEY
- The core pipeline must run without an LLM key using rule-based fallback agents.
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

## Required outputs

outputs/
- planned_queries.json
- raw_openalex_results.json
- raw_semantic_scholar_results.json
- merged_papers.csv
- evidence_table.csv
- ranked_papers_before_feedback.csv
- ranked_papers_after_feedback.csv, if feedback is provided
- evaluation.json
- report.md

## Testing

Run:

pytest

Before finishing a task, ensure:
- tests pass,
- CLI still works,
- no API keys are printed,
- no raw cache or output files are committed accidentally.