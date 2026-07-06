# Human-in-the-loop Multi-agent LLM System for Scientific Literature Screening

This repository is a lightweight, reproducible research prototype for evidence-grounded scientific literature screening. It is intentionally simple: the Streamlit UI is a thin wrapper around the core pipeline, with no PDF parsing, no vector database, and no mandatory LLM key.

The MVP pipeline:

1. plans academic search queries from a research question,
2. retrieves metadata from OpenAlex and Semantic Scholar,
3. normalizes and deduplicates papers,
4. extracts claim-level evidence from abstracts,
5. verifies whether evidence is grounded in the abstract,
6. ranks papers with a transparent scoring formula,
7. optionally applies human feedback,
8. writes CSV, JSON, and Markdown outputs.

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

Optional API keys can be supplied through environment variables:

```bash
cp .env.example .env
```

The package reads environment variables from the process environment. If you use
`.env`, load it with your shell or preferred environment manager before running
the CLI.

The rule-based core pipeline works without `DEEPSEEK_API_KEY`.

## Optional DeepSeek LLM Enhancement

The default pipeline is fully rule-based. You can optionally use an
OpenAI-compatible LLM backend for query planning, evidence extraction, and/or
verification. DeepSeek support is built in through `DEEPSEEK_API_KEY`.

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
  --feedback examples/human_feedback.csv \
  --gold-labels examples/gold_labels.csv \
  --output-dir outputs
```

## Run The UI

```bash
streamlit run app.py
```

The UI calls the same core pipeline functions as the CLI. Feedback changes are
applied to the in-memory ranking and do not rerun provider API calls.

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
- `outputs/raw_openalex_results.json`
- `outputs/raw_semantic_scholar_results.json`
- `outputs/merged_papers.csv`
- `outputs/evidence_table.csv`
- `outputs/ranked_papers_before_feedback.csv`
- `outputs/ranked_papers_after_feedback.csv`, when feedback is provided
- `outputs/ranked_papers.csv`
- `outputs/evaluation.json`
- `outputs/report.md`

Raw cache files are stored under `data/cache/` and ignored by git.

## Scoring

```text
final_score =
0.40 * relevance_score
+ 0.25 * evidence_score
+ 0.15 * recency_score
+ 0.15 * quality_score
+ 0.05 * diversity_score
+ human_feedback_adjustment
```

Base scores are clamped to `[0, 1]`. Human feedback is an explicit additive adjustment.

## Test

```bash
pytest
```

The tests use fake retrievers for pipeline behavior, so they do not require internet access or API keys.
