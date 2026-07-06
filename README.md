# Human-in-the-loop Multi-agent LLM System for Scientific Literature Screening

This repository is a lightweight, reproducible research prototype for evidence-grounded scientific literature screening. It is intentionally simple: the Streamlit UI is a thin wrapper around the core pipeline, with no PDF parsing, no vector database, and no mandatory LLM key.

The MVP pipeline:

1. builds a structured, provider-aware query plan from a research question,
2. retrieves metadata from OpenAlex and Semantic Scholar using provider-specific queries,
3. normalizes and deduplicates papers,
4. extracts claim-level evidence from abstracts,
5. verifies whether evidence is grounded in the abstract with strict span validation,
6. reranks papers with hybrid TF-IDF/API/field relevance plus transparent scoring,
7. optionally applies human feedback,
8. writes CSV, JSON, Markdown outputs, and an agent decision trace.

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
  or edit `core_terms`, `must_terms`, `optional_terms`, `exclude_terms`, OpenAlex
  queries, and Semantic Scholar queries, then click `Step 4: Run Retrieval`.
  The preview step does not call literature provider APIs, which helps avoid
  spending requests on a search direction that does not match the user's intent.
- Search mode controls for strictness, OpenAlex mode, sort preference, and ranking
  profile (`relevance_first`, `balanced`, `high_quality_review`).
- A collapsible run-status panel shows what the pipeline is doing during
  screening: retrieval by provider/query, deduplication, evidence extraction,
  grounding verification, ranking, evaluation, and artifact writing.
- Runtime API key entry for `OPENALEX_API_KEY`, `S2_API_KEY`, and `DEEPSEEK_API_KEY`.
  Keys are applied to the current Streamlit process and are not written into project files.
- Adjustable scoring weights for relevance, evidence, recency, quality, and diversity,
  with tooltip explanations for how each weight affects ranking.
- Year filtering through the `From year` setting.

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
- `outputs/agent_trace.json`
- `outputs/retrieval_diagnostics.json`
- `outputs/report.md`

Raw cache files are stored under `data/cache/` and ignored by git.

## Query Planning And Scoring

The planner writes a structured `QueryPlan` into `planned_queries.json`, including
topic terms, provider-specific queries, and search controls. OpenAlex queries use
quoted multi-word core terms plus boolean-style `AND` / `OR` / `NOT` where useful.
Semantic Scholar queries use quoted phrases, `+required` terms, `-excluded` terms,
and OR alternatives.

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

Evidence score combines grounding and relevance:

```text
evidence_score =
0.60 * verifier_confidence
+ 0.40 * evidence_question_relevance
```

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
