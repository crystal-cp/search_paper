# 人在回路的多智能体 LLM 科学文献筛选系统

[English README](README.md)

这是一个轻量、可复现、面向研究展示的科学文献筛选原型。它的重点不是做商业产品，而是展示一条透明的文献筛选流水线：从研究问题出发，规划检索 query，调用 OpenAlex / Semantic Scholar，去重，抽取 evidence，验证 evidence 是否真的来自摘要，最后用可解释的评分函数排序，并支持人工反馈。

项目默认不依赖 LLM。没有 `DEEPSEEK_API_KEY` 时，系统会自动使用规则版 planner / extractor / verifier。

## 目前能做什么

核心 pipeline 会执行：

1. 将用户问题解释成 `SearchBrief`，识别检索意图、纳入/排除标准、需要覆盖的方面。
2. 生成结构化 `QueryPlan`，包含 `core_terms`、`must_terms`、`optional_terms`、`exclude_terms` 和不同 provider 的 query。
3. 对中文研究问题生成英文 planning question 和英文检索 query。
4. 调用 OpenAlex 和 Semantic Scholar 检索论文元数据。
5. 规范化论文字段并按 DOI / 标题去重。
6. 从摘要中抽取 claim-level evidence。
7. 用 span validation 验证 evidence sentence 是否能在 abstract 中 exact match 或 high-confidence fuzzy match。
8. 将 unsupported、weak_support、strict_support、missing_abstract、llm_invalid_evidence 区分开。
9. 用混合相关性、证据质量、年份、引用质量、多样性和人工反馈计算排序。
10. 输出 CSV、JSON、Markdown report、agent trace、run events 和 retrieval diagnostics。
11. 在 Streamlit UI 中展示 query plan、ranking、evidence chain、反馈重排、run history 和项目输出。

## 安装

建议使用独立环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ui.txt
```

如果使用 conda，也可以在已有环境里安装依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-ui.txt
```

## API Key

复制环境变量模板：

```bash
cp .env.example .env
```

支持的环境变量：

```text
OPENALEX_API_KEY=
S2_API_KEY=
DEEPSEEK_API_KEY=
```

说明：

- 当前 OpenAlex API 需要免费的 `OPENALEX_API_KEY`。
- Semantic Scholar 可以不填 `S2_API_KEY`，但容易遇到 rate limit，建议填写。
- DeepSeek 只在使用 LLM 模式时需要；没有 key 时系统会回退到规则流程。
- UI 中也可以临时填写 API key。key 只写入当前 Streamlit 进程，不保存到项目文件。

## 命令行运行

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
  --feedback examples/human_feedback.csv \
  --gold-labels examples/gold_labels.csv \
  --output-dir outputs
```

离线烟雾测试可以避免真实 API 调用：

```bash
python -m lit_screening.pipeline run \
  --question "How can human-in-the-loop systems improve literature screening?" \
  --providers openalex semantic_scholar \
  --max-per-query 0 \
  --output-dir outputs
```

## Streamlit UI

```bash
streamlit run app.py
```

UI 的推荐流程：

1. 在侧边栏选择 provider、max papers per query、是否使用 cache、LLM backend、检索模式、排序 profile 和 scoring weights。
2. 如果需要限制年份，勾选 `Apply year filter`，再设置 `From year`。
3. 在主页面输入 research question。
4. 点击 `Generate Query Plan`，先检查并可编辑 SearchBrief、core terms、must terms、exclude terms、OpenAlex queries 和 Semantic Scholar queries。
5. 确认 query 方向没偏后，再点击 `Run Retrieval`。
6. 查看 ranked papers、evidence chain、result groups、paper cards、metrics 和 trace。
7. 对论文标记 include / exclude / uncertain，导入或导出 feedback CSV，并在不重新调用 API 的情况下 rerank。

Step 3 是一个可折叠检查点，里面分为字段说明、研究意图、术语与 provider queries。第 4 步检索完成后，Step 3 会默认收起，但仍然可以展开查看本次实际使用的 query plan。

注意：`From year` 只有在 `Apply year filter` 勾选后才生效。启用后，pipeline 会在 provider 返回结果之后再做一次本地硬过滤，早于该年份的论文不会进入 dedup、evidence extraction 或 ranking。缺少年份元数据的论文也会被排除，因为无法证明它满足年份条件。

## Optional DeepSeek LLM

开启 DeepSeek：

```bash
export DEEPSEEK_API_KEY="your-key"

python -m lit_screening.pipeline run \
  --question "表面磁化的重要性" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --from-year 2020 \
  --llm-backend deepseek \
  --planner-mode llm \
  --extractor-mode llm \
  --verifier-mode llm \
  --output-dir outputs
```

如果 `DEEPSEEK_API_KEY` 缺失，系统不会失败，会记录 LLM inactive 并自动使用规则版 agents。

## 输出文件

一次运行会生成：

```text
outputs/
  planned_queries.json
  search_brief.json
  question_refinement.json
  raw_openalex_results.json
  raw_semantic_scholar_results.json
  merged_papers.csv
  evidence_table.csv
  aspect_coverage.csv
  ranked_papers_before_feedback.csv
  ranked_papers_after_feedback.csv
  ranked_papers.csv
  evaluation.json
  agent_trace.json
  run_events.jsonl
  retrieval_diagnostics.json
  result_groups.json
  prisma_like_flow.json
  paper_cards.md
  reading_path.md
  report.md
```

其中：

- `run_events.jsonl`：运行过程日志，记录每个阶段、provider 错误和 fatal exception，方便分析为什么搜不到论文。
- `retrieval_diagnostics.json`：记录 query plan、每个 provider 的 query、每个 query 返回多少、provider error、top titles、top score breakdown 和年份过滤审计。
- `agent_trace.json`：记录 planner、retriever、extractor、verifier、ranker 的关键决策。
- `ranked_papers.csv`：最终排序结果。
- `report.md`：面向展示和复盘的 Markdown 报告。

缓存文件写入 `data/cache/`，输出文件写入 `outputs/`。这些运行产物默认被 `.gitignore` 忽略。

## Evidence Audit

本项目的一个重点是：evidence 不能只由 LLM 或规则“声称支持”，必须能回到 abstract。

支持级别包括：

- `strict_support`：evidence sentence 在 abstract 中 exact match 或 high-confidence fuzzy match。
- `weak_support`：只达到较弱 overlap，不等同于严格引用。
- `unverified`：无法验证 evidence 来自 abstract。
- `missing_abstract`：没有摘要，不能支持 claim。
- `llm_invalid_evidence`：LLM 给出的 evidence 不能在 abstract 中匹配。

相关字段会写入 CSV、UI、report 和 trace：

```text
support_level
span_match_type
span_match_confidence
matched_text
strict_span_validated
llm_invalid_evidence
missing_abstract
```

## Scoring

混合相关性：

```text
hybrid_relevance_score =
0.30 * title_similarity
+ 0.25 * abstract_similarity
+ 0.15 * evidence_similarity
+ 0.10 * api_relevance_score
+ 0.10 * must_term_coverage
+ 0.10 * field_match_score
```

证据分数：

```text
evidence_score =
0.60 * verifier_confidence
+ 0.40 * evidence_question_relevance
```

最终分数：

```text
final_score =
0.40 * relevance_score
+ 0.25 * evidence_score
+ 0.15 * recency_score
+ 0.15 * quality_score
+ 0.05 * diversity_score
+ human_feedback_adjustment
```

UI 和 CLI 都支持调整权重。UI 中每个权重旁边有说明，方便展示这个权重如何影响论文排序。

## 测试

```bash
pytest
```

测试使用 fake retrievers 和 mocked LLM responses，不需要真实 API key 或网络。

当前重点测试包括：

- DOI 和 title 规范化
- 去重
- scoring formula 和用户权重
- feedback adjustment
- missing abstract 时 extractor 不 hallucinate
- verifier 对 missing evidence 的处理
- fake pipeline end-to-end
- query planner 不把 LLM-agent 词强行注入无关科学问题
- TF-IDF reranking
- retrieval diagnostics
- provider 错误日志
- 本地硬年份过滤
