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
5. 可导入已有 BibTeX、RIS 或 CSV 文献库导出文件。
6. 可从已知种子论文出发，通过 references、citations 和 recommendations 做引用滚雪球扩展。
7. 规范化论文字段并按 DOI / 标题去重。
8. 从摘要中抽取 claim-level evidence。
9. 用 span validation 验证 evidence sentence 是否能在 abstract 中 exact match 或 high-confidence fuzzy match。
10. 将 unsupported、weak_support、strict_support、missing_abstract、llm_invalid_evidence 区分开。
11. 用混合相关性、证据质量、年份、引用质量、多样性和人工反馈计算排序。
12. 输出 CSV、JSON、Markdown report、agent trace、run events 和 retrieval diagnostics。
13. 在 Streamlit UI 中展示 query plan、ranking、evidence chain、反馈重排、run history 和项目输出。

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

也可以筛选已有文献库，例如 Zotero、Web of Science、Scopus、Google Scholar 或手工表格导出的文件：

```bash
python -m lit_screening.pipeline run \
  --question "surface magnetization boundary spin signals" \
  --providers openalex semantic_scholar \
  --max-per-query 10 \
  --input-file examples/my_library.bib \
  --input-format auto \
  --output-dir outputs
```

支持 BibTeX（`.bib` / `.bibtex`）、RIS（`.ris`）和 CSV（`.csv`）。CSV 常用列包括 `title`、`abstract`、`authors`、`year`、`venue`、`doi`、`url`、`citation_count`。

也可以从已知论文开始检索。Seed Paper Mode 支持 DOI、Semantic Scholar paper ID、OpenAlex ID 或论文标题。引用滚雪球默认关闭，只有显式加入 `--enable-snowballing` 时才会通过 Semantic Scholar 查询 references、citations 和 recommendations。如果没有手动提供 seed，系统会尽量从高置信度 top ranked papers 里自动选 seed。

功能性 citation snowballing 会使用 Semantic Scholar 的论文解析、references、citations 和 recommendations 接口。如果没有 `S2_API_KEY`，系统不会失败，而是保存 seed 记录用于审计，并安全跳过扩展。

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

Seed CSV 列为 `seed_id`、`seed_type`、`title`、`doi`、`note`。常用 `seed_type` 包括 `doi`、`semantic_scholar`、`openalex`、`title`。

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
5. 如已有文献库导出文件，可在 `Import Existing Literature Library` 上传 BibTeX、RIS 或 CSV。
6. 如果已有关键论文，可在 `Seed Paper Mode` 输入 DOI / 标题 / Semantic Scholar ID / OpenAlex ID，或上传 `seed_papers.csv`；是否执行引用滚雪球由侧边栏控制。
7. 确认 query 方向没偏后，再点击 `Run Retrieval`。
8. 查看 ranked papers、evidence chain、result groups、paper cards、metrics 和 trace。
9. 对论文标记 include / exclude / uncertain，导入或导出 feedback CSV，并在不重新调用 API 的情况下 rerank。

Step 3 是一个可折叠检查点，里面分为字段说明、研究意图、术语与 provider queries。第 4 步检索完成后，Step 3 会默认收起，但仍然可以展开查看本次实际使用的 query plan。

UI 还包含 `Research Whiteboard` tab，用来在正式阅读 ranked papers 前查看系统的研究理解过程：concept map、research lenses、query families、seed hints、query provenance、paper role distribution、evidence function distribution、research tensions 和 suggested next searches。缺失的 artifact 会显示 `Not available for this run.`，不会让 UI 报错。

注意：`From year` 只有在 `Apply year filter` 勾选后才生效。启用后，pipeline 会在 provider 返回结果之后再做一次本地硬过滤，早于该年份的论文不会进入 dedup、evidence extraction 或 ranking。缺少年份元数据的论文也会被排除，因为无法证明它满足年份条件。

OpenAlex mode 中的 `keyword`、`exact`、`semantic` 会分别映射到 OpenAlex 请求参数 `search`、`search.exact`、`search.semantic`。`keyword+semantic` 会拆成两次独立检索，并在 diagnostics 里记录为不同 retrieval stage。

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

## Research Sensemaking Mode

Research Sensemaking Mode 是现有筛选 pipeline 外层的解释层。它负责说明系统如何理解研究问题、如何拆概念、为什么开这些检索路线、哪些结论已经由摘要 span 支持、哪些只是由标题/query/规则推断出来的待验证线索。

它不会替换旧的 `QueryPlan`。默认检索仍然使用 `planned_queries.json`；只有显式开启 `use_query_families=True` 或 CLI 的 `--use-query-families` 时，`QueryFamilyPlan` 里的 query 才会参与检索。

### Domain Packs

Domain pack 是放在 `lit_screening/domain_packs/` 里的轻量 JSON 领域知识文件，不依赖 PyYAML，也不引入复杂 ontology。第一版领域包是 `materials_magnetism.json`，包含表面磁化、boundary magnetization、surface spin polarization、local magnetoelectric response、Cr2O3 / chromia、FeF2、NiO、CuMnAs、SPLEEM、XMCD-PEEM、SP-STM、NV magnetometry 等术语、材料、方法、应用和误检排除词。

开发时使用 `lit_screening.domain_packs.loader` 里的 `load_domain_pack("materials_magnetism")` 和 `list_domain_packs()`。新增 domain pack 应保持 JSON/dataclass 结构简单、可测试、不依赖网络或 API key。

### ResearchLens and QueryFamily

`ResearchLens` 表示研究者式调研视角，例如 `theory_origin`、`spaldin_framework`、`direct_surface_detection`、`nanoscale_readout`、`local_magnetoelectric_predictor`、`applications`、`limitations` 和 `frontier`。`ConceptMapper` 会把这些 lens 写入 `concept_map.json`。

`QueryFamily` 表示某个 lens 下为什么要开一组 provider query。例如 `direct_surface_detection` 会组织 SPLEEM、XMCD-PEEM、spin-resolved photoemission、SP-STM 相关检索；`nanoscale_readout` 会组织 NV magnetometry 和 scanning diamond magnetometry 相关检索。`QueryFamilyPlanner` 会写出 `query_families.json`。

默认情况下，query families 只作为解释 artifact，不改变检索。当开启 query-family 检索时，pipeline 会额外写出 `query_provenance.json`，记录每条 query 的 provider、source、family name、lens name 和 purpose。

### Seed hints

`SeedExtractionAgent` 会从用户问题中抽取轻量 seed hints，包括显式论文标题、DOI、arXiv ID 和 Nicola A. Spaldin / Spaldin 这样的作者提示，并写入 `seed_hints.json`。这些 seed hints 可以增强 research lenses 和 query families，但不会自动做 citation expansion；只有用户显式启用已有 Seed Paper Mode 和 snowballing flags 时才会滚雪球。

### Paper roles

`PaperRoleClassifier` 会基于 title、abstract、venue、year 和 query provenance 为论文分配研究角色，包括 `theory_origin`、`conceptual_framework`、`experimental_proof`、`surface_probe_method`、`nanoscale_readout`、`material_case`、`application_bridge`、`frontier_extension`、`limitation_or_challenge`、`review_background`。结果写入 `paper_roles.json`，用于 report 和 Research Whiteboard 按研究脉络组织论文，而不只是按 final score 排序。

### Research Process Report

Research Process Report 目前是 `report.md` 中的 `# Research Process` 章节；pipeline 不会单独生成 `research_process_report.md` 文件。该章节包括：

- research question interpretation
- concept decomposition
- search lenses and query families
- screening and inclusion criteria
- paper roles and why they matter
- research lineage
- controversies, limitations, and gaps
- missing keywords, methods, authors, or schools
- suggested next searches
- verified vs uncertain findings

如果某个 artifact 没有生成，report 会写 `Not generated in this run.`。报告不能编造 citation relation；没有真实 citation / snowballing artifact 支持时，会保守写明 citation relation not verified。

### Exploration quality metrics

`lit_screening/evaluation/exploration_quality.py` 新增 exploration-quality 指标，但不替代 precision@k、nDCG 等原有 ranking metrics。输出文件是 `exploration_quality.json`，包含 concept coverage、query family coverage、paper role diversity、seed hint utilization、evidence function diversity、gap specificity 和 research tension count。

### 材料磁学示例

可以用 Spaldin surface magnetization 问题做离线检查：

```bash
python -m lit_screening.pipeline run \
  --question "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets 和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order 相关的探测表面磁化和自旋极化重要性的文章" \
  --providers openalex semantic_scholar \
  --max-per-query 0 \
  --output-dir outputs/spaldin_surface_demo
```

预期可以看到这些 sensemaking outputs：

- `outputs/spaldin_surface_demo/concept_map.json`
- `outputs/spaldin_surface_demo/query_families.json`
- `outputs/spaldin_surface_demo/seed_hints.json`
- `outputs/spaldin_surface_demo/paper_roles.json`
- `outputs/spaldin_surface_demo/exploration_quality.json`
- `outputs/spaldin_surface_demo/report.md`，其中包含 Research Process 章节

因为 `--max-per-query 0` 不调用 provider，`paper_roles.json` 等依赖论文结果的 artifact 可能为空；但 concept map、seed hints、query-family rationale 和 report 结构应该仍然可用于审计。

## 输出文件

一次运行会生成：

```text
outputs/
  planned_queries.json
  search_brief.json
  search_contract.json
  ambiguity_analysis.json
  question_refinement.json
  concept_map.json
  query_families.json
  seed_hints.json
  query_provenance.json
  raw_openalex_results.json
  raw_semantic_scholar_results.json
  merged_papers.csv
  evidence_table.csv
  evidence_functions.json
  aspect_coverage.csv
  domain_assessments.json
  paper_roles.json
  research_tensions.json
  screening_decisions.csv
  screening_decisions.json
  preference_learning.json
  feedback_query_refinement.json
  seed_papers.json
  citation_expansion.csv
  retrieval_paths.csv
  method_comparison_matrix.csv
  method_comparison_matrix.md
  research_gap_matrix.csv
  research_gap_matrix.md
  suggested_next_searches.json
  suggested_next_searches.md
  ranked_papers_before_feedback.csv
  ranked_papers_after_feedback.csv
  ranked_papers.csv
  evaluation.json
  agent_trace.json
  run_events.jsonl
  imported_papers.csv
  import_diagnostics.json
  retrieval_diagnostics.json
  query_pilot_diagnostics.json
  query_repair_suggestions.json
  result_groups.json
  prisma_like_flow.json
  paper_cards.md
  reading_path.md
  report.md
  exploration_quality.json
```

其中：

- `run_events.jsonl`：运行过程日志，记录每个阶段、provider 错误和 fatal exception，方便分析为什么搜不到论文。
- `imported_papers.csv`：当导入外部文献库时，保存导入并规范化后的论文。
- `import_diagnostics.json`：当导入外部文献库时，保存导入格式、数量、跳过记录和错误信息。
- `retrieval_diagnostics.json`：记录 query plan、每个 provider 的 query、每个 query 返回多少、provider error、导入文献库数量、top titles、top score breakdown 和年份过滤审计。
- `search_contract.json`：记录系统理解的领域边界、必须包含概念、必须排除概念和 field-of-study guardrails。
- `ambiguity_analysis.json`：记录 `screening`、`agent`、`evidence` 等歧义词如何被解释，以及建议排除的错误含义。
- `query_pilot_diagnostics.json`、`query_repair_suggestions.json`：当运行 Pilot Search 时，记录小样本检索漂移、修复建议，以及是否应用修复后的 queries。
- `agent_trace.json`：记录 planner、retriever、extractor、verifier、ranker 的关键决策。
- `screening_decisions.csv/json`：给每篇论文输出 include / maybe / exclude、置信度、阅读优先级、建议动作和排除原因。
- `preference_learning.json`、`feedback_query_refinement.json`：记录从人工 include/exclude 反馈里学到的正/负 terms，以及下一轮检索建议。
- `concept_map.json`、`query_families.json`、`seed_hints.json`、`query_provenance.json`、`paper_roles.json`、`evidence_functions.json`、`research_tensions.json`：Research Sensemaking artifacts，供 Research Whiteboard 和 `report.md` 的 Research Process 章节使用。
- `seed_papers.json`、`citation_expansion.csv`、`retrieval_paths.csv`：记录种子论文、引用扩展得到的候选论文，以及每篇扩展论文是通过 reference、citation 还是 recommendation 进入候选集。
- `ranked_papers.csv`：最终排序结果，并包含 `retrieval_provider`、`retrieval_stage`、`retrieval_query`、`source_stage`、`seed_paper_id`、`seed_title`、`seed_reason` 等 provenance 字段。
- `method_comparison_matrix.*`、`research_gap_matrix.*`、`suggested_next_searches.*`：把排序结果整理成方法比较、研究空白和下一步检索建议。
- `report.md`：面向展示和复盘的 Markdown 报告。
- `exploration_quality.json`：记录 concept coverage、query family coverage、paper role diversity、seed hint utilization、evidence function diversity、gap specificity 和 research tension count。

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

代码对应：`lit_screening/reranking.py::compute_hybrid_relevance_features`。

证据分数：

```text
evidence_score =
0.60 * verifier_confidence
+ 0.40 * evidence_question_relevance
```

代码对应：`lit_screening/scoring.py::score_evidence`。

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

代码对应：`lit_screening/scoring.py::compute_score_breakdown` 是 `RankerAgent` 使用的主评分入口；`compute_final_score` 是组合已有分数组件的底层公式 helper。`score_paper` 只保留为向后兼容 alias。

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
- BibTeX / RIS / CSV 外部文献库导入
- seed_papers.csv 解析
- fake Semantic Scholar responses 下的 citation snowballing
- Seed Paper Mode 输出 `citation_expansion.csv`、`retrieval_paths.csv`、`seed_papers.json`
- scoring weights 和 ranking profile 确实改变排序
