"""Streamlit UI for the literature-screening pipeline."""

from __future__ import annotations

import csv
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from lit_screening.agents.human_feedback import HumanFeedbackAgent
from lit_screening.models import FeedbackRecord, PipelineResult, QueryPlan, RankedPaper
from lit_screening.pipeline import (
    apply_feedback_to_pipeline_result,
    plan_screening_queries,
    ranked_to_row,
    run_pipeline,
)


PROVIDER_OPTIONS = {
    "Both": ["openalex", "semantic_scholar"],
    "OpenAlex": ["openalex"],
    "Semantic Scholar": ["semantic_scholar"],
}
PROJECTS_DIR = Path("outputs/projects")
HISTORY_PATH = PROJECTS_DIR / "history.json"
DEFAULT_UI_WEIGHTS = {
    "relevance": 0.40,
    "evidence": 0.25,
    "recency": 0.15,
    "quality": 0.15,
    "diversity": 0.05,
}
WEIGHT_HELP = {
    "relevance": {
        "en": "How strongly the paper title, abstract, claim, and evidence overlap with the research question.",
        "zh": "论文标题、摘要、claim 和 evidence 与研究问题越相关，这项分数越高。",
    },
    "evidence": {
        "en": "How much verified evidence matters. Only strict abstract span matches receive full evidence credit.",
        "zh": "可信 evidence 的重要性。只有能在 abstract 中严格匹配的 evidence 才会拿到高分。",
    },
    "recency": {
        "en": "How much recent publication year matters in ranking.",
        "zh": "年份越近的论文是否应该被排得更靠前。",
    },
    "quality": {
        "en": "How much citation count and venue metadata matter as a rough quality signal.",
        "zh": "引用数和期刊/会议元数据作为粗略质量信号的权重。",
    },
    "diversity": {
        "en": "How much the ranker should avoid over-concentrating on the same venue.",
        "zh": "避免结果过度集中在同一 venue 的权重。",
    },
}
PIPELINE_STAGE_PROGRESS = {
    "planning": 5,
    "retrieval": 25,
    "dedup": 45,
    "extraction": 58,
    "verification": 72,
    "ranking": 84,
    "feedback": 88,
    "evaluation": 92,
    "artifacts": 96,
    "complete": 100,
}
PIPELINE_STAGE_LABELS = {
    "planning": "Query planning",
    "retrieval": "Literature retrieval",
    "dedup": "Metadata merge",
    "extraction": "Evidence extraction",
    "verification": "Evidence grounding",
    "ranking": "Ranking",
    "feedback": "Human feedback",
    "evaluation": "Evaluation",
    "artifacts": "Output artifacts",
    "complete": "Complete",
}


def ui_text(language: str, english: str, chinese: str) -> str:
    """Return UI text in the selected interface language."""

    return chinese if language == "中文" else english


def truncate_text(text: Any, max_length: int = 180) -> str:
    """Return a compact single-line display string."""

    value = str(text)
    return value if len(value) <= max_length else f"{value[: max_length - 3]}..."


def format_progress_details(details: dict[str, Any]) -> str:
    """Format callback details without exposing secrets or overwhelming the UI."""

    if not details:
        return ""
    parts = []
    for key, value in details.items():
        parts.append(f"{key}: {truncate_text(value)}")
    return " | ".join(parts)


def progress_percent(stage: str, message: str) -> int:
    """Map pipeline events to a stable progress percentage."""

    base = PIPELINE_STAGE_PROGRESS.get(stage, 10)
    if stage == "retrieval" and "returned" in message.lower():
        return min(base + 10, 40)
    if "finished" in message.lower() or "ready" in message.lower():
        return min(base + 8, 99)
    return base


def slugify(text: str, max_length: int = 48) -> str:
    """Create a filesystem-friendly project slug."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (slug or "screening-project")[:max_length]


def load_history() -> list[dict[str, Any]]:
    """Load saved screening project history."""

    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_history(history: list[dict[str, Any]]) -> None:
    """Persist saved screening project history."""

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def new_project_output_dir(question: str) -> Path:
    """Create a unique output directory for a screening project."""

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECTS_DIR / f"{run_id}-{slugify(question)}"


def save_project_manifest(
    result: PipelineResult,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Save one run as a reusable screening project manifest."""

    manifest = {
        "run_id": Path(result.output_dir).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "question": result.question,
        "output_dir": result.output_dir,
        "providers": settings["providers"],
        "max_per_query": settings["max_per_query"],
        "from_year": settings["from_year"],
        "llm_backend": settings["llm_backend"],
        "strictness": settings.get("strictness", "balanced"),
        "openalex_mode": settings.get("openalex_mode", "keyword+semantic"),
        "sort_preference": settings.get("sort_preference", "relevance"),
        "ranking_profile": settings.get("ranking_profile", "balanced"),
        "scoring_weights": result.scoring_weights,
        "merged_paper_count": result.merged_paper_count,
        "duplicate_count": result.duplicate_count,
        "ranked_papers_path": result.ranked_papers_path,
        "report_path": result.report_path,
        "evaluation_path": result.evaluation_path,
        "agent_trace_path": str(Path(result.output_dir) / "agent_trace.json"),
    }
    output_dir = Path(result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "project_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    history = [item for item in load_history() if item.get("run_id") != manifest["run_id"]]
    history.insert(0, manifest)
    save_history(history[:50])
    return manifest


def ranked_dataframe(ranked_papers: list[RankedPaper]) -> pd.DataFrame:
    """Convert ranked papers to a display-friendly dataframe."""

    rows = [ranked_to_row(item) for item in ranked_papers]
    if not rows:
        return pd.DataFrame()
    columns = [
        "rank",
        "final_score",
        "title",
        "year",
        "venue",
        "source_provider",
        "support_level",
        "span_match_type",
        "span_match_confidence",
        "strict_span_validated",
        "llm_invalid_evidence",
        "missing_abstract",
        "confidence",
        "paper_id",
    ]
    return pd.DataFrame(rows)[columns]


def file_bytes(path: str | Path) -> bytes | None:
    """Read a file for download if it exists."""

    candidate = Path(path)
    if not candidate.exists():
        return None
    return candidate.read_bytes()


def read_json_file(path: str | Path) -> dict[str, Any]:
    """Read JSON from disk with a safe fallback."""

    candidate = Path(path)
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def set_runtime_api_keys() -> None:
    """Apply user-entered API keys to the current Streamlit process only."""

    key_map = {
        "OPENALEX_API_KEY": st.session_state.get("openalex_api_key_input", ""),
        "S2_API_KEY": st.session_state.get("s2_api_key_input", ""),
        "DEEPSEEK_API_KEY": st.session_state.get("deepseek_api_key_input", ""),
    }
    for env_name, value in key_map.items():
        if value:
            os.environ[env_name] = value


def render_key_warnings(providers: list[str], llm_backend: str, language: str) -> None:
    """Show missing-key status without exposing key values."""

    if "openalex" in providers and not os.getenv("OPENALEX_API_KEY"):
        st.sidebar.info(
            ui_text(
                language,
                "OPENALEX_API_KEY is not set; OpenAlex will use public access.",
                "OPENALEX_API_KEY 未设置；OpenAlex 会使用公开访问。",
            )
        )
    if "semantic_scholar" in providers and not os.getenv("S2_API_KEY"):
        st.sidebar.warning(
            ui_text(
                language,
                "S2_API_KEY is not set; Semantic Scholar may be rate-limited.",
                "S2_API_KEY 未设置；Semantic Scholar 可能更容易限流。",
            )
        )
    if llm_backend == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        st.sidebar.warning(
            ui_text(
                language,
                "DEEPSEEK_API_KEY is not set; LLM modes will fall back to rules.",
                "DEEPSEEK_API_KEY 未设置；LLM 模式会退回规则模式。",
            )
        )


def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return pipeline settings."""

    language = st.sidebar.radio("Language / 界面语言", ["English", "中文"], horizontal=True)
    st.sidebar.header(ui_text(language, "Screening Settings", "筛选设置"))
    provider_label = st.sidebar.radio(ui_text(language, "Providers", "数据源"), list(PROVIDER_OPTIONS))
    max_per_query = st.sidebar.number_input(
        ui_text(language, "Max papers per query", "每个 query 最大论文数"),
        min_value=0,
        max_value=100,
        value=10,
        step=1,
    )
    from_year_value = st.sidebar.number_input(
        ui_text(language, "From year", "起始年份"),
        min_value=1900,
        max_value=2100,
        value=2020,
        step=1,
        help=ui_text(
            language,
            "Only retrieve papers published from this year onward when the provider supports year filtering.",
            "只检索该年份及之后发表的论文；具体效果取决于数据源是否支持年份过滤。",
        ),
    )
    use_cache = st.sidebar.checkbox(ui_text(language, "Use cache", "使用缓存"), value=True)
    llm_backend = st.sidebar.selectbox("LLM backend", ["none", "deepseek"], index=0)

    with st.sidebar.expander(ui_text(language, "Search Mode", "检索模式"), expanded=True):
        strictness = st.selectbox(
            ui_text(language, "Strictness", "检索严格度"),
            ["strict", "balanced", "broad"],
            index=1,
            help=ui_text(
                language,
                "Strict uses fewer required terms, broad allows more alternatives.",
                "strict 更聚焦，broad 会放宽主题替代词。",
            ),
        )
        openalex_mode = st.selectbox(
            "OpenAlex mode",
            ["keyword", "semantic", "keyword+semantic"],
            index=2,
        )
        sort_preference = st.selectbox(
            ui_text(language, "Sort preference", "排序偏好"),
            ["relevance", "recent", "cited"],
            index=0,
        )
        ranking_profile = st.selectbox(
            ui_text(language, "Ranking profile", "排序配置"),
            ["relevance_first", "balanced", "high_quality_review"],
            index=1,
        )

    with st.sidebar.expander(ui_text(language, "API Keys", "API 设置")):
        st.text_input(
            "OPENALEX_API_KEY",
            type="password",
            key="openalex_api_key_input",
            help=ui_text(language, "Optional for OpenAlex.", "OpenAlex 可选。"),
        )
        st.text_input(
            "S2_API_KEY",
            type="password",
            key="s2_api_key_input",
            help=ui_text(language, "Recommended for Semantic Scholar rate limits.", "建议填写，可降低 Semantic Scholar 限流风险。"),
        )
        st.text_input(
            "DEEPSEEK_API_KEY",
            type="password",
            key="deepseek_api_key_input",
            help=ui_text(language, "Required only when using DeepSeek LLM modes.", "仅在使用 DeepSeek LLM 模式时需要。"),
        )
        set_runtime_api_keys()
        st.caption(ui_text(language, "Keys are applied to this Streamlit process and are not saved to project files.", "Key 只写入当前 Streamlit 进程，不保存到项目文件。"))

    with st.sidebar.expander("LLM Agent Modes"):
        default_mode = "rule" if llm_backend == "none" else "llm"
        planner_mode = st.selectbox(
            "Planner",
            ["rule", "llm"],
            index=["rule", "llm"].index(default_mode),
        )
        extractor_mode = st.selectbox(
            "Extractor",
            ["rule", "llm"],
            index=["rule", "llm"].index(default_mode),
        )
        verifier_mode = st.selectbox(
            "Verifier",
            ["rule", "llm"],
            index=["rule", "llm"].index(default_mode),
        )

    with st.sidebar.expander(ui_text(language, "Scoring Weights", "评分权重")):
        weights = {
            "relevance": st.slider(
                "Relevance",
                0.0,
                1.0,
                DEFAULT_UI_WEIGHTS["relevance"],
                0.01,
                help=WEIGHT_HELP["relevance"]["zh" if language == "中文" else "en"],
            ),
            "evidence": st.slider(
                "Evidence",
                0.0,
                1.0,
                DEFAULT_UI_WEIGHTS["evidence"],
                0.01,
                help=WEIGHT_HELP["evidence"]["zh" if language == "中文" else "en"],
            ),
            "recency": st.slider(
                "Recency",
                0.0,
                1.0,
                DEFAULT_UI_WEIGHTS["recency"],
                0.01,
                help=WEIGHT_HELP["recency"]["zh" if language == "中文" else "en"],
            ),
            "quality": st.slider(
                "Quality",
                0.0,
                1.0,
                DEFAULT_UI_WEIGHTS["quality"],
                0.01,
                help=WEIGHT_HELP["quality"]["zh" if language == "中文" else "en"],
            ),
            "diversity": st.slider(
                "Diversity",
                0.0,
                1.0,
                DEFAULT_UI_WEIGHTS["diversity"],
                0.01,
                help=WEIGHT_HELP["diversity"]["zh" if language == "中文" else "en"],
            ),
        }
        weight_sum = sum(weights.values())
        st.caption(
            ui_text(
                language,
                f"Current weight sum: {weight_sum:.2f}. Scores use these values directly.",
                f"当前权重总和：{weight_sum:.2f}。系统会直接使用这些权重计算 final_score。",
            )
        )

    providers = PROVIDER_OPTIONS[provider_label]
    render_key_warnings(providers, llm_backend, language)

    return {
        "language": language,
        "providers": providers,
        "max_per_query": int(max_per_query),
        "from_year": int(from_year_value),
        "use_cache": bool(use_cache),
        "llm_backend": llm_backend,
        "planner_mode": planner_mode,
        "extractor_mode": extractor_mode,
        "verifier_mode": verifier_mode,
        "scoring_weights": weights,
        "strictness": strictness,
        "openalex_mode": openalex_mode,
        "sort_preference": sort_preference,
        "ranking_profile": ranking_profile,
    }


def render_history_sidebar(language: str) -> None:
    """Render saved screening project history."""

    st.sidebar.header(ui_text(language, "Run History", "运行历史"))
    history = load_history()
    if not history:
        st.sidebar.caption(ui_text(language, "No saved projects yet.", "还没有保存的项目。"))
        return

    labels = [
        f"{item.get('created_at', '')} | {item.get('question', '')[:42]}"
        for item in history
    ]
    selected_label = st.sidebar.selectbox(ui_text(language, "Saved projects", "已保存项目"), labels)
    selected_index = labels.index(selected_label)
    selected = history[selected_index]
    if st.sidebar.button(ui_text(language, "Open Saved Project", "打开保存项目")):
        st.session_state.saved_project = selected
        st.session_state.pipeline_result = None


def ensure_state() -> None:
    """Initialize Streamlit session state keys."""

    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("feedback_records", {})
    st.session_state.setdefault("active_manifest", None)
    st.session_state.setdefault("saved_project", None)
    st.session_state.setdefault("query_preview", None)
    st.session_state.setdefault("query_preview_signature", None)
    st.session_state.setdefault("query_editor_text", "")
    st.session_state.setdefault("core_terms_text", "")
    st.session_state.setdefault("must_terms_text", "")
    st.session_state.setdefault("optional_terms_text", "")
    st.session_state.setdefault("exclude_terms_text", "")
    st.session_state.setdefault("openalex_queries_text", "")
    st.session_state.setdefault("semantic_scholar_queries_text", "")


def query_preview_signature(question: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Return the settings that affect query generation."""

    return {
        "question": question,
        "llm_backend": settings["llm_backend"],
        "planner_mode": settings["planner_mode"],
        "strictness": settings["strictness"],
        "openalex_mode": settings["openalex_mode"],
        "sort_preference": settings["sort_preference"],
        "ranking_profile": settings["ranking_profile"],
    }


def generate_query_preview(question: str, settings: dict[str, Any]) -> None:
    """Generate planned queries without running retrieval."""

    with st.spinner("Planning search queries..."):
        plan = plan_screening_queries(
            question=question,
            llm_backend=settings["llm_backend"],
            planner_mode=settings["planner_mode"],
            strictness=settings["strictness"],
            openalex_mode=settings["openalex_mode"],
            sort_preference=settings["sort_preference"],
            ranking_profile=settings["ranking_profile"],
        )
    st.session_state.query_preview = plan
    st.session_state.query_preview_signature = query_preview_signature(question, settings)
    st.session_state.query_editor_text = "\n".join(plan["queries"])
    query_plan = plan["query_plan"]
    st.session_state.core_terms_text = "\n".join(query_plan.core_terms)
    st.session_state.must_terms_text = "\n".join(query_plan.must_terms)
    st.session_state.optional_terms_text = "\n".join(query_plan.optional_terms)
    st.session_state.exclude_terms_text = "\n".join(query_plan.exclude_terms)
    st.session_state.openalex_queries_text = "\n".join(query_plan.openalex_queries)
    st.session_state.semantic_scholar_queries_text = "\n".join(
        query_plan.semantic_scholar_queries
    )
    st.session_state.pipeline_result = None
    st.session_state.saved_project = None


def parse_query_editor_text(text: str) -> list[str]:
    """Parse one-query-per-line text into a unique query list."""

    queries: list[str] = []
    for line in text.splitlines():
        cleaned = " ".join(line.split())
        if cleaned and cleaned not in queries:
            queries.append(cleaned)
    return queries


def preview_query_plan(preview: dict[str, Any]) -> QueryPlan:
    """Return a QueryPlan object from a preview payload."""

    value = preview.get("query_plan")
    if isinstance(value, QueryPlan):
        return value
    value = value or {}
    return QueryPlan(
        original_question=value.get("original_question", preview.get("question", "")),
        detected_language=value.get("detected_language", "en"),
        translated_question=value.get("translated_question", ""),
        core_terms=list(value.get("core_terms", [])),
        must_terms=list(value.get("must_terms", [])),
        optional_terms=list(value.get("optional_terms", [])),
        exclude_terms=list(value.get("exclude_terms", [])),
        openalex_queries=list(value.get("openalex_queries", [])),
        semantic_scholar_queries=list(value.get("semantic_scholar_queries", [])),
        filters=dict(value.get("filters", {})),
    )


def edited_query_plan(preview: dict[str, Any], settings: dict[str, Any]) -> QueryPlan:
    """Build a QueryPlan from the current editable UI fields."""

    base = preview_query_plan(preview)
    return QueryPlan(
        original_question=base.original_question,
        detected_language=base.detected_language,
        translated_question=base.translated_question,
        core_terms=parse_query_editor_text(st.session_state.core_terms_text),
        must_terms=parse_query_editor_text(st.session_state.must_terms_text),
        optional_terms=parse_query_editor_text(st.session_state.optional_terms_text),
        exclude_terms=parse_query_editor_text(st.session_state.exclude_terms_text),
        openalex_queries=parse_query_editor_text(st.session_state.openalex_queries_text),
        semantic_scholar_queries=parse_query_editor_text(
            st.session_state.semantic_scholar_queries_text
        ),
        filters={
            **base.filters,
            "strictness": settings["strictness"],
            "openalex_mode": settings["openalex_mode"],
            "sort_preference": settings["sort_preference"],
            "ranking_profile": settings["ranking_profile"],
        },
    )


def run_screening(
    question: str,
    settings: dict[str, Any],
    planned_queries: list[str] | None = None,
    query_plan: QueryPlan | None = None,
    planner_metadata: dict[str, Any] | None = None,
) -> None:
    """Run the core pipeline and store the result in session state."""

    output_dir = new_project_output_dir(question)
    language = settings["language"]
    status_label = ui_text(
        language,
        "Running literature screening...",
        "正在运行文献筛选...",
    )
    with st.status(status_label, expanded=True) as status:
        progress_bar = st.progress(0)

        def progress_callback(stage: str, message: str, details: dict[str, Any]) -> None:
            stage_label = PIPELINE_STAGE_LABELS.get(stage, stage.title())
            progress_bar.progress(progress_percent(stage, message))
            st.write(f"**{stage_label}**: {message}")
            detail_text = format_progress_details(details)
            if detail_text:
                st.caption(detail_text)

        try:
            result = run_pipeline(
                question=question,
                providers=settings["providers"],
                max_per_query=settings["max_per_query"],
                from_year=settings["from_year"],
                output_dir=str(output_dir),
                use_cache=settings["use_cache"],
                llm_backend=settings["llm_backend"],
                planner_mode=settings["planner_mode"],
                extractor_mode=settings["extractor_mode"],
                verifier_mode=settings["verifier_mode"],
                scoring_weights=settings["scoring_weights"],
                planned_queries_override=planned_queries,
                query_plan_override=query_plan,
                planner_metadata_override=planner_metadata,
                strictness=settings["strictness"],
                openalex_mode=settings["openalex_mode"],
                sort_preference=settings["sort_preference"],
                ranking_profile=settings["ranking_profile"],
                progress_callback=progress_callback,
            )
        except Exception:
            status.update(
                label=ui_text(
                    language,
                    "Literature screening failed",
                    "文献筛选失败",
                ),
                state="error",
                expanded=True,
            )
            raise
        progress_bar.progress(100)
        status.update(
            label=ui_text(
                language,
                "Literature screening complete",
                "文献筛选完成",
            ),
            state="complete",
            expanded=False,
        )
    st.session_state.pipeline_result = result
    st.session_state.feedback_records = {}
    st.session_state.active_manifest = save_project_manifest(result, settings)
    st.session_state.saved_project = None


def feedback_records_to_csv(records: dict[str, FeedbackRecord]) -> str:
    """Serialize feedback records as CSV text."""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["paper_id", "label", "adjustment", "note"])
    writer.writeheader()
    for record in records.values():
        writer.writerow(
            {
                "paper_id": record.paper_id,
                "label": record.label,
                "adjustment": record.adjustment,
                "note": record.note,
            }
        )
    return buffer.getvalue()


def parse_feedback_csv(uploaded_file: Any) -> dict[str, FeedbackRecord]:
    """Parse uploaded feedback CSV into feedback records."""

    if uploaded_file is None:
        return {}
    text = uploaded_file.getvalue().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    feedback_agent = HumanFeedbackAgent()
    records: dict[str, FeedbackRecord] = {}
    for row in reader:
        paper_id = (row.get("paper_id") or "").strip()
        if not paper_id:
            continue
        label = (row.get("label") or "uncertain").strip().lower()
        if label not in {"include", "exclude", "uncertain"}:
            label = "uncertain"
        adjustment_text = (row.get("adjustment") or "").strip()
        adjustment = (
            float(adjustment_text)
            if adjustment_text
            else feedback_agent.default_adjustment(label)
        )
        records[paper_id] = FeedbackRecord(
            paper_id=paper_id,
            label=label,
            adjustment=adjustment,
            note=row.get("note") or "",
        )
    return records


def apply_feedback_records(result: PipelineResult, records: dict[str, FeedbackRecord]) -> None:
    """Apply feedback records to the active project."""

    st.session_state.feedback_records = records
    updated = apply_feedback_to_pipeline_result(result, records)
    st.session_state.pipeline_result = updated
    if st.session_state.active_manifest:
        st.session_state.active_manifest = save_project_manifest(
            updated,
            {
                "providers": st.session_state.active_manifest.get("providers", []),
                "max_per_query": st.session_state.active_manifest.get("max_per_query", 0),
                "from_year": st.session_state.active_manifest.get("from_year", 0),
                "llm_backend": st.session_state.active_manifest.get("llm_backend", "none"),
                "strictness": st.session_state.active_manifest.get("strictness", "balanced"),
                "openalex_mode": st.session_state.active_manifest.get(
                    "openalex_mode",
                    "keyword+semantic",
                ),
                "sort_preference": st.session_state.active_manifest.get(
                    "sort_preference",
                    "relevance",
                ),
                "ranking_profile": st.session_state.active_manifest.get(
                    "ranking_profile",
                    "balanced",
                ),
                "scoring_weights": result.scoring_weights,
            },
        )


def render_feedback_tools(result: PipelineResult, language: str) -> None:
    """Render CSV import/export feedback controls."""

    st.subheader("Feedback CSV")
    uploaded = st.file_uploader(
        ui_text(language, "Import feedback CSV", "导入 feedback CSV"),
        type=["csv"],
        help=ui_text(
            language,
            "Expected columns: paper_id,label,adjustment,note",
            "需要列：paper_id,label,adjustment,note",
        ),
    )
    columns = st.columns(2)
    with columns[0]:
        if st.button(ui_text(language, "Apply Imported Feedback", "应用导入的反馈")):
            records = parse_feedback_csv(uploaded)
            apply_feedback_records(result, records)
            st.success(
                ui_text(
                    language,
                    f"Imported {len(records)} feedback records without rerunning retrieval.",
                    f"已导入 {len(records)} 条反馈，未重新请求 API。",
                )
            )
    with columns[1]:
        csv_text = feedback_records_to_csv(st.session_state.feedback_records)
        st.download_button(
            ui_text(language, "Export feedback CSV", "导出 feedback CSV"),
            data=csv_text,
            file_name="feedback.csv",
            mime="text/csv",
        )


def render_manual_feedback(selected: RankedPaper, result: PipelineResult, language: str) -> None:
    """Render feedback controls and rerank through core pipeline helpers."""

    feedback_agent = HumanFeedbackAgent()
    existing = st.session_state.feedback_records.get(selected.paper.paper_id)
    labels = ["include", "exclude", "uncertain"]
    default_label = existing.label if existing else "uncertain"
    label = st.radio(
        ui_text(language, "Feedback label", "反馈标签"),
        labels,
        index=labels.index(default_label),
        horizontal=True,
        key=f"feedback_label_{selected.paper.paper_id}",
    )
    note = st.text_area(
        ui_text(language, "Feedback note", "反馈备注"),
        value=existing.note if existing else "",
        key=f"feedback_note_{selected.paper.paper_id}",
        height=90,
    )

    if st.button(ui_text(language, "Apply Feedback And Rerank", "应用反馈并重新排序"), type="primary"):
        feedback_record = FeedbackRecord(
            paper_id=selected.paper.paper_id,
            label=label,
            adjustment=feedback_agent.default_adjustment(label),
            note=note,
        )
        updated_feedback = dict(st.session_state.feedback_records)
        updated_feedback[selected.paper.paper_id] = feedback_record
        apply_feedback_records(result, updated_feedback)
        st.success(ui_text(language, "Feedback applied without rerunning retrieval.", "反馈已应用，未重新请求 API。"))


def render_planned_queries(queries: list[str]) -> None:
    """Render planned queries as a clean table instead of raw JSON."""

    st.dataframe(
        pd.DataFrame({"query": queries}),
        use_container_width=True,
        hide_index=True,
    )


def render_question_preprocessing(
    question: str,
    planner_metadata: dict[str, Any],
    language: str,
) -> None:
    """Show how the research question was prepared for English retrieval."""

    planning_question = planner_metadata.get("planning_question") or question
    translated_question = planner_metadata.get("translated_question") or ""
    translation_mode = planner_metadata.get("translation_mode") or "none"
    question_language = planner_metadata.get("question_language") or "en_or_other"
    warning = planner_metadata.get("translation_warning") or ""

    rows = [
        {
            "field": "original_question",
            "value": question,
        },
        {
            "field": "planning_question",
            "value": planning_question,
        },
        {
            "field": "question_language",
            "value": question_language,
        },
        {
            "field": "translation_mode",
            "value": translation_mode,
        },
    ]
    if translated_question:
        rows.insert(
            2,
            {
                "field": "translated_question",
                "value": translated_question,
            },
        )
    st.caption(
        ui_text(
            language,
            "The pipeline uses the English planning question for retrieval, evidence extraction, and ranking.",
            "系统会用英文 planning question 进行检索、evidence 抽取和排序。",
        )
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if warning:
        st.info(
            ui_text(
                language,
                f"Translation note: {warning}",
                f"翻译提示：{warning}",
            )
        )


def render_query_preview_editor(
    question: str,
    settings: dict[str, Any],
    language: str,
) -> None:
    """Render the human checkpoint between planning and retrieval."""

    preview = st.session_state.query_preview
    if not preview:
        return

    current_signature = query_preview_signature(question, settings)
    preview_signature = st.session_state.query_preview_signature
    if preview_signature != current_signature:
        st.warning(
            ui_text(
                language,
                "The query preview was generated for an older question or planner setting. Refresh queries before running search.",
                "当前 query preview 来自旧问题或旧 planner 设置。正式检索前请先重新生成 queries。",
            )
        )

    st.subheader("Step 3: Review Query Plan")
    render_question_preprocessing(
        preview.get("question", question),
        preview.get("planner_metadata", {}),
        language,
    )
    term_cols = st.columns(4)
    with term_cols[0]:
        st.text_area("core_terms", key="core_terms_text", height=140)
    with term_cols[1]:
        st.text_area("must_terms", key="must_terms_text", height=140)
    with term_cols[2]:
        st.text_area("optional_terms", key="optional_terms_text", height=140)
    with term_cols[3]:
        st.text_area("exclude_terms", key="exclude_terms_text", height=140)

    query_cols = st.columns(2)
    with query_cols[0]:
        st.text_area(
            "OpenAlex queries",
            key="openalex_queries_text",
            height=180,
            help=ui_text(
                language,
                "One OpenAlex query per line. Multi-word core terms should usually stay quoted.",
                "每行一条 OpenAlex query。多词核心术语通常建议保留引号。",
            ),
        )
    with query_cols[1]:
        st.text_area(
            "Semantic Scholar queries",
            key="semantic_scholar_queries_text",
            height=180,
            help=ui_text(
                language,
                "One Semantic Scholar query per line. Required terms can use + and excluded terms can use -.",
                "每行一条 Semantic Scholar query。必选词可以用 +，排除词可以用 -。",
            ),
        )
    openalex_count = len(parse_query_editor_text(st.session_state.openalex_queries_text))
    semantic_count = len(
        parse_query_editor_text(st.session_state.semantic_scholar_queries_text)
    )
    st.caption(
        ui_text(
            language,
            f"{openalex_count} OpenAlex queries and {semantic_count} Semantic Scholar queries ready. No provider API calls have been made at this preview step.",
            f"当前有 {openalex_count} 条 OpenAlex queries、{semantic_count} 条 Semantic Scholar queries。预览阶段还没有请求文献 API。",
        )
    )


def render_downloads(ranked_path: str, report_path: str, language: str) -> None:
    """Render artifact download buttons."""

    ranked_bytes = file_bytes(ranked_path)
    report_bytes = file_bytes(report_path)
    download_cols = st.columns(2)
    with download_cols[0]:
        if ranked_bytes is not None:
            st.download_button(
                ui_text(language, "Download ranked_papers.csv", "下载 ranked_papers.csv"),
                data=ranked_bytes,
                file_name="ranked_papers.csv",
                mime="text/csv",
            )
    with download_cols[1]:
        if report_bytes is not None:
            st.download_button(
                ui_text(language, "Download report.md", "下载 report.md"),
                data=report_bytes,
                file_name="report.md",
                mime="text/markdown",
            )


def render_result(result: PipelineResult, language: str) -> None:
    """Render active project results."""

    st.caption(ui_text(language, f"Project folder: `{result.output_dir}`", f"项目目录：`{result.output_dir}`"))
    metric_columns = st.columns(4)
    metric_columns[0].metric("Merged papers", result.merged_paper_count)
    metric_columns[1].metric("Duplicates", result.duplicate_count)
    metric_columns[2].metric(
        "Strict evidence",
        result.evaluation_metrics.get("strict_supported_count", 0),
    )
    metric_columns[3].metric(
        "Weak support",
        result.evaluation_metrics.get("weak_support_count", 0),
    )

    tabs = st.tabs(["Queries", "Ranked Papers", "Evidence", "Feedback", "Trace", "Metrics"])
    with tabs[0]:
        planner_metadata = result.agent_trace.get("planner", {}).get("metadata", {})
        render_question_preprocessing(result.question, planner_metadata, language)
        render_planned_queries(result.planned_queries)

    with tabs[1]:
        if result.merged_paper_count == 0:
            st.warning(
                ui_text(
                    language,
                    "No papers were retrieved. Try increasing max papers per query, changing providers, or checking API/network access.",
                    "没有检索到论文。可以尝试增加每个 query 的论文数、切换数据源，或检查 API/网络。",
                )
            )
        table = ranked_dataframe(result.ranked_final)
        st.dataframe(table, use_container_width=True, hide_index=True)
        render_downloads(result.ranked_papers_path, result.report_path, language)

    selected: RankedPaper | None = None
    if result.ranked_final:
        options = {
            f"{item.rank}. {item.paper.title[:100]}": item
            for item in result.ranked_final
        }
        selected_label = st.selectbox(ui_text(language, "Selected paper", "选择论文"), list(options))
        selected = options[selected_label]

    with tabs[2]:
        if selected is None:
            st.info(ui_text(language, "Run a screening project and select a paper to inspect evidence.", "先运行一个筛选项目，并选择一篇论文查看 evidence。"))
        else:
            evidence_table = pd.DataFrame(
                [
                    {
                        "paper_id": selected.paper.paper_id,
                        "claim": selected.evidence.claim,
                        "evidence_sentence": selected.evidence.evidence_sentence,
                        "support_level": selected.verification.support_level,
                        "span_match_type": selected.verification.span_match_type,
                        "span_match_confidence": selected.verification.span_match_confidence,
                        "strict_span_validated": selected.verification.support_level == "strict_support",
                        "llm_invalid_evidence": selected.verification.support_level == "llm_invalid_evidence",
                        "missing_abstract": selected.verification.support_level == "missing_abstract",
                        "matched_text": selected.verification.matched_text,
                        "rationale": selected.verification.rationale,
                    }
                ]
            )
            st.dataframe(evidence_table, use_container_width=True, hide_index=True)

    with tabs[3]:
        render_feedback_tools(result, language)
        if selected is not None:
            render_manual_feedback(selected, result, language)

    with tabs[4]:
        st.json(result.agent_trace)

    with tabs[5]:
        st.json(result.evaluation_metrics)


def render_saved_project(manifest: dict[str, Any], language: str) -> None:
    """Render a saved project summary from disk artifacts."""

    st.subheader(ui_text(language, "Saved Project", "已保存项目"))
    st.caption(ui_text(language, f"Project folder: `{manifest.get('output_dir', '')}`", f"项目目录：`{manifest.get('output_dir', '')}`"))
    st.write(manifest.get("question", ""))

    output_dir = Path(manifest.get("output_dir", ""))
    planned = read_json_file(output_dir / "planned_queries.json")
    ranked_path = output_dir / "ranked_papers.csv"
    evaluation = read_json_file(output_dir / "evaluation.json")
    trace = read_json_file(output_dir / "agent_trace.json")

    tabs = st.tabs(["Queries", "Ranked Papers", "Trace", "Metrics", "Downloads"])
    with tabs[0]:
        metadata = planned.get("planner_metadata") or planned.get("llm", {})
        render_question_preprocessing(planned.get("question", ""), metadata, language)
        render_planned_queries(planned.get("queries", []))
    with tabs[1]:
        if ranked_path.exists():
            st.dataframe(pd.read_csv(ranked_path), use_container_width=True)
        else:
            st.warning(ui_text(language, "Saved ranked_papers.csv is missing.", "保存的 ranked_papers.csv 不存在。"))
    with tabs[2]:
        st.json(trace)
    with tabs[3]:
        st.json(evaluation)
    with tabs[4]:
        render_downloads(str(ranked_path), str(output_dir / "report.md"), language)


def main() -> None:
    """Streamlit app entrypoint."""

    st.set_page_config(page_title="Literature Screening Project", layout="wide")
    ensure_state()
    settings = render_sidebar()
    language = settings["language"]
    render_history_sidebar(language)

    st.title(ui_text(language, "Literature Screening Project Workspace", "文献筛选项目工作台"))
    st.caption(
        ui_text(
            language,
            "Evidence-grounded search, extraction, verification, ranking, and feedback.",
            "基于 evidence 的检索、抽取、验证、排序与人工反馈。",
        )
    )
    question = st.text_area(
        ui_text(language, "Research question", "研究问题"),
        value="",
        placeholder=ui_text(
            language,
            "Example: the significance of surface magnetization",
            "例如：the significance of surface magnetization",
        ),
        height=110,
    )

    cleaned_question = question.strip()
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button(ui_text(language, "Step 2: Generate Query Plan", "第 2 步：生成 Query Plan"), type="primary"):
            if not cleaned_question:
                st.error(ui_text(language, "Please enter a research question.", "请先输入研究问题。"))
            else:
                try:
                    generate_query_preview(cleaned_question, settings)
                except Exception as exc:
                    st.error(ui_text(language, f"Query planning failed: {exc}", f"Query 规划失败：{exc}"))

    preview_ready = bool(st.session_state.query_preview)
    preview_is_current = (
        st.session_state.query_preview_signature
        == query_preview_signature(cleaned_question, settings)
    )
    openalex_queries = parse_query_editor_text(st.session_state.openalex_queries_text)
    semantic_queries = parse_query_editor_text(st.session_state.semantic_scholar_queries_text)
    selected_query_count = 0
    if "openalex" in settings["providers"]:
        selected_query_count += len(openalex_queries)
    if "semantic_scholar" in settings["providers"]:
        selected_query_count += len(semantic_queries)
    with action_cols[1]:
        if st.button(
            ui_text(language, "Step 4: Run Retrieval", "第 4 步：开始检索"),
            disabled=not preview_ready or not preview_is_current or selected_query_count == 0,
        ):
            try:
                preview = st.session_state.query_preview or {}
                current_query_plan = edited_query_plan(preview, settings)
                run_screening(
                    cleaned_question,
                    settings,
                    query_plan=current_query_plan,
                    planner_metadata=preview.get("planner_metadata", {}),
                )
            except Exception as exc:
                st.error(ui_text(language, f"Screening failed: {exc}", f"筛选失败：{exc}"))

    render_query_preview_editor(cleaned_question, settings, language)

    result = st.session_state.pipeline_result
    saved_project = st.session_state.saved_project
    if result:
        render_result(result, language)
    elif saved_project:
        render_saved_project(saved_project, language)


if __name__ == "__main__":
    main()
