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
import requests
import streamlit as st

from lit_screening.agents.human_feedback import HumanFeedbackAgent
from lit_screening.models import (
    FeedbackRecord,
    PipelineResult,
    QueryPlan,
    RankedPaper,
    SearchBrief,
)
from lit_screening.pipeline import (
    apply_feedback_to_pipeline_result,
    plan_screening_queries,
    ranked_to_row,
    run_pipeline,
)
from lit_screening.run_logging import ScreeningRunLogger


PROVIDER_OPTIONS = {
    "Both": ["openalex", "semantic_scholar"],
    "OpenAlex": ["openalex"],
    "Semantic Scholar": ["semantic_scholar"],
}
PROVIDER_HEALTH_URLS = {
    "openalex": "https://api.openalex.org/works",
    "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/search",
}
API_ENV_NAMES = ["OPENALEX_API_KEY", "S2_API_KEY", "DEEPSEEK_API_KEY"]
ORIGINAL_API_ENV_VALUES = {name: os.environ.get(name) for name in API_ENV_NAMES}
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
SEARCH_INTENTS = [
    "overview",
    "frontier",
    "implementation",
    "evidence_verification",
    "proposal",
    "systematic_review",
]
STEP3_FIELD_GUIDE = [
    {
        "field": "Refined English question",
        "en": "The English planning question used for retrieval, evidence extraction, and ranking.",
        "zh": "用于检索、evidence 抽取和排序的英文规划问题。",
        "edit": {
            "en": "Edit when the translation or topic focus is wrong.",
            "zh": "当翻译不准或主题焦点偏了时修改。",
        },
    },
    {
        "field": "Search intent",
        "en": "Tells the system whether this is an overview, frontier scan, implementation search, systematic review, and so on.",
        "zh": "告诉系统这是背景综述、前沿扫描、实现方法、系统综述等哪类检索。",
        "edit": {
            "en": "Change it when the paper type or ranking emphasis should change.",
            "zh": "当你希望论文类型或排序侧重点改变时修改。",
        },
    },
    {
        "field": "Time window",
        "en": "A human-readable time preference for the search brief. The hard year filter is controlled in the sidebar.",
        "zh": "检索意图中的时间偏好说明。真正硬性的年份过滤在侧边栏控制。",
        "edit": {
            "en": "Use it to describe recent-only, historical, or no-strict-window searches.",
            "zh": "用于说明只看近期、看历史脉络，或不严格限制时间。",
        },
    },
    {
        "field": "User goal",
        "en": "Plain-language goal that guides ranking and the final report.",
        "zh": "用户目标说明，会影响排序解释和最终 report。",
        "edit": {
            "en": "Edit when the search purpose is not captured.",
            "zh": "当系统没有理解你的检索目的时修改。",
        },
    },
    {
        "field": "inclusion_criteria",
        "en": "Topics or properties papers should include. These guide query terms and aspect coverage.",
        "zh": "希望论文包含的主题或性质，会影响 query 和 aspect coverage。",
        "edit": {
            "en": "Add one item per line for concepts that must stay in scope.",
            "zh": "每行写一个必须保留在范围内的概念。",
        },
    },
    {
        "field": "exclusion_criteria",
        "en": "Topics to avoid. These can become excluded terms in provider queries when supported.",
        "zh": "希望避开的主题，在数据源支持时会转成排除词。",
        "edit": {
            "en": "Add noisy meanings or unrelated domains here.",
            "zh": "把容易干扰的含义或无关领域写在这里。",
        },
    },
    {
        "field": "required_aspects",
        "en": "Aspects used by the aspect-coverage agent to check what each paper covers.",
        "zh": "aspect coverage agent 会用它判断每篇论文覆盖了哪些方面。",
        "edit": {
            "en": "Use this for dimensions you want the final result set to cover.",
            "zh": "用于定义最终结果集应该覆盖的维度。",
        },
    },
    {
        "field": "preferred_paper_types",
        "en": "Preferred paper categories such as review, survey, benchmark, method, or tutorial.",
        "zh": "偏好的论文类型，例如 review、survey、benchmark、method、tutorial。",
        "edit": {
            "en": "Edit when you need background papers, methods, benchmarks, or recent frontier work.",
            "zh": "当你想偏向背景论文、方法论文、benchmark 或前沿工作时修改。",
        },
    },
    {
        "field": "core_terms",
        "en": "Central topic phrases. Multi-word core terms are important for provider query construction.",
        "zh": "中心主题短语，多词术语会影响 provider query 的构造。",
        "edit": {
            "en": "Keep the real scientific object here; remove drift terms.",
            "zh": "这里保留真正的科学对象，删掉跑偏的词。",
        },
    },
    {
        "field": "must_terms",
        "en": "High-priority terms that should appear in focused searches.",
        "zh": "高优先级术语，用于更聚焦的检索。",
        "edit": {
            "en": "Use fewer must terms for broad searches and more for strict searches.",
            "zh": "宽泛检索少放 must terms，严格检索多放。",
        },
    },
    {
        "field": "optional_terms",
        "en": "Useful expansions such as mechanisms, applications, or review-related terms.",
        "zh": "有用的扩展词，例如机制、应用、review/survey 等。",
        "edit": {
            "en": "Edit to broaden or redirect the search without making terms mandatory.",
            "zh": "用于拓宽或调整方向，但不把它们变成必选词。",
        },
    },
    {
        "field": "exclude_terms",
        "en": "Negative terms used to reduce off-topic results.",
        "zh": "负向词，用来减少跑题结果。",
        "edit": {
            "en": "Add ambiguous domains or meanings you do not want.",
            "zh": "把你不想要的歧义领域或含义加在这里。",
        },
    },
    {
        "field": "OpenAlex queries",
        "en": "Provider-specific search strings sent to OpenAlex.",
        "zh": "实际发送给 OpenAlex 的 provider-specific 检索式。",
        "edit": {
            "en": "Edit these when OpenAlex search direction is too narrow, too broad, or off-topic.",
            "zh": "当 OpenAlex 检索方向过窄、过宽或跑题时修改。",
        },
    },
    {
        "field": "Semantic Scholar queries",
        "en": "Provider-specific search strings sent to Semantic Scholar.",
        "zh": "实际发送给 Semantic Scholar 的 provider-specific 检索式。",
        "edit": {
            "en": "Use +required and -excluded terms when you want stronger control.",
            "zh": "需要更强控制时可以使用 +必选词 和 -排除词。",
        },
    },
]
PIPELINE_STAGE_PROGRESS = {
    "planning": 5,
    "retrieval": 25,
    "dedup": 45,
    "extraction": 58,
    "verification": 72,
    "aspect_coverage": 78,
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
    "aspect_coverage": "Aspect coverage",
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
        "imported_library": result.evaluation_metrics.get("imported_library", {}),
        "ranked_papers_path": result.ranked_papers_path,
        "report_path": result.report_path,
        "evaluation_path": result.evaluation_path,
        "search_brief_path": str(Path(result.output_dir) / "search_brief.json"),
        "question_refinement_path": str(Path(result.output_dir) / "question_refinement.json"),
        "aspect_coverage_path": str(Path(result.output_dir) / "aspect_coverage.csv"),
        "retrieval_diagnostics_path": str(Path(result.output_dir) / "retrieval_diagnostics.json"),
        "result_groups_path": str(Path(result.output_dir) / "result_groups.json"),
        "prisma_like_flow_path": str(Path(result.output_dir) / "prisma_like_flow.json"),
        "paper_cards_path": str(Path(result.output_dir) / "paper_cards.md"),
        "reading_path_path": str(Path(result.output_dir) / "reading_path.md"),
        "agent_trace_path": str(Path(result.output_dir) / "agent_trace.json"),
        "run_events_path": str(Path(result.output_dir) / "run_events.jsonl"),
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


def uploaded_library_format(file_name: str) -> str:
    """Return the import format for an uploaded literature-library file."""

    suffix = Path(file_name).suffix.lower()
    if suffix in {".bib", ".bibtex"}:
        return "bibtex"
    if suffix == ".ris":
        return "ris"
    if suffix == ".csv":
        return "csv"
    return "auto"


def save_uploaded_library(uploaded_file: Any, output_dir: Path) -> tuple[str | None, str]:
    """Save an uploaded library file into the project folder."""

    if uploaded_file is None:
        return None, "auto"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".txt"
    destination = output_dir / f"imported_library{suffix}"
    destination.write_bytes(uploaded_file.getvalue())
    return str(destination), uploaded_library_format(uploaded_file.name)


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
        "doi",
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


def file_text(path: str | Path) -> str:
    """Read a text artifact if it exists."""

    candidate = Path(path)
    if not candidate.exists():
        return ""
    return candidate.read_text(encoding="utf-8")


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
        elif ORIGINAL_API_ENV_VALUES.get(env_name):
            os.environ[env_name] = ORIGINAL_API_ENV_VALUES[env_name] or ""
        else:
            os.environ.pop(env_name, None)


def render_key_warnings(providers: list[str], llm_backend: str, language: str) -> None:
    """Show missing-key status without exposing key values."""

    if "openalex" in providers and not os.getenv("OPENALEX_API_KEY"):
        st.sidebar.warning(
            ui_text(
                language,
                "OPENALEX_API_KEY is not set; current OpenAlex API access requires a free key.",
                "OPENALEX_API_KEY 未设置；当前 OpenAlex API 需要免费 key。",
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


def check_provider_connectivity(
    providers: list[str],
    timeout: float = 3.0,
) -> list[dict[str, Any]]:
    """Check whether selected metadata providers are reachable."""

    results: list[dict[str, Any]] = []
    for provider in providers:
        try:
            if provider == "openalex":
                if not os.getenv("OPENALEX_API_KEY"):
                    results.append(
                        {
                            "provider": provider,
                            "reachable": False,
                            "status_code": "",
                            "message": "missing OPENALEX_API_KEY",
                        }
                    )
                    continue
                response = requests.get(
                    PROVIDER_HEALTH_URLS[provider],
                    params={"per-page": 1, "api_key": os.getenv("OPENALEX_API_KEY")},
                    timeout=timeout,
                )
            elif provider == "semantic_scholar":
                headers = {"x-api-key": os.getenv("S2_API_KEY")} if os.getenv("S2_API_KEY") else None
                response = requests.get(
                    PROVIDER_HEALTH_URLS[provider],
                    params={"query": "test", "limit": 1, "fields": "title"},
                    headers=headers,
                    timeout=timeout,
                )
            else:
                continue
            reachable = response.status_code < 500
            results.append(
                {
                    "provider": provider,
                    "reachable": reachable,
                    "status_code": response.status_code,
                    "message": response.text[:180] if not reachable else "reachable",
                }
            )
        except requests.RequestException as exc:
            results.append(
                {
                    "provider": provider,
                    "reachable": False,
                    "status_code": "",
                    "message": str(exc)[:180],
                }
            )
    return results


def render_connectivity_result(rows: list[dict[str, Any]], language: str) -> None:
    """Render provider connectivity diagnostics."""

    if not rows:
        return
    if all(not row["reachable"] for row in rows):
        st.error(
            ui_text(
                language,
                "Selected metadata providers are not reachable from this process.",
                "当前进程无法连接选中的论文数据源。",
            )
        )
    elif any(not row["reachable"] for row in rows):
        st.warning(
            ui_text(
                language,
                "At least one selected provider is not reachable.",
                "至少有一个选中的数据源不可连接。",
            )
        )
    else:
        st.success(
            ui_text(
                language,
                "Selected providers are reachable.",
                "选中的数据源可以连接。",
            )
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
    use_year_filter = st.sidebar.checkbox(
        ui_text(language, "Apply year filter", "启用年份过滤"),
        value=False,
        help=ui_text(
            language,
            "Leave this off for broad background searches. Turn it on for recent-only searches.",
            "调研背景文献时建议先关闭；只想看近期文献时再打开。",
        ),
    )
    from_year_value = st.sidebar.number_input(
        ui_text(language, "From year", "起始年份"),
        min_value=1900,
        max_value=2100,
        value=2020,
        step=1,
        disabled=not use_year_filter,
        help=ui_text(
            language,
            "Keep only papers published from this year onward. The pipeline enforces this locally after provider retrieval.",
            "只保留该年份及之后发表的论文；pipeline 会在数据源返回后执行本地硬过滤。",
        ),
    )
    if not use_year_filter:
        st.sidebar.caption(
            ui_text(
                language,
                "Year filtering is off. The year shown above will not be applied.",
                "年份过滤未启用。上方显示的年份不会参与本次筛选。",
            )
        )
    else:
        st.sidebar.caption(
            ui_text(
                language,
                "The pipeline also applies a local hard year filter after retrieval.",
                "pipeline 会在检索后再执行一次本地硬年份过滤。",
            )
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
            help=ui_text(
                language,
                "Required by the current OpenAlex API. Free keys have a daily budget.",
                "当前 OpenAlex API 需要填写。免费 key 有每日额度。",
            ),
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
    with st.sidebar.expander(ui_text(language, "Provider Connectivity", "数据源连通性")):
        if st.button(ui_text(language, "Check providers", "检查数据源")):
            st.session_state.provider_connectivity = check_provider_connectivity(providers)
        if st.session_state.get("provider_connectivity"):
            render_connectivity_result(st.session_state.provider_connectivity, language)

    return {
        "language": language,
        "providers": providers,
        "max_per_query": int(max_per_query),
        "from_year": int(from_year_value) if use_year_filter else None,
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
    st.session_state.setdefault("required_aspects_text", "")
    st.session_state.setdefault("openalex_queries_text", "")
    st.session_state.setdefault("semantic_scholar_queries_text", "")
    st.session_state.setdefault("refined_question_text", "")
    st.session_state.setdefault("search_intent_value", "overview")
    st.session_state.setdefault("user_goal_text", "")
    st.session_state.setdefault("inclusion_criteria_text", "")
    st.session_state.setdefault("exclusion_criteria_text", "")
    st.session_state.setdefault("preferred_paper_types_text", "")
    st.session_state.setdefault("time_window_text", "")
    st.session_state.setdefault("success_definition_text", "")
    st.session_state.setdefault("provider_connectivity", None)


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


def preview_search_brief(preview: dict[str, Any], question: str = "") -> SearchBrief:
    """Return a SearchBrief object from a preview payload."""

    value = preview.get("search_brief")
    if isinstance(value, SearchBrief):
        return value
    data = value or {}
    return SearchBrief(
        original_question=data.get("original_question", preview.get("question", question)),
        refined_question=data.get("refined_question", preview.get("planning_question", question)),
        search_intent=data.get("search_intent", "overview"),
        user_goal=data.get("user_goal", ""),
        inclusion_criteria=list(data.get("inclusion_criteria", [])),
        exclusion_criteria=list(data.get("exclusion_criteria", [])),
        required_aspects=list(data.get("required_aspects", [])),
        preferred_paper_types=list(data.get("preferred_paper_types", [])),
        time_window=data.get("time_window", ""),
        success_definition=data.get("success_definition", ""),
    )


def edited_search_brief(preview: dict[str, Any], question: str) -> SearchBrief:
    """Build a SearchBrief from the editable UI fields."""

    base = preview_search_brief(preview, question)
    return SearchBrief(
        original_question=base.original_question or question,
        refined_question=" ".join(st.session_state.refined_question_text.split())
        or base.refined_question
        or question,
        search_intent=st.session_state.search_intent_value
        if st.session_state.search_intent_value in SEARCH_INTENTS
        else base.search_intent,
        user_goal=" ".join(st.session_state.user_goal_text.split()) or base.user_goal,
        inclusion_criteria=parse_query_editor_text(st.session_state.inclusion_criteria_text),
        exclusion_criteria=parse_query_editor_text(st.session_state.exclusion_criteria_text),
        required_aspects=parse_query_editor_text(st.session_state.required_aspects_text),
        preferred_paper_types=parse_query_editor_text(
            st.session_state.preferred_paper_types_text
        ),
        time_window=" ".join(st.session_state.time_window_text.split()),
        success_definition=" ".join(st.session_state.success_definition_text.split()),
    )


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
    search_brief = preview_search_brief(plan, question)
    st.session_state.refined_question_text = search_brief.refined_question
    st.session_state.search_intent_value = (
        search_brief.search_intent
        if search_brief.search_intent in SEARCH_INTENTS
        else "overview"
    )
    st.session_state.user_goal_text = search_brief.user_goal
    st.session_state.inclusion_criteria_text = "\n".join(search_brief.inclusion_criteria)
    st.session_state.exclusion_criteria_text = "\n".join(search_brief.exclusion_criteria)
    st.session_state.required_aspects_text = "\n".join(search_brief.required_aspects)
    st.session_state.preferred_paper_types_text = "\n".join(
        search_brief.preferred_paper_types
    )
    st.session_state.time_window_text = search_brief.time_window
    st.session_state.success_definition_text = search_brief.success_definition
    query_plan = plan["query_plan"]
    st.session_state.core_terms_text = "\n".join(query_plan.core_terms)
    st.session_state.must_terms_text = "\n".join(query_plan.must_terms)
    st.session_state.optional_terms_text = "\n".join(query_plan.optional_terms)
    st.session_state.exclude_terms_text = "\n".join(query_plan.exclude_terms)
    if not st.session_state.required_aspects_text:
        st.session_state.required_aspects_text = "\n".join(query_plan.required_aspects)
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
        required_aspects=list(value.get("required_aspects", [])),
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
        translated_question=" ".join(st.session_state.refined_question_text.split())
        or base.translated_question,
        core_terms=parse_query_editor_text(st.session_state.core_terms_text),
        must_terms=parse_query_editor_text(st.session_state.must_terms_text),
        optional_terms=parse_query_editor_text(st.session_state.optional_terms_text),
        exclude_terms=parse_query_editor_text(st.session_state.exclude_terms_text),
        required_aspects=parse_query_editor_text(st.session_state.required_aspects_text),
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
    search_brief: SearchBrief | None = None,
    planner_metadata: dict[str, Any] | None = None,
    uploaded_library: Any = None,
) -> None:
    """Run the core pipeline and store the result in session state."""

    output_dir = new_project_output_dir(question)
    language = settings["language"]
    input_file, input_format = save_uploaded_library(uploaded_library, output_dir)
    connectivity = check_provider_connectivity(settings["providers"])
    st.session_state.provider_connectivity = connectivity
    if any(not row["reachable"] for row in connectivity):
        render_connectivity_result(connectivity, language)
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
                search_brief_override=search_brief,
                planned_queries_override=planned_queries,
                query_plan_override=query_plan,
                planner_metadata_override=planner_metadata,
                strictness=settings["strictness"],
                openalex_mode=settings["openalex_mode"],
                sort_preference=settings["sort_preference"],
                ranking_profile=settings["ranking_profile"],
                input_file=input_file,
                input_format=input_format,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            ScreeningRunLogger(output_dir).log_exception(
                "fatal",
                exc,
                {"output_dir": str(output_dir)},
            )
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


def step3_field_guide_dataframe(language: str) -> pd.DataFrame:
    """Build the Step 3 field guide as a localized table."""

    return pd.DataFrame(
        [
            {
                "field": item["field"],
                "what_it_controls": item["zh" if language == "中文" else "en"],
                "when_to_edit": item["edit"]["zh" if language == "中文" else "en"],
            }
            for item in STEP3_FIELD_GUIDE
        ]
    )


def render_step3_summary(language: str, openalex_count: int, semantic_count: int) -> None:
    """Render a compact Step 3 summary before the editable details."""

    st.caption(
        ui_text(
            language,
            (
                "This checkpoint is where you inspect and edit the search plan "
                "before spending provider API calls."
            ),
            "这里是在真正请求文献 API 之前检查和修改检索计划的检查点。",
        )
    )
    summary_cols = st.columns(3)
    summary_cols[0].metric("OpenAlex queries", openalex_count)
    summary_cols[1].metric("Semantic Scholar queries", semantic_count)
    summary_cols[2].metric(
        ui_text(language, "Provider calls made", "已请求文献 API"),
        ui_text(language, "No", "否"),
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

    openalex_count = len(parse_query_editor_text(st.session_state.openalex_queries_text))
    semantic_count = len(
        parse_query_editor_text(st.session_state.semantic_scholar_queries_text)
    )
    step3_label = ui_text(
        language,
        f"Step 3: Review Query Plan ({openalex_count} OpenAlex / {semantic_count} Semantic Scholar)",
        f"第 3 步：检查 Query Plan（OpenAlex {openalex_count} 条 / Semantic Scholar {semantic_count} 条）",
    )
    with st.expander(
        step3_label,
        expanded=st.session_state.pipeline_result is None,
    ):
        render_step3_summary(language, openalex_count, semantic_count)
        tabs = st.tabs(
            [
                ui_text(language, "Field Guide", "字段说明"),
                ui_text(language, "Research Intent", "研究意图"),
                ui_text(language, "Terms And Provider Queries", "术语与 Provider Queries"),
            ]
        )

        with tabs[0]:
            st.markdown(
                ui_text(
                    language,
                    (
                        "Use this table as a checklist before Step 4. "
                        "The most important fields to check are usually "
                        "`Refined English question`, `core_terms`, `must_terms`, "
                        "and the provider queries."
                    ),
                    (
                        "第 4 步之前可以把这张表当作检查清单。通常最值得检查的是 "
                        "`Refined English question`、`core_terms`、`must_terms` "
                        "和两个 provider 的 queries。"
                    ),
                )
            )
            st.dataframe(
                step3_field_guide_dataframe(language),
                use_container_width=True,
                hide_index=True,
            )

        with tabs[1]:
            render_question_preprocessing(
                preview.get("question", question),
                preview.get("planner_metadata", {}),
                language,
            )
            st.text_area(
                ui_text(language, "Refined English question", "英文 refined question"),
                key="refined_question_text",
                height=90,
                help=ui_text(
                    language,
                    "This English question is used by retrieval, extraction, and ranking.",
                    "系统会用这个英文问题进行检索、evidence 抽取和排序。",
                ),
            )
            intent_cols = st.columns(3)
            with intent_cols[0]:
                st.selectbox(
                    ui_text(language, "Search intent", "搜索意图"),
                    SEARCH_INTENTS,
                    key="search_intent_value",
                    help=ui_text(
                        language,
                        "Controls preferred paper types, aspects, and ranking emphasis.",
                        "影响偏好的论文类型、required aspects 和排序侧重点。",
                    ),
                )
            with intent_cols[1]:
                st.text_input(
                    ui_text(language, "Time window", "时间窗口"),
                    key="time_window_text",
                    help=ui_text(
                        language,
                        "A soft time preference in the search brief. Use the sidebar year filter for a hard cutoff.",
                        "SearchBrief 里的软性时间偏好。硬性年份截止请用侧边栏年份过滤。",
                    ),
                )
            with intent_cols[2]:
                st.text_input(
                    ui_text(language, "User goal", "用户目标"),
                    key="user_goal_text",
                    help=ui_text(
                        language,
                        "The practical reason for the search. This appears in the report and helps interpret ranking.",
                        "检索的实际目标，会进入 report 并帮助解释排序。",
                    ),
                )
            logic_cols = st.columns(4)
            with logic_cols[0]:
                st.text_area(
                    "inclusion_criteria",
                    key="inclusion_criteria_text",
                    height=130,
                    help=ui_text(
                        language,
                        "One criterion per line. These become required or high-priority query terms.",
                        "每行一条。系统会把它们作为高优先级检索条件。",
                    ),
                )
            with logic_cols[1]:
                st.text_area(
                    "exclusion_criteria",
                    key="exclusion_criteria_text",
                    height=130,
                    help=ui_text(
                        language,
                        "One exclusion per line. These become provider-specific negative terms when possible.",
                        "每行一条。系统会尽量转换成数据源支持的排除词。",
                    ),
                )
            with logic_cols[2]:
                st.text_area(
                    "required_aspects",
                    key="required_aspects_text",
                    height=130,
                    help=ui_text(
                        language,
                        "Aspects that each paper should cover. These are used by aspect coverage and ranking.",
                        "论文最好覆盖的方面，会用于 aspect coverage 和排序。",
                    ),
                )
            with logic_cols[3]:
                st.text_area(
                    "preferred_paper_types",
                    key="preferred_paper_types_text",
                    height=130,
                    help=ui_text(
                        language,
                        "Preferred literature types, such as review, method paper, benchmark paper.",
                        "偏好的论文类型，例如 review、method paper、benchmark paper。",
                    ),
                )
            st.text_area(
                ui_text(language, "Success definition", "成功标准"),
                key="success_definition_text",
                height=70,
                help=ui_text(
                    language,
                    "What a good result set should look like.",
                    "什么样的结果集算是满足本次检索目标。",
                ),
            )

        with tabs[2]:
            st.caption(
                ui_text(
                    language,
                    "Edit one item per line. Step 4 will use these exact values unless you regenerate the query plan.",
                    "每行一个条目。除非重新生成 Query Plan，第 4 步会直接使用这里的值。",
                )
            )
            term_cols = st.columns(4)
            with term_cols[0]:
                st.text_area(
                    "core_terms",
                    key="core_terms_text",
                    height=140,
                    help=ui_text(
                        language,
                        "Central scientific objects or phrases. Remove terms that drift away from the topic.",
                        "中心科学对象或短语。删掉明显跑偏的词。",
                    ),
                )
            with term_cols[1]:
                st.text_area(
                    "must_terms",
                    key="must_terms_text",
                    height=140,
                    help=ui_text(
                        language,
                        "High-priority terms used for focused query construction.",
                        "用于聚焦检索式的高优先级术语。",
                    ),
                )
            with term_cols[2]:
                st.text_area(
                    "optional_terms",
                    key="optional_terms_text",
                    height=140,
                    help=ui_text(
                        language,
                        "Expansion terms that broaden the search without becoming mandatory.",
                        "扩展检索范围，但不作为必选词。",
                    ),
                )
            with term_cols[3]:
                st.text_area(
                    "exclude_terms",
                    key="exclude_terms_text",
                    height=140,
                    help=ui_text(
                        language,
                        "Terms or meanings to avoid when the provider supports exclusion.",
                        "当 provider 支持时，用于排除不想要的词或含义。",
                    ),
                )

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


def render_search_brief_result(result: PipelineResult, language: str) -> None:
    """Render the intent-level interpretation used by the pipeline."""

    brief = result.search_brief
    if brief is None:
        st.info(ui_text(language, "No SearchBrief was saved for this run.", "此运行没有保存 SearchBrief。"))
        return
    rows = [
        ("refined_question", brief.refined_question),
        ("search_intent", brief.search_intent),
        ("user_goal", brief.user_goal),
        ("inclusion_criteria", "; ".join(brief.inclusion_criteria)),
        ("exclusion_criteria", "; ".join(brief.exclusion_criteria)),
        ("required_aspects", "; ".join(brief.required_aspects)),
        ("preferred_paper_types", "; ".join(brief.preferred_paper_types)),
        ("time_window", brief.time_window),
        ("success_definition", brief.success_definition),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["field", "value"]),
        use_container_width=True,
        hide_index=True,
    )
    if result.question_refinement:
        st.subheader(ui_text(language, "Refined Subquestions", "细化子问题"))
        subquestions = result.question_refinement.get("subquestions", [])
        if subquestions:
            for item in subquestions:
                st.markdown(f"- {item}")
        else:
            st.caption(ui_text(language, "No subquestions were needed.", "无需拆分子问题。"))


def render_aspect_coverage(result: PipelineResult, language: str) -> None:
    """Render aspect coverage records as a matrix-like table."""

    if not result.aspect_coverage_records:
        st.info(ui_text(language, "No aspect coverage records were generated.", "没有生成 aspect coverage 记录。"))
        return
    rows = [
        {
            "paper_id": record.paper_id,
            "title": record.title,
            "covered_aspects": "; ".join(record.covered_aspects),
            "missing_aspects": "; ".join(record.missing_aspects),
            "aspect_coverage_score": record.aspect_coverage_score,
        }
        for record in result.aspect_coverage_records
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_result_groups(result: PipelineResult, language: str) -> None:
    """Render grouped result lists."""

    groups = result.result_groups or {}
    if not groups:
        st.info(ui_text(language, "No grouped results were generated.", "没有生成结果分组。"))
        return
    for group_name, rows in groups.items():
        with st.expander(f"{group_name} ({len(rows)})", expanded=group_name == "must_read"):
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption(ui_text(language, "No papers in this group.", "这个分组没有论文。"))


def render_markdown_artifact(path: str | Path, empty_message: str) -> None:
    """Render a generated Markdown artifact."""

    text = file_text(path)
    if text:
        st.markdown(text)
    else:
        st.info(empty_message)


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
    year_filter = result.evaluation_metrics.get("year_filter", {})
    imported_library = result.evaluation_metrics.get("imported_library", {})
    if imported_library and imported_library.get("paper_count", 0):
        st.caption(
            ui_text(
                language,
                (
                    f"Imported library: {imported_library.get('paper_count', 0)} papers "
                    f"from {imported_library.get('detected_format', 'unknown')}."
                ),
                (
                    f"外部文献库导入：{imported_library.get('paper_count', 0)} 篇，"
                    f"格式 {imported_library.get('detected_format', 'unknown')}。"
                ),
            )
        )
    if year_filter:
        if year_filter.get("enabled"):
            st.caption(
                ui_text(
                    language,
                    (
                        f"Year filter: from {year_filter.get('from_year')}; "
                        f"kept {year_filter.get('kept_count', 0)} / "
                        f"{year_filter.get('input_count', 0)} retrieved records; "
                        f"excluded {year_filter.get('excluded_before_year_count', 0)} older "
                        f"and {year_filter.get('excluded_missing_year_count', 0)} missing-year records."
                    ),
                    (
                        f"年份过滤：从 {year_filter.get('from_year')} 年开始；"
                        f"保留 {year_filter.get('kept_count', 0)} / "
                        f"{year_filter.get('input_count', 0)} 条检索记录；"
                        f"排除旧年份 {year_filter.get('excluded_before_year_count', 0)} 条，"
                        f"缺少年份 {year_filter.get('excluded_missing_year_count', 0)} 条。"
                    ),
                )
            )
        else:
            st.caption(
                ui_text(
                    language,
                    "Year filter was off for this run.",
                    "本次运行没有启用年份过滤。",
                )
            )

    selected: RankedPaper | None = None
    if result.ranked_final:
        options = {
            f"{item.rank}. {item.paper.title[:100]}": item
            for item in result.ranked_final
        }
        selected_label = st.selectbox(
            ui_text(language, "Selected paper", "选择论文"),
            list(options),
        )
        selected = options[selected_label]

    tabs = st.tabs(
        [
            "Research Intent",
            "Search Strategy",
            "Ranked Papers",
            "Evidence Chain",
            "Results Map",
            "Paper Cards",
            "Feedback",
            "Report & Export",
            "Trace",
            "Metrics",
        ]
    )
    with tabs[0]:
        render_search_brief_result(result, language)

    with tabs[1]:
        planner_metadata = result.agent_trace.get("planner", {}).get("metadata", {})
        render_question_preprocessing(result.question, planner_metadata, language)
        if result.query_plan:
            st.subheader(ui_text(language, "Structured Query Plan", "结构化 Query Plan"))
            plan_rows = [
                ("core_terms", "; ".join(result.query_plan.core_terms)),
                ("must_terms", "; ".join(result.query_plan.must_terms)),
                ("optional_terms", "; ".join(result.query_plan.optional_terms)),
                ("exclude_terms", "; ".join(result.query_plan.exclude_terms)),
                ("required_aspects", "; ".join(result.query_plan.required_aspects)),
                ("filters", json.dumps(result.query_plan.filters, ensure_ascii=False)),
            ]
            st.dataframe(
                pd.DataFrame(plan_rows, columns=["field", "value"]),
                use_container_width=True,
                hide_index=True,
            )
            query_cols = st.columns(2)
            with query_cols[0]:
                st.caption("OpenAlex")
                render_planned_queries(result.query_plan.openalex_queries)
            with query_cols[1]:
                st.caption("Semantic Scholar")
                render_planned_queries(result.query_plan.semantic_scholar_queries)
        st.subheader(ui_text(language, "Combined Planned Queries", "合并后的 Planned Queries"))
        render_planned_queries(result.planned_queries)

    with tabs[2]:
        if result.merged_paper_count == 0:
            st.warning(
                ui_text(
                    language,
                    "No papers were retrieved. Try increasing max papers per query, changing providers, or checking API/network access.",
                    "没有检索到论文。可以尝试增加每个 query 的论文数、切换数据源，或检查 API/网络。",
                )
            )
            if (
                year_filter.get("enabled")
                and year_filter.get("input_count", 0) > 0
                and year_filter.get("kept_count", 0) == 0
            ):
                st.warning(
                    ui_text(
                        language,
                        "The providers returned records, but the local year filter removed all of them.",
                        "数据源返回了记录，但本地年份过滤把它们全部排除了。",
                    )
                )
            diagnostics = read_json_file(Path(result.output_dir) / "retrieval_diagnostics.json")
            provider_errors = diagnostics.get("provider_errors", {})
            error_rows = [
                {
                    "provider": provider,
                    "query": row.get("query", ""),
                    "error": row.get("error", ""),
                    "status_code": row.get("status_code", ""),
                    "error_message": row.get("error_message", ""),
                }
                for provider, rows in provider_errors.items()
                for row in rows
            ]
            if error_rows:
                st.error(
                    ui_text(
                        language,
                        "Provider API errors were detected. This run failed to retrieve metadata for at least some queries.",
                        "检测到数据源 API 错误。本次运行至少有部分 query 没有成功返回元数据。",
                    )
                )
                st.dataframe(
                    pd.DataFrame(error_rows),
                    use_container_width=True,
                    hide_index=True,
                )
        table = ranked_dataframe(result.ranked_final)
        st.dataframe(table, use_container_width=True, hide_index=True)

    with tabs[3]:
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

    with tabs[4]:
        st.subheader(ui_text(language, "Aspect Coverage Matrix", "Aspect Coverage 矩阵"))
        render_aspect_coverage(result, language)
        st.subheader(ui_text(language, "Grouped Result Lists", "结果分组"))
        render_result_groups(result, language)
        st.subheader(ui_text(language, "PRISMA-like Screening Flow", "PRISMA-like 筛选流程"))
        st.json(result.evaluation_metrics.get("prisma_like_flow", {}))
        st.subheader(ui_text(language, "Year Filter Audit", "年份过滤审计"))
        st.json(result.evaluation_metrics.get("year_filter", {}))
        st.subheader(ui_text(language, "Imported Library Audit", "导入文献库审计"))
        st.json(result.evaluation_metrics.get("imported_library", {}))

    with tabs[5]:
        st.subheader(ui_text(language, "Recommended Reading Path", "推荐阅读路径"))
        render_markdown_artifact(
            Path(result.output_dir) / "reading_path.md",
            ui_text(language, "reading_path.md was not generated.", "没有生成 reading_path.md。"),
        )
        st.subheader(ui_text(language, "Top Paper Evidence Cards", "Top Paper Evidence Cards"))
        render_markdown_artifact(
            Path(result.output_dir) / "paper_cards.md",
            ui_text(language, "paper_cards.md was not generated.", "没有生成 paper_cards.md。"),
        )

    with tabs[6]:
        render_feedback_tools(result, language)
        if selected is not None:
            render_manual_feedback(selected, result, language)

    with tabs[7]:
        render_downloads(result.ranked_papers_path, result.report_path, language)
        extra_artifacts = [
            ("aspect_coverage.csv", Path(result.output_dir) / "aspect_coverage.csv", "text/csv"),
            ("imported_papers.csv", Path(result.output_dir) / "imported_papers.csv", "text/csv"),
            ("import_diagnostics.json", Path(result.output_dir) / "import_diagnostics.json", "application/json"),
            ("retrieval_diagnostics.json", Path(result.output_dir) / "retrieval_diagnostics.json", "application/json"),
            ("run_events.jsonl", Path(result.output_dir) / "run_events.jsonl", "application/jsonl"),
            ("paper_cards.md", Path(result.output_dir) / "paper_cards.md", "text/markdown"),
            ("reading_path.md", Path(result.output_dir) / "reading_path.md", "text/markdown"),
        ]
        for file_name, path, mime in extra_artifacts:
            data = file_bytes(path)
            if data is not None:
                st.download_button(
                    file_name,
                    data=data,
                    file_name=file_name,
                    mime=mime,
                )

    with tabs[8]:
        st.json(result.agent_trace)

    with tabs[9]:
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
    search_brief = read_json_file(output_dir / "search_brief.json")
    question_refinement = read_json_file(output_dir / "question_refinement.json")
    result_groups = read_json_file(output_dir / "result_groups.json")
    prisma_flow = read_json_file(output_dir / "prisma_like_flow.json")

    tabs = st.tabs(
        [
            "Research Intent",
            "Search Strategy",
            "Ranked Papers",
            "Results Map",
            "Paper Cards",
            "Report & Export",
            "Trace",
            "Metrics",
        ]
    )
    with tabs[0]:
        if search_brief:
            st.dataframe(
                pd.DataFrame(
                    [
                        ("refined_question", search_brief.get("refined_question", "")),
                        ("search_intent", search_brief.get("search_intent", "")),
                        ("user_goal", search_brief.get("user_goal", "")),
                        (
                            "inclusion_criteria",
                            "; ".join(search_brief.get("inclusion_criteria", [])),
                        ),
                        (
                            "exclusion_criteria",
                            "; ".join(search_brief.get("exclusion_criteria", [])),
                        ),
                        (
                            "required_aspects",
                            "; ".join(search_brief.get("required_aspects", [])),
                        ),
                    ],
                    columns=["field", "value"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        if question_refinement:
            st.json(question_refinement)

    with tabs[1]:
        metadata = planned.get("planner_metadata") or planned.get("llm", {})
        render_question_preprocessing(planned.get("question", ""), metadata, language)
        render_planned_queries(planned.get("queries", []))
    with tabs[2]:
        if ranked_path.exists():
            st.dataframe(pd.read_csv(ranked_path), use_container_width=True)
        else:
            st.warning(ui_text(language, "Saved ranked_papers.csv is missing.", "保存的 ranked_papers.csv 不存在。"))
    with tabs[3]:
        aspect_path = output_dir / "aspect_coverage.csv"
        if aspect_path.exists():
            st.dataframe(pd.read_csv(aspect_path), use_container_width=True)
        st.subheader(ui_text(language, "Grouped Result Lists", "结果分组"))
        st.json(result_groups)
        st.subheader(ui_text(language, "PRISMA-like Screening Flow", "PRISMA-like 筛选流程"))
        st.json(prisma_flow)
    with tabs[4]:
        render_markdown_artifact(
            output_dir / "reading_path.md",
            ui_text(language, "reading_path.md was not generated.", "没有生成 reading_path.md。"),
        )
        render_markdown_artifact(
            output_dir / "paper_cards.md",
            ui_text(language, "paper_cards.md was not generated.", "没有生成 paper_cards.md。"),
        )
    with tabs[5]:
        render_downloads(str(ranked_path), str(output_dir / "report.md"), language)
    with tabs[6]:
        st.json(trace)
    with tabs[7]:
        st.json(evaluation)


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
    with st.expander(
        ui_text(
            language,
            "Import Existing Literature Library",
            "导入已有文献库",
        ),
        expanded=False,
    ):
        uploaded_library = st.file_uploader(
            ui_text(
                language,
                "Upload BibTeX, RIS, or CSV export",
                "上传 BibTeX、RIS 或 CSV 导出文件",
            ),
            type=["bib", "bibtex", "ris", "csv"],
            help=ui_text(
                language,
                "Use exports from Zotero, Web of Science, Scopus, Google Scholar, or a curated CSV. Imported records are merged with provider retrieval results.",
                "可使用 Zotero、Web of Science、Scopus、Google Scholar 或手工 CSV 导出。导入记录会和在线检索结果合并筛选。",
            ),
        )
        st.caption(
            ui_text(
                language,
                "CSV columns can include: title, abstract, authors, year, venue, doi, url, citation_count.",
                "CSV 可包含列：title、abstract、authors、year、venue、doi、url、citation_count。",
            )
        )
        if uploaded_library is not None:
            st.success(
                ui_text(
                    language,
                    f"Ready to import {uploaded_library.name}.",
                    f"已准备导入 {uploaded_library.name}。",
                )
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
                current_search_brief = edited_search_brief(preview, cleaned_question)
                run_screening(
                    cleaned_question,
                    settings,
                    query_plan=current_query_plan,
                    search_brief=current_search_brief,
                    planner_metadata=preview.get("planner_metadata", {}),
                    uploaded_library=uploaded_library,
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
