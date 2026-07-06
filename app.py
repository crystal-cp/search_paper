"""Streamlit UI for the literature-screening pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from lit_screening.agents.human_feedback import HumanFeedbackAgent
from lit_screening.models import FeedbackRecord, PipelineResult, RankedPaper
from lit_screening.pipeline import (
    apply_feedback_to_pipeline_result,
    ranked_to_row,
    run_pipeline,
)


PROVIDER_OPTIONS = {
    "Both": ["openalex", "semantic_scholar"],
    "OpenAlex": ["openalex"],
    "Semantic Scholar": ["semantic_scholar"],
}


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
        "supported",
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


def render_key_warnings(providers: list[str], llm_backend: str) -> None:
    """Show missing-key status without exposing key values."""

    if "openalex" in providers and not os.getenv("OPENALEX_API_KEY"):
        st.sidebar.info("OPENALEX_API_KEY is not set; OpenAlex will use public access.")
    if "semantic_scholar" in providers and not os.getenv("S2_API_KEY"):
        st.sidebar.warning("S2_API_KEY is not set; Semantic Scholar may be rate-limited.")
    if llm_backend == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        st.sidebar.warning("DEEPSEEK_API_KEY is not set; LLM modes will fall back to rules.")


def render_sidebar() -> dict:
    """Render sidebar controls and return pipeline settings."""

    st.sidebar.header("Screening Settings")
    provider_label = st.sidebar.radio(
        "Providers",
        list(PROVIDER_OPTIONS),
        horizontal=False,
    )
    max_per_query = st.sidebar.number_input(
        "Max papers per query",
        min_value=0,
        max_value=100,
        value=10,
        step=1,
    )
    from_year_value = st.sidebar.number_input(
        "From year",
        min_value=1900,
        max_value=2100,
        value=2020,
        step=1,
    )
    use_cache = st.sidebar.checkbox("Use cache", value=True)
    llm_backend = st.sidebar.selectbox("LLM backend", ["none", "deepseek"], index=0)

    with st.sidebar.expander("LLM Agent Modes"):
        default_mode = "rule" if llm_backend == "none" else "llm"
        planner_mode = st.selectbox("Planner", ["rule", "llm"], index=["rule", "llm"].index(default_mode))
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

    with st.sidebar.expander("Scoring Weights"):
        st.slider("Relevance", 0.0, 1.0, 0.40, 0.05, disabled=True)
        st.slider("Evidence", 0.0, 1.0, 0.25, 0.05, disabled=True)
        st.slider("Recency", 0.0, 1.0, 0.15, 0.05, disabled=True)
        st.slider("Quality", 0.0, 1.0, 0.15, 0.05, disabled=True)
        st.slider("Diversity", 0.0, 1.0, 0.05, 0.05, disabled=True)

    providers = PROVIDER_OPTIONS[provider_label]
    render_key_warnings(providers, llm_backend)

    return {
        "providers": providers,
        "max_per_query": int(max_per_query),
        "from_year": int(from_year_value),
        "use_cache": bool(use_cache),
        "llm_backend": llm_backend,
        "planner_mode": planner_mode,
        "extractor_mode": extractor_mode,
        "verifier_mode": verifier_mode,
    }


def ensure_state() -> None:
    """Initialize Streamlit session state keys."""

    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("feedback_records", {})


def run_screening(question: str, settings: dict) -> None:
    """Run the core pipeline and store the result in session state."""

    with st.spinner("Running literature screening..."):
        result = run_pipeline(
            question=question,
            providers=settings["providers"],
            max_per_query=settings["max_per_query"],
            from_year=settings["from_year"],
            output_dir="outputs/streamlit",
            use_cache=settings["use_cache"],
            llm_backend=settings["llm_backend"],
            planner_mode=settings["planner_mode"],
            extractor_mode=settings["extractor_mode"],
            verifier_mode=settings["verifier_mode"],
        )
    st.session_state.pipeline_result = result
    st.session_state.feedback_records = {}


def render_feedback(selected: RankedPaper, result: PipelineResult) -> None:
    """Render feedback controls and rerank through core pipeline helpers."""

    feedback_agent = HumanFeedbackAgent()
    existing = st.session_state.feedback_records.get(selected.paper.paper_id)
    labels = ["include", "exclude", "uncertain"]
    default_label = existing.label if existing else "uncertain"
    label = st.radio(
        "Feedback label",
        labels,
        index=labels.index(default_label),
        horizontal=True,
        key=f"feedback_label_{selected.paper.paper_id}",
    )
    note = st.text_area(
        "Feedback note",
        value=existing.note if existing else "",
        key=f"feedback_note_{selected.paper.paper_id}",
        height=90,
    )

    if st.button("Apply Feedback And Rerank", type="primary"):
        feedback_record = FeedbackRecord(
            paper_id=selected.paper.paper_id,
            label=label,
            adjustment=feedback_agent.default_adjustment(label),
            note=note,
        )
        updated_feedback = dict(st.session_state.feedback_records)
        updated_feedback[selected.paper.paper_id] = feedback_record
        st.session_state.feedback_records = updated_feedback
        st.session_state.pipeline_result = apply_feedback_to_pipeline_result(
            result,
            updated_feedback,
        )
        st.success("Feedback applied without rerunning retrieval.")


def render_result(result: PipelineResult) -> None:
    """Render planned queries, ranked papers, evidence, metrics, and downloads."""

    st.subheader("Planned Queries")
    st.write(result.planned_queries)

    if result.merged_paper_count == 0:
        st.warning("No papers were retrieved. Try increasing max papers per query, changing providers, or checking API/network access.")

    st.subheader("Ranked Papers")
    table = ranked_dataframe(result.ranked_final)
    if table.empty:
        st.dataframe(table, use_container_width=True)
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)

    if result.ranked_final:
        options = {
            f"{item.rank}. {item.paper.title[:100]}": item
            for item in result.ranked_final
        }
        selected_label = st.selectbox("Selected paper", list(options))
        selected = options[selected_label]

        st.subheader("Evidence Chain")
        evidence_table = pd.DataFrame(
            [
                {
                    "paper_id": selected.paper.paper_id,
                    "claim": selected.evidence.claim,
                    "evidence_sentence": selected.evidence.evidence_sentence,
                    "supported": selected.verification.supported,
                    "confidence": selected.verification.confidence,
                    "error_type": selected.verification.error_type,
                    "rationale": selected.verification.rationale,
                }
            ]
        )
        st.dataframe(evidence_table, use_container_width=True, hide_index=True)

        render_feedback(selected, result)

    st.subheader("Evaluation Metrics")
    metrics = result.evaluation_metrics
    st.json(metrics)

    ranked_bytes = file_bytes(result.ranked_papers_path)
    report_bytes = file_bytes(result.report_path)
    download_cols = st.columns(2)
    with download_cols[0]:
        if ranked_bytes is not None:
            st.download_button(
                "Download ranked_papers.csv",
                data=ranked_bytes,
                file_name="ranked_papers.csv",
                mime="text/csv",
            )
    with download_cols[1]:
        if report_bytes is not None:
            st.download_button(
                "Download report.md",
                data=report_bytes,
                file_name="report.md",
                mime="text/markdown",
            )


def main() -> None:
    """Streamlit app entrypoint."""

    st.set_page_config(page_title="Literature Screening", layout="wide")
    ensure_state()
    settings = render_sidebar()

    st.title("Scientific Literature Screening")
    question = st.text_area(
        "Research question",
        value=(
            "How can human-in-the-loop multi-agent LLM systems improve "
            "scientific literature screening and evidence verification?"
        ),
        height=130,
    )

    if st.button("Run Screening", type="primary"):
        if not question.strip():
            st.error("Please enter a research question.")
        else:
            try:
                run_screening(question.strip(), settings)
            except Exception as exc:
                st.error(f"Screening failed: {exc}")

    result = st.session_state.pipeline_result
    if result:
        render_result(result)


if __name__ == "__main__":
    main()
