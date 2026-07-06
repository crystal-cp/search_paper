"""Command-line pipeline for the literature-screening MVP."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agents.extractor import ExtractorAgent
from .agents.human_feedback import HumanFeedbackAgent
from .agents.planner import PlannerAgent
from .agents.ranker import RankerAgent
from .agents.retriever import RetrieverAgent
from .agents.verifier import VerifierAgent
from .config import PipelineConfig
from .dedup import deduplicate_with_stats
from .evaluation import compute_evaluation, save_evaluation
from .llm_client import GenericLLMClient
from .models import (
    EvidenceRecord,
    FeedbackRecord,
    Paper,
    PipelineResult,
    RankedPaper,
    VerificationResult,
)
from .report import generate_report
from .utils import ensure_dir, write_csv, write_json


PAPER_FIELDS = [
    "paper_id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "source_provider",
    "citation_count",
]

EVIDENCE_FIELDS = [
    "paper_id",
    "title",
    "claim",
    "evidence_sentence",
    "relevance_reason",
    "limitation",
    "keyword_overlap",
    "supported",
    "confidence",
    "error_type",
    "rationale",
    "extraction_mode",
    "evidence_llm_used",
    "evidence_invalid_llm_output",
    "evidence_llm_error_type",
    "verification_mode",
    "verification_llm_used",
    "verification_invalid_llm_output",
    "verification_llm_error_type",
]

RANKED_FIELDS = [
    "rank",
    "paper_id",
    "title",
    "year",
    "venue",
    "doi",
    "url",
    "source_provider",
    "citation_count",
    "relevance_score",
    "evidence_score",
    "recency_score",
    "quality_score",
    "diversity_score",
    "human_feedback_adjustment",
    "final_score",
    "supported",
    "confidence",
    "error_type",
    "claim",
    "evidence_sentence",
    "feedback_label",
    "feedback_note",
    "extraction_mode",
    "evidence_llm_used",
    "evidence_invalid_llm_output",
    "evidence_llm_error_type",
    "verification_mode",
    "verification_llm_used",
    "verification_invalid_llm_output",
    "verification_llm_error_type",
]


def paper_to_row(paper: Paper) -> dict[str, Any]:
    """Convert a Paper into a flat CSV row."""

    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": "; ".join(paper.authors),
        "year": paper.year or "",
        "venue": paper.venue,
        "doi": paper.doi,
        "url": paper.url,
        "source_provider": paper.source_provider,
        "citation_count": paper.citation_count,
    }


def evidence_to_row(
    evidence: EvidenceRecord,
    verification: VerificationResult,
) -> dict[str, Any]:
    """Convert evidence plus verification into a flat CSV row."""

    return {
        "paper_id": evidence.paper_id,
        "title": evidence.title,
        "claim": evidence.claim,
        "evidence_sentence": evidence.evidence_sentence,
        "relevance_reason": evidence.relevance_reason,
        "limitation": evidence.limitation,
        "keyword_overlap": f"{evidence.keyword_overlap:.4f}",
        "supported": verification.supported,
        "confidence": f"{verification.confidence:.4f}",
        "error_type": verification.error_type,
        "rationale": verification.rationale,
        "extraction_mode": evidence.extraction_mode,
        "evidence_llm_used": evidence.llm_used,
        "evidence_invalid_llm_output": evidence.invalid_llm_output,
        "evidence_llm_error_type": evidence.llm_error_type,
        "verification_mode": verification.verification_mode,
        "verification_llm_used": verification.llm_used,
        "verification_invalid_llm_output": verification.invalid_llm_output,
        "verification_llm_error_type": verification.llm_error_type,
    }


def ranked_to_row(item: RankedPaper) -> dict[str, Any]:
    """Convert a RankedPaper into a flat CSV row."""

    feedback = item.feedback
    return {
        "rank": item.rank,
        "paper_id": item.paper.paper_id,
        "title": item.paper.title,
        "year": item.paper.year or "",
        "venue": item.paper.venue,
        "doi": item.paper.doi,
        "url": item.paper.url,
        "source_provider": item.paper.source_provider,
        "citation_count": item.paper.citation_count,
        "relevance_score": f"{item.scores.relevance_score:.4f}",
        "evidence_score": f"{item.scores.evidence_score:.4f}",
        "recency_score": f"{item.scores.recency_score:.4f}",
        "quality_score": f"{item.scores.quality_score:.4f}",
        "diversity_score": f"{item.scores.diversity_score:.4f}",
        "human_feedback_adjustment": f"{item.scores.human_feedback_adjustment:.4f}",
        "final_score": f"{item.scores.final_score:.4f}",
        "supported": item.verification.supported,
        "confidence": f"{item.verification.confidence:.4f}",
        "error_type": item.verification.error_type,
        "claim": item.evidence.claim,
        "evidence_sentence": item.evidence.evidence_sentence,
        "feedback_label": feedback.label if feedback else "",
        "feedback_note": feedback.note if feedback else "",
        "extraction_mode": item.evidence.extraction_mode,
        "evidence_llm_used": item.evidence.llm_used,
        "evidence_invalid_llm_output": item.evidence.invalid_llm_output,
        "evidence_llm_error_type": item.evidence.llm_error_type,
        "verification_mode": item.verification.verification_mode,
        "verification_llm_used": item.verification.llm_used,
        "verification_invalid_llm_output": item.verification.invalid_llm_output,
        "verification_llm_error_type": item.verification.llm_error_type,
    }


def build_llm_client(
    config: PipelineConfig,
    llm_backend: str,
    llm_client: GenericLLMClient | None = None,
) -> GenericLLMClient | None:
    """Create an optional LLM client from config."""

    if llm_client is not None:
        return llm_client if llm_client.is_available else None
    if llm_backend == "none":
        return None
    if llm_backend == "deepseek":
        client = GenericLLMClient(
            provider_name="deepseek",
            api_key_env_var=config.deepseek_api_key_env,
            base_url=config.deepseek_base_url,
            model=config.deepseek_model,
            timeout=config.llm_timeout,
        )
        return client if client.is_available else None
    raise ValueError(f"Unknown LLM backend: {llm_backend}")


def write_pipeline_csvs(
    output_dir: Path,
    merged_papers: list[Paper],
    evidence_records: list[EvidenceRecord],
    verification_results: list[VerificationResult],
    ranked_before_feedback: list[RankedPaper],
    ranked_final: list[RankedPaper],
    ranked_after_feedback: list[RankedPaper] | None,
) -> None:
    """Write all required tabular outputs."""

    write_csv(
        output_dir / "merged_papers.csv",
        [paper_to_row(paper) for paper in merged_papers],
        PAPER_FIELDS,
    )
    verification_by_id = {result.paper_id: result for result in verification_results}
    write_csv(
        output_dir / "evidence_table.csv",
        [
            evidence_to_row(record, verification_by_id[record.paper_id])
            for record in evidence_records
            if record.paper_id in verification_by_id
        ],
        EVIDENCE_FIELDS,
    )
    write_csv(
        output_dir / "ranked_papers_before_feedback.csv",
        [ranked_to_row(item) for item in ranked_before_feedback],
        RANKED_FIELDS,
    )
    if ranked_after_feedback is not None:
        write_csv(
            output_dir / "ranked_papers_after_feedback.csv",
            [ranked_to_row(item) for item in ranked_after_feedback],
            RANKED_FIELDS,
        )
    write_csv(
        output_dir / "ranked_papers.csv",
        [ranked_to_row(item) for item in ranked_final],
        RANKED_FIELDS,
    )


def _llm_metrics(
    llm_backend: str,
    active_llm_backend: str,
    planner_mode: str,
    extractor_mode: str,
    verifier_mode: str,
    planner_metadata: dict[str, Any],
    evidence_records: list[EvidenceRecord],
    verification_results: list[VerificationResult],
) -> dict[str, Any]:
    """Build a compact diagnostics block for optional LLM usage."""

    planner_invalid = bool(planner_metadata.get("invalid_llm_output"))
    evidence_invalid_count = sum(1 for record in evidence_records if record.invalid_llm_output)
    verification_invalid_count = sum(
        1 for result in verification_results if result.invalid_llm_output
    )
    return {
        "backend_requested": llm_backend,
        "backend_active": active_llm_backend,
        "planner_mode": planner_mode,
        "extractor_mode": extractor_mode,
        "verifier_mode": verifier_mode,
        "planner": planner_metadata,
        "evidence_invalid_llm_output_count": evidence_invalid_count,
        "verification_invalid_llm_output_count": verification_invalid_count,
        "invalid_llm_output_count": int(planner_invalid)
        + evidence_invalid_count
        + verification_invalid_count,
    }


def apply_feedback_to_pipeline_result(
    result: PipelineResult,
    feedback_records: dict[str, FeedbackRecord],
    gold_labels_path: str | None = None,
) -> PipelineResult:
    """Apply feedback to an in-memory pipeline result without rerunning retrieval."""

    feedback_agent = HumanFeedbackAgent()
    ranked_after_feedback = feedback_agent.apply(
        result.ranked_before_feedback,
        feedback_records,
    )
    metrics = compute_evaluation(
        retrieval_counts=result.retrieval_counts,
        original_paper_count=result.raw_paper_count,
        merged_papers=result.merged_papers,
        evidence_records=result.evidence_records,
        verification_results=result.verification_results,
        ranked_before_feedback=result.ranked_before_feedback,
        ranked_after_feedback=ranked_after_feedback,
        gold_labels_path=gold_labels_path,
    )
    metrics["duplicate_count"] = result.duplicate_count
    if "llm" in result.evaluation_metrics:
        metrics["llm"] = result.evaluation_metrics["llm"]

    output_dir = Path(result.output_dir)
    save_evaluation(output_dir / "evaluation.json", metrics)
    write_pipeline_csvs(
        output_dir,
        result.merged_papers,
        result.evidence_records,
        result.verification_results,
        result.ranked_before_feedback,
        ranked_after_feedback,
        ranked_after_feedback,
    )
    generate_report(
        path=output_dir / "report.md",
        research_question=result.question,
        planned_queries=result.planned_queries,
        retrieval_statistics=metrics,
        ranked_papers=ranked_after_feedback,
        evidence_records=result.evidence_records,
        evaluation_metrics=metrics,
        feedback_applied=bool(feedback_records),
    )

    return replace(
        result,
        ranked_after_feedback=ranked_after_feedback,
        ranked_final=ranked_after_feedback,
        evaluation_metrics=metrics,
    )


def run_pipeline(
    question: str,
    providers: list[str],
    max_per_query: int = 10,
    from_year: int | None = None,
    feedback_path: str | None = None,
    gold_labels_path: str | None = None,
    output_dir: str = "outputs",
    use_cache: bool = True,
    llm_backend: str = "none",
    planner_mode: str = "rule",
    extractor_mode: str = "rule",
    verifier_mode: str = "rule",
    llm_client: GenericLLMClient | None = None,
    retriever_agent: RetrieverAgent | None = None,
) -> PipelineResult:
    """Run the full MVP pipeline and write output artifacts."""

    out = ensure_dir(output_dir)
    ensure_dir("data/cache")

    config = PipelineConfig(
        providers=providers,
        max_per_query=max_per_query,
        from_year=from_year,
        output_dir=str(out),
        use_cache=use_cache,
        llm_backend=llm_backend,
    )

    active_llm_client = build_llm_client(config, llm_backend, llm_client)

    planner = PlannerAgent(mode=planner_mode, llm_client=active_llm_client)
    queries = planner.plan(question)
    write_json(
        out / "planned_queries.json",
        {
            "question": question,
            "queries": queries,
            "llm": planner.last_llm_metadata,
        },
    )

    retriever = retriever_agent or RetrieverAgent(config=config)
    raw_papers, _raw_by_provider, retrieval_counts = retriever.retrieve(
        queries=queries,
        providers=providers,
        max_per_query=max_per_query,
        from_year=from_year,
        output_dir=out,
    )

    merged_papers, duplicate_count = deduplicate_with_stats(raw_papers)

    extractor = ExtractorAgent(mode=extractor_mode, llm_client=active_llm_client)
    evidence_records = extractor.extract_many(merged_papers, question)

    verifier = VerifierAgent(mode=verifier_mode, llm_client=active_llm_client)
    verification_results = verifier.verify_many(merged_papers, evidence_records)

    ranker = RankerAgent()
    ranked_before_feedback = ranker.rank(
        merged_papers,
        evidence_records,
        verification_results,
        question,
    )

    feedback_agent = HumanFeedbackAgent()
    ranked_after_feedback: list[RankedPaper] | None = None
    ranked_final = ranked_before_feedback
    feedback_applied = False
    if feedback_path:
        feedback_records = feedback_agent.read_feedback(feedback_path)
        ranked_after_feedback = feedback_agent.apply(ranked_before_feedback, feedback_records)
        ranked_final = ranked_after_feedback
        feedback_applied = bool(feedback_records)

    metrics = compute_evaluation(
        retrieval_counts=retrieval_counts,
        original_paper_count=len(raw_papers),
        merged_papers=merged_papers,
        evidence_records=evidence_records,
        verification_results=verification_results,
        ranked_before_feedback=ranked_before_feedback,
        ranked_after_feedback=ranked_after_feedback,
        gold_labels_path=gold_labels_path,
    )
    metrics["duplicate_count"] = duplicate_count
    metrics["llm"] = _llm_metrics(
        llm_backend=llm_backend,
        active_llm_backend=active_llm_client.provider_name if active_llm_client else "none",
        planner_mode=planner_mode,
        extractor_mode=extractor_mode,
        verifier_mode=verifier_mode,
        planner_metadata=planner.last_llm_metadata,
        evidence_records=evidence_records,
        verification_results=verification_results,
    )
    save_evaluation(out / "evaluation.json", metrics)

    write_pipeline_csvs(
        out,
        merged_papers,
        evidence_records,
        verification_results,
        ranked_before_feedback,
        ranked_final,
        ranked_after_feedback,
    )

    generate_report(
        path=out / "report.md",
        research_question=question,
        planned_queries=queries,
        retrieval_statistics=metrics,
        ranked_papers=ranked_final,
        evidence_records=evidence_records,
        evaluation_metrics=metrics,
        feedback_applied=feedback_applied,
    )

    return PipelineResult(
        output_dir=str(out),
        planned_queries=queries,
        retrieval_counts=retrieval_counts,
        merged_paper_count=len(merged_papers),
        duplicate_count=duplicate_count,
        report_path=str(out / "report.md"),
        evaluation_path=str(out / "evaluation.json"),
        ranked_papers_path=str(out / "ranked_papers.csv"),
        question=question,
        raw_paper_count=len(raw_papers),
        merged_papers=merged_papers,
        evidence_records=evidence_records,
        verification_results=verification_results,
        ranked_before_feedback=ranked_before_feedback,
        ranked_after_feedback=ranked_after_feedback,
        ranked_final=ranked_final,
        evaluation_metrics=metrics,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Run the literature-screening MVP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the screening pipeline.")
    run.add_argument("--question", required=True, help="Research question to screen for.")
    run.add_argument(
        "--providers",
        nargs="+",
        default=["openalex", "semantic_scholar"],
        help="Retrieval providers to use.",
    )
    run.add_argument("--max-per-query", type=int, default=10)
    run.add_argument("--from-year", type=int, default=None)
    run.add_argument("--feedback", dest="feedback_path", default=None)
    run.add_argument("--gold-labels", dest="gold_labels_path", default=None)
    run.add_argument("--output-dir", default="outputs")
    run.add_argument(
        "--llm-backend",
        choices=["none", "deepseek"],
        default="none",
        help="Optional OpenAI-compatible LLM backend.",
    )
    run.add_argument(
        "--planner-mode",
        choices=["rule", "llm"],
        default="rule",
        help="Use rule-based or LLM-enhanced query planning.",
    )
    run.add_argument(
        "--extractor-mode",
        choices=["rule", "llm"],
        default="rule",
        help="Use rule-based or LLM-enhanced evidence extraction.",
    )
    run.add_argument(
        "--verifier-mode",
        choices=["rule", "llm"],
        default="rule",
        help="Use rule-based or LLM-enhanced evidence verification.",
    )
    run.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable local API response cache.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_pipeline(
            question=args.question,
            providers=args.providers,
            max_per_query=args.max_per_query,
            from_year=args.from_year,
            feedback_path=args.feedback_path,
            gold_labels_path=args.gold_labels_path,
            output_dir=args.output_dir,
            use_cache=not args.disable_cache,
            llm_backend=args.llm_backend,
            planner_mode=args.planner_mode,
            extractor_mode=args.extractor_mode,
            verifier_mode=args.verifier_mode,
        )
        print(f"Report: {result.report_path}")
        print(f"Ranking: {result.ranked_papers_path}")
        print(f"Evaluation: {result.evaluation_path}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
