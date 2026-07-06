"""Evaluation metrics for the literature-screening pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .models import EvidenceRecord, Paper, RankedPaper, VerificationResult
from .utils import write_json


def load_gold_labels(path: str | Path | None) -> dict[str, str]:
    """Load gold labels from CSV if a path is provided."""

    if not path:
        return {}
    gold_path = Path(path)
    if not gold_path.exists():
        return {}

    labels: dict[str, str] = {}
    with gold_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            paper_id = (row.get("paper_id") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if paper_id and label:
                labels[paper_id] = label
    return labels


def precision_at_k(
    ranked_papers: list[RankedPaper],
    gold_labels: dict[str, str],
    k: int = 10,
) -> float | None:
    """Compute Precision@k for papers with available gold labels."""

    if not gold_labels:
        return None
    top_k = ranked_papers[:k]
    if not top_k:
        return 0.0
    positives = 0
    judged = 0
    for item in top_k:
        label = gold_labels.get(item.paper.paper_id)
        if not label:
            continue
        judged += 1
        if label == "include":
            positives += 1
    if judged == 0:
        return None
    return positives / judged


def ranking_changes(
    before: list[RankedPaper],
    after: list[RankedPaper] | None,
) -> dict[str, Any]:
    """Summarize rank shifts caused by human feedback."""

    if not after:
        return {"feedback_applied": False, "moved_count": 0, "top_changes": []}

    before_rank = {item.paper.paper_id: item.rank for item in before}
    changes = []
    for item in after:
        old_rank = before_rank.get(item.paper.paper_id)
        if old_rank is None or old_rank == item.rank:
            continue
        changes.append(
            {
                "paper_id": item.paper.paper_id,
                "title": item.paper.title,
                "before_rank": old_rank,
                "after_rank": item.rank,
                "delta": old_rank - item.rank,
            }
        )
    changes.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return {
        "feedback_applied": True,
        "moved_count": len(changes),
        "top_changes": changes[:10],
    }


def compute_evaluation(
    retrieval_counts: dict[str, int],
    original_paper_count: int,
    merged_papers: list[Paper],
    evidence_records: list[EvidenceRecord],
    verification_results: list[VerificationResult],
    ranked_before_feedback: list[RankedPaper],
    ranked_after_feedback: list[RankedPaper] | None = None,
    gold_labels_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute pipeline-level evaluation and audit metrics."""

    merged_count = len(merged_papers)
    duplicate_count = max(0, original_paper_count - merged_count)
    missing_abstract_count = sum(1 for paper in merged_papers if not paper.abstract)
    unsupported_count = sum(1 for result in verification_results if not result.supported)
    support_level_counts: dict[str, int] = {}
    for result in verification_results:
        support_level_counts[result.support_level] = (
            support_level_counts.get(result.support_level, 0) + 1
        )
    gold_labels = load_gold_labels(gold_labels_path)
    final_ranking = ranked_after_feedback or ranked_before_feedback

    return {
        "retrieval_counts_by_provider": retrieval_counts,
        "raw_retrieved_paper_count": original_paper_count,
        "merged_paper_count": merged_count,
        "duplicate_count": duplicate_count,
        "missing_abstract_ratio": missing_abstract_count / merged_count if merged_count else 0.0,
        "unsupported_claim_rate": unsupported_count / len(verification_results)
        if verification_results
        else 0.0,
        "support_level_counts": support_level_counts,
        "strict_supported_count": support_level_counts.get("strict_support", 0),
        "weak_support_count": support_level_counts.get("weak_support", 0),
        "unverified_count": support_level_counts.get("unverified", 0),
        "llm_invalid_evidence_count": support_level_counts.get("llm_invalid_evidence", 0),
        "precision_at_10": precision_at_k(final_ranking, gold_labels, 10),
        "ranking_changes": ranking_changes(ranked_before_feedback, ranked_after_feedback),
        "evidence_record_count": len(evidence_records),
    }


def save_evaluation(path: str | Path, metrics: dict[str, Any]) -> None:
    """Save evaluation metrics as JSON."""

    write_json(path, metrics)
