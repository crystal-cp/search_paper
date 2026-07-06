"""Evaluation metrics for the literature-screening pipeline."""

from __future__ import annotations

import csv
import math
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


def _is_positive_label(label: str | None) -> bool:
    return (label or "").strip().lower() == "include"


def _relevance_vector(
    ranked_papers: list[RankedPaper],
    gold_labels: dict[str, str],
    k: int | None = None,
) -> list[int]:
    """Return binary relevance values for a ranked list."""

    subset = ranked_papers[:k] if k else ranked_papers
    return [
        1 if _is_positive_label(gold_labels.get(item.paper.paper_id)) else 0
        for item in subset
    ]


def ndcg_at_k(
    ranked_papers: list[RankedPaper],
    gold_labels: dict[str, str],
    k: int = 10,
) -> float | None:
    """Compute binary nDCG@k using include labels as relevant."""

    if not gold_labels:
        return None
    gains = _relevance_vector(ranked_papers, gold_labels, k)
    if not gains:
        return 0.0
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    total_relevant = sum(1 for label in gold_labels.values() if _is_positive_label(label))
    ideal_gains = [1] * min(total_relevant, k)
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    if idcg == 0:
        return None
    return dcg / idcg


def average_precision(
    ranked_papers: list[RankedPaper],
    gold_labels: dict[str, str],
) -> float | None:
    """Compute average precision over available include labels."""

    if not gold_labels:
        return None
    total_relevant = sum(1 for label in gold_labels.values() if _is_positive_label(label))
    if total_relevant == 0:
        return None
    hits = 0
    precision_sum = 0.0
    for index, item in enumerate(ranked_papers, start=1):
        if _is_positive_label(gold_labels.get(item.paper.paper_id)):
            hits += 1
            precision_sum += hits / index
    return precision_sum / total_relevant


def recall_at_k(
    ranked_papers: list[RankedPaper],
    gold_labels: dict[str, str],
    k: int = 10,
) -> float | None:
    """Compute Recall@k using include labels as relevant."""

    if not gold_labels:
        return None
    total_relevant = sum(1 for label in gold_labels.values() if _is_positive_label(label))
    if total_relevant == 0:
        return None
    hits = sum(_relevance_vector(ranked_papers, gold_labels, k))
    return hits / total_relevant


def ranking_changes(
    before: list[RankedPaper],
    after: list[RankedPaper] | None,
) -> dict[str, Any]:
    """Summarize rank shifts caused by human feedback."""

    if not after:
        return {
            "feedback_applied": False,
            "moved_count": 0,
            "mean_abs_rank_delta": 0.0,
            "max_abs_rank_delta": 0,
            "top10_changed_count": 0,
            "top_changes": [],
        }

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
    abs_deltas = [abs(row["delta"]) for row in changes]
    top10_before = {item.paper.paper_id for item in before[:10]}
    top10_after = {item.paper.paper_id for item in after[:10]}
    return {
        "feedback_applied": True,
        "moved_count": len(changes),
        "mean_abs_rank_delta": sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0,
        "max_abs_rank_delta": max(abs_deltas) if abs_deltas else 0,
        "top10_changed_count": len(top10_before ^ top10_after),
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
    verification_count = len(verification_results)
    strict_count = support_level_counts.get("strict_support", 0)
    weak_count = support_level_counts.get("weak_support", 0)
    llm_invalid_count = support_level_counts.get("llm_invalid_evidence", 0)
    auditable_count = sum(
        1
        for result in verification_results
        if result.support_level not in {"missing_abstract", "missing_evidence"}
    )
    changes = ranking_changes(ranked_before_feedback, ranked_after_feedback)

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
        "strict_supported_count": strict_count,
        "weak_support_count": weak_count,
        "unverified_count": support_level_counts.get("unverified", 0),
        "llm_invalid_evidence_count": llm_invalid_count,
        "grounding_accuracy": strict_count / auditable_count if auditable_count else 0.0,
        "strict_support_rate": strict_count / verification_count if verification_count else 0.0,
        "weak_support_rate": weak_count / verification_count if verification_count else 0.0,
        "llm_invalid_evidence_rate": llm_invalid_count / verification_count
        if verification_count
        else 0.0,
        "precision_at_10": precision_at_k(final_ranking, gold_labels, 10),
        "ndcg_at_10": ndcg_at_k(final_ranking, gold_labels, 10),
        "map": average_precision(final_ranking, gold_labels),
        "recall_at_10": recall_at_k(final_ranking, gold_labels, 10),
        "ranking_changes": changes,
        "feedback_before_after_ranking_delta": changes,
        "evidence_record_count": len(evidence_records),
    }


def save_evaluation(path: str | Path, metrics: dict[str, Any]) -> None:
    """Save evaluation metrics as JSON."""

    write_json(path, metrics)
