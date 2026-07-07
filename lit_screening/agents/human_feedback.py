"""Human feedback loading and score adjustment."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from lit_screening.models import FeedbackRecord, RankedPaper
from lit_screening.scoring import compute_final_score


VALID_LABELS = {"include", "exclude", "uncertain"}


class HumanFeedbackAgent:
    """Apply simple human score adjustments from a CSV file."""

    def read_feedback(self, path: str | Path) -> dict[str, FeedbackRecord]:
        """Read feedback rows keyed by paper_id."""

        feedback_path = Path(path)
        if not feedback_path.exists():
            return {}

        records: dict[str, FeedbackRecord] = {}
        with feedback_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                paper_id = (row.get("paper_id") or "").strip()
                if not paper_id:
                    continue
                label = (row.get("label") or "uncertain").strip().lower()
                if label not in VALID_LABELS:
                    label = "uncertain"
                adjustment_text = (row.get("adjustment") or "").strip()
                adjustment = (
                    float(adjustment_text)
                    if adjustment_text
                    else self.default_adjustment(label)
                )
                records[paper_id] = FeedbackRecord(
                    paper_id=paper_id,
                    label=label,
                    adjustment=adjustment,
                    note=row.get("note") or "",
                )
        return records

    def default_adjustment(self, label: str) -> float:
        """Map labels to conservative default score adjustments."""

        if label == "include":
            return 0.10
        if label == "exclude":
            return -0.20
        return 0.0

    def apply(
        self,
        ranked_papers: list[RankedPaper],
        feedback_records: dict[str, FeedbackRecord],
        scoring_weights: dict[str, float] | None = None,
        preference_scores: dict[str, float] | None = None,
    ) -> list[RankedPaper]:
        """Apply human feedback and return a newly sorted ranking."""

        adjusted: list[RankedPaper] = []
        learned_scores = preference_scores or {}
        for item in ranked_papers:
            feedback = feedback_records.get(item.paper.paper_id)
            preference_score = learned_scores.get(item.paper.paper_id)
            if not feedback and preference_score is None:
                adjusted.append(item)
                continue

            human_adjustment = (
                feedback.adjustment
                if feedback
                else item.scores.human_feedback_adjustment
            )
            scores = compute_final_score(
                item.scores.relevance_score,
                item.scores.evidence_score,
                item.scores.recency_score,
                item.scores.quality_score,
                item.scores.diversity_score,
                human_adjustment,
                weights=scoring_weights,
                aspect_coverage_score=item.scores.aspect_coverage_score,
                domain_penalty_multiplier=item.scores.domain_penalty_multiplier,
                preference_score=preference_score,
            )
            adjusted.append(replace(item, scores=scores, feedback=feedback))

        adjusted.sort(key=lambda item: item.scores.final_score, reverse=True)
        return [replace(item, rank=index + 1) for index, item in enumerate(adjusted)]
