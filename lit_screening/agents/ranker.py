"""Ranking agent for screened papers."""

from __future__ import annotations

from dataclasses import replace

from lit_screening.dedup import normalize_title
from lit_screening.models import EvidenceRecord, Paper, QueryPlan, RankedPaper, VerificationResult
from lit_screening.scoring import score_paper


class RankerAgent:
    """Rank papers with transparent score components."""

    def rank(
        self,
        papers: list[Paper],
        evidence_records: list[EvidenceRecord],
        verification_results: list[VerificationResult],
        question: str,
        scoring_weights: dict[str, float] | None = None,
        query_plan: QueryPlan | None = None,
        ranking_profile: str = "balanced",
    ) -> list[RankedPaper]:
        """Return papers sorted by final score."""

        evidence_by_id = {record.paper_id: record for record in evidence_records}
        verification_by_id = {result.paper_id: result for result in verification_results}

        preliminary: list[RankedPaper] = []
        for paper in papers:
            evidence = evidence_by_id[paper.paper_id]
            verification = verification_by_id[paper.paper_id]
            scores = score_paper(
                paper,
                evidence,
                verification,
                question,
                diversity_score=0.5,
                weights=scoring_weights,
                query_plan=query_plan,
                ranking_profile=ranking_profile,
            )
            preliminary.append(RankedPaper(0, paper, evidence, verification, scores))

        preliminary.sort(key=lambda item: item.scores.final_score, reverse=True)

        venue_seen: dict[str, int] = {}
        reranked: list[RankedPaper] = []
        for item in preliminary:
            venue_key = normalize_title(item.paper.venue) or "unknown"
            count = venue_seen.get(venue_key, 0)
            diversity = 1.0 / (1.0 + count)
            venue_seen[venue_key] = count + 1
            scores = score_paper(
                item.paper,
                item.evidence,
                item.verification,
                question,
                diversity_score=diversity,
                weights=scoring_weights,
                query_plan=query_plan,
                ranking_profile=ranking_profile,
            )
            reranked.append(replace(item, scores=scores))

        reranked.sort(key=lambda item: item.scores.final_score, reverse=True)
        return [replace(item, rank=index + 1) for index, item in enumerate(reranked)]
