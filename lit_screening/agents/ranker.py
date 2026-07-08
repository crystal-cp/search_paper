"""Ranking agent for screened papers."""

from __future__ import annotations

from dataclasses import replace

from lit_screening.dedup import normalize_title
from lit_screening.models import (
    AspectCoverageRecord,
    DomainAssessment,
    EvidenceRecord,
    Paper,
    QueryPlan,
    RankedPaper,
    VerificationResult,
    is_user_seed_paper,
    source_stage_values,
)
from lit_screening.scoring import compute_score_breakdown


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
        aspect_coverage_records: list[AspectCoverageRecord] | None = None,
        domain_assessments: list[DomainAssessment] | None = None,
        use_intent_centrality: bool = True,
        use_group_coverage_ranking: bool = True,
    ) -> list[RankedPaper]:
        """Return papers sorted by final score."""

        evidence_by_id = {record.paper_id: record for record in evidence_records}
        verification_by_id = {result.paper_id: result for result in verification_results}
        aspect_by_id = {
            record.paper_id: record
            for record in aspect_coverage_records or []
        }
        domain_by_id = {
            record.paper_id: record
            for record in domain_assessments or []
        }

        preliminary: list[RankedPaper] = []
        for paper in papers:
            evidence = evidence_by_id[paper.paper_id]
            verification = verification_by_id[paper.paper_id]
            domain_assessment = domain_by_id.get(paper.paper_id)
            scores = compute_score_breakdown(
                paper,
                evidence,
                verification,
                question,
                diversity_score=0.5,
                weights=scoring_weights,
                query_plan=query_plan,
                ranking_profile=ranking_profile,
                aspect_coverage_score=aspect_by_id.get(
                    paper.paper_id,
                    AspectCoverageRecord(paper.paper_id, paper.title),
                ).aspect_coverage_score,
                domain_assessment=domain_assessment,
                use_intent_centrality=use_intent_centrality,
                use_group_coverage_ranking=use_group_coverage_ranking,
            )
            scores = seed_score_floor(paper, scores)
            preliminary.append(
                RankedPaper(
                    0,
                    paper,
                    evidence,
                    verification,
                    scores,
                    domain_assessment=domain_assessment,
                )
            )

        preliminary.sort(key=lambda item: item.scores.final_score, reverse=True)

        venue_seen: dict[str, int] = {}
        reranked: list[RankedPaper] = []
        for item in preliminary:
            venue_key = normalize_title(item.paper.venue) or "unknown"
            count = venue_seen.get(venue_key, 0)
            diversity = 1.0 / (1.0 + count)
            venue_seen[venue_key] = count + 1
            scores = compute_score_breakdown(
                item.paper,
                item.evidence,
                item.verification,
                question,
                diversity_score=diversity,
                weights=scoring_weights,
                query_plan=query_plan,
                ranking_profile=ranking_profile,
                aspect_coverage_score=aspect_by_id.get(
                    item.paper.paper_id,
                    AspectCoverageRecord(item.paper.paper_id, item.paper.title),
                ).aspect_coverage_score,
                domain_assessment=item.domain_assessment,
                use_intent_centrality=use_intent_centrality,
                use_group_coverage_ranking=use_group_coverage_ranking,
            )
            scores = seed_score_floor(item.paper, scores)
            reranked.append(replace(item, scores=scores))

        reranked.sort(key=lambda item: item.scores.final_score, reverse=True)
        return [replace(item, rank=index + 1) for index, item in enumerate(reranked)]


def seed_score_floor(paper: Paper, scores):
    """Keep user-provided seed anchors at the top of the ranked list."""

    if not is_user_seed_paper(paper):
        return scores
    stages = set(source_stage_values(paper.source_stage))
    floor = 0.995 if "seed_exact" in stages else 0.985
    if scores.final_score >= floor:
        return scores
    return replace(
        scores,
        final_score=floor,
        pre_domain_final_score=max(scores.pre_domain_final_score, floor),
        domain_penalty_multiplier=1.0,
    )
