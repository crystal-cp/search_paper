"""Aspect coverage classifier for intent-aware paper screening."""

from __future__ import annotations

from lit_screening.models import AspectCoverageRecord, EvidenceRecord, Paper
from lit_screening.reranking import tfidf_cosine_similarity
from lit_screening.utils import tokenize


class AspectCoverageAgent:
    """Classify which required research aspects are covered by each paper."""

    def classify_many(
        self,
        papers: list[Paper],
        evidence_records: list[EvidenceRecord],
        required_aspects: list[str],
    ) -> list[AspectCoverageRecord]:
        """Return aspect-coverage records for a paper list."""

        evidence_by_id = {record.paper_id: record for record in evidence_records}
        return [
            self.classify(
                paper,
                evidence_by_id.get(paper.paper_id),
                required_aspects,
            )
            for paper in papers
        ]

    def classify(
        self,
        paper: Paper,
        evidence: EvidenceRecord | None,
        required_aspects: list[str],
    ) -> AspectCoverageRecord:
        """Classify aspect coverage for one paper."""

        aspects = [aspect for aspect in required_aspects if aspect]
        if not aspects:
            return AspectCoverageRecord(
                paper_id=paper.paper_id,
                title=paper.title,
                covered_aspects=[],
                missing_aspects=[],
                aspect_coverage_score=0.0,
            )
        evidence_text = ""
        if evidence:
            evidence_text = " ".join([evidence.claim, evidence.evidence_sentence])
        paper_text = " ".join(
            [
                paper.title,
                paper.abstract,
                paper.tldr,
                " ".join(paper.fields_of_study),
                evidence_text,
            ]
        )
        covered: list[str] = []
        missing: list[str] = []
        paper_tokens = set(tokenize(paper_text))
        for aspect in aspects:
            aspect_tokens = set(tokenize(aspect))
            overlap = bool(aspect_tokens and aspect_tokens <= paper_tokens)
            similarity = tfidf_cosine_similarity(aspect, paper_text)
            if overlap or similarity >= 0.18:
                covered.append(aspect)
            else:
                missing.append(aspect)
        return AspectCoverageRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            covered_aspects=covered,
            missing_aspects=missing,
            aspect_coverage_score=len(covered) / len(aspects) if aspects else 0.0,
        )
