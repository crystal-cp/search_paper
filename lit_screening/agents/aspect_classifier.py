"""Aspect coverage classifier for intent-aware paper screening."""

from __future__ import annotations

import re

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
            label, synonyms = _parse_aspect(aspect)
            candidates = synonyms or [label]
            overlap = any(_term_matches(candidate, paper_text, paper_tokens) for candidate in candidates)
            similarity = max(
                [tfidf_cosine_similarity(candidate, paper_text) for candidate in candidates]
                or [0.0]
            )
            if overlap or similarity >= 0.18:
                covered.append(label)
            else:
                missing.append(label)
        return AspectCoverageRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            covered_aspects=covered,
            missing_aspects=missing,
            aspect_coverage_score=len(covered) / len(aspects) if aspects else 0.0,
        )


def _parse_aspect(aspect: str) -> tuple[str, list[str]]:
    """Parse optional 'label: synonym; synonym' aspect groups."""

    cleaned = " ".join(str(aspect or "").split())
    if not cleaned:
        return "", []
    if ":" not in cleaned:
        return cleaned, [cleaned]
    label, raw_terms = cleaned.split(":", 1)
    terms = [
        " ".join(term.split())
        for term in re.split(r"[;,|]", raw_terms)
        if " ".join(term.split())
    ]
    return " ".join(label.split()), terms


def _term_matches(term: str, paper_text: str, paper_tokens: set[str]) -> bool:
    """Return True when a term or all its tokens are present in paper text."""

    cleaned = " ".join(str(term or "").lower().split())
    if not cleaned:
        return False
    if cleaned in paper_text.lower():
        return True
    term_tokens = set(tokenize(cleaned))
    return bool(term_tokens and term_tokens <= paper_tokens)
