"""Claim and evidence extractor with optional LLM enhancement."""

from __future__ import annotations

import json
from dataclasses import replace

from lit_screening.llm_client import GenericLLMClient
from lit_screening.models import EvidenceRecord, Paper
from lit_screening.utils import keyword_overlap_score, overlap_terms, split_sentences


class ExtractorAgent:
    """Extract the abstract sentence most relevant to the research question."""

    def __init__(
        self,
        mode: str = "rule",
        llm_client: GenericLLMClient | None = None,
    ) -> None:
        self.mode = mode
        self.llm_client = llm_client

    def extract(self, paper: Paper, question: str) -> EvidenceRecord:
        """Return one evidence record for a paper without inventing evidence."""

        if self.mode == "llm" and self.llm_client and self.llm_client.is_available:
            return self._extract_with_llm(paper, question)
        fallback = self._extract_rule(paper, question)
        if self.mode == "llm":
            return replace(fallback, llm_error_type="llm_unavailable")
        return fallback

    def _extract_rule(self, paper: Paper, question: str) -> EvidenceRecord:
        """Rule-based extraction fallback."""

        if not paper.abstract:
            return EvidenceRecord(
                paper_id=paper.paper_id,
                title=paper.title,
                claim="",
                evidence_sentence="",
                relevance_reason="No abstract available for rule-based extraction.",
                limitation="Missing abstract; no claim was generated.",
                keyword_overlap=0.0,
            )

        sentences = split_sentences(paper.abstract)
        if not sentences:
            return EvidenceRecord(
                paper_id=paper.paper_id,
                title=paper.title,
                claim="",
                evidence_sentence="",
                relevance_reason="Abstract could not be split into evidence sentences.",
                limitation="No sentence-level evidence was extracted.",
                keyword_overlap=0.0,
            )

        best_sentence = max(sentences, key=lambda sentence: keyword_overlap_score(sentence, question))
        overlap = keyword_overlap_score(best_sentence, question)
        shared_terms = overlap_terms(best_sentence, question)
        reason_terms = ", ".join(shared_terms[:8]) if shared_terms else "no direct keyword overlap"

        return EvidenceRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            claim=best_sentence,
            evidence_sentence=best_sentence,
            relevance_reason=f"Selected for overlap with question terms: {reason_terms}.",
            limitation="" if overlap > 0 else "Selected sentence has weak lexical overlap.",
            keyword_overlap=overlap,
        )

    def _extract_with_llm(self, paper: Paper, question: str) -> EvidenceRecord:
        """Use an LLM for extraction, falling back when JSON is unsafe."""

        fallback = self._extract_rule(paper, question)
        if not paper.abstract:
            return fallback

        system_prompt = (
            "You extract claim-level evidence from paper abstracts. Return JSON only "
            "with keys: claim, evidence_sentence, relevance_reason, limitation. "
            "The evidence_sentence must be copied from the abstract. If no evidence "
            "is available, return empty claim and evidence_sentence with a limitation."
        )
        user_prompt = json.dumps(
            {
                "research_question": question,
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
            },
            ensure_ascii=False,
        )
        result = self.llm_client.chat_json(system_prompt, user_prompt)
        if result.invalid_llm_output:
            return replace(
                fallback,
                llm_used=True,
                invalid_llm_output=True,
                llm_error_type=result.error_type,
            )

        evidence_sentence = str(result.data.get("evidence_sentence") or "").strip()
        claim = str(result.data.get("claim") or evidence_sentence).strip()
        relevance_reason = str(result.data.get("relevance_reason") or "").strip()
        limitation = str(result.data.get("limitation") or "").strip()

        if not evidence_sentence:
            return replace(
                fallback,
                llm_used=True,
                invalid_llm_output=True,
                llm_error_type="missing_evidence_sentence",
            )

        return EvidenceRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            claim=claim,
            evidence_sentence=evidence_sentence,
            relevance_reason=relevance_reason or "LLM selected abstract evidence.",
            limitation=limitation,
            keyword_overlap=keyword_overlap_score(evidence_sentence, question),
            extraction_mode="llm",
            llm_used=True,
            invalid_llm_output=False,
            llm_error_type="",
        )

    def extract_many(self, papers: list[Paper], question: str) -> list[EvidenceRecord]:
        """Extract evidence records for multiple papers."""

        return [self.extract(paper, question) for paper in papers]
