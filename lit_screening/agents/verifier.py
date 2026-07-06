"""Evidence verifier with optional LLM enhancement."""

from __future__ import annotations

import json
from dataclasses import replace

from lit_screening.llm_client import GenericLLMClient
from lit_screening.models import EvidenceRecord, Paper, VerificationResult
from lit_screening.utils import clamp, keyword_overlap_score, tokenize


class VerifierAgent:
    """Check whether extracted evidence is grounded in a paper abstract."""

    def __init__(
        self,
        mode: str = "rule",
        llm_client: GenericLLMClient | None = None,
    ) -> None:
        self.mode = mode
        self.llm_client = llm_client

    def verify(self, paper: Paper, evidence: EvidenceRecord) -> VerificationResult:
        """Verify one evidence record against the paper abstract."""

        if self.mode == "llm" and self.llm_client and self.llm_client.is_available:
            return self._verify_with_llm(paper, evidence)
        fallback = self._verify_rule(paper, evidence)
        if self.mode == "llm":
            return replace(fallback, llm_error_type="llm_unavailable")
        return fallback

    def _verify_rule(self, paper: Paper, evidence: EvidenceRecord) -> VerificationResult:
        """Rule-based verification fallback."""

        if not paper.abstract:
            return VerificationResult(
                paper_id=paper.paper_id,
                supported=False,
                confidence=0.0,
                error_type="missing_abstract",
                rationale="The paper has no abstract, so the claim cannot be grounded.",
            )
        if not evidence.evidence_sentence:
            return VerificationResult(
                paper_id=paper.paper_id,
                supported=False,
                confidence=0.0,
                error_type="missing_evidence",
                rationale="No evidence sentence was extracted.",
            )

        abstract = " ".join(paper.abstract.lower().split())
        sentence = " ".join(evidence.evidence_sentence.lower().split())
        if sentence and sentence in abstract:
            return VerificationResult(
                paper_id=paper.paper_id,
                supported=True,
                confidence=1.0,
                error_type="",
                rationale="The evidence sentence appears verbatim in the abstract.",
            )

        sentence_terms = set(tokenize(evidence.evidence_sentence))
        abstract_terms = set(tokenize(paper.abstract))
        if not sentence_terms:
            overlap = 0.0
        else:
            overlap = len(sentence_terms & abstract_terms) / len(sentence_terms)
        if overlap >= 0.65:
            return VerificationResult(
                paper_id=paper.paper_id,
                supported=True,
                confidence=clamp(0.5 + 0.5 * overlap),
                error_type="",
                rationale="The evidence sentence has high keyword overlap with the abstract.",
            )

        return VerificationResult(
            paper_id=paper.paper_id,
            supported=False,
            confidence=keyword_overlap_score(evidence.evidence_sentence, paper.abstract),
            error_type="low_overlap",
            rationale="The evidence sentence is not sufficiently grounded in the abstract.",
        )

    def _verify_with_llm(
        self,
        paper: Paper,
        evidence: EvidenceRecord,
    ) -> VerificationResult:
        """Use an LLM for verification, falling back when JSON is unsafe."""

        fallback = self._verify_rule(paper, evidence)
        if not paper.abstract or not evidence.evidence_sentence:
            return fallback

        system_prompt = (
            "You verify whether extracted evidence is grounded in a paper abstract. "
            "Return JSON only with keys: supported (boolean), confidence (0 to 1), "
            "error_type, rationale. Use error_type '' when supported; otherwise use "
            "missing_abstract, missing_evidence, low_overlap, or unsupported_claim."
        )
        user_prompt = json.dumps(
            {
                "title": paper.title,
                "abstract": paper.abstract,
                "claim": evidence.claim,
                "evidence_sentence": evidence.evidence_sentence,
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

        supported_value = result.data.get("supported")
        if isinstance(supported_value, bool):
            supported = supported_value
        elif isinstance(supported_value, str):
            supported = supported_value.strip().lower() == "true"
        else:
            return replace(
                fallback,
                llm_used=True,
                invalid_llm_output=True,
                llm_error_type="missing_supported",
            )

        try:
            confidence = clamp(float(result.data.get("confidence", 0.0)))
        except (TypeError, ValueError):
            return replace(
                fallback,
                llm_used=True,
                invalid_llm_output=True,
                llm_error_type="invalid_confidence",
            )

        error_type = str(result.data.get("error_type") or "").strip()
        rationale = str(result.data.get("rationale") or "").strip()
        return VerificationResult(
            paper_id=paper.paper_id,
            supported=supported,
            confidence=confidence,
            error_type=error_type if not supported else "",
            rationale=rationale or "LLM verification result.",
            verification_mode="llm",
            llm_used=True,
            invalid_llm_output=False,
            llm_error_type="",
        )

    def verify_many(
        self,
        papers: list[Paper],
        evidence_records: list[EvidenceRecord],
    ) -> list[VerificationResult]:
        """Verify multiple papers by paper_id."""

        paper_by_id = {paper.paper_id: paper for paper in papers}
        return [
            self.verify(paper_by_id[evidence.paper_id], evidence)
            for evidence in evidence_records
            if evidence.paper_id in paper_by_id
        ]
