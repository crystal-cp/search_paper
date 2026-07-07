"""Screening decisions for include/maybe/exclude recommendations."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from lit_screening.models import (
    AspectCoverageRecord,
    RankedPaper,
    ScreeningDecision,
    is_user_seed_paper,
    source_stage_values,
)
from lit_screening.utils import clamp


class ScreeningDecisionAgent:
    """Turn ranked papers into explicit screening recommendations."""

    def decide_many(
        self,
        ranked_papers: list[RankedPaper],
        aspect_coverage_records: list[AspectCoverageRecord] | None = None,
        duplicate_paper_ids: set[str] | None = None,
    ) -> tuple[list[ScreeningDecision], list[RankedPaper]]:
        """Return decisions and ranked papers carrying those decisions."""

        aspect_by_id = {
            record.paper_id: record
            for record in aspect_coverage_records or []
        }
        duplicate_ids = duplicate_paper_ids or set()
        decisions: list[ScreeningDecision] = []
        updated: list[RankedPaper] = []
        for item in ranked_papers:
            decision = self.decide(
                item,
                aspect_by_id.get(item.paper.paper_id),
                is_duplicate=item.paper.paper_id in duplicate_ids,
            )
            decisions.append(decision)
            updated.append(replace(item, screening_decision=decision))
        return decisions, updated

    def decide(
        self,
        item: RankedPaper,
        aspect: AspectCoverageRecord | None = None,
        is_duplicate: bool = False,
    ) -> ScreeningDecision:
        """Return one screening decision."""

        domain = item.domain_assessment
        covered = aspect.covered_aspects if aspect else []
        missing = aspect.missing_aspects if aspect else []
        aspect_score = aspect.aspect_coverage_score if aspect else 1.0
        reasons = exclusion_reasons_for_item(item, missing, is_duplicate)
        score = item.scores.final_score
        support_level = item.verification.support_level
        domain_decision = domain.domain_decision if domain else "unknown"
        domain_score = domain.domain_match_score if domain else 0.0

        if is_user_seed_paper(item.paper):
            stages = set(source_stage_values(item.paper.source_stage))
            primary = "seed_exact_match" if "seed_exact" in stages else "user_seed"
            return ScreeningDecision(
                paper_id=item.paper.paper_id,
                decision="include",
                decision_confidence=0.98,
                primary_reason=primary,
                exclusion_reasons=[],
                required_aspects_covered=covered,
                required_aspects_missing=missing,
                domain_match_score=max(domain_score, 1.0),
                domain_decision="in_scope"
                if domain_decision in {"unknown", "out_of_scope", "borderline"}
                else domain_decision,
                reading_priority="must_read",
                suggested_action="include",
            )

        if is_duplicate:
            decision = "exclude"
            primary = "duplicate"
        elif domain_decision == "out_of_scope":
            decision = "exclude"
            primary = "off_topic_domain"
        elif support_level in {"missing_abstract", "missing_evidence"}:
            decision = "maybe" if score >= 0.35 and domain_decision != "out_of_scope" else "exclude"
            primary = "missing_abstract" if support_level == "missing_abstract" else "weak_evidence_grounding"
        elif support_level in {"unverified", "llm_invalid_evidence"}:
            decision = "maybe" if score >= 0.45 and domain_decision == "in_scope" else "exclude"
            primary = "weak_evidence_grounding"
        elif domain_decision == "borderline":
            decision = "maybe"
            primary = "off_topic_domain" if reasons else "borderline_domain"
        elif score >= 0.55 and support_level == "strict_support" and aspect_score >= 0.4:
            decision = "include"
            primary = "high_score_in_scope_grounded_evidence"
        elif score >= 0.45 and support_level in {"strict_support", "weak_support"}:
            decision = "maybe"
            primary = "partial_match_needs_inspection"
        else:
            decision = "exclude" if score < 0.25 else "maybe"
            primary = "low_metadata_quality" if score < 0.25 else "partial_match_needs_inspection"

        if primary not in reasons and primary not in {
            "high_score_in_scope_grounded_evidence",
            "partial_match_needs_inspection",
            "borderline_domain",
        }:
            reasons.insert(0, primary)
        confidence = decision_confidence(item, aspect_score, decision)
        return ScreeningDecision(
            paper_id=item.paper.paper_id,
            decision=decision,
            decision_confidence=confidence,
            primary_reason=primary,
            exclusion_reasons=reasons,
            required_aspects_covered=covered,
            required_aspects_missing=missing,
            domain_match_score=domain_score,
            domain_decision=domain_decision,
            reading_priority=reading_priority(decision, score, domain_decision),
            suggested_action=suggested_action(decision, support_level, item.paper.abstract),
        )


def exclusion_reasons_for_item(
    item: RankedPaper,
    missing_aspects: list[str],
    is_duplicate: bool,
) -> list[str]:
    """Collect normalized exclusion reasons for one item."""

    reasons: list[str] = []
    if is_duplicate:
        reasons.append("duplicate")
    domain = item.domain_assessment
    if domain:
        if domain.domain_decision == "out_of_scope":
            reasons.append("off_topic_domain")
        if domain.forbidden_concepts_found:
            reasons.extend(reason_from_forbidden(term) for term in domain.forbidden_concepts_found)
        if domain.missing_required_concepts:
            reasons.append("missing_required_concept")
    if not item.paper.abstract:
        reasons.append("missing_abstract")
    if item.verification.support_level in {
        "missing_evidence",
        "unverified",
        "weak_support",
        "llm_invalid_evidence",
    }:
        reasons.append("weak_evidence_grounding")
    for aspect in missing_aspects:
        mapped = reason_from_missing_aspect(aspect)
        if mapped:
            reasons.append(mapped)
    if item.scores.final_score < 0.2:
        reasons.append("low_metadata_quality")
    return unique_reasons(reasons)


def reason_from_forbidden(term: str) -> str:
    """Map forbidden concept hits to normalized exclusion reasons."""

    lowered = term.lower()
    if "patient screening" in lowered or "drug screening" in lowered or "biomarker" in lowered:
        return "ambiguous_screening_meaning"
    if "materials screening" in lowered or "clinical" in lowered or "medicine" in lowered:
        return "wrong_application_domain"
    if "llm" in lowered or "large language model" in lowered:
        return "general_llm_only"
    return "off_topic_domain"


def reason_from_missing_aspect(aspect: str) -> str:
    """Map missing required aspects to normalized exclusion reasons."""

    lowered = aspect.lower()
    if "multi" in lowered and "agent" in lowered:
        return "no_multi_agent_component"
    if "human feedback" in lowered or "human-in-the-loop" in lowered:
        return "no_human_feedback_component"
    if "evidence" in lowered or "verification" in lowered:
        return "no_evidence_verification_component"
    return "missing_required_concept"


def decision_confidence(
    item: RankedPaper,
    aspect_score: float,
    decision: str,
) -> float:
    """Estimate confidence in the screening decision."""

    domain_score = item.domain_assessment.domain_match_score if item.domain_assessment else 0.5
    evidence = item.scores.evidence_score
    score = item.scores.final_score
    if decision == "include":
        return clamp(0.35 * score + 0.30 * domain_score + 0.20 * evidence + 0.15 * aspect_score)
    if decision == "exclude":
        penalty_confidence = 1.0 - item.scores.domain_penalty_multiplier
        return clamp(0.35 * penalty_confidence + 0.25 * (1 - score) + 0.25 * (1 - domain_score) + 0.15 * (1 - evidence))
    return clamp(0.45 + 0.25 * (1 - abs(score - 0.5)) + 0.15 * (1 - abs(domain_score - 0.5)))


def reading_priority(decision: str, score: float, domain_decision: str) -> str:
    """Map screening decision to reading priority."""

    if decision == "exclude":
        return "exclude"
    if decision == "include" and score >= 0.7:
        return "must_read"
    if decision == "include":
        return "read_later"
    if domain_decision == "borderline":
        return "optional"
    return "read_later"


def suggested_action(decision: str, support_level: str, abstract: str) -> str:
    """Suggest the next human action."""

    if decision == "include":
        return "include"
    if decision == "exclude":
        return "exclude"
    if not abstract or support_level in {"missing_abstract", "unverified", "weak_support"}:
        return "inspect_full_text"
    return "uncertain"


def summarize_screening_decisions(
    decisions: list[ScreeningDecision],
) -> dict[str, Any]:
    """Build counts and common exclusion reasons for evaluation/reporting."""

    decision_counts = Counter(decision.decision for decision in decisions)
    reason_counts = Counter(
        reason
        for decision in decisions
        for reason in decision.exclusion_reasons
    )
    return {
        "decision_counts": dict(decision_counts),
        "common_exclusion_reasons": dict(reason_counts.most_common(12)),
    }


def unique_reasons(values: list[str]) -> list[str]:
    """Return normalized unique reason codes."""

    result: list[str] = []
    for value in values:
        cleaned = "_".join(str(value).strip().lower().split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
