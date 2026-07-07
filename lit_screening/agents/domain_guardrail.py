"""Domain guardrails for filtering off-topic retrieved papers."""

from __future__ import annotations

from typing import Iterable

from lit_screening.models import (
    DomainAssessment,
    DomainProfile,
    Paper,
    QueryPlan,
    SearchContract,
    is_user_seed_paper,
)
from lit_screening.utils import clamp, tokenize


class DomainGuardrailAgent:
    """Assess whether papers are in scope for a SearchContract."""

    def assess(
        self,
        paper: Paper,
        contract_or_profile: SearchContract | DomainProfile,
        query_plan: QueryPlan | None = None,
    ) -> DomainAssessment:
        """Return a domain decision for one paper."""

        if is_user_seed_paper(paper):
            return DomainAssessment(
                paper_id=paper.paper_id,
                domain_match_score=1.0,
                domain_decision="in_scope",
                off_topic_reason="User-provided seed paper retained as an in-scope anchor.",
                positive_domain_matches=["user_seed"],
                negative_domain_matches=[],
                missing_required_concepts=[],
                forbidden_concepts_found=[],
            )

        contract = contract_or_profile if isinstance(contract_or_profile, SearchContract) else None
        profile = (
            contract_or_profile.domain_profile
            if isinstance(contract_or_profile, SearchContract)
            else contract_or_profile
        )
        text = paper_domain_text(paper)
        must_groups = _must_term_groups(query_plan, profile.domain_name)
        if must_groups:
            required = [
                "any of: " + " OR ".join(group)
                for group in must_groups
                if group
            ]
            required_match_flags = [
                any(concept_matches(term, text) for term in group)
                for group in must_groups
                if group
            ]
            missing_required = [
                required[index]
                for index, matched in enumerate(required_match_flags)
                if not matched
            ]
            flattened_group_terms = [
                term for group in must_groups for term in group
            ]
        else:
            required = _unique(
                [
                    *(contract.must_include_concepts if contract else []),
                    *profile.required_concepts,
                    *(query_plan.must_terms if query_plan else []),
                ],
                limit=14,
            )
            missing_required = [
                item for item in required if not concept_matches(item, text)
            ]
            flattened_group_terms = []
        forbidden = _unique(
            [
                *(contract.must_exclude_concepts if contract else []),
                *profile.forbidden_concepts,
                *profile.negative_domains,
                *profile.field_of_study_blacklist,
            ],
            limit=20,
        )
        positive_candidates = _unique(
            [
                *profile.positive_domains,
                *profile.field_of_study_whitelist,
                *profile.preferred_venues,
                *required,
                *flattened_group_terms,
                *terminology_synonyms(profile),
            ],
            limit=40,
        )
        negative_candidates = _unique(
            [
                *profile.negative_domains,
                *profile.field_of_study_blacklist,
                *profile.excluded_venues,
                *forbidden,
            ],
            limit=40,
        )
        positive_matches = [
            item for item in positive_candidates if concept_matches(item, text)
        ]
        negative_matches = [
            item for item in negative_candidates if concept_matches(item, text)
        ]
        forbidden_found = [
            item for item in forbidden if concept_matches(item, text)
        ]
        required_coverage = (
            (len(required) - len(missing_required)) / len(required)
            if required
            else 0.5
        )
        positive_signal = min(1.0, len(positive_matches) / max(3, len(required) or 3))
        negative_penalty = min(1.0, (len(negative_matches) + len(forbidden_found)) / 3)
        score = clamp(0.75 * required_coverage + 0.25 * positive_signal - 0.55 * negative_penalty)

        if forbidden_found:
            score = min(score, 0.29)
            decision = "out_of_scope"
        elif negative_matches and score < 0.75:
            score = min(score, 0.34)
            decision = "out_of_scope"
        elif score >= 0.75:
            decision = "in_scope"
        elif score >= 0.65 and not missing_required:
            decision = "in_scope"
        elif score >= 0.45 and positive_matches:
            decision = "borderline"
        elif score >= 0.55:
            decision = "borderline"
        else:
            decision = "out_of_scope"

        reason = off_topic_reason(
            decision,
            missing_required,
            forbidden_found,
            negative_matches,
            profile.domain_name,
        )
        return DomainAssessment(
            paper_id=paper.paper_id,
            domain_match_score=round(score, 4),
            domain_decision=decision,
            off_topic_reason=reason,
            positive_domain_matches=positive_matches,
            negative_domain_matches=negative_matches,
            missing_required_concepts=missing_required,
            forbidden_concepts_found=forbidden_found,
        )

    def assess_many(
        self,
        papers: list[Paper],
        contract_or_profile: SearchContract | DomainProfile,
        query_plan: QueryPlan | None = None,
    ) -> list[DomainAssessment]:
        """Return domain decisions for many papers."""

        return [
            self.assess(paper, contract_or_profile, query_plan=query_plan)
            for paper in papers
        ]


def paper_domain_text(paper: Paper) -> str:
    """Build searchable text for domain matching."""

    raw_text = ""
    if isinstance(paper.raw, dict):
        raw_text = " ".join(str(value) for value in paper.raw.values() if isinstance(value, str))
    return " ".join(
        [
            paper.title,
            paper.abstract,
            paper.venue,
            " ".join(paper.fields_of_study),
            " ".join(paper.publication_types),
            paper.tldr,
            paper.source_provider,
            raw_text,
        ]
    ).lower()


def concept_matches(concept: str, text: str) -> bool:
    """Return True when a concept appears in paper domain text."""

    cleaned = " ".join(str(concept or "").lower().split())
    if not cleaned:
        return False
    if cleaned in text:
        return True
    concept_tokens = set(tokenize(cleaned))
    if not concept_tokens:
        return False
    text_tokens = set(tokenize(text))
    if len(concept_tokens) == 1:
        return bool(concept_tokens & text_tokens)
    return concept_tokens.issubset(text_tokens)


def terminology_synonyms(profile: DomainProfile) -> list[str]:
    """Flatten domain terminology synonyms."""

    values: list[str] = []
    for term, synonyms in profile.terminology_map.items():
        values.append(term)
        values.extend(synonyms)
    return _unique(values, limit=40)


def _must_term_groups(query_plan: QueryPlan | None, domain_name: str) -> list[list[str]]:
    """Return any-of required term groups for domains that support them."""

    if not query_plan or domain_name != "materials_magnetism":
        return []
    raw_groups = query_plan.filters.get("must_term_groups", [])
    groups: list[list[str]] = []
    if not isinstance(raw_groups, list):
        return []
    for group in raw_groups:
        if not isinstance(group, list):
            continue
        cleaned = _unique([str(item) for item in group], limit=12)
        if cleaned:
            groups.append(cleaned)
    return groups


def off_topic_reason(
    decision: str,
    missing_required: list[str],
    forbidden_found: list[str],
    negative_matches: list[str],
    domain_name: str,
) -> str:
    """Explain why a paper is in scope or demoted."""

    if forbidden_found:
        return f"Forbidden concept(s) found for {domain_name}: {', '.join(forbidden_found[:5])}."
    if negative_matches:
        return f"Negative domain signal(s) found for {domain_name}: {', '.join(negative_matches[:5])}."
    if decision == "borderline" and missing_required:
        return f"Missing some required concept(s): {', '.join(missing_required[:5])}."
    if decision == "out_of_scope" and missing_required:
        return f"Insufficient domain evidence; missing required concept(s): {', '.join(missing_required[:5])}."
    return "Paper matches the search contract domain." if decision == "in_scope" else "Weak domain match."


def _unique(values: Iterable[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result[:limit] if limit else result
