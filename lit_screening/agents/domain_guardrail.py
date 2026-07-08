"""Domain guardrails for filtering off-topic retrieved papers."""

from __future__ import annotations

import html
import re
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
                required_group_matches={"user_seed": True},
                optional_group_matches={},
                intent_centrality_score=1.0,
                required_group_coverage_score=1.0,
                missing_required_group_count=0,
                group_coverage_explanation=["user_seed: matched"],
                matched_groups=["user_seed"],
                missing_groups=[],
                target_context_match=["user_seed"],
                negative_context_match=[],
                topic_focus_score=1.0,
                aspect_match_score=1.0,
                target_context_score=1.0,
                negative_context_penalty=0.0,
                peripheral_context_reason="",
            )

        contract = contract_or_profile if isinstance(contract_or_profile, SearchContract) else None
        profile = (
            contract_or_profile.domain_profile
            if isinstance(contract_or_profile, SearchContract)
            else contract_or_profile
        )
        text = paper_domain_text(paper)
        must_groups = _must_term_groups(contract, query_plan)
        optional_groups = _optional_term_groups(contract)
        if must_groups:
            required = [
                f"{name}: " + " OR ".join(group)
                for name, group in must_groups
                if group
            ]
            required_group_matches = {
                name: any(concept_matches(term, text) for term in group)
                for name, group in must_groups
                if group
            }
            optional_group_matches = {
                name: any(concept_matches(term, text) for term in group)
                for name, group in optional_groups
                if group
            }
            required_match_flags = list(required_group_matches.values())
            missing_required = [
                required[index]
                for index, matched in enumerate(required_match_flags)
                if not matched
            ]
            flattened_group_terms = [
                term for _name, group in must_groups for term in group
            ]
        else:
            required_group_matches = {}
            optional_group_matches = {
                name: any(concept_matches(term, text) for term in group)
                for name, group in optional_groups
                if group
            }
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
        matched_groups = [
            name for name, matched in required_group_matches.items() if matched
        ]
        missing_groups = [
            name for name, matched in required_group_matches.items() if not matched
        ]
        target_context_match, negative_context_match, target_context_score = (
            _target_context_signals(contract, text)
        )
        lithium_drift = _lithium_context_drift(
            contract,
            text,
            required_group_matches,
            negative_context_match=negative_context_match,
        )
        if lithium_drift:
            weak_negative_candidates = _unique(
                [
                    *profile.negative_domains,
                    *profile.field_of_study_blacklist,
                    *profile.excluded_venues,
                    *lithium_drift,
                ],
                limit=24,
            )
        else:
            weak_negative_candidates = _unique(
                [
                    *profile.negative_domains,
                    *profile.field_of_study_blacklist,
                    *profile.excluded_venues,
                ],
                limit=20,
            )
        hard_forbidden = _unique(
            [
                *(contract.must_exclude_concepts if contract else []),
                *profile.forbidden_concepts,
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
        positive_matches = [
            item for item in positive_candidates if concept_matches(item, text)
        ]
        weak_negative_matches = [
            item for item in weak_negative_candidates if concept_matches(item, text)
        ]
        forbidden_found = [
            item for item in hard_forbidden if concept_matches(item, text)
        ]
        required_coverage = (
            (len(required) - len(missing_required)) / len(required)
            if required
            else 0.5
        )
        required_group_coverage = (
            sum(1 for matched in required_group_matches.values() if matched)
            / len(required_group_matches)
            if required_group_matches
            else required_coverage
        )
        optional_group_coverage = (
            sum(1 for matched in optional_group_matches.values() if matched)
            / len(optional_group_matches)
            if optional_group_matches
            else 0.0
        )
        missing_required_group_count = sum(
            1 for matched in required_group_matches.values() if not matched
        )
        topic_focus_score = _topic_focus_score(
            paper=paper,
            must_groups=must_groups,
            required_group_matches=required_group_matches,
            text=text,
            negative_context_match=negative_context_match,
        )
        aspect_match_score = optional_group_coverage if optional_groups else required_group_coverage
        negative_context_penalty = min(0.55, len(negative_context_match) * 0.22)
        peripheral_context_reason = _peripheral_context_reason(
            contract=contract,
            target_context_match=target_context_match,
            negative_context_match=negative_context_match,
            topic_focus_score=topic_focus_score,
        )
        intent_centrality = clamp(
            0.45 * required_group_coverage
            + 0.30 * topic_focus_score
            + 0.15 * aspect_match_score
            + 0.10 * target_context_score
            - negative_context_penalty
        )
        if lithium_drift or negative_context_match:
            intent_centrality = min(intent_centrality, 0.45)
        if peripheral_context_reason:
            intent_centrality = min(intent_centrality, 0.62)
        positive_signal = min(1.0, len(positive_matches) / max(3, len(required) or 3))
        strong_positive_evidence = bool(positive_matches) and (
            required_coverage >= 0.5 or len(positive_matches) >= 2
        )
        weak_negative_penalty = 0.0 if strong_positive_evidence else min(
            0.35,
            len(weak_negative_matches) * 0.12,
        )
        hard_negative_penalty = min(0.75, len(forbidden_found) * 0.35)
        negative_penalty = weak_negative_penalty + hard_negative_penalty
        score = clamp(0.70 * intent_centrality + 0.20 * positive_signal - 0.55 * negative_penalty)

        if forbidden_found:
            score = min(score, 0.29)
            decision = "out_of_scope"
        elif must_groups and missing_required:
            score = min(score, 0.58)
            decision = "borderline" if positive_matches and required_group_coverage >= 0.5 else "out_of_scope"
        elif lithium_drift or negative_context_match:
            score = min(score, 0.42)
            decision = "borderline"
        elif peripheral_context_reason and topic_focus_score < 0.5:
            score = min(score, 0.56)
            decision = "borderline"
        elif weak_negative_matches and not strong_positive_evidence and score < 0.40:
            score = min(score, 0.44)
            decision = "borderline" if positive_matches else "out_of_scope"
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
            weak_negative_matches,
            profile.domain_name,
        )
        return DomainAssessment(
            paper_id=paper.paper_id,
            domain_match_score=round(score, 4),
            domain_decision=decision,
            off_topic_reason=reason,
            positive_domain_matches=positive_matches,
            negative_domain_matches=weak_negative_matches,
            missing_required_concepts=missing_required,
            forbidden_concepts_found=forbidden_found,
            required_group_matches=required_group_matches,
            optional_group_matches=optional_group_matches,
            intent_centrality_score=round(intent_centrality, 4),
            required_group_coverage_score=round(required_group_coverage, 4),
            missing_required_group_count=missing_required_group_count,
            group_coverage_explanation=_group_coverage_explanation(
                required_group_matches,
                optional_group_matches,
                lithium_drift,
                target_context_match=target_context_match,
                negative_context_match=negative_context_match,
                topic_focus_score=topic_focus_score,
                peripheral_context_reason=peripheral_context_reason,
            ),
            matched_groups=matched_groups,
            missing_groups=missing_groups,
            target_context_match=target_context_match,
            negative_context_match=negative_context_match,
            topic_focus_score=round(topic_focus_score, 4),
            aspect_match_score=round(aspect_match_score, 4),
            target_context_score=round(target_context_score, 4),
            negative_context_penalty=round(negative_context_penalty, 4),
            peripheral_context_reason=peripheral_context_reason,
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

    return normalize_domain_text(
        " ".join(
            [
                paper.title,
                paper.abstract,
                paper.venue,
                " ".join(paper.fields_of_study),
                " ".join(paper.publication_types),
                paper.tldr,
                paper.source_provider,
            ]
        )
    )


def normalize_domain_text(value: str) -> str:
    """Normalize paper text before rule-based group matching."""

    text = html.unescape(str(value or ""))
    text = re.sub(r"</?\s*(sub|sup)\s*>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(
        str.maketrans(
            {
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
                "\u00ad": "-",
                "\u00a0": " ",
            }
        )
    )
    text = re.sub(r"([a-z])-\s+([a-z])", r"\1-\2", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def paper_title_text(paper: Paper) -> str:
    return normalize_domain_text(paper.title)


def paper_abstract_text(paper: Paper) -> str:
    return normalize_domain_text(
        " ".join(
        [
            paper.abstract,
            paper.tldr,
        ]
        )
    )


def concept_matches(concept: str, text: str) -> bool:
    """Return True when a concept appears in paper domain text."""

    cleaned = " ".join(str(concept or "").lower().split())
    if not cleaned:
        return False
    found_variant = False
    for variant in _concept_variants(cleaned):
        if variant not in text:
            continue
        found_variant = True
        if not _all_occurrences_negated(variant, text):
            return True
    if found_variant:
        return False
    if cleaned in text and not _all_occurrences_negated(cleaned, text):
        return True
    concept_tokens = set(tokenize(cleaned))
    if not concept_tokens:
        return False
    text_tokens = _expanded_tokens(text)
    expanded_concept_tokens = _expanded_tokens(cleaned)
    if len(concept_tokens) == 1:
        token = next(iter(concept_tokens))
        if token not in text_tokens:
            return False
        return not _all_occurrences_negated(token, text)
    if not expanded_concept_tokens.issubset(text_tokens):
        return False
    return not any(_all_occurrences_negated(token, text) for token in expanded_concept_tokens)


def _concept_variants(cleaned: str) -> list[str]:
    variants = [cleaned]
    normalized = normalize_domain_text(cleaned)
    variants.append(normalized)
    if "-" in normalized:
        variants.append(normalized.replace("-", " "))
    if " " in normalized:
        variants.append(normalized.replace(" ", "-"))
    if cleaned.endswith(" battery"):
        variants.append(cleaned[:-7] + " batteries")
    if cleaned.endswith(" batteries"):
        variants.append(cleaned[:-10] + " battery")
    compact_formula = re.sub(r"\s+", "", normalized)
    if compact_formula != normalized:
        variants.append(compact_formula)
    return _unique(variants, limit=8)


def _expanded_tokens(text: str) -> set[str]:
    result: set[str] = set()
    for token in tokenize(text):
        result.add(token)
        if token.endswith("ies") and len(token) > 4:
            result.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            result.add(token[:-1])
    return result


def _all_occurrences_negated(needle: str, text: str) -> bool:
    starts = [index for index in _find_occurrences(text, needle)]
    if not starts:
        return False
    return all(_is_negated_at(text, start, start + len(needle)) for start in starts)


def _find_occurrences(text: str, needle: str) -> Iterable[int]:
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        yield index
        start = index + max(1, len(needle))


def _is_negated_at(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 28):start]
    after = text[end:end + 36]
    before_markers = ["not ", "no ", "without ", "lacking ", "lack of ", "but not "]
    after_markers = [" not studied", " not addressed", " not considered", " absent"]
    return any(marker in before for marker in before_markers) or any(
        marker in after for marker in after_markers
    )


def terminology_synonyms(profile: DomainProfile) -> list[str]:
    """Flatten domain terminology synonyms."""

    values: list[str] = []
    for term, synonyms in profile.terminology_map.items():
        values.append(term)
        values.extend(synonyms)
    return _unique(values, limit=40)


def _must_term_groups(
    contract: SearchContract | None,
    query_plan: QueryPlan | None,
) -> list[tuple[str, list[str]]]:
    """Return generic any-of required term groups from contract/query metadata."""

    raw_groups: list[tuple[str, list[str]]] = []
    if contract:
        raw_groups.extend(
            (group.group_name, group.terms)
            for group in contract.constraint_groups
            if group.required
            and group.terms
            and group.group_name not in {"must_concepts", "optional_high_confidence"}
        )
    if query_plan:
        raw_groups.extend(
            (f"query_plan_group_{index}", group)
            for index, group in enumerate(query_plan.filters.get("must_term_groups", []), start=1)
        )
    groups: list[tuple[str, list[str]]] = []
    for name, group in raw_groups:
        if not isinstance(group, list):
            continue
        cleaned = _unique([str(item) for item in group], limit=12)
        if cleaned:
            group_name = str(name or "required_group")
            groups.append((group_name, _expand_group_terms(group_name, cleaned)))
    return groups


def _optional_term_groups(contract: SearchContract | None) -> list[tuple[str, list[str]]]:
    if not contract:
        return []
    groups: list[tuple[str, list[str]]] = []
    for group in contract.constraint_groups:
        if group.required or not group.terms:
            continue
        if group.group_name == "negative_context_group":
            continue
        cleaned = _unique([str(item) for item in group.terms], limit=12)
        if cleaned:
            groups.append((group.group_name, _expand_group_terms(group.group_name, cleaned)))
    return groups


def _expand_group_terms(group_name: str, terms: list[str]) -> list[str]:
    """Add deterministic synonym variants for active contract groups."""

    lowered = normalize_domain_text(" ".join([group_name, *terms]))
    additions: list[str] = []
    has_oer = "oer" in lowered or "oxygen evolution" in lowered or "water oxidation" in lowered
    if has_oer:
        additions.extend(["OER", "oxygen evolution reaction", "oxygen evolution", "water oxidation"])
        return _unique([*terms, *additions], limit=36)
    has_lithium_context = (
        "context" in group_name.lower()
        and ("lithium" in lowered or "li-ion" in lowered or "li ion" in lowered)
    )
    if has_lithium_context:
        return _unique(
            [
                "lithium-ion battery",
                "lithium ion battery",
                "li-ion battery",
                "li ion battery",
                "lithium battery",
                "lithium metal",
                "lithium-metal",
                "graphite anode",
                "silicon anode",
            ],
            limit=24,
        )
    if "sei" in lowered or "solid electrolyte interphase" in lowered:
        additions.extend(
            [
                "SEI",
                "solid electrolyte interphase",
                "solid-electrolyte interphase",
                "solid electrolyte interface",
                "solid-electrolyte interface",
            ]
        )
    if "lithium" in lowered or "li-ion" in lowered:
        additions.extend(
            [
                "lithium-ion battery",
                "lithium ion battery",
                "li-ion battery",
                "li ion battery",
                "lithium battery",
                "lithium metal",
                "lithium-metal",
                "graphite anode",
                "silicon anode",
            ]
        )
    if "spin" in lowered or "electronic" in lowered or "covalency" in lowered or "orbital" in lowered:
        additions.extend(
            [
                "spin state",
                "spin-state",
                "spin transition",
                "spin-state transition",
                "low-spin",
                "low spin",
                "high-spin",
                "high spin",
                "electronic spin state",
                "spin density",
                "spin polarization",
                "spin modulation",
                "spin-selective",
                "electronic structure",
                "orbital occupancy",
                "eg occupancy",
                "e g occupancy",
                "metal-oxygen covalency",
                "metal oxygen covalency",
                "co 3d-o 2p covalency",
                "co 3d o 2p covalency",
                "3d-o 2p covalency",
                "3d o 2p covalency",
                "covalency",
            ]
        )
    if (
        "catalyst" in lowered
        or "oxide" in lowered
        or "material" in lowered
        or "transition metal" in lowered
    ):
        additions.extend(
            [
                "transition metal oxide",
                "oxide catalyst",
                "electrocatalyst",
                "catalyst",
                "oxyhydroxide",
                "perovskite",
                "cobalt oxide",
                "NiOOH",
                "CoOOH",
                "LaCoO3",
                "LaCoO 3",
                "MnOOH",
                "cobaltite",
                "single-atom catalyst",
                "single atom catalyst",
            ]
        )
    return _unique([*terms, *additions], limit=36)


def _lithium_context_drift(
    contract: SearchContract | None,
    text: str,
    required_group_matches: dict[str, bool],
    negative_context_match: list[str] | None = None,
) -> list[str]:
    if not contract:
        return []
    lithium_required = any(
        "lithium" in " ".join(group.terms).lower() or "li-ion" in " ".join(group.terms).lower()
        for group in contract.constraint_groups
        if group.required
    )
    lithium_matched = any(
        matched
        for name, matched in required_group_matches.items()
        if "context" in name
    )
    if not lithium_required:
        return []
    markers = [
        "sodium",
        "potassium",
        "zinc",
        "zn",
        "azib",
        "azibs",
        "magnesium",
        "beyond lithium-ion",
        "beyond lithium ion",
        "aqueous zinc-ion",
        "aqueous zinc ion",
        "sodium-ion battery",
        "sodium ion battery",
        "potassium-ion battery",
        "potassium ion battery",
        "zinc-ion battery",
        "zinc ion battery",
        "zinc-based battery",
        "zinc based battery",
        "zinc battery",
        "magnesium-ion battery",
        "magnesium ion battery",
        "magnesium battery",
        "na-ion battery",
        "k-ion battery",
        "zn-ion battery",
        "zn-based battery",
        "zn based battery",
        "zn battery",
        "mg-ion battery",
        "mg battery",
        "sodium battery",
        "potassium battery",
    ]
    found = _unique(
        [
            marker
            for marker in markers
            if _chemistry_context_marker_matches(marker, text)
        ]
    )
    if negative_context_match:
        found = _unique([*found, *negative_context_match])
    if lithium_matched and not found:
        return []
    return found


def _target_context_signals(
    contract: SearchContract | None,
    text: str,
) -> tuple[list[str], list[str], float]:
    if not contract:
        return [], [], 1.0
    lithium_required = any(
        "lithium" in normalize_domain_text(" ".join(group.terms))
        or "li-ion" in normalize_domain_text(" ".join(group.terms))
        for group in contract.constraint_groups
        if group.required
    )
    if not lithium_required:
        return [], [], 1.0
    target_terms = [
        *_contract_group_terms(contract, "target_chemistry_group"),
        "lithium-ion battery",
        "lithium ion battery",
        "li-ion battery",
        "li ion battery",
        "lithium battery",
        "lithium",
        "lithium metal",
        "lithium-metal",
        "lithium metal battery",
        "lithium metal anode",
        "graphite anode",
        "silicon anode",
    ]
    negative_terms = [
        *_contract_group_terms(contract, "negative_context_group"),
        "beyond lithium-ion",
        "beyond lithium ion",
        "aqueous zinc-ion",
        "aqueous zinc ion",
        "azib",
        "azibs",
        "zinc-ion battery",
        "zinc ion battery",
        "zinc-based battery",
        "zinc based battery",
        "zinc battery",
        "zn-ion battery",
        "zn based battery",
        "zn-based battery",
        "zn battery",
        "sodium-ion battery",
        "sodium ion battery",
        "sodium battery",
        "na-ion battery",
        "potassium-ion battery",
        "potassium ion battery",
        "potassium battery",
        "k-ion battery",
        "magnesium-ion battery",
        "magnesium ion battery",
        "magnesium battery",
        "mg-ion battery",
    ]
    target = _unique([term for term in target_terms if concept_matches(term, text)])
    negative = _unique(
        [term for term in negative_terms if _chemistry_context_marker_matches(term, text)]
    )
    if re.search(r"\bbeyond\s+li\s*-?\s*ion\b", text):
        negative.append("beyond li-ion")
    negative = _unique(negative)
    if negative:
        return target, negative, 0.2 if target else 0.0
    return target, negative, 1.0 if target else 0.35


def _strict_context_marker_matches(marker: str, text: str) -> bool:
    cleaned = normalize_domain_text(marker)
    if not cleaned:
        return False
    variants = _concept_variants(cleaned)
    for variant in variants:
        if len(variant) < 4:
            continue
        if variant in text and not _all_occurrences_negated(variant, text):
            return True
    return False


def _chemistry_context_marker_matches(marker: str, text: str) -> bool:
    cleaned = normalize_domain_text(marker)
    if not cleaned:
        return False
    if cleaned in {"zn", "na", "k", "mg"}:
        return re.search(rf"(?<![a-z0-9]){re.escape(cleaned)}(?![a-z0-9])", text) is not None
    if cleaned in {"sodium", "potassium", "zinc", "magnesium", "azib", "azibs"}:
        return concept_matches(cleaned, text)
    return _strict_context_marker_matches(cleaned, text)


def _contract_group_terms(contract: SearchContract, group_name: str) -> list[str]:
    return _unique(
        [
            term
            for group in contract.constraint_groups
            if group.group_name == group_name
            for term in group.terms
        ],
        limit=24,
    )


def _topic_focus_score(
    paper: Paper,
    must_groups: list[tuple[str, list[str]]],
    required_group_matches: dict[str, bool],
    text: str,
    negative_context_match: list[str],
) -> float:
    if not must_groups:
        return 0.55
    title = paper_title_text(paper)
    abstract = paper_abstract_text(paper)
    title_matches = {
        name: any(concept_matches(term, title) for term in terms)
        for name, terms in must_groups
    }
    abstract_matches = {
        name: any(concept_matches(term, abstract) for term in terms)
        for name, terms in must_groups
    }
    title_coverage = (
        sum(1 for matched in title_matches.values() if matched) / len(title_matches)
        if title_matches
        else 0.0
    )
    abstract_coverage = (
        sum(1 for matched in abstract_matches.values() if matched) / len(abstract_matches)
        if abstract_matches
        else 0.0
    )
    full_coverage = (
        sum(1 for matched in required_group_matches.values() if matched)
        / len(required_group_matches)
        if required_group_matches
        else 0.0
    )
    core_group_names = [
        name
        for name, terms in must_groups
        if "core" in name.lower()
        or any(
            normalize_domain_text(term) in {"sei", "solid electrolyte interphase"}
            for term in terms
        )
    ]
    core_title_hit = any(title_matches.get(name) for name in core_group_names)
    core_full_hit = any(required_group_matches.get(name) for name in core_group_names)
    title_is_broad_battery = _looks_like_broad_battery_overview(title)
    score = clamp(
        0.45 * title_coverage
        + 0.25 * abstract_coverage
        + 0.20 * full_coverage
        + 0.10 * (1.0 if core_title_hit else 0.35 if core_full_hit else 0.0)
    )
    if title_is_broad_battery:
        score = min(score, 0.38)
    if negative_context_match:
        score = min(score, 0.55)
    return score


def _looks_like_broad_battery_overview(title: str) -> bool:
    compact = title.strip(" .:-")
    if compact in {"lithium-ion batteries", "lithium ion batteries", "lithium batteries"}:
        return True
    if compact.startswith("beyond lithium-ion") or compact.startswith("beyond lithium ion"):
        return True
    if (
        ("lithium-ion batteries" in compact or "lithium ion batteries" in compact)
        and not any(term in compact for term in ("sei", "solid electrolyte", "interphase", "interface"))
    ):
        return len(tokenize(compact)) <= 6
    return False


def _peripheral_context_reason(
    contract: SearchContract | None,
    target_context_match: list[str],
    negative_context_match: list[str],
    topic_focus_score: float,
) -> str:
    if negative_context_match:
        return "negative_or_beyond_target_chemistry_context"
    if _lithium_target_required(contract) and not target_context_match:
        return "missing_target_lithium_context"
    if topic_focus_score < 0.45:
        return "low_topic_focus"
    return ""


def _lithium_target_required(contract: SearchContract | None) -> bool:
    if not contract:
        return False
    return any(
        "lithium" in normalize_domain_text(" ".join(group.terms))
        or "li-ion" in normalize_domain_text(" ".join(group.terms))
        for group in contract.constraint_groups
        if group.required
    )


def _group_coverage_explanation(
    required_group_matches: dict[str, bool],
    optional_group_matches: dict[str, bool],
    drift_terms: list[str],
    target_context_match: list[str] | None = None,
    negative_context_match: list[str] | None = None,
    topic_focus_score: float = 0.0,
    peripheral_context_reason: str = "",
) -> list[str]:
    values: list[str] = []
    for name, matched in required_group_matches.items():
        values.append(f"required:{name}:{'matched' if matched else 'missing'}")
    for name, matched in optional_group_matches.items():
        values.append(f"optional:{name}:{'matched' if matched else 'missing'}")
    for term in drift_terms:
        values.append(f"downrank:non_target_battery_context:{term}")
    for term in target_context_match or []:
        values.append(f"target_context:{term}")
    for term in negative_context_match or []:
        values.append(f"negative_context:{term}")
    values.append(f"topic_focus_score:{topic_focus_score:.4f}")
    if peripheral_context_reason:
        values.append(f"peripheral_context_reason:{peripheral_context_reason}")
    return values


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
