"""Pack-driven domain routing with generic fallback."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from lit_screening.domain_packs import list_domain_packs, load_domain_pack
from lit_screening.models import DomainActivationResult, DomainPack


class DomainRouter:
    """Select an optional domain pack without making packs a hard dependency."""

    def route(self, text: str, domain_hint: str = "") -> DomainActivationResult:
        context = " ".join(str(text or "").lower().split())
        candidates: list[dict[str, Any]] = []
        activation_evidence: dict[str, list[str]] = {}
        negative_evidence: dict[str, list[str]] = {}
        packs = _available_packs()
        hinted = str(domain_hint or "").strip()
        for pack in packs:
            candidate = _score_pack(pack, context)
            if hinted and hinted == pack.domain_name:
                candidate["hinted"] = True
                candidate["score"] = max(float(candidate["score"]), 1.0)
                candidate["confidence"] = max(float(candidate["confidence"]), 0.7)
            candidates.append(candidate)
            activation_evidence[pack.domain_name] = list(candidate["activation_evidence"])
            negative_evidence[pack.domain_name] = list(candidate["negative_evidence"])

        viable = [
            item
            for item in candidates
            if item.get("activated")
            and not item.get("blocked_by_negative")
        ]
        viable.sort(key=lambda item: (-float(item.get("score", 0.0)), item["domain_name"]))
        if viable:
            selected = viable[0]
            return DomainActivationResult(
                selected_domain=str(selected["domain_name"]),
                candidate_domains=candidates,
                activation_evidence=activation_evidence,
                negative_evidence=negative_evidence,
                confidence=float(selected.get("confidence", 0.0)),
                fallback_reason="",
                domain_pack_enhancement_used=True,
            )
        return DomainActivationResult(
            selected_domain="general_science",
            candidate_domains=candidates,
            activation_evidence=activation_evidence,
            negative_evidence=negative_evidence,
            confidence=0.35 if context else 0.0,
            fallback_reason="no_domain_pack_met_group_activation",
            domain_pack_enhancement_used=False,
        )

    def route_dict(self, text: str, domain_hint: str = "") -> dict[str, Any]:
        return asdict(self.route(text, domain_hint=domain_hint))


def _available_packs() -> list[DomainPack]:
    packs: list[DomainPack] = []
    for name in list_domain_packs():
        try:
            packs.append(load_domain_pack(name))
        except ValueError:
            continue
    return packs


def _score_pack(pack: DomainPack, context: str) -> dict[str, Any]:
    activation = dict(getattr(pack, "activation", {}) or {})
    positive_groups = _group_list(activation.get("positive_groups"))
    negative_groups = _group_list(activation.get("negative_groups"))
    min_positive = int(activation.get("min_positive_groups") or 1)
    if not positive_groups:
        positive_groups = _default_positive_groups(pack)
        min_positive = max(1, int(activation.get("min_positive_groups") or 2))
    matched_groups: list[list[str]] = []
    activation_terms: list[str] = []
    for group in positive_groups:
        matched = _matched_terms(context, group)
        if matched:
            matched_groups.append(matched)
            activation_terms.extend(matched)
    negative_terms: list[str] = []
    for group in negative_groups:
        negative_terms.extend(_matched_terms(context, group))
    matched_count = len(matched_groups)
    blocked = bool(negative_terms)
    activated = matched_count >= min_positive
    score = matched_count / max(1, len(positive_groups))
    if blocked:
        score *= 0.35
    confidence = min(0.95, 0.35 + score * 0.6)
    return {
        "domain_name": pack.domain_name,
        "score": round(score, 4),
        "confidence": round(confidence if activated and not blocked else 0.0, 4),
        "activated": bool(activated),
        "blocked_by_negative": blocked,
        "matched_positive_group_count": matched_count,
        "min_positive_groups": min_positive,
        "activation_evidence": _unique(activation_terms, 32),
        "negative_evidence": _unique(negative_terms, 32),
    }


def _default_positive_groups(pack: DomainPack) -> list[list[str]]:
    groups: list[list[str]] = []
    anchors = list(getattr(pack, "domain_anchors", []) or [])
    if anchors:
        groups.append(anchors)
    concept_terms: list[str] = []
    for concept in getattr(pack, "concepts", {}).values():
        concept_terms.extend(concept.synonyms)
    if concept_terms:
        groups.append(concept_terms)
    if getattr(pack, "mechanisms", None):
        groups.append(list(pack.mechanisms))
    return groups


def _group_list(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    groups: list[list[str]] = []
    for group in value:
        if isinstance(group, list):
            cleaned = _unique([str(term) for term in group], 32)
            if cleaned:
                groups.append(cleaned)
    return groups


def _matched_terms(context: str, terms: list[str]) -> list[str]:
    matches: list[str] = []
    for term in terms:
        cleaned = " ".join(str(term or "").lower().split())
        if cleaned and cleaned in context:
            matches.append(term)
    return _unique(matches, 16)


def _unique(values: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value or "").split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result[:limit] if limit else result
