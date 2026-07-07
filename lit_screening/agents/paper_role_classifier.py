"""Rule-based paper role classification for literature-screening sensemaking."""

from __future__ import annotations

from typing import Any

from lit_screening.models import Paper, PaperRoleRecord


ROLE_PRIORITY = [
    "theory_origin",
    "conceptual_framework",
    "nanoscale_readout",
    "experimental_proof",
    "surface_probe_method",
    "material_case",
    "application_bridge",
    "frontier_extension",
    "limitation_or_challenge",
    "review_background",
]

TEXT_RULES = [
    (
        "theory_origin",
        [
            "boundary magnetization",
            "equilibrium magnetization",
            "magnetoelectric antiferromagnet",
        ],
    ),
    (
        "conceptual_framework",
        [
            "classification",
            "magnetoelectric responses",
            "surface magnetization in antiferromagnets",
            "local magnetoelectric effects",
            "surface magnetic order",
        ],
    ),
    (
        "experimental_proof",
        [
            "imaging",
            "surface magnetization domains",
            "spin polarization asymmetry",
            "magnetic domain",
            "domain imaging",
        ],
    ),
    (
        "surface_probe_method",
        [
            "spleem",
            "xmcd-peem",
            "xmld-peem",
            "spin-polarized low-energy electron microscopy",
            "spin-resolved photoemission",
            "spin-polarized photoemission",
            "sp-stm",
            "spin-polarized stm",
            "sarpes",
        ],
    ),
    (
        "nanoscale_readout",
        [
            "nv magnetometry",
            "scanning diamond magnetometry",
            "nanoscale readout",
            "diamond magnetometry",
        ],
    ),
    (
        "material_case",
        [
            "cr2o3",
            "chromia",
            "fef2",
            "nio",
            "cum nas",
            "cumnas",
            "mn2au",
            "fesn",
        ],
    ),
    (
        "application_bridge",
        [
            "exchange bias",
            "memory",
            "spintronics",
            "readout",
            "magnetoelectric memory",
        ],
    ),
    (
        "frontier_extension",
        [
            "altermagnetism",
            "spin splitting",
            "sarpes",
            "spin-resolved arpes",
        ],
    ),
    (
        "limitation_or_challenge",
        [
            "roughness",
            "finite temperature",
            "defects",
            "paramagnetism",
            "parasitic magnetization",
        ],
    ),
    (
        "review_background",
        [
            "review",
            "rev. mod. phys.",
            "reviews of modern physics",
            "survey",
        ],
    ),
]

QUERY_FAMILY_ROLE_MAP = {
    "theory_origin": ["theory_origin"],
    "spaldin_framework": ["conceptual_framework"],
    "surface_magnetization_classification": ["conceptual_framework"],
    "local_magnetoelectric_predictor": ["conceptual_framework"],
    "local_magnetic_order": ["conceptual_framework"],
    "direct_surface_detection": ["experimental_proof", "surface_probe_method"],
    "nanoscale_readout": ["nanoscale_readout"],
    "applications": ["application_bridge"],
    "limitations": ["limitation_or_challenge"],
    "frontier": ["frontier_extension"],
    "seed_context": ["conceptual_framework"],
}


class PaperRoleClassifier:
    """Assign research roles from paper metadata and query provenance."""

    def classify_many(
        self,
        papers: list[Paper],
        query_provenance: dict[str, Any] | None = None,
    ) -> list[PaperRoleRecord]:
        """Classify a list of papers without changing ranking."""

        return [
            self.classify(paper, query_provenance=query_provenance)
            for paper in papers
        ]

    def classify(
        self,
        paper: Paper,
        query_provenance: dict[str, Any] | None = None,
    ) -> PaperRoleRecord:
        """Return one role record for a paper."""

        roles: list[str] = []
        reasons: list[str] = []
        linked_lenses, linked_query_families = _linked_provenance(
            paper,
            query_provenance,
        )
        text = _paper_text(paper)

        for role, markers in TEXT_RULES:
            matches = [marker for marker in markers if marker in text]
            if matches:
                roles.append(role)
                reasons.append(
                    f"{role}: matched " + ", ".join(matches[:3])
                )

        for family in linked_query_families:
            for role in QUERY_FAMILY_ROLE_MAP.get(family, []):
                roles.append(role)
                reasons.append(f"{role}: linked query family {family}")

        for lens in linked_lenses:
            for role in QUERY_FAMILY_ROLE_MAP.get(lens, []):
                roles.append(role)
                reasons.append(f"{role}: linked lens {lens}")

        roles = _unique_role_order(roles)
        if not roles:
            roles = ["review_background"]
            reasons.append("review_background: no stronger role markers found")

        primary_role = _primary_role(roles)
        confidence = _confidence(roles, reasons, linked_lenses, linked_query_families)
        return PaperRoleRecord(
            paper_id=paper.paper_id,
            title=paper.title,
            roles=roles,
            primary_role=primary_role,
            confidence=confidence,
            reasons=_unique_strings(reasons),
            linked_lenses=linked_lenses,
            linked_query_families=linked_query_families,
        )


def _paper_text(paper: Paper) -> str:
    fields = [
        paper.title,
        paper.abstract,
        paper.venue,
        str(paper.year or ""),
        paper.retrieval_query,
        paper.source_stage,
        str(paper.raw.get("matched_query") or ""),
        str(paper.raw.get("matched_query_purpose") or ""),
    ]
    return " ".join(fields).lower()


def _linked_provenance(
    paper: Paper,
    query_provenance: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    lenses: list[str] = []
    families: list[str] = []
    raw_lens = str(paper.raw.get("matched_lens") or "").strip()
    raw_family = str(paper.raw.get("matched_query_family") or "").strip()
    if raw_lens:
        lenses.append(raw_lens)
    if raw_family:
        families.append(raw_family)

    if query_provenance:
        provider_candidates = _split_values(
            ";".join([paper.retrieval_provider, paper.source_provider])
        )
        query_candidates = _split_values(paper.retrieval_query)
        for record in query_provenance.get("records", []):
            if not isinstance(record, dict):
                continue
            raw_query = str(record.get("raw_query") or "")
            provider = str(record.get("provider") or "")
            if raw_query not in query_candidates:
                continue
            if provider_candidates and provider and provider not in provider_candidates:
                continue
            lens = str(record.get("lens_name") or "").strip()
            family = str(record.get("family_name") or "").strip()
            if lens:
                lenses.append(lens)
            if family:
                families.append(family)

    return _unique_strings(lenses), _unique_strings(families)


def _split_values(value: str) -> list[str]:
    return [
        cleaned
        for cleaned in (" ".join(part.split()) for part in str(value or "").split(";"))
        if cleaned
    ]


def _unique_role_order(roles: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for role in ROLE_PRIORITY:
        if role in roles and role not in seen:
            result.append(role)
            seen.add(role)
    for role in roles:
        if role not in seen:
            result.append(role)
            seen.add(role)
    return result


def _primary_role(roles: list[str]) -> str:
    for role in ROLE_PRIORITY:
        if role in roles:
            return role
    return roles[0] if roles else "review_background"


def _confidence(
    roles: list[str],
    reasons: list[str],
    linked_lenses: list[str],
    linked_query_families: list[str],
) -> float:
    if roles == ["review_background"] and len(reasons) == 1:
        return 0.2
    score = 0.42
    score += min(len(roles), 4) * 0.08
    score += min(len(reasons), 5) * 0.04
    if linked_lenses:
        score += 0.06
    if linked_query_families:
        score += 0.06
    return round(min(score, 0.95), 3)


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
