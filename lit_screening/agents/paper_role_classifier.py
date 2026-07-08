"""Rule-based paper role classification for literature-screening sensemaking."""

from __future__ import annotations

import re
from typing import Any

from lit_screening.models import Paper, PaperRoleRecord, SearchContract


ROLE_PRIORITY = [
    "theory_mechanism",
    "characterization_method",
    "in_situ_operando",
    "ex_situ_post_mortem",
    "application_performance",
    "failure_limitation",
    "controversy_debate",
    "peripheral_background",
    "out_of_scope",
    "theory_origin",
    "conceptual_framework",
    "nanoscale_readout",
    "experimental_proof",
    "direct_probe_method",
    "surface_probe_method",
    "interface_screening",
    "material_case",
    "device_application",
    "application_bridge",
    "frontier_extension",
    "limitation_or_challenge",
    "review_background",
]

GENERIC_TEXT_RULES = [
    ("review_background", ["review", "survey", "perspective", "overview", "roadmap"]),
    ("theory_mechanism", ["theory", "mechanism", "model", "descriptor", "electronic structure", "orbital occupancy"]),
    ("characterization_method", ["characterization", "spectroscopy", "microscopy", "measurement", "TEM", "STEM", "XPS", "XAS", "RIXS"]),
    ("in_situ_operando", ["in situ", "operando", "real-time", "real time"]),
    ("ex_situ_post_mortem", ["ex situ", "post mortem"]),
    ("material_case", ["case study", "representative material", "catalyst", "oxide", "anode", "perovskite", "mof"]),
    ("application_performance", ["application", "performance", "activity", "accuracy", "recall", "capacity", "efficiency"]),
    ("failure_limitation", ["failure", "limitation", "degradation", "aging", "stability", "reconstruction"]),
    ("controversy_debate", ["controversy", "debate", "competing mechanism", "open question"]),
]

DOMAIN_TEXT_RULES = [
    (
        "theory_origin",
        [
            "boundary magnetization",
            "equilibrium magnetization",
            "magnetoelectric antiferromagnet",
            "modern theory of polarization",
            "electric polarization as a bulk quantity",
            "electric polarization",
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
        "direct_probe_method",
        [
            "piezoresponse force microscopy",
            "pfm",
            "second harmonic generation",
            "shg",
            "kelvin probe force microscopy",
            "kpfm",
            "xps",
            "photoemission",
            "tem",
            "stem",
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
        "interface_screening",
        [
            "depolarization field",
            "screening charge",
            "interface screening",
            "surface screening",
            "surface charge",
            "electrode screening",
            "imprint",
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
            "batio3",
            "pzt",
            "pbtio3",
            "bifeo3",
            "hfo2",
            "hfzro2",
            "linbo3",
            "pvdf",
        ],
    ),
    (
        "device_application",
        [
            "ferroelectric memory",
            "ferroelectric tunnel junction",
            "fefet",
            "nonvolatile memory",
            "oxide electronics",
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
            "dead layer",
            "domain pinning",
            "leakage",
        ],
    ),
    (
        "review_background",
        [
            "review",
            "rev. mod. phys.",
            "reviews of modern physics",
            "survey",
            "perspective",
            "roadmap",
            "overview",
            "annual review",
            "reports on progress",
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
        search_contract: SearchContract | None = None,
    ) -> list[PaperRoleRecord]:
        """Classify a list of papers without changing ranking."""

        records = [
            self.classify(
                paper,
                query_provenance=query_provenance,
                search_contract=search_contract,
            )
            for paper in papers
        ]
        _mark_overbroad_roles(records)
        return records

    def classify(
        self,
        paper: Paper,
        query_provenance: dict[str, Any] | None = None,
        search_contract: SearchContract | None = None,
    ) -> PaperRoleRecord:
        """Return one role record for a paper."""

        roles: list[str] = []
        reasons: list[str] = []
        linked_lenses, linked_query_families = _linked_provenance(
            paper,
            query_provenance,
        )
        text = _paper_text(paper)
        active_domain = _active_domain(query_provenance, search_contract, text)
        text_rules = list(GENERIC_TEXT_RULES)
        if active_domain in {"materials_magnetism", "ferroelectric_polarization"}:
            text_rules.extend(DOMAIN_TEXT_RULES)

        for role, markers in text_rules:
            matches = [marker for marker in markers if _marker_matches(marker, text)]
            if matches:
                roles.append(role)
                reasons.append(
                    f"{role}: matched " + ", ".join(matches[:3])
                )

        roles = _unique_role_order(roles)
        if linked_query_families or linked_lenses:
            reasons.append(
                "retrieval_lane: linked query family/lens recorded as provenance only"
            )

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
            content_roles=list(roles),
            retrieval_lanes=_unique_strings([*linked_query_families, *linked_lenses]),
        )


def _paper_text(paper: Paper) -> str:
    fields = [
        paper.title,
        paper.abstract,
    ]
    return " ".join(fields).lower()


def _active_domain(
    query_provenance: dict[str, Any] | None,
    search_contract: SearchContract | None,
    text: str = "",
) -> str:
    if search_contract is not None:
        return str(search_contract.domain_profile.domain_name or "").strip().lower()
    if query_provenance:
        domain = str(query_provenance.get("domain") or "").strip().lower()
        if domain:
            return domain
    if _looks_like_materials_magnetism_text(text):
        return "materials_magnetism"
    if _looks_like_ferroelectric_text(text):
        return "ferroelectric_polarization"
    return ""


def _looks_like_materials_magnetism_text(text: str) -> bool:
    markers = [
        "surface magnetization",
        "boundary magnetization",
        "magnetoelectric antiferromagnet",
        "antiferromagnetic surface",
        "spin polarization asymmetry",
        "cr2o3",
        "chromia",
    ]
    return any(marker in text for marker in markers)


def _looks_like_ferroelectric_text(text: str) -> bool:
    markers = [
        "ferroelectric",
        "depolarization field",
        "piezoresponse force microscopy",
        "second harmonic generation",
        "kelvin probe force microscopy",
    ]
    return any(marker in text for marker in markers)


def _marker_matches(marker: str, text: str) -> bool:
    raw_marker = str(marker or "").strip()
    if not raw_marker:
        return False
    marker_lower = raw_marker.lower()
    if re.fullmatch(r"[A-Z0-9\-]{2,8}", raw_marker):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(marker_lower)}(?![a-z0-9])",
                text,
            )
        )
    if len(marker_lower) <= 5 and marker_lower.replace("-", "").isalnum():
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(marker_lower)}(?![a-z0-9])",
                text,
            )
        )
    return marker_lower in text


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
    return roles[0] if roles else "unclassified"


def _confidence(
    roles: list[str],
    reasons: list[str],
    linked_lenses: list[str],
    linked_query_families: list[str],
) -> float:
    if not roles:
        return 0.1
    score = 0.42
    score += min(len(roles), 4) * 0.08
    score += min(len(reasons), 5) * 0.04
    if linked_lenses:
        score += 0.02
    if linked_query_families:
        score += 0.02
    return round(min(score, 0.95), 3)


def _mark_overbroad_roles(records: list[PaperRoleRecord]) -> None:
    """Flag roles that appear on more than 30 percent of records."""

    if not records:
        return
    role_counts: dict[str, int] = {}
    for record in records:
        for role in record.roles:
            role_counts[role] = role_counts.get(role, 0) + 1
    total = len(records)
    overbroad_roles = {
        role
        for role, count in role_counts.items()
        if total > 1 and (count / total) > 0.30
    }
    if not overbroad_roles:
        return
    for record in records:
        matched = [role for role in record.roles if role in overbroad_roles]
        if matched:
            record.overbroad_role_warning = (
                "Role appears on more than 30% of papers; treat as broad context, not a decisive role: "
                + ", ".join(matched)
            )


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
