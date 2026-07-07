"""Rule-based controversy and boundary-condition detection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from lit_screening.models import Paper, RankedPaper, ResearchTension


@dataclass(frozen=True)
class TensionRule:
    """Rule definition for one research tension."""

    key: str
    label: str
    description: str
    markers: list[str] = field(default_factory=list)
    side_a_markers: list[str] = field(default_factory=list)
    side_b_markers: list[str] = field(default_factory=list)
    why_it_matters: str = ""
    suggested_next_searches: list[str] = field(default_factory=list)


MATERIALS_MAGNETISM_TENSION_RULES = [
    TensionRule(
        key="intrinsic_vs_defect_magnetism",
        label="Intrinsic surface magnetism vs defect/parasitic magnetism",
        description=(
            "Papers may attribute boundary magnetic signals either to intrinsic "
            "symmetry-allowed surface magnetization or to defects, impurities, "
            "ferrimagnetic contamination, and other parasitic sources."
        ),
        markers=[
            "defects",
            "defect",
            "impurity",
            "parasitic",
            "parasitic magnetization",
            "ferrimagnetism",
        ],
        why_it_matters=(
            "A demo or follow-up review should separate intrinsic boundary order "
            "from extrinsic magnetic signals before treating a surface signal as "
            "evidence for the predicted mechanism."
        ),
        suggested_next_searches=[
            "defects parasitic magnetization antiferromagnetic thin films",
            "Cr2O3 surface magnetization oxygen vacancy impurity",
        ],
    ),
    TensionRule(
        key="ideal_surface_vs_real_rough_surface",
        label="Ideal surface theory vs real rough/terminated surfaces",
        description=(
            "Theory often assumes a clean termination, while experiments may face "
            "roughness, reconstruction, step edges, and compensated or "
            "uncompensated terminations."
        ),
        markers=[
            "roughness",
            "termination",
            "reconstruction",
            "step",
            "step edge",
            "compensated",
            "uncompensated",
        ],
        why_it_matters=(
            "Surface preparation can decide whether a predicted boundary moment "
            "survives or is averaged away in measurements."
        ),
        suggested_next_searches=[
            "surface roughness robust magnetization antiferromagnet",
            "antiferromagnet surface termination uncompensated magnetization",
        ],
    ),
    TensionRule(
        key="zero_kelvin_dft_vs_finite_temperature",
        label="Zero-temperature calculations vs finite-temperature surfaces",
        description=(
            "First-principles or symmetry arguments may be formulated near ideal "
            "zero-temperature order, while experiments operate near finite "
            "temperature, thermal disorder, or surface paramagnetism."
        ),
        markers=[
            "finite temperature",
            "thermal",
            "neel",
            "neel temperature",
            "paramagnetic",
            "surface paramagnetism",
        ],
        why_it_matters=(
            "Temperature controls magnetic order, surface compensation, and signal "
            "stability, so it is a boundary condition for using theory as a "
            "screening rule."
        ),
        suggested_next_searches=[
            "Cr2O3 surface paramagnetism finite temperature magnetoelectric antiferromagnet",
            "finite temperature boundary magnetization magnetoelectric antiferromagnet",
        ],
    ),
    TensionRule(
        key="direct_spin_polarization_vs_stray_field_probe",
        label="Direct spin-polarization probes vs stray-field readout",
        description=(
            "Some papers probe spin-polarized surface states or local spin order "
            "directly, while others infer boundary magnetism from stray fields or "
            "magnetic-domain contrast."
        ),
        side_a_markers=[
            "nv",
            "nv magnetometry",
            "mfm",
            "stray field",
            "magnetic force microscopy",
        ],
        side_b_markers=[
            "spleem",
            "sp-stm",
            "spin-resolved",
            "spin-polarized",
            "spin polarized",
            "photoemission",
        ],
        why_it_matters=(
            "The measurement channel determines whether a paper demonstrates "
            "spin polarization itself or only a magnetic field consistent with it."
        ),
        suggested_next_searches=[
            "SPLEEM chromia spin polarization asymmetry",
            "NV magnetometry Cr2O3 boundary magnetization",
            "spin-resolved photoemission antiferromagnet surface spin polarization",
        ],
    ),
    TensionRule(
        key="material_specific_cr2o3_vs_general_antiferromagnets",
        label="Cr2O3-specific evidence vs general antiferromagnet claims",
        description=(
            "The literature may be rich for Cr2O3/chromia while broader claims "
            "refer to antiferromagnets in general."
        ),
        markers=[
            "cr2o3",
            "chromia",
            "general antiferromagnets",
            "antiferromagnets",
            "antiferromagnetic",
        ],
        why_it_matters=(
            "A convincing research map should distinguish material-specific case "
            "studies from mechanisms expected to transfer across compounds."
        ),
        suggested_next_searches=[
            "Cr2O3 FeF2 NiO CuMnAs Mn2Au surface magnetization antiferromagnet",
            "surface magnetization general antiferromagnets material comparison",
        ],
    ),
    TensionRule(
        key="net_surface_magnetization_vs_local_spin_order",
        label="Net surface magnetization vs local spin order",
        description=(
            "A surface may have local magnetic order, spin polarization, or "
            "compensated antiferromagnetic structure without producing a large "
            "net surface magnetization."
        ),
        markers=[
            "local magnetic order",
            "local spin order",
            "no net magnetization",
            "net magnetization",
            "compensated",
            "surface spin polarization",
            "spin polarization",
        ],
        why_it_matters=(
            "This boundary condition prevents over-reading local order or spin "
            "polarization as a macroscopic boundary moment."
        ),
        suggested_next_searches=[
            "local magnetic order no net magnetization compensated antiferromagnetic surface",
            "surface spin polarization boundary magnetization compensated antiferromagnet",
        ],
    ),
]


class ControversyAndBoundaryAgent:
    """Identify research tensions from paper metadata without using an LLM."""

    def analyze(
        self,
        papers: list[Paper] | list[RankedPaper],
        domain: str = "materials_magnetism",
        query_provenance: dict[str, Any] | None = None,
    ) -> list[ResearchTension]:
        """Return controversy and boundary-condition records for a domain."""

        normalized_papers = [_paper_from_item(item) for item in papers]
        if domain != "materials_magnetism":
            return []
        return [
            tension
            for rule in MATERIALS_MAGNETISM_TENSION_RULES
            if (
                tension := _apply_materials_rule(
                    rule,
                    normalized_papers,
                    query_provenance=query_provenance,
                )
            )
        ]


def _apply_materials_rule(
    rule: TensionRule,
    papers: list[Paper],
    query_provenance: dict[str, Any] | None = None,
) -> ResearchTension | None:
    paper_matches: list[tuple[Paper, list[str]]] = []
    side_a_found = False
    side_b_found = False
    for paper in papers:
        text = _paper_text(paper, query_provenance=query_provenance)
        matches = _matched_markers(text, rule.markers)
        side_a_matches = _matched_markers(text, rule.side_a_markers)
        side_b_matches = _matched_markers(text, rule.side_b_markers)
        if side_a_matches:
            side_a_found = True
        if side_b_matches:
            side_b_found = True
        all_matches = _unique_strings(matches + side_a_matches + side_b_matches)
        if all_matches:
            paper_matches.append((paper, all_matches))

    if not paper_matches:
        return None

    if rule.side_a_markers and rule.side_b_markers:
        confidence = 0.82 if side_a_found and side_b_found else 0.56
    else:
        confidence = min(0.9, 0.5 + 0.1 * len(paper_matches))

    return ResearchTension(
        tension_key=rule.key,
        tension_label=rule.label,
        description=rule.description,
        supporting_paper_ids=_unique_strings(
            [paper.paper_id for paper, _ in paper_matches if paper.paper_id]
        ),
        evidence_snippets=_evidence_snippets(paper_matches),
        why_it_matters=rule.why_it_matters,
        suggested_next_searches=list(rule.suggested_next_searches),
        confidence=round(confidence, 2),
    )


def _paper_from_item(item: Paper | RankedPaper) -> Paper:
    if isinstance(item, RankedPaper):
        return item.paper
    return item


def _paper_text(
    paper: Paper,
    query_provenance: dict[str, Any] | None = None,
) -> str:
    fields = [
        paper.title,
        paper.abstract,
        paper.venue,
        str(paper.year or ""),
        paper.retrieval_query,
        paper.source_stage,
        paper.seed_title,
        paper.seed_reason,
        paper.tldr,
        str(paper.raw.get("matched_query") or ""),
        str(paper.raw.get("matched_query_family") or ""),
        str(paper.raw.get("matched_lens") or ""),
        str(paper.raw.get("matched_query_purpose") or ""),
    ]
    if query_provenance:
        for record in query_provenance.get("records", []):
            if not isinstance(record, dict):
                continue
            raw_query = str(record.get("raw_query") or "")
            if raw_query and raw_query == paper.retrieval_query:
                fields.extend(
                    [
                        raw_query,
                        str(record.get("family_name") or ""),
                        str(record.get("lens_name") or ""),
                        str(record.get("purpose") or ""),
                    ]
                )
    return _normalize_text(" ".join(fields))


def _normalize_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", folded.lower()).strip()


def _matched_markers(text: str, markers: list[str]) -> list[str]:
    matches: list[str] = []
    for marker in markers:
        normalized_marker = _normalize_text(marker)
        if normalized_marker and _marker_in_text(text, normalized_marker):
            matches.append(marker)
    return matches


def _marker_in_text(text: str, marker: str) -> bool:
    if len(marker) <= 3 and marker.isalnum():
        return bool(re.search(rf"\b{re.escape(marker)}\b", text))
    return marker in text


def _evidence_snippets(paper_matches: list[tuple[Paper, list[str]]]) -> list[str]:
    snippets: list[str] = []
    for paper, markers in paper_matches[:6]:
        marker_text = ", ".join(markers[:4])
        title = paper.title or paper.paper_id
        context = _first_relevant_sentence(paper.abstract, markers)
        if context:
            snippets.append(f"{paper.paper_id}: {title} -- {context}")
        else:
            snippets.append(f"{paper.paper_id}: {title} -- matched {marker_text}")
    return snippets


def _first_relevant_sentence(abstract: str, markers: list[str]) -> str:
    if not abstract:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    for sentence in sentences:
        normalized_sentence = _normalize_text(sentence)
        if any(_normalize_text(marker) in normalized_sentence for marker in markers):
            return sentence[:300]
    return sentences[0][:300] if sentences else ""


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique
