"""Rule-based classification of evidence function in a research argument."""

from __future__ import annotations

from lit_screening.models import EvidenceFunction


def classify_evidence_function(
    evidence_text: str,
    title: str = "",
    abstract: str = "",
) -> EvidenceFunction:
    """Classify what an evidence snippet contributes to the argument."""

    text = " ".join([evidence_text or "", title or "", abstract or ""]).lower()
    if not text.strip():
        return EvidenceFunction.UNKNOWN

    if _has_any(text, ["review", "overview"]):
        return EvidenceFunction.REVIEW_BACKGROUND
    if _has_any(
        text,
        [
            "limitation",
            "defect",
            "defects",
            "roughness",
            "finite temperature",
            "paramagnetic",
            "paramagnetism",
        ],
    ):
        return EvidenceFunction.REPORTS_LIMITATION
    if _has_any(
        text,
        [
            "spin polarization",
            "spin-resolved",
            "spin-polarized photoemission",
            "spin polarized photoemission",
        ],
    ):
        return EvidenceFunction.MEASURES_SPIN_POLARIZATION
    if _has_any(text, ["imaging", "image", "images", "domains"]):
        return EvidenceFunction.DIRECTLY_IMAGES_SIGNAL
    if _has_any(text, ["peem", "mfm", "spleem", "stm", "sp-stm"]):
        return EvidenceFunction.REPORTS_SURFACE_PROBE
    if _has_any(
        text,
        ["we predict", "calculation", "first-principles", "first principles", "symmetry"],
    ):
        return EvidenceFunction.PREDICTS_EFFECT
    if _has_any(text, ["exchange bias", "memory", "readout", "spintronics"]):
        return EvidenceFunction.CONNECTS_TO_APPLICATION
    if _has_any(
        text,
        [
            "we report",
            "we observe",
            "experiment",
            "experimental",
            "measured",
            "measurement",
            "measurements",
        ],
    ):
        return EvidenceFunction.REPORTS_EXPERIMENT
    if _has_any(
        text,
        [
            "define",
            "defines",
            "defined as",
            "definition",
            "concept",
            "classification",
        ],
    ):
        return EvidenceFunction.DEFINES_CONCEPT
    return EvidenceFunction.UNKNOWN


def _has_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)
