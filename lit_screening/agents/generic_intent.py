"""Generic scientific intent framing for domains without a dedicated pack."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lit_screening.models import (
    ExpertResearchIntent,
    GenericResearchIntentFrame,
    IntentConcept,
)
from lit_screening.utils import tokenize


RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
GLOSSARY_PATH = RESOURCE_DIR / "generic_science_glossary.json"

GENERIC_NOISY_TERMS = {
    "important",
    "importance",
    "significance",
    "background",
    "paper",
    "papers",
    "study",
    "studies",
    "research",
    "related work",
    "literature",
    "review",
    "survey",
}

ASPECT_CUES = {
    "mechanism": [
        "mechanism",
        "mechanisms",
        "theory",
        "theoretical",
        "why",
        "effect",
        "effects",
        "affect",
        "influence",
        "机制",
        "理论",
        "为什么",
        "影响",
    ],
    "method": [
        "method",
        "methods",
        "characterization",
        "characterize",
        "probe",
        "detect",
        "measurement",
        "experimental",
        "workflow",
        "表征",
        "探测",
        "实验",
        "方法",
        "流程",
    ],
    "in_situ": ["in situ", "operando", "real-time", "实时", "原位", "动态"],
    "ex_situ": ["ex situ", "post mortem", "offline", "非原位", "离线"],
    "case": [
        "case",
        "cases",
        "material",
        "materials",
        "system",
        "systems",
        "benchmark",
        "typical",
        "representative",
        "典型",
        "材料",
        "体系",
        "案例",
    ],
    "application": [
        "application",
        "applications",
        "performance",
        "activity",
        "accuracy",
        "recall",
        "metric",
        "device",
        "应用",
        "性能",
        "活性",
        "准确率",
        "召回率",
        "指标",
        "器件",
    ],
    "failure": [
        "failure",
        "limitation",
        "limitations",
        "degradation",
        "aging",
        "stability",
        "失效",
        "限制",
        "局限",
        "降解",
        "老化",
        "稳定性",
    ],
    "controversy": [
        "controversy",
        "controversies",
        "debate",
        "competing",
        "open question",
        "争议",
        "分歧",
    ],
    "review": [
        "review",
        "background",
        "overview",
        "introduction",
        "综述",
        "背景",
        "入门",
        "重要",
        "意义",
    ],
    "comparison": [
        "comparison",
        "compare",
        "advantages",
        "disadvantages",
        "pros",
        "cons",
        "比较",
        "优缺点",
    ],
}


def build_generic_intent_frame(
    question: str,
    expert_intent: ExpertResearchIntent | None = None,
) -> GenericResearchIntentFrame:
    """Infer a generic research-intent frame without requiring a domain pack."""

    text = " ".join(str(question or "").split())
    lowered = text.lower()
    glossary_terms = _glossary_terms(text)
    explicit_terms = _explicit_english_terms(text)
    concept_terms = _concept_terms(expert_intent)
    all_terms = _unique([*concept_terms, *glossary_terms, *explicit_terms], 40)
    downweighted = _unique(
        [
            term
            for term in [*GENERIC_NOISY_TERMS, "importance", "significance"]
            if term in lowered or _contains_chinese_noise(text, term)
        ],
        16,
    )
    active_terms = [
        term for term in all_terms if term.lower() not in GENERIC_NOISY_TERMS
    ]
    concept_by_category = _concepts_by_category(expert_intent)
    core_terms = _core_terms(active_terms, concept_by_category)
    context_terms = _context_terms(active_terms, core_terms, concept_by_category)
    method_terms = _terms_for_categories(concept_by_category, {"method"})
    mechanism_terms = _terms_for_categories(concept_by_category, {"mechanism"})
    material_terms = _terms_for_categories(concept_by_category, {"material"})
    application_terms = _terms_for_categories(concept_by_category, {"application"})
    if _has_cue(lowered, text, ASPECT_CUES["method"]):
        method_terms = _unique([*method_terms, *_aspect_terms(active_terms, "method")], 12)
    if _has_cue(lowered, text, ASPECT_CUES["mechanism"]):
        mechanism_terms = _unique([*mechanism_terms, *_aspect_terms(active_terms, "mechanism")], 12)
    if _has_cue(lowered, text, ASPECT_CUES["case"]):
        material_terms = _unique([*material_terms, *_aspect_terms(active_terms, "case")], 12)
    if _has_cue(lowered, text, ASPECT_CUES["application"]):
        application_terms = _unique([*application_terms, *_aspect_terms(active_terms, "application")], 12)
    failure_terms = _aspect_terms(active_terms, "failure")
    controversy_terms = _aspect_terms(active_terms, "controversy")
    target_terms = _target_terms(active_terms, core_terms, context_terms)
    relation_terms = _relation_terms(lowered, text)
    frame = GenericResearchIntentFrame(
        research_object=core_terms[:4],
        domain_context=context_terms[:6],
        target_process_or_property=target_terms[:6],
        relation_or_effect=relation_terms,
        mechanism_need=bool(mechanism_terms) or _has_cue(lowered, text, ASPECT_CUES["mechanism"]),
        method_need=bool(method_terms) or _has_cue(lowered, text, ASPECT_CUES["method"]),
        in_situ_or_operando_need=_has_cue(lowered, text, ASPECT_CUES["in_situ"]),
        ex_situ_need=_has_cue(lowered, text, ASPECT_CUES["ex_situ"]),
        material_case_need=bool(material_terms) or _has_cue(lowered, text, ASPECT_CUES["case"]),
        application_or_performance_need=bool(application_terms) or _has_cue(lowered, text, ASPECT_CUES["application"]),
        failure_or_limitation_need=bool(failure_terms) or _has_cue(lowered, text, ASPECT_CUES["failure"]),
        controversy_need=bool(controversy_terms) or _has_cue(lowered, text, ASPECT_CUES["controversy"]),
        review_background_need=_has_cue(lowered, text, ASPECT_CUES["review"]),
        core_terms=_unique([*core_terms, *target_terms], 12),
        method_terms=method_terms,
        mechanism_terms=mechanism_terms,
        material_or_case_terms=material_terms,
        application_or_metric_terms=application_terms,
        failure_or_limitation_terms=failure_terms,
        controversy_terms=controversy_terms,
        downweighted_terms=downweighted,
        term_sources=_term_sources(
            concept_terms=concept_terms,
            glossary_terms=glossary_terms,
            explicit_terms=explicit_terms,
        ),
    )
    return _repair_sparse_frame(frame, active_terms)


def generic_aspect_groups(frame: GenericResearchIntentFrame) -> list[str]:
    """Return aspect groups consumable by AspectCoverageAgent."""

    groups = [
        "review_background: review; background; overview; introduction",
        "theory_mechanism: theory; mechanism; model; explanation",
    ]
    if frame.method_need:
        groups.append("methods_characterization: method; characterization; measurement; probe; experiment")
    if frame.in_situ_or_operando_need:
        groups.append("in_situ_operando: in situ; operando; real-time; dynamic evolution")
    if frame.ex_situ_need:
        groups.append("ex_situ: ex situ; post mortem; offline characterization")
    if frame.material_case_need:
        groups.append("materials_cases: material; system; case study; benchmark")
    if frame.application_or_performance_need:
        groups.append("application_performance: application; performance; activity; accuracy; recall; metric")
    if frame.failure_or_limitation_need:
        groups.append("failure_limitation: failure; limitation; degradation; aging; stability")
    if frame.controversy_need:
        groups.append("controversy_debate: controversy; debate; competing mechanism; open question")
    return groups


def is_generic_noisy_term(term: str) -> bool:
    return " ".join(str(term or "").lower().split()) in GENERIC_NOISY_TERMS


def is_single_acronym_query(query: str) -> bool:
    """Return True for a final provider query that is only one ambiguous token."""

    cleaned = re.sub(r"[+\"()]", " ", str(query or ""))
    cleaned = re.sub(r"\b(AND|OR|NOT)\b", " ", cleaned, flags=re.IGNORECASE)
    tokens = [token for token in cleaned.split() if token]
    if len(tokens) != 1:
        return False
    token = tokens[0].strip()
    if len(token) <= 1:
        return True
    if token.upper() == token and re.fullmatch(r"[A-Z0-9][A-Z0-9\-]{1,8}", token):
        return True
    return len(token) <= 12 and token.lower() not in {"review", "mechanism", "method"}


def _glossary_terms(text: str) -> list[str]:
    glossary = _load_glossary()
    terms: list[str] = []
    for source, target in glossary.get("translations", {}).items():
        if str(source) in text:
            terms.extend(_split_expansion(str(target)))
    for source, target in glossary.get("phrase_expansions", {}).items():
        if str(source) in text:
            terms.extend(_split_expansion(str(target)))
    return _unique(terms, 32)


def _load_glossary() -> dict[str, Any]:
    if not GLOSSARY_PATH.exists():
        return {"translations": {}, "phrase_expansions": {}}
    try:
        data = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"translations": {}, "phrase_expansions": {}}
    if not isinstance(data, dict):
        return {"translations": {}, "phrase_expansions": {}}
    return data


def _split_expansion(value: str) -> list[str]:
    return [
        " ".join(part.split())
        for part in re.split(r"\s*[|;]\s*", value)
        if " ".join(part.split())
    ]


def _explicit_english_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Z0-9\-]{1,8}\b", text):
        terms.append(match.group(0))
    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){1,4}\b", text):
        phrase = " ".join(match.group(0).split())
        lower = phrase.lower()
        if any(word in GENERIC_NOISY_TERMS for word in tokenize(lower)):
            continue
        if len(tokenize(lower)) >= 2:
            terms.append(phrase)
    return _unique(terms, 24)


def _concept_terms(expert_intent: ExpertResearchIntent | None) -> list[str]:
    if expert_intent is None:
        return []
    return _unique(
        [
            concept.term
            for concept in expert_intent.structured_concepts
            if concept.term
            and concept.query_role in {"must", "optional"}
            and concept.should_use_in_provider_query
        ],
        32,
    )


def _concepts_by_category(
    expert_intent: ExpertResearchIntent | None,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if expert_intent is None:
        return result
    for concept in expert_intent.structured_concepts:
        if (
            concept.term
            and concept.query_role in {"must", "optional"}
            and concept.should_use_in_provider_query
        ):
            result.setdefault(concept.category, []).append(concept.term)
    return {key: _unique(values, 16) for key, values in result.items()}


def _terms_for_categories(
    concept_by_category: dict[str, list[str]],
    categories: set[str],
) -> list[str]:
    values: list[str] = []
    for category in categories:
        values.extend(concept_by_category.get(category, []))
    return _unique(values, 16)


def _core_terms(
    active_terms: list[str],
    concept_by_category: dict[str, list[str]],
) -> list[str]:
    preferred = _unique(
        [
            *concept_by_category.get("object", []),
            *concept_by_category.get("property", []),
            *concept_by_category.get("mechanism", []),
        ],
        12,
    )
    if preferred:
        return preferred
    ranked = sorted(
        active_terms,
        key=lambda term: (
            0 if _looks_like_scientific_acronym(term) else 1,
            0 if len(tokenize(term)) >= 2 else 1,
            len(term),
        ),
    )
    return _unique(ranked, 6)


def _context_terms(
    active_terms: list[str],
    core_terms: list[str],
    concept_by_category: dict[str, list[str]],
) -> list[str]:
    core_lower = {term.lower() for term in core_terms}
    values = [
        *concept_by_category.get("material", []),
        *concept_by_category.get("application", []),
    ]
    for term in active_terms:
        lower = term.lower()
        if lower in core_lower:
            continue
        if any(marker in lower for marker in ["battery", "catalyst", "oxide", "review", "screening", "solar cell", "thin film", "mof", "co2", "interface"]):
            values.append(term)
    return _unique(values, 8)


def _target_terms(
    active_terms: list[str],
    core_terms: list[str],
    context_terms: list[str],
) -> list[str]:
    excluded = {term.lower() for term in [*core_terms, *context_terms]}
    selected = [
        term
        for term in active_terms
        if term.lower() not in excluded
        and any(
            marker in term.lower()
            for marker in [
                "activity",
                "accuracy",
                "recall",
                "composition",
                "structure",
                "evolution",
                "stability",
                "performance",
                "defect",
                "recombination",
                "adsorption",
            ]
        )
    ]
    return _unique(selected, 8)


def _aspect_terms(active_terms: list[str], aspect: str) -> list[str]:
    markers = {
        "method": ["method", "characterization", "measurement", "probe", "microscopy", "spectroscopy", "screening", "workflow"],
        "mechanism": ["mechanism", "theory", "model", "effect", "electronic structure"],
        "case": ["material", "system", "case", "catalyst", "oxide", "anode", "mof", "solar cell", "thin film"],
        "application": ["application", "performance", "activity", "accuracy", "recall", "metric", "capture", "adsorption"],
        "failure": ["failure", "limitation", "degradation", "aging", "stability", "defect"],
        "controversy": ["controversy", "debate", "competing"],
    }.get(aspect, [])
    return _unique(
        [
            term
            for term in active_terms
            if any(marker in term.lower() for marker in markers)
        ],
        10,
    )


def _relation_terms(lowered: str, text: str) -> list[str]:
    values: list[str] = []
    if _has_cue(lowered, text, ["why", "important", "importance", "significance", "为什么", "重要", "意义"]):
        values.append("why important")
    if _has_cue(lowered, text, ["effect", "affect", "influence", "impact", "影响"]):
        values.append("effect relationship")
    if _has_cue(lowered, text, ["characterize", "probe", "detect", "表征", "探测"]):
        values.append("how to characterize")
    return _unique(values, 6)


def _repair_sparse_frame(
    frame: GenericResearchIntentFrame,
    active_terms: list[str],
) -> GenericResearchIntentFrame:
    if not frame.research_object and active_terms:
        frame.research_object = active_terms[:2]
    if not frame.core_terms:
        frame.core_terms = _unique([*frame.research_object, *active_terms[:4]], 8)
    if not frame.domain_context:
        frame.domain_context = [
            term
            for term in active_terms
            if term.lower() not in {value.lower() for value in frame.core_terms}
        ][:4]
    return frame


def _term_sources(
    concept_terms: list[str],
    glossary_terms: list[str],
    explicit_terms: list[str],
) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for term in concept_terms:
        sources.setdefault(term, []).append("structured_concepts")
    for term in glossary_terms:
        sources.setdefault(term, []).append("generic_glossary")
    for term in explicit_terms:
        sources.setdefault(term, []).append("user_text")
    return {term: _unique(values) for term, values in sources.items()}


def _contains_chinese_noise(text: str, term: str) -> bool:
    if term in {"importance", "significance"}:
        return "重要" in text or "意义" in text
    if term == "background":
        return "背景" in text
    if term == "review":
        return "综述" in text
    if term == "literature":
        return "文献" in text
    return False


def _has_cue(lowered: str, text: str, cues: list[str]) -> bool:
    return any(cue in lowered or cue in text for cue in cues)


def _looks_like_scientific_acronym(term: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9\-]{1,8}", term or ""))


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
