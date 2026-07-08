"""Plan provider query families from research lenses."""

from __future__ import annotations

from lit_screening.agents.generic_intent import is_single_acronym_query
from lit_screening.domain_packs import load_domain_pack
from lit_screening.models import (
    QueryFamily,
    QueryFamilyPlan,
    ResearchLens,
    ResearchLensPlan,
    SeedHint,
)


DOMAIN_TEMPLATE_CACHE: dict[str, dict[str, dict[str, object]]] = {}


class QueryFamilyPlanner:
    """Generate QueryFamilyPlan objects from research lenses."""

    def plan(
        self,
        lens_plan: ResearchLensPlan,
        seed_hints: list[SeedHint] | None = None,
    ) -> QueryFamilyPlan:
        """Return query families for a lens plan without replacing PlannerAgent."""

        families = [
            self._family_for_lens(lens_plan.domain, lens)
            for lens in lens_plan.lenses
        ]
        seed_context = _seed_context_family(seed_hints)
        if seed_context is not None:
            families.append(seed_context)
        return QueryFamilyPlan(
            domain=lens_plan.domain,
            central_question=lens_plan.central_question,
            families=families,
        )

    def _family_for_lens(self, domain: str, lens: ResearchLens) -> QueryFamily:
        template = _domain_query_template(domain, lens.name)
        if template is not None:
            return QueryFamily(
                name=lens.name,
                purpose=str(template["purpose"]),
                lens_name=lens.name,
                queries_by_provider={
                    "openalex": list(template["openalex"]),
                    "semantic_scholar": list(template["semantic_scholar"]),
                },
                expected_paper_roles=list(template["roles"]),
                expected_evidence_types=_unique(
                    [*lens.expected_evidence_types, *list(template["roles"])]
                ),
                exclusion_terms=list(lens.exclusion_risks),
                stop_condition=str(template["stop"]),
                priority=_family_priority(lens.name, domain=domain),
                budget=_family_budget(lens.name, domain=domain),
            )
        return _fallback_family(lens)


def _fallback_family(lens: ResearchLens) -> QueryFamily:
    queries = _generic_queries_for_lens(lens)
    return QueryFamily(
        name=lens.name,
        purpose=lens.role or f"retrieve papers for {lens.name}",
        lens_name=lens.name,
        queries_by_provider={
            "openalex": list(queries),
            "semantic_scholar": list(queries),
        },
        expected_paper_roles=_generic_expected_roles(lens.name),
        expected_evidence_types=list(lens.expected_evidence_types),
        exclusion_terms=list(lens.exclusion_risks),
        stop_condition="Stop when the lens has at least a small set of relevant papers.",
        priority=_generic_family_priority(lens.name),
        budget=min(3, max(1, len(queries))),
    )


def _domain_query_template(domain: str, lens_name: str) -> dict[str, object] | None:
    """Return optional query-template metadata from an active domain pack."""

    if not domain or not lens_name:
        return None
    templates = DOMAIN_TEMPLATE_CACHE.get(domain)
    if templates is None:
        try:
            pack = load_domain_pack(domain)
        except ValueError:
            templates = {}
        else:
            raw_templates = getattr(pack, "query_templates", {}) or {}
            templates = {
                str(name): dict(payload)
                for name, payload in raw_templates.items()
                if isinstance(payload, dict)
            }
        DOMAIN_TEMPLATE_CACHE[domain] = templates
    template = templates.get(lens_name)
    return dict(template) if isinstance(template, dict) else None


def _generic_queries_for_lens(lens: ResearchLens) -> list[str]:
    """Compose provider-neutral fallback queries from lens structure."""

    core = _query_terms(lens.core_concepts, limit=4)
    context = _query_terms(lens.materials, limit=4)
    methods = _query_terms(lens.methods, limit=8)
    applications = _query_terms(lens.applications, limit=4)
    synonyms = _query_terms(lens.synonyms, limit=3)
    base = _join_query_terms([*(core[:2] or synonyms[:1]), *context[:2]])
    if not base:
        base = _join_query_terms([*core[:2], *synonyms[:1]])
    lens_name = lens.name
    queries: list[str] = []
    if base:
        queries.append(base)
    if lens_name in {"background_review", "core_topic"}:
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:2], "review"]),
                _join_query_terms([*core[:2], *context[:2], "background"]),
            ]
        )
    elif lens_name == "theory_mechanism":
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:2], "mechanism"]),
                _join_query_terms([*core[:2], *context[:2], "theory"]),
            ]
        )
    elif lens_name == "characterization_methods":
        method_seed = methods[:2] or ["characterization", "measurement"]
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:1], method_seed[0]]),
                _join_query_terms([*core[:2], *context[:1], "experimental characterization"]),
            ]
        )
    elif lens_name == "in_situ_or_operando_methods":
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:1], "in situ characterization"]),
                _join_query_terms([*core[:2], *context[:1], "operando characterization"]),
            ]
        )
    elif lens_name == "ex_situ_methods":
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:1], "ex situ characterization"]),
                _join_query_terms([*core[:2], *context[:1], "post mortem characterization"]),
            ]
        )
    elif lens_name == "materials_or_cases":
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:3], "case study"]),
                _join_query_terms([*core[:2], *context[:3], "representative materials"]),
            ]
        )
    elif lens_name == "application_or_performance":
        app_seed = applications[:2] or ["performance", "application"]
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:2], app_seed[0]]),
                _join_query_terms([*core[:2], *context[:2], "performance"]),
            ]
        )
    elif lens_name == "failure_or_limitation":
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:2], "failure mechanism"]),
                _join_query_terms([*core[:2], *context[:2], "limitation degradation"]),
            ]
        )
    elif lens_name == "controversy_debate":
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:2], "controversy"]),
                _join_query_terms([*core[:2], *context[:2], "competing mechanism"]),
            ]
        )
    elif lens_name == "method_comparison":
        queries.extend(
            [
                _join_query_terms([*core[:2], *methods[:8], "comparison review"]),
                _join_query_terms([*core[:2], *methods[:8], "advantages disadvantages"]),
            ]
        )
    elif lens_name in {"human_feedback", "evidence_validation", "performance_metrics"}:
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:2], *methods[:1], *applications[:1]]),
                _join_query_terms([*core[:2], *context[:2], lens_name.replace("_", " ")]),
            ]
        )
    else:
        queries.extend(
            [
                _join_query_terms([*core[:2], *context[:2], *methods[:1]]),
                _join_query_terms([*core[:2], *context[:2], *applications[:1]]),
            ]
        )
    return _sanitize_queries(queries, fallback_terms=[*core, *context, *methods, *applications])


def _query_terms(values: list[str], limit: int = 4) -> list[str]:
    return _unique(
        [
            value
            for value in values
            if value and str(value).strip().lower() not in {"why important", "importance"}
        ]
    )[:limit]


def _join_query_terms(values: list[str]) -> str:
    return " ".join(_quote_if_needed(value) for value in values if str(value or "").strip())


def _quote_if_needed(value: str) -> str:
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        return ""
    if " " in cleaned and not (cleaned.startswith('"') and cleaned.endswith('"')):
        return f'"{cleaned}"'
    return cleaned


def _sanitize_queries(queries: list[str], fallback_terms: list[str]) -> list[str]:
    cleaned = _unique([" ".join(query.split()) for query in queries if query])
    repaired: list[str] = []
    context = [
        term
        for term in fallback_terms
        if term and not is_single_acronym_query(term)
    ]
    context_term = context[0] if context else ""
    for query in cleaned:
        if is_single_acronym_query(query):
            if context_term:
                repaired.append(_join_query_terms([query, context_term]))
            continue
        repaired.append(query)
    return _unique([query for query in repaired if query and not is_single_acronym_query(query)], 3)


def _generic_expected_roles(lens_name: str) -> list[str]:
    mapping = {
        "background_review": ["review_background"],
        "theory_mechanism": ["theory_mechanism"],
        "characterization_methods": ["experimental_characterization", "method_development"],
        "in_situ_or_operando_methods": ["experimental_characterization"],
        "ex_situ_methods": ["experimental_characterization"],
        "materials_or_cases": ["material_case"],
        "application_or_performance": ["application_performance"],
        "failure_or_limitation": ["failure_limitation"],
        "controversy_debate": ["controversy_debate"],
        "method_comparison": ["method_development", "review_background"],
    }
    return mapping.get(lens_name, ["lens_specific_paper"])


def _generic_family_priority(lens_name: str) -> int:
    priorities = {
        "core_topic": 92,
        "theory_mechanism": 88,
        "characterization_methods": 86,
        "in_situ_or_operando_methods": 84,
        "ex_situ_methods": 80,
        "background_review": 76,
        "materials_or_cases": 74,
        "application_or_performance": 70,
        "failure_or_limitation": 68,
        "controversy_debate": 66,
        "method_comparison": 82,
    }
    return priorities.get(lens_name, 50)


def _seed_context_family(seed_hints: list[SeedHint] | None) -> QueryFamily | None:
    if not seed_hints:
        return None
    title_hints = [hint for hint in seed_hints if hint.title]
    if not title_hints:
        return None

    titles = _unique([str(hint.title) for hint in title_hints if hint.title])
    confidence_values = [hint.confidence for hint in title_hints if hint.confidence]
    confidence = max(confidence_values) if confidence_values else 0.0
    queries: list[str] = []
    for title in titles:
        queries.append(f'"{title}"')
        lower_title = title.lower()
        if "surface magnetization in antiferromagnets" in lower_title:
            queries.extend(
                [
                    "surface magnetization antiferromagnets",
                    'Spaldin "surface magnetization" antiferromagnets',
                ]
            )
        if "local magnetoelectric effects" in lower_title:
            queries.extend(
                [
                    '"local magnetoelectric effects" "surface magnetic order"',
                    'Spaldin "local magnetoelectric effects" "surface magnetic order"',
                ]
            )

    author_queries = _author_seed_queries(seed_hints, titles)
    queries = _unique([*queries, *author_queries])
    if not queries:
        return None

    return QueryFamily(
        name="seed_context",
        purpose="retrieve and contextualize papers explicitly mentioned by the user",
        lens_name="seed_context",
        queries_by_provider={
            "openalex": queries,
            "semantic_scholar": list(queries),
        },
        expected_paper_roles=[
            "seed paper",
            "seed citation context",
            "framework anchor",
        ],
        expected_evidence_types=[
            "exact title match",
            "seed-paper context",
            "author-linked framework",
        ],
        exclusion_terms=[],
        stop_condition="Stop when the explicitly mentioned seed titles and close context papers are represented.",
        priority=100,
        budget=4,
        linked_seed_titles=titles,
        seed_hint_confidence=confidence,
    )


def _author_seed_queries(seed_hints: list[SeedHint], titles: list[str]) -> list[str]:
    authors = _unique(
        [
            author
            for hint in seed_hints
            for author in hint.authors
            if author
        ]
    )
    if not authors:
        return []
    author_term = "Spaldin" if any("spaldin" in author.lower() for author in authors) else authors[0]
    queries: list[str] = []
    for title in titles:
        lower_title = title.lower()
        if "surface magnetization" in lower_title:
            queries.append(f'{author_term} "surface magnetization" antiferromagnets')
        if "local magnetoelectric effects" in lower_title:
            queries.append(
                f'{author_term} "local magnetoelectric effects" "surface magnetic order"'
            )
    return queries


def _semantic_required(term: str) -> str:
    cleaned = " ".join(str(term).split())
    if not cleaned:
        return ""
    if " " in cleaned:
        return f'+"{cleaned}"'
    return f"+{cleaned}"


def _family_priority(name: str, domain: str = "materials_magnetism") -> int:
    if domain == "ferroelectric_polarization":
        priorities = {
            "theory_origin": 90,
            "direct_probe_methods": 88,
            "interface_screening": 86,
            "materials_cases": 80,
            "device_applications": 72,
            "limitations": 68,
            "background_reviews": 58,
            "seed_context": 100,
        }
        return priorities.get(name, 50)
    priorities = {
        "seed_context": 100,
        "spaldin_framework": 95,
        "theory_origin": 90,
        "local_magnetoelectric_predictor": 88,
        "direct_surface_detection": 85,
        "nanoscale_readout": 82,
        "applications": 72,
        "limitations": 70,
        "frontier": 60,
    }
    return priorities.get(name, 50)


def _family_budget(name: str, domain: str = "materials_magnetism") -> int:
    if domain == "ferroelectric_polarization":
        budgets = {
            "theory_origin": 3,
            "direct_probe_methods": 4,
            "interface_screening": 3,
            "materials_cases": 4,
            "device_applications": 3,
            "limitations": 2,
            "background_reviews": 2,
            "seed_context": 4,
        }
        return budgets.get(name, 2)
    budgets = {
        "seed_context": 4,
        "spaldin_framework": 3,
        "theory_origin": 3,
        "local_magnetoelectric_predictor": 3,
        "direct_surface_detection": 3,
        "nanoscale_readout": 2,
        "applications": 2,
        "limitations": 2,
        "frontier": 2,
    }
    return budgets.get(name, 2)


def _unique(values: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result[:limit] if limit else result
