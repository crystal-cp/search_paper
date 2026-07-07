"""Plan provider query families from research lenses."""

from __future__ import annotations

from lit_screening.models import (
    QueryFamily,
    QueryFamilyPlan,
    ResearchLens,
    ResearchLensPlan,
    SeedHint,
)


MATERIALS_QUERY_TEMPLATES = {
    "theory_origin": {
        "purpose": "recover theory papers explaining boundary and surface magnetization",
        "openalex": [
            '"boundary magnetization" "magnetoelectric antiferromagnet"',
            '"equilibrium magnetization" boundary magnetoelectric antiferromagnet',
            '"magnetoelectric multipolization" "surface magnetization"',
        ],
        "semantic_scholar": [
            '+"boundary magnetization" +"magnetoelectric antiferromagnet"',
            '+"equilibrium magnetization" +boundary +magnetoelectric +antiferromagnet',
            '+"magnetoelectric multipolization" +"surface magnetization"',
        ],
        "roles": ["theory origin", "classification", "mechanism"],
        "stop": "Stop when core theory papers and at least two material examples are found.",
    },
    "spaldin_framework": {
        "purpose": "trace work connected to the Spaldin surface-magnetization framework",
        "openalex": [
            '"surface magnetization" antiferromagnets classification magnetoelectric responses',
            '"local magnetoelectric effects" "surface magnetic order"',
            '"atomic-site magnetic multipoles" antiferromagnet surface',
        ],
        "semantic_scholar": [
            '+"surface magnetization" +antiferromagnets +classification +"magnetoelectric responses"',
            '+"local magnetoelectric effects" +"surface magnetic order"',
            '+"atomic-site magnetic multipoles" +antiferromagnet +surface',
        ],
        "roles": ["seed-related theory", "framework extension", "citation bridge"],
        "stop": "Stop when papers cite or extend both Spaldin seed-paper ideas.",
    },
    "surface_magnetization_classification": {
        "purpose": "find classification and example-material papers for antiferromagnetic surface magnetization",
        "openalex": [
            '"surface magnetization" antiferromagnets classification',
            '"boundary magnetization" antiferromagnetic surface classification',
            '"uncompensated surface magnetization" antiferromagnet',
        ],
        "semantic_scholar": [
            '+"surface magnetization" +antiferromagnets +classification',
            '+"boundary magnetization" +"antiferromagnetic surface" +classification',
            '+"uncompensated surface magnetization" +antiferromagnet',
        ],
        "roles": ["classification", "example material", "surface termination rule"],
        "stop": "Stop when classification rules and representative antiferromagnets are covered.",
    },
    "local_magnetoelectric_predictor": {
        "purpose": "find papers using local magnetoelectric response as a surface-order predictor",
        "openalex": [
            '"local magnetoelectric effects" "surface magnetic order"',
            '"atomic-site magnetic multipoles" antiferromagnet surface',
            '"local magnetoelectric response" "magnetic multipole" antiferromagnet',
        ],
        "semantic_scholar": [
            '+"local magnetoelectric effects" +"surface magnetic order"',
            '+"atomic-site magnetic multipoles" +antiferromagnet +surface',
            '+"local magnetoelectric response" +"magnetic multipole" +antiferromagnet',
        ],
        "roles": ["predictor theory", "multipole analysis", "surface-order mechanism"],
        "stop": "Stop when local-response or multipole evidence connects to surface order.",
    },
    "local_magnetic_order": {
        "purpose": "find papers connecting local magnetoelectric descriptors to surface magnetic order",
        "openalex": [
            '"local magnetoelectric effects" "surface magnetic order"',
            '"local magnetoelectric response" "surface magnetic order"',
            '"magnetic multipole" "surface magnetic order" antiferromagnet',
        ],
        "semantic_scholar": [
            '+"local magnetoelectric effects" +"surface magnetic order"',
            '+"local magnetoelectric response" +"surface magnetic order"',
            '+"magnetic multipole" +"surface magnetic order" +antiferromagnet',
        ],
        "roles": ["surface-order predictor", "local order descriptor", "multipole analysis"],
        "stop": "Stop when local descriptors are tied to predicted surface magnetic order.",
    },
    "direct_surface_detection": {
        "purpose": "find direct surface-sensitive detection of magnetization or spin polarization",
        "openalex": [
            "Cr2O3 surface magnetization spin-polarized photoemission",
            "chromia surface spin polarization SPLEEM",
            "Cr2O3 surface magnetization domains XMCD PEEM MFM",
        ],
        "semantic_scholar": [
            "+Cr2O3 +\"surface magnetization\" +\"spin-polarized photoemission\"",
            "+chromia +\"surface spin polarization\" +SPLEEM",
            "+Cr2O3 +\"surface magnetization\" +domains +XMCD +PEEM +MFM",
        ],
        "roles": ["experimental detection", "surface-sensitive measurement", "method reference"],
        "stop": "Stop when at least one surface-sensitive method reports a direct signal.",
    },
    "nanoscale_readout": {
        "purpose": "find nanoscale readout routes for boundary magnetization",
        "openalex": [
            "NV magnetometry Cr2O3 boundary magnetization",
            "scanning diamond magnetometry antiferromagnetic domains Cr2O3",
        ],
        "semantic_scholar": [
            '+"NV magnetometry" +Cr2O3 +"boundary magnetization"',
            '+"scanning diamond magnetometry" +"antiferromagnetic domains" +Cr2O3',
        ],
        "roles": ["nanoscale readout", "magnetometry evidence", "domain imaging"],
        "stop": "Stop when nanoscale magnetic-field evidence is found or clearly absent.",
    },
    "applications": {
        "purpose": "connect surface magnetization to memory, readout, and spintronics use cases",
        "openalex": [
            "Cr2O3 exchange bias surface magnetization magnetoelectric memory",
            "magnetoelectric antiferromagnet boundary magnetization memory readout",
            "antiferromagnetic spintronics surface magnetization readout",
        ],
        "semantic_scholar": [
            "+Cr2O3 +\"exchange bias\" +\"surface magnetization\" +\"magnetoelectric memory\"",
            '+"magnetoelectric antiferromagnet" +"boundary magnetization" +memory +readout',
            '+"antiferromagnetic spintronics" +"surface magnetization" +readout',
        ],
        "roles": ["application motivation", "device concept", "readout mechanism"],
        "stop": "Stop when application papers explain why the surface signal matters.",
    },
    "limitations": {
        "purpose": "separate true surface magnetization from artifacts and limitations",
        "openalex": [
            "Cr2O3 surface paramagnetism finite temperature magnetoelectric antiferromagnet",
            "surface roughness robust magnetization antiferromagnet",
            "defects parasitic magnetization antiferromagnetic thin films",
        ],
        "semantic_scholar": [
            "+Cr2O3 +\"surface paramagnetism\" +\"finite temperature\" +\"magnetoelectric antiferromagnet\"",
            '+"surface roughness" +robust +magnetization +antiferromagnet',
            "+defects +\"parasitic magnetization\" +\"antiferromagnetic thin films\"",
        ],
        "roles": ["control", "failure mode", "artifact risk"],
        "stop": "Stop when roughness, defects, or finite-temperature confounds are characterized.",
    },
    "frontier": {
        "purpose": "scan recent frontiers around surface spin splitting and altermagnetism",
        "openalex": [
            "surface altermagnetism antiferromagnet spin-resolved ARPES",
            "surface spin splitting antiferromagnet altermagnetism",
        ],
        "semantic_scholar": [
            '+"surface altermagnetism" +antiferromagnet +"spin-resolved ARPES"',
            '+"surface spin splitting" +antiferromagnet +altermagnetism',
        ],
        "roles": ["frontier", "recent extension", "open problem"],
        "stop": "Stop when recent frontier papers show whether this route is active.",
    },
}


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
        if domain == "materials_magnetism" and lens.name in MATERIALS_QUERY_TEMPLATES:
            template = MATERIALS_QUERY_TEMPLATES[lens.name]
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
            )
        return _fallback_family(lens)


def _fallback_family(lens: ResearchLens) -> QueryFamily:
    concepts = lens.core_concepts or lens.synonyms or [lens.question]
    base_query = " ".join(concepts[:4])
    semantic_query = " ".join(_semantic_required(term) for term in concepts[:3])
    return QueryFamily(
        name=lens.name,
        purpose=lens.role or f"retrieve papers for {lens.name}",
        lens_name=lens.name,
        queries_by_provider={
            "openalex": [base_query],
            "semantic_scholar": [semantic_query or base_query],
        },
        expected_paper_roles=["lens-specific paper"],
        expected_evidence_types=list(lens.expected_evidence_types),
        exclusion_terms=list(lens.exclusion_risks),
        stop_condition="Stop when the lens has at least a small set of relevant papers.",
    )


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


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
