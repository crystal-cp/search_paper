"""Rule-based concept mapping into research lenses."""

from __future__ import annotations

from lit_screening.domain_packs import load_domain_pack
from lit_screening.models import DomainPack, ResearchLens, ResearchLensPlan, SeedHint


SPALDIN_SEED_HINTS = [
    "Surface Magnetization in Antiferromagnets",
    "Surface Magnetization in Antiferromagnets: Classification, Example Materials, and Relation to Magnetoelectric Responses",
    "Local Magnetoelectric Effects as Predictors of Surface Magnetic Order",
]


class ConceptMapper:
    """Map a user question to a rule-based ResearchLensPlan."""

    def map_question(
        self,
        question: str,
        domain: str = "materials_magnetism",
        search_brief: object | None = None,
        search_contract: object | None = None,
        seed_hints: list[SeedHint] | None = None,
    ) -> ResearchLensPlan:
        """Return research lenses without calling LLMs, APIs, or the pipeline."""

        pack = load_domain_pack(domain)
        if domain == "ferroelectric_polarization":
            return _map_ferroelectric_polarization(
                question=question,
                pack=pack,
                search_brief=search_brief,
                search_contract=search_contract,
                seed_hints=seed_hints,
            )
        if domain != "materials_magnetism":
            raise ValueError(
                "ConceptMapper currently supports materials_magnetism and "
                "ferroelectric_polarization."
            )

        seed_title_hints = _seed_hint_titles(seed_hints)
        context = " ".join(
            [
                _context_text(question, search_brief, search_contract),
                " ".join(seed_title_hints).lower(),
            ]
        )
        concepts = _materials_concepts(context, pack)
        materials = _materials_for_context(context, pack)
        exclusions = _unique([*pack.false_positive_terms, *_contract_exclusions(search_contract)])
        seed_paper_hints = _unique([*_seed_hints(context), *seed_title_hints])
        lenses: list[ResearchLens] = []

        if _has_any(
            context,
            [
                "surface magnetization",
                "表面磁化",
                "boundary magnetization",
                "antiferromagnet",
                "反铁磁",
            ],
        ):
            lenses.append(
                ResearchLens(
                    name="theory_origin",
                    role="trace the theoretical origin and classification of surface magnetization",
                    question="What theory explains surface or boundary magnetization in antiferromagnets?",
                    core_concepts=concepts,
                    synonyms=_concept_synonyms(pack, ["surface_magnetization"]),
                    materials=materials,
                    methods=[],
                    applications=pack.applications,
                    seed_paper_hints=seed_paper_hints,
                    expected_evidence_types=[
                        "classification",
                        "symmetry argument",
                        "theoretical prediction",
                    ],
                    exclusion_risks=exclusions,
                )
            )

        if _has_any(
            context,
            ["spaldin", *[hint.lower() for hint in SPALDIN_SEED_HINTS]],
        ):
            lenses.append(
                ResearchLens(
                    name="spaldin_framework",
                    role="connect the question to the Spaldin surface-magnetization framework",
                    question="Which papers build on Spaldin-style surface magnetization and magnetoelectric-response arguments?",
                    core_concepts=concepts,
                    synonyms=_concept_synonyms(
                        pack,
                        ["surface_magnetization", "local_magnetoelectricity"],
                    ),
                    materials=materials,
                    methods=[],
                    applications=pack.applications,
                    seed_paper_hints=seed_paper_hints,
                    expected_evidence_types=[
                        "citation link",
                        "theory extension",
                        "example material",
                    ],
                    exclusion_risks=exclusions,
                )
            )

        if _has_any(
            context,
            [
                "detect",
                "detection",
                "probe",
                "探测",
                "photoemission",
                "microscopy",
                "surface spin",
                "自旋极化",
            ],
        ):
            lenses.append(
                ResearchLens(
                    name="direct_surface_detection",
                    role="find direct surface-sensitive probes of magnetization or spin polarization",
                    question="Which experiments directly detect surface magnetization or surface spin polarization?",
                    core_concepts=concepts,
                    synonyms=_concept_synonyms(
                        pack,
                        ["surface_magnetization", "surface_spin_polarization"],
                    ),
                    materials=materials,
                    methods=[
                        "SPLEEM",
                        "XMCD-PEEM",
                        "spin-resolved photoemission",
                        "SP-STM",
                    ],
                    applications=pack.applications,
                    seed_paper_hints=seed_paper_hints,
                    expected_evidence_types=[
                        "surface-sensitive measurement",
                        "imaging contrast",
                        "spin-resolved signal",
                    ],
                    exclusion_risks=exclusions,
                )
            )

        if _has_any(context, ["nanoscale", "readout", "probe", "探测", "magnetometry", "磁化"]):
            lenses.append(
                ResearchLens(
                    name="nanoscale_readout",
                    role="find nanoscale magnetic readout routes for antiferromagnetic surfaces",
                    question="Which nanoscale readout methods can detect boundary magnetization or spin signals?",
                    core_concepts=concepts,
                    synonyms=_concept_synonyms(pack, ["surface_magnetization"]),
                    materials=materials,
                    methods=["NV magnetometry", "scanning diamond magnetometry"],
                    applications=[
                        "Neel vector readout",
                        "Néel vector readout",
                        "antiferromagnetic spintronics",
                    ],
                    seed_paper_hints=seed_paper_hints,
                    expected_evidence_types=[
                        "local magnetic field map",
                        "nanoscale readout",
                        "surface stray-field signal",
                    ],
                    exclusion_risks=exclusions,
                )
            )

        if _has_any(
            context,
            [
                "magnetoelectric",
                "磁电",
                "local magnetoelectric",
                "multipole",
                "surface magnetic order",
            ],
        ):
            lenses.append(
                ResearchLens(
                    name="local_magnetoelectric_predictor",
                    role="map local magnetoelectric response to surface magnetic order predictions",
                    question="How do local magnetoelectric effects or magnetic multipoles predict surface magnetic order?",
                    core_concepts=concepts,
                    synonyms=_concept_synonyms(pack, ["local_magnetoelectricity"]),
                    materials=materials,
                    methods=[],
                    applications=["magnetoelectric memory", "antiferromagnetic spintronics"],
                    seed_paper_hints=seed_paper_hints,
                    expected_evidence_types=[
                        "local response tensor",
                        "magnetic multipole",
                        "surface-order predictor",
                    ],
                    exclusion_risks=exclusions,
                )
            )

        if _has_any(context, ["application", "应用", "memory", "spintronics", "readout", "important", "重要"]):
            lenses.append(
                ResearchLens(
                    name="applications",
                    role="identify device or spintronics motivations for surface magnetization",
                    question="Why is surface magnetization or spin polarization important for antiferromagnetic applications?",
                    core_concepts=concepts,
                    synonyms=_concept_synonyms(
                        pack,
                        ["surface_magnetization", "surface_spin_polarization"],
                    ),
                    materials=materials,
                    methods=[],
                    applications=pack.applications,
                    seed_paper_hints=seed_paper_hints,
                    expected_evidence_types=[
                        "application motivation",
                        "readout mechanism",
                        "device implication",
                    ],
                    exclusion_risks=exclusions,
                )
            )

        if _has_any(context, ["roughness", "termination", "limitation", "limits", "限制", "局限"]):
            lenses.append(
                ResearchLens(
                    name="limitations",
                    role="separate robust surface signals from termination or roughness artifacts",
                    question="When are surface magnetization claims robust or sensitive to termination and roughness?",
                    core_concepts=concepts,
                    synonyms=_concept_synonyms(pack, ["surface_magnetization"]),
                    materials=materials,
                    methods=[],
                    applications=[],
                    seed_paper_hints=seed_paper_hints,
                    expected_evidence_types=[
                        "control comparison",
                        "surface termination analysis",
                        "roughness sensitivity",
                    ],
                    exclusion_risks=exclusions,
                )
            )

        if _has_any(context, ["frontier", "recent", "latest", "前沿", "最新", "important", "重要"]):
            lenses.append(
                ResearchLens(
                    name="frontier",
                    role="find recent extensions and open problems",
                    question="What recent work extends surface magnetization, spin polarization, or local magnetoelectric predictors?",
                    core_concepts=concepts,
                    synonyms=_all_synonyms(pack),
                    materials=materials,
                    methods=_methods_for_context(context, pack),
                    applications=pack.applications,
                    seed_paper_hints=seed_paper_hints,
                    expected_evidence_types=[
                        "recent result",
                        "open problem",
                        "new material proposal",
                    ],
                    exclusion_risks=exclusions,
                )
            )

        if not lenses:
            lenses.append(
                ResearchLens(
                    name="theory_origin",
                    role="start from core materials-magnetism concepts",
                    question="What papers explain surface magnetization in antiferromagnetic materials?",
                    core_concepts=concepts,
                    synonyms=_all_synonyms(pack),
                    materials=materials,
                    methods=[],
                    applications=pack.applications,
                    seed_paper_hints=seed_paper_hints,
                    expected_evidence_types=["theory", "example material", "review"],
                    exclusion_risks=exclusions,
                )
            )

        if _has_surface_magnetization_seed(seed_title_hints):
            lenses.extend(
                [
                    ResearchLens(
                        name="spaldin_framework",
                        role="connect the question to the Spaldin surface-magnetization framework",
                        question="Which papers build on the Spaldin surface-magnetization framework?",
                        core_concepts=concepts,
                        synonyms=_concept_synonyms(
                            pack,
                            ["surface_magnetization", "local_magnetoelectricity"],
                        ),
                        materials=materials,
                        methods=[],
                        applications=pack.applications,
                        seed_paper_hints=seed_paper_hints,
                        expected_evidence_types=[
                            "seed-paper context",
                            "classification",
                            "theory extension",
                        ],
                        exclusion_risks=exclusions,
                    ),
                    ResearchLens(
                        name="theory_origin",
                        role="trace the theory origin of boundary and surface magnetization",
                        question="What theory classifies surface magnetization in antiferromagnets?",
                        core_concepts=concepts,
                        synonyms=_concept_synonyms(pack, ["surface_magnetization"]),
                        materials=materials,
                        methods=[],
                        applications=pack.applications,
                        seed_paper_hints=seed_paper_hints,
                        expected_evidence_types=[
                            "classification",
                            "symmetry argument",
                            "example material",
                        ],
                        exclusion_risks=exclusions,
                    ),
                    ResearchLens(
                        name="surface_magnetization_classification",
                        role="classify surface magnetization examples and boundary conditions",
                        question="How are antiferromagnetic surfaces classified by their surface magnetization?",
                        core_concepts=concepts,
                        synonyms=_concept_synonyms(pack, ["surface_magnetization"]),
                        materials=materials,
                        methods=[],
                        applications=pack.applications,
                        seed_paper_hints=seed_paper_hints,
                        expected_evidence_types=[
                            "classification table",
                            "example material",
                            "surface termination rule",
                        ],
                        exclusion_risks=exclusions,
                    ),
                ]
            )

        if _has_local_magnetoelectric_seed(seed_title_hints):
            lenses.extend(
                [
                    ResearchLens(
                        name="local_magnetoelectric_predictor",
                        role="map local magnetoelectric response to surface magnetic order predictions",
                        question="How do local magnetoelectric effects predict surface magnetic order?",
                        core_concepts=concepts,
                        synonyms=_concept_synonyms(pack, ["local_magnetoelectricity"]),
                        materials=materials,
                        methods=[],
                        applications=["magnetoelectric memory", "antiferromagnetic spintronics"],
                        seed_paper_hints=seed_paper_hints,
                        expected_evidence_types=[
                            "local response tensor",
                            "magnetic multipole",
                            "surface-order predictor",
                        ],
                        exclusion_risks=exclusions,
                    ),
                    ResearchLens(
                        name="local_magnetic_order",
                        role="connect local magnetoelectric descriptors to surface magnetic order",
                        question="Which local magnetic-order descriptors predict surface spin or magnetic order?",
                        core_concepts=concepts,
                        synonyms=_concept_synonyms(pack, ["local_magnetoelectricity"]),
                        materials=materials,
                        methods=[],
                        applications=["magnetoelectric memory", "antiferromagnetic spintronics"],
                        seed_paper_hints=seed_paper_hints,
                        expected_evidence_types=[
                            "local order parameter",
                            "magnetic multipole",
                            "surface magnetic order",
                        ],
                        exclusion_risks=exclusions,
                    ),
                ]
            )

        return ResearchLensPlan(
            domain=pack.domain_name,
            central_question=_central_question(question, search_brief, search_contract),
            lenses=_dedupe_lenses(lenses),
        )


def _map_ferroelectric_polarization(
    question: str,
    pack: DomainPack,
    search_brief: object | None,
    search_contract: object | None,
    seed_hints: list[SeedHint] | None,
) -> ResearchLensPlan:
    """Map ferroelectric-polarization questions to domain-pack-driven lenses."""

    seed_title_hints = _seed_hint_titles(seed_hints)
    context = " ".join(
        [
            _context_text(question, search_brief, search_contract),
            " ".join(seed_title_hints).lower(),
        ]
    )
    concepts = _ferroelectric_concepts(context, pack)
    materials = _ferroelectric_materials(context, pack)
    methods = _ferroelectric_methods(context, pack)
    exclusions = _unique([*pack.false_positive_terms, *_contract_exclusions(search_contract)])
    seed_paper_hints = _unique(seed_title_hints)
    lenses: list[ResearchLens] = []

    if _has_any(context, ["why", "important", "重要", "background", "theory", "理论", "depolarization", "surface charge"]):
        lenses.append(
            ResearchLens(
                name="theory_origin",
                role="recover theory explaining surface polarization and depolarization in ferroelectric thin films",
                question="What theory explains surface polarization, surface charge, and depolarization fields in ferroelectric thin films?",
                core_concepts=concepts,
                synonyms=_concept_synonyms(pack, ["ferroelectric_polarization", "surface_polarization", "screening_effects"]),
                materials=materials,
                methods=[],
                applications=pack.applications,
                seed_paper_hints=seed_paper_hints,
                expected_evidence_types=["theory", "model", "surface-charge relation"],
                exclusion_risks=exclusions,
            )
        )

    if _has_any(context, ["probe", "detect", "characterize", "探测", "表征", "method", "实验", "pfm", "shg", "microscopy"]):
        lenses.append(
            ResearchLens(
                name="direct_probe_methods",
                role="find surface-sensitive and direct experimental probes of ferroelectric polarization",
                question="Which experiments directly probe or characterize ferroelectric surface or interface polarization?",
                core_concepts=concepts,
                synonyms=_concept_synonyms(pack, ["surface_polarization"]),
                materials=materials,
                methods=methods,
                applications=[],
                seed_paper_hints=seed_paper_hints,
                expected_evidence_types=["direct probe", "domain imaging", "surface potential measurement"],
                exclusion_risks=exclusions,
            )
        )

    if _has_any(context, ["screening", "屏蔽", "interface", "界面", "depolarization", "surface charge", "charge"]):
        lenses.append(
            ResearchLens(
                name="interface_screening",
                role="connect surface or interface screening to ferroelectric polarization stability",
                question="How do screening charges and interfaces control depolarization and surface polarization in ferroelectric thin films?",
                core_concepts=concepts,
                synonyms=_concept_synonyms(pack, ["screening_effects"]),
                materials=materials,
                methods=methods,
                applications=pack.applications,
                seed_paper_hints=seed_paper_hints,
                expected_evidence_types=["screening mechanism", "interface effect", "depolarization-field evidence"],
                exclusion_risks=exclusions,
            )
        )

    if _has_any(context, ["material", "材料", "case", "案例", "batio3", "pzt", "bifeo3", "hfo2"]) or materials:
        lenses.append(
            ResearchLens(
                name="materials_cases",
                role="find representative material systems for ferroelectric surface polarization",
                question="Which ferroelectric materials provide clear surface-polarization or thin-film screening case studies?",
                core_concepts=concepts,
                synonyms=_all_synonyms(pack),
                materials=materials,
                methods=methods,
                applications=pack.applications,
                seed_paper_hints=seed_paper_hints,
                expected_evidence_types=["material case", "thin-film experiment", "polarization switching"],
                exclusion_risks=exclusions,
            )
        )

    if _has_any(context, ["application", "应用", "device", "器件", "memory", "fefet", "tunnel", "important", "重要"]):
        lenses.append(
            ResearchLens(
                name="device_applications",
                role="connect surface polarization and screening to device motivation",
                question="Why do surface polarization and screening matter for ferroelectric devices?",
                core_concepts=concepts,
                synonyms=_concept_synonyms(pack, ["ferroelectric_polarization", "screening_effects"]),
                materials=materials,
                methods=[],
                applications=pack.applications,
                seed_paper_hints=seed_paper_hints,
                expected_evidence_types=["device motivation", "readout or memory effect", "interface constraint"],
                exclusion_risks=exclusions,
            )
        )

    if _has_any(context, ["limitation", "limits", "限制", "imprint", "pinning", "dead layer", "leakage", "screening"]):
        lenses.append(
            ResearchLens(
                name="limitations",
                role="separate robust polarization claims from screening, imprint, and interface limitations",
                question="What limitations or boundary conditions affect surface polarization in ferroelectric thin films?",
                core_concepts=concepts,
                synonyms=_concept_synonyms(pack, ["screening_effects"]),
                materials=materials,
                methods=methods,
                applications=[],
                seed_paper_hints=seed_paper_hints,
                expected_evidence_types=["limitation", "boundary condition", "artifact risk"],
                exclusion_risks=exclusions,
            )
        )

    if _has_any(context, ["background", "review", "overview", "背景", "综述", "important", "重要"]):
        lenses.append(
            ResearchLens(
                name="background_reviews",
                role="find conservative background or review papers without treating importance as only a review request",
                question="Which background papers frame why ferroelectric surface polarization and screening matter?",
                core_concepts=concepts,
                synonyms=_all_synonyms(pack),
                materials=materials,
                methods=methods,
                applications=pack.applications,
                seed_paper_hints=seed_paper_hints,
                expected_evidence_types=["review", "background", "field framing"],
                exclusion_risks=exclusions,
            )
        )

    if not lenses:
        lenses.append(
            ResearchLens(
                name="theory_origin",
                role="start from ferroelectric surface-polarization concepts",
                question="What papers explain ferroelectric thin-film surface polarization?",
                core_concepts=concepts,
                synonyms=_all_synonyms(pack),
                materials=materials,
                methods=methods,
                applications=pack.applications,
                seed_paper_hints=seed_paper_hints,
                expected_evidence_types=["theory", "experiment", "material case"],
                exclusion_risks=exclusions,
            )
        )

    return ResearchLensPlan(
        domain=pack.domain_name,
        central_question=_central_question(question, search_brief, search_contract),
        lenses=_dedupe_lenses(lenses),
    )


def _context_text(
    question: str,
    search_brief: object | None,
    search_contract: object | None,
) -> str:
    parts = [question]
    parts.extend(_object_strings(search_brief))
    parts.extend(_object_strings(search_contract))
    return " ".join(parts).lower()


def _object_strings(value: object | None) -> list[str]:
    if value is None:
        return []
    fields = [
        "refined_question",
        "original_question",
        "user_goal",
        "search_intent",
        "time_window",
        "success_definition",
        "inclusion_criteria",
        "exclusion_criteria",
        "required_aspects",
        "preferred_paper_types",
        "must_include_concepts",
        "must_exclude_concepts",
    ]
    strings: list[str] = []
    for field_name in fields:
        if isinstance(value, dict):
            item = value.get(field_name)
        else:
            item = getattr(value, field_name, None)
        if isinstance(item, list):
            strings.extend(str(entry) for entry in item)
        elif item:
            strings.append(str(item))
    return strings


def _central_question(
    question: str,
    search_brief: object | None,
    search_contract: object | None,
) -> str:
    for value in [search_contract, search_brief]:
        if isinstance(value, dict):
            refined = value.get("refined_question", "")
        else:
            refined = getattr(value, "refined_question", "") if value is not None else ""
        if isinstance(refined, str) and refined.strip():
            return " ".join(refined.split())
    return " ".join(question.split())


def _ferroelectric_concepts(context: str, pack: DomainPack) -> list[str]:
    concepts: list[str] = []
    if _has_any(context, ["ferroelectric", "ferroelectricity", "铁电"]):
        concepts.extend(_concept_synonyms(pack, ["ferroelectric_polarization"])[:2])
    if _has_any(context, ["surface polarization", "表面极化", "surface charge"]):
        concepts.extend(_concept_synonyms(pack, ["surface_polarization"])[:2])
    if _has_any(context, ["depolarization", "screening", "屏蔽", "interface", "界面"]):
        concepts.extend(_concept_synonyms(pack, ["screening_effects"])[:3])
    if not concepts:
        concepts.extend([*pack.domain_anchors[:2], *_concept_synonyms(pack, ["ferroelectric_polarization"])[:1]])
    return _unique(concepts)


def _ferroelectric_materials(context: str, pack: DomainPack) -> list[str]:
    selected = [material for material in pack.materials if material.lower() in context]
    defaults = ["BaTiO3", "PZT", "PbTiO3", "BiFeO3", "HfO2", "HfZrO2", "LiNbO3"]
    if _has_any(context, ["material", "材料", "case", "案例", "thin film", "薄膜"]):
        selected.extend(defaults)
    return _unique([*selected, *pack.materials])


def _ferroelectric_methods(context: str, pack: DomainPack) -> list[str]:
    selected = [method for method in pack.methods if method.lower() in context]
    if _has_any(context, ["probe", "detect", "characterize", "探测", "表征", "microscopy", "experiment"]):
        selected.extend(
            [
                "piezoresponse force microscopy",
                "second harmonic generation",
                "Kelvin probe force microscopy",
                "XPS",
                "TEM",
                "STEM",
            ]
        )
    return _unique([*selected, *pack.methods])


def _materials_concepts(context: str, pack: DomainPack) -> list[str]:
    concepts: list[str] = []
    if _has_any(context, ["surface magnetization", "表面磁化", "boundary magnetization"]):
        concepts.extend(_concept_synonyms(pack, ["surface_magnetization"])[:2])
    if _has_any(context, ["spin polarization", "surface spin", "自旋极化"]):
        concepts.extend(
            [
                "spin polarization",
                *_concept_synonyms(pack, ["surface_spin_polarization"])[:1],
            ]
        )
    if _has_any(context, ["antiferromagnet", "antiferromagnetic", "反铁磁"]):
        concepts.append("antiferromagnet")
    if _has_any(context, ["magnetoelectric", "磁电", "multipole", "surface magnetic order"]):
        concepts.extend(["local magnetoelectric response", "magnetic multipole"])
    if not concepts:
        concepts.extend(_concept_synonyms(pack, ["surface_magnetization"])[:2])
    return _unique(concepts)


def _materials_for_context(context: str, pack: DomainPack) -> list[str]:
    selected = [material for material in pack.materials if material.lower() in context]
    defaults = ["Cr2O3", "chromia", "FeF2", "NiO", "CuMnAs"]
    return _unique([*selected, *defaults, *pack.materials])


def _methods_for_context(context: str, pack: DomainPack) -> list[str]:
    selected = [method for method in pack.methods if method.lower() in context]
    if _has_any(
        context,
        ["detect", "detection", "probe", "探测", "photoemission", "microscopy"],
    ):
        selected.extend(["SPLEEM", "XMCD-PEEM", "spin-resolved photoemission", "SP-STM"])
    if _has_any(context, ["nanoscale", "readout", "magnetometry", "探测"]):
        selected.extend(["NV magnetometry", "scanning diamond magnetometry"])
    return _unique([*selected, *pack.methods])


def _seed_hints(context: str) -> list[str]:
    if _has_any(
        context,
        [
            "spaldin",
            "surface magnetization in antiferromagnets",
            "local magnetoelectric effects as predictors",
        ],
    ):
        return SPALDIN_SEED_HINTS
    return []


def _seed_hint_titles(seed_hints: list[SeedHint] | None) -> list[str]:
    if not seed_hints:
        return []
    return _unique(
        [
            str(seed_hint.title)
            for seed_hint in seed_hints
            if seed_hint.title
        ]
    )


def _has_surface_magnetization_seed(seed_titles: list[str]) -> bool:
    return any(
        "surface magnetization in antiferromagnets" in title.lower()
        for title in seed_titles
    )


def _has_local_magnetoelectric_seed(seed_titles: list[str]) -> bool:
    return any(
        "local magnetoelectric effects" in title.lower()
        for title in seed_titles
    )


def _contract_exclusions(search_contract: object | None) -> list[str]:
    if search_contract is None:
        return []
    values = getattr(search_contract, "must_exclude_concepts", [])
    return [str(value) for value in values] if isinstance(values, list) else []


def _concept_synonyms(pack: DomainPack, concept_names: list[str]) -> list[str]:
    synonyms: list[str] = []
    for name in concept_names:
        concept = pack.concepts.get(name)
        if concept:
            synonyms.extend(concept.synonyms)
            synonyms.extend(concept.related)
    return _unique(synonyms)


def _all_synonyms(pack: DomainPack) -> list[str]:
    synonyms: list[str] = []
    for concept in pack.concepts.values():
        synonyms.extend(concept.synonyms)
        synonyms.extend(concept.related)
    return _unique(synonyms)


def _dedupe_lenses(lenses: list[ResearchLens]) -> list[ResearchLens]:
    result: list[ResearchLens] = []
    seen: set[str] = set()
    for lens in lenses:
        if lens.name in seen:
            continue
        result.append(lens)
        seen.add(lens.name)
    return result


def _has_any(text: str, markers: list[str]) -> bool:
    return any(marker.lower() in text for marker in markers)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
