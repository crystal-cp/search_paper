from lit_screening.agents.domain_guardrail import DomainGuardrailAgent
from lit_screening.models import Paper
from lit_screening.pipeline import plan_screening_queries


FERROELECTRIC_QUESTION = (
    "我想了解铁电薄膜表面极化为什么重要，以及有哪些实验方法可以直接探测或表征它。"
    "最好能帮我找到理论背景、实验探测方法、典型材料案例、器件应用，"
    "以及表面/界面屏蔽效应相关的论文。"
)


def _contract():
    return plan_screening_queries(FERROELECTRIC_QUESTION)["search_contract"]


def _decision(paper: Paper) -> str:
    return DomainGuardrailAgent().assess(paper, _contract()).domain_decision


def test_ferroelectric_guardrail_excludes_screening_and_surface_false_positives():
    false_positive_papers = [
        Paper(
            paper_id="drug",
            title="Drug screening and cognitive screening workflow",
            abstract="Drug screening and cognitive screening are evaluated in patients.",
            fields_of_study=["Medicine"],
        ),
        Paper(
            paper_id="cell",
            title="Cell surface polarization during migration",
            abstract="Cell surface polarization is a biological process.",
            fields_of_study=["Biology"],
        ),
        Paper(
            paper_id="plasmon",
            title="Polarization-controlled coupling of surface plasmon polaritons",
            abstract="Generic SHG plasmonics controls surface plasmon polaritons.",
            fields_of_study=["Physics"],
        ),
        Paper(
            paper_id="screening",
            title="COSMO solvent screening for molecular design",
            abstract="This paper studies generic solvent screening.",
        ),
    ]

    assert all(_decision(paper) != "in_scope" for paper in false_positive_papers)


def test_ferroelectric_guardrail_keeps_core_ferroelectric_papers_in_scope():
    in_scope_papers = [
        Paper(
            paper_id="batio3-pfm",
            title="PFM study of BaTiO3 ferroelectric thin film surface polarization",
            abstract=(
                "Piezoresponse force microscopy measures ferroelectric domains, "
                "surface polarization, and depolarization field screening charge."
            ),
            fields_of_study=["Materials Science", "Physics"],
        ),
        Paper(
            paper_id="screening",
            title="Surface-screening mechanisms in ferroelectric thin films",
            abstract=(
                "Interface screening and screening charge control the depolarization "
                "field in ferroelectric thin films."
            ),
            fields_of_study=["Materials Science"],
        ),
    ]

    assert all(_decision(paper) == "in_scope" for paper in in_scope_papers)
