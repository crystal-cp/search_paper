from lit_screening.agents.intent_repair import NoviceIntentInterpreter
from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.agents.seed_extraction import SeedExtractionAgent
from lit_screening.pipeline import run_pipeline
from lit_screening.retrieval.base import RetrievalResult


QUESTION = (
    "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets: "
    "Classification, Example Materials, and Relation to Magnetoelectric Responses "
    "和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order "
    "相关的有关探测表面磁化和自旋极化的重要性的文章"
)


class EmptyOpenAlexClient:
    provider_name = "openalex"

    def search(self, *args, **kwargs):
        return RetrievalResult(raw={"results": []}, papers=[])


def repaired_intent():
    seed_hints = SeedExtractionAgent().extract(QUESTION)
    return NoviceIntentInterpreter().repair(
        QUESTION,
        seed_hints=seed_hints,
        use_llm=False,
    )


def test_related_papers_are_not_interpreted_as_same_author_only():
    intent = repaired_intent()

    assert any("same-author" in item for item in intent.possible_interpretations)
    assert "topic + method + theory-lineage" in intent.selected_interpretation
    assert "same-author" not in intent.selected_interpretation
    assert any("citation relation" in item for item in intent.unsafe_or_overbroad_assumptions)


def test_importance_is_not_collapsed_to_review_only():
    intent = repaired_intent()
    interpretations = " ".join(intent.possible_interpretations).lower()

    assert "background" in interpretations
    assert "evidence papers" in interpretations
    assert "application" in interpretations
    assert "review" not in intent.selected_interpretation.lower()
    assert any("not only review" in item.lower() for item in intent.assumptions)


def test_probe_wording_triggers_method_and_evidence_lane():
    intent = repaired_intent()
    interpretations = " ".join(intent.possible_interpretations).lower()
    assumptions = " ".join(intent.assumptions).lower()

    assert "direct experimental probes" in interpretations
    assert "indirect readout methods" in interpretations
    assert "surface-sensitive techniques" in interpretations
    assert "method-and-evidence lane" in assumptions


def test_spin_polarization_generates_possible_interpretations():
    intent = repaired_intent()
    interpretations = " ".join(intent.possible_interpretations).lower()

    assert "surface spin polarization" in interpretations
    assert "spin-resolved electronic structure" in interpretations
    assert "spin-polarized probe method" in interpretations
    assert "magnetic moment or surface magnetization" in interpretations
    assert intent.assumptions


def test_report_includes_intent_assumptions_section(tmp_path):
    output_dir = tmp_path / "outputs"
    run_pipeline(
        QUESTION,
        providers=["openalex"],
        max_per_query=0,
        output_dir=str(output_dir),
        retriever_agent=RetrieverAgent(clients={"openalex": EmptyOpenAlexClient()}),
    )

    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "## Intent assumptions and possible misreadings" in report
    assert "Possible interpretations of the user's wording" in report
    assert "User words not mechanically used as hard query terms" in report
