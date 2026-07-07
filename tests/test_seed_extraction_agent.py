import json

from lit_screening.agents.seed_extraction import SeedExtractionAgent
from lit_screening.pipeline import run_pipeline


SPALDIN_QUESTION = (
    "有没有和 Nicola A. Spaldin 的 Surface Magnetization in Antiferromagnets: "
    "Classification, Example Materials, and Relation to Magnetoelectric Responses "
    "和 Local Magnetoelectric Effects as Predictors of Surface Magnetic Order "
    "相关的有关探测表面磁化和自旋极化的重要性的文章"
)


def test_seed_extraction_agent_extracts_two_spaldin_titles():
    hints = SeedExtractionAgent().extract(SPALDIN_QUESTION)
    titles = {hint.title for hint in hints if hint.title}
    author_values = {
        author
        for hint in hints
        for author in hint.authors
    }

    assert (
        "Surface Magnetization in Antiferromagnets: Classification, Example "
        "Materials, and Relation to Magnetoelectric Responses"
    ) in titles
    assert "Local Magnetoelectric Effects as Predictors of Surface Magnetic Order" in titles
    assert "Nicola A. Spaldin" in author_values or "Spaldin" in author_values
    assert all(hint.doi is None for hint in hints if hint.title)


def test_seed_extraction_agent_extracts_doi_and_arxiv_without_api():
    question = (
        'Find papers related to "Surface Magnetization in Antiferromagnets" '
        "and DOI 10.48550/arXiv.2301.10140, also arXiv:2301.10140."
    )
    hints = SeedExtractionAgent().extract(question)

    assert any(hint.doi == "10.48550/arXiv.2301.10140" for hint in hints)
    assert any(hint.arxiv_id == "2301.10140" for hint in hints)


def test_pipeline_writes_seed_hints_artifact_without_retrieval(tmp_path):
    output_dir = tmp_path / "outputs"

    result = run_pipeline(
        question=SPALDIN_QUESTION,
        providers=[],
        max_per_query=0,
        output_dir=str(output_dir),
    )
    seed_hints = json.loads((output_dir / "seed_hints.json").read_text())
    trace = json.loads((output_dir / "agent_trace.json").read_text())

    assert len(seed_hints) >= 2
    assert result.seed_hints
    assert result.retrieval_counts == {}
    assert trace["seed_extraction"]["executed"] is True
    assert trace["seed_extraction"]["artifact_only"] is True
