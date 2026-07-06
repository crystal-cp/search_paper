from lit_screening.dedup import deduplicate_papers, normalize_doi, normalize_title
from lit_screening.models import Paper


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1234/ABC.Def.") == "10.1234/abc.def"
    assert normalize_doi("doi: 10.5555/Test") == "10.5555/test"
    assert normalize_doi("") == ""


def test_normalize_title():
    assert normalize_title("Human-in-the-loop: Literature Screening!") == (
        "humanintheloop literature screening"
    )
    assert normalize_title("  A   Study, of Claims. ") == "a study of claims"


def test_deduplicate_papers():
    first = Paper(
        paper_id="p1",
        title="Evidence Verification with LLM Agents",
        abstract="Short abstract.",
        doi="https://doi.org/10.1000/test",
        source_provider="openalex",
    )
    second = Paper(
        paper_id="p2",
        title="Evidence verification with LLM agents",
        abstract="A longer abstract about evidence verification and human feedback.",
        doi="10.1000/TEST",
        source_provider="semantic_scholar",
        citation_count=10,
    )

    deduped = deduplicate_papers([first, second])

    assert len(deduped) == 1
    assert deduped[0].citation_count == 10
    assert "openalex" in deduped[0].source_provider
    assert "semantic_scholar" in deduped[0].source_provider
