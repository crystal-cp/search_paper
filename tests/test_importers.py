from lit_screening.importers import import_papers_from_file


def test_import_bibtex_file(tmp_path):
    path = tmp_path / "library.bib"
    path.write_text(
        """
@article{surface2024,
  title = {Surface magnetization in antiferromagnets},
  author = {Ada Researcher and Ben Scientist},
  year = {2024},
  journal = {Demo Journal},
  doi = {https://doi.org/10.1234/Surface},
  url = {https://example.test/surface},
  abstract = {Surface magnetization controls boundary spin signals.}
}
""",
        encoding="utf-8",
    )

    result = import_papers_from_file(path)

    assert result.detected_format == "bibtex"
    assert result.raw_count == 1
    assert result.papers[0].title == "Surface magnetization in antiferromagnets"
    assert result.papers[0].doi == "10.1234/surface"
    assert result.papers[0].authors == ["Ada Researcher", "Ben Scientist"]
    assert result.papers[0].source_provider == "imported_bibtex"


def test_import_ris_file(tmp_path):
    path = tmp_path / "library.ris"
    path.write_text(
        """
TY  - JOUR
TI  - Surface spin signals
AU  - Ada Researcher
AU  - Ben Scientist
PY  - 2023
JO  - Demo Journal
DO  - 10.5678/Spin
AB  - Surface spin signals are linked to magnetization.
ER  -
""",
        encoding="utf-8",
    )

    result = import_papers_from_file(path)

    assert result.detected_format == "ris"
    assert result.papers[0].title == "Surface spin signals"
    assert result.papers[0].year == 2023
    assert result.papers[0].doi == "10.5678/spin"
    assert result.papers[0].authors == ["Ada Researcher", "Ben Scientist"]


def test_import_csv_file(tmp_path):
    path = tmp_path / "library.csv"
    path.write_text(
        (
            "title,abstract,authors,year,venue,doi,url,citation_count\n"
            "Imported surface paper,Imported abstract,Ada; Ben,2022,Demo,10.9999/demo,https://example.test,12\n"
        ),
        encoding="utf-8",
    )

    result = import_papers_from_file(path, "csv")

    assert result.detected_format == "csv"
    assert result.papers[0].title == "Imported surface paper"
    assert result.papers[0].citation_count == 12
    assert result.papers[0].source_provider == "imported_csv"
