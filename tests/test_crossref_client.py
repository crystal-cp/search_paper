import requests

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.retrieval.crossref_client import CrossrefClient


class DummyCrossrefWorksResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "status": "ok",
            "message": {
                "items": [
                    {
                        "title": ["Surface magnetization in antiferromagnets"],
                        "author": [
                            {"given": "A.", "family": "Researcher"},
                            {"name": "Consortium Author"},
                        ],
                        "issued": {"date-parts": [[2024, 1, 1]]},
                        "DOI": "10.1234/Surface",
                        "container-title": ["Demo Journal"],
                        "URL": "https://doi.org/10.1234/surface",
                        "type": "journal-article",
                        "reference-count": 12,
                    }
                ]
            },
        }


class DummyCrossrefDoiResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "status": "ok",
            "message": {
                "title": ["Resolved DOI metadata"],
                "author": [{"given": "C.", "family": "Curie"}],
                "published-online": {"date-parts": [[2023, 5, 1]]},
                "DOI": "10.5555/Resolved",
                "container-title": ["Metadata Letters"],
                "URL": "https://doi.org/10.5555/resolved",
                "type": "journal-article",
            },
        }


def test_crossref_works_parses_metadata(monkeypatch):
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyCrossrefWorksResponse()

    monkeypatch.setattr("lit_screening.retrieval.crossref_client.requests.get", fake_get)
    client = CrossrefClient(timeout=3.0, retries=0)

    result = client.works("surface magnetization", rows=1)

    assert captured["params"] == {"query": "surface magnetization", "rows": 1}
    assert result.raw["source"] == "crossref"
    assert len(result.papers) == 1
    paper = result.papers[0]
    assert paper.title == "Surface magnetization in antiferromagnets"
    assert paper.authors == ["A. Researcher", "Consortium Author"]
    assert paper.year == 2024
    assert paper.doi == "10.1234/surface"
    assert paper.venue == "Demo Journal"
    assert paper.url == "https://doi.org/10.1234/surface"
    assert paper.source_provider == "crossref"
    assert paper.reference_count == 12


def test_crossref_resolve_doi_parses_single_work(monkeypatch):
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        return DummyCrossrefDoiResponse()

    monkeypatch.setattr("lit_screening.retrieval.crossref_client.requests.get", fake_get)
    client = CrossrefClient(retries=0)

    result = client.resolve_doi("https://doi.org/10.5555/Resolved")

    assert captured["url"].endswith("/10.5555%2Fresolved")
    assert captured["params"] == {}
    assert len(result.papers) == 1
    paper = result.papers[0]
    assert paper.title == "Resolved DOI metadata"
    assert paper.authors == ["C. Curie"]
    assert paper.year == 2023
    assert paper.doi == "10.5555/resolved"
    assert paper.venue == "Metadata Letters"
    assert paper.source_provider == "crossref"


def test_crossref_client_handles_request_error(monkeypatch):
    def fake_get(url, params, timeout):
        raise requests.RequestException("crossref unavailable")

    monkeypatch.setattr("lit_screening.retrieval.crossref_client.requests.get", fake_get)
    client = CrossrefClient(retries=0)

    result = client.works("surface magnetization", rows=1)

    assert result.papers == []
    assert result.raw["provider"] == "crossref"
    assert result.raw["error"] == "RequestException"
    assert "crossref unavailable" in result.raw["error_message"]


def test_crossref_is_not_enabled_in_default_retriever():
    assert "crossref" not in RetrieverAgent().clients
