import requests

from lit_screening.agents.retriever import RetrieverAgent
from lit_screening.retrieval.arxiv_client import ArxivClient


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2401.01234v1</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Surface magnetization in antiferromagnets</title>
    <summary>
      We study boundary magnetization in antiferromagnetic materials.
    </summary>
    <author><name>A. Researcher</name></author>
    <author><name>B. Scientist</name></author>
    <category term="cond-mat.mtrl-sci" />
    <link href="http://arxiv.org/abs/2401.01234v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.01234v1" title="pdf" type="application/pdf"/>
  </entry>
</feed>
"""


class DummyArxivResponse:
    status_code = 200
    text = ARXIV_FEED

    def raise_for_status(self):
        return None


def test_arxiv_client_parses_atom_feed(monkeypatch):
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyArxivResponse()

    monkeypatch.setattr("lit_screening.retrieval.arxiv_client.requests.get", fake_get)
    client = ArxivClient(timeout=3.0, retries=0)

    result = client.search("surface magnetization", max_results=1)

    assert captured["params"]["search_query"] == "all:surface magnetization"
    assert result.raw["source"] == "arxiv"
    assert len(result.papers) == 1
    paper = result.papers[0]
    assert paper.title == "Surface magnetization in antiferromagnets"
    assert paper.authors == ["A. Researcher", "B. Scientist"]
    assert paper.year == 2024
    assert paper.abstract == (
        "We study boundary magnetization in antiferromagnetic materials."
    )
    assert paper.url == "http://arxiv.org/abs/2401.01234v1"
    assert paper.source_provider == "arxiv"
    assert paper.provider_ids["arxiv"] == "2401.01234v1"


def test_arxiv_client_handles_request_error(monkeypatch):
    def fake_get(url, params, timeout):
        raise requests.RequestException("network down")

    monkeypatch.setattr("lit_screening.retrieval.arxiv_client.requests.get", fake_get)
    client = ArxivClient(retries=0)

    result = client.search("surface magnetization", max_results=1)

    assert result.papers == []
    assert result.raw["provider"] == "arxiv"
    assert result.raw["error"] == "RequestException"
    assert "network down" in result.raw["error_message"]


def test_arxiv_is_not_enabled_in_default_retriever():
    assert "arxiv" not in RetrieverAgent().clients
