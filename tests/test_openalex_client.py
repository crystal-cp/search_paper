from lit_screening.retrieval.openalex_client import OpenAlexClient


class DummyOpenAlexResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Surface magnetization in antiferromagnets",
                    "authorships": [],
                    "publication_year": 2024,
                }
            ]
        }


def test_openalex_client_reports_missing_api_key_without_request(tmp_path):
    client = OpenAlexClient(api_key="", cache_dir=str(tmp_path), use_cache=False)

    result = client.search("surface magnetization", max_results=1)

    assert result.papers == []
    assert result.raw["error"] == "missing_api_key"
    assert "OPENALEX_API_KEY" in result.raw["error_message"]


def test_openalex_normalizer_handles_null_collection_fields():
    client = OpenAlexClient(api_key="test", use_cache=False)
    raw = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "Null-safe OpenAlex record",
                "authorships": None,
                "concepts": None,
                "topics": None,
                "primary_location": None,
                "host_venue": None,
                "open_access": None,
                "best_oa_location": None,
            }
        ]
    }

    papers = client._normalize_many(raw)

    assert len(papers) == 1
    assert papers[0].title == "Null-safe OpenAlex record"
    assert papers[0].authors == []
    assert papers[0].fields_of_study == []


def test_openalex_client_uses_keyword_search_parameter(monkeypatch, tmp_path):
    captured = {}

    def fake_get(url, params, timeout):
        captured.update(params)
        return DummyOpenAlexResponse()

    monkeypatch.setattr(
        "lit_screening.retrieval.openalex_client.requests.get",
        fake_get,
    )
    client = OpenAlexClient(api_key="test-key", cache_dir=str(tmp_path), use_cache=False)

    result = client.search("surface magnetization", max_results=1, search_mode="keyword")

    assert captured["search"] == "surface magnetization"
    assert "search.exact" not in captured
    assert "search.semantic" not in captured
    assert result.papers[0].retrieval_stage == "openalex_keyword"
    assert result.papers[0].retrieval_query == "surface magnetization"


def test_openalex_client_uses_exact_search_parameter(monkeypatch, tmp_path):
    captured = {}

    def fake_get(url, params, timeout):
        captured.update(params)
        return DummyOpenAlexResponse()

    monkeypatch.setattr(
        "lit_screening.retrieval.openalex_client.requests.get",
        fake_get,
    )
    client = OpenAlexClient(api_key="test-key", cache_dir=str(tmp_path), use_cache=False)

    result = client.search("surface magnetization", max_results=1, search_mode="exact")

    assert captured["search.exact"] == "surface magnetization"
    assert "search" not in captured
    assert "search.semantic" not in captured
    assert result.papers[0].retrieval_stage == "openalex_exact"


def test_openalex_client_uses_semantic_search_parameter(monkeypatch, tmp_path):
    captured = {}

    def fake_get(url, params, timeout):
        captured.update(params)
        return DummyOpenAlexResponse()

    monkeypatch.setattr(
        "lit_screening.retrieval.openalex_client.requests.get",
        fake_get,
    )
    client = OpenAlexClient(api_key="test-key", cache_dir=str(tmp_path), use_cache=False)

    result = client.search("surface magnetization", max_results=1, search_mode="semantic")

    assert captured["search.semantic"] == "surface magnetization"
    assert "search" not in captured
    assert "search.exact" not in captured
    assert result.papers[0].retrieval_stage == "openalex_semantic"
