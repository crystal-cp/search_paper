from lit_screening.retrieval.openalex_client import OpenAlexClient


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
