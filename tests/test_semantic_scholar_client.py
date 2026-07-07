from lit_screening.retrieval.semantic_scholar_client import (
    SemanticScholarClient,
    semantic_scholar_expansion_fields,
)


def test_semantic_scholar_normalizer_handles_null_collection_fields():
    client = SemanticScholarClient(api_key="", use_cache=False)
    raw = {
        "data": [
            {
                "paperId": "s2-1",
                "title": "Null-safe Semantic Scholar record",
                "authors": None,
                "externalIds": None,
                "openAccessPdf": None,
                "fieldsOfStudy": None,
                "s2FieldsOfStudy": None,
                "publicationTypes": None,
                "tldr": None,
            }
        ]
    }

    papers = client._normalize_many(raw)

    assert len(papers) == 1
    assert papers[0].title == "Null-safe Semantic Scholar record"
    assert papers[0].authors == []
    assert papers[0].publication_types == []
    assert papers[0].fields_of_study == []


def test_semantic_scholar_expansion_fields_drop_tldr():
    fields = semantic_scholar_expansion_fields(
        "paperId,title,tldr,citedPaper.tldr,abstract"
    )

    assert fields == "paperId,title,abstract"


def test_semantic_scholar_sends_api_key_header(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setattr(
        "lit_screening.retrieval.semantic_scholar_client.requests.get",
        fake_get,
    )
    SemanticScholarClient._last_request_at = 0.0
    client = SemanticScholarClient(
        api_key="test-key",
        use_cache=False,
        min_interval_seconds=0.0,
    )

    client.search("surface magnetization", max_results=1)

    assert calls
    assert calls[0]["headers"] == {"x-api-key": "test-key"}


def test_semantic_scholar_rate_limit_is_shared_across_endpoints(monkeypatch):
    calls = []
    sleeps = []
    clock = {"now": 100.0}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": []}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "time": clock["now"]})
        return FakeResponse()

    def fake_monotonic():
        return clock["now"]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(
        "lit_screening.retrieval.semantic_scholar_client.requests.get",
        fake_get,
    )
    monkeypatch.setattr(
        "lit_screening.retrieval.semantic_scholar_client.time.monotonic",
        fake_monotonic,
    )
    monkeypatch.setattr(
        "lit_screening.retrieval.semantic_scholar_client.time.sleep",
        fake_sleep,
    )
    SemanticScholarClient._last_request_at = 0.0
    first = SemanticScholarClient(
        api_key="test-key",
        use_cache=False,
        min_interval_seconds=1.1,
    )
    second = SemanticScholarClient(
        api_key="test-key",
        use_cache=False,
        min_interval_seconds=1.1,
    )

    first.search("surface magnetization", max_results=1)
    second.get_references("S2PAPER", limit=1)

    assert len(calls) == 2
    assert calls[0]["time"] == 100.0
    assert calls[1]["time"] >= 101.1
    assert sleeps == [1.1]
