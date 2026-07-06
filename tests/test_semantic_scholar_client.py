from lit_screening.retrieval.semantic_scholar_client import SemanticScholarClient


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
