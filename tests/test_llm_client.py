from lit_screening.llm_client import GenericLLMClient, parse_json_safely, strip_markdown_code_fences


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_strip_markdown_code_fences_and_parse_json():
    text = """```json
{"queries": ["one", "two"]}
```"""

    assert strip_markdown_code_fences(text) == '{"queries": ["one", "two"]}'
    result = parse_json_safely(text)
    assert result.invalid_llm_output is False
    assert result.data["queries"] == ["one", "two"]


def test_invalid_json_returns_safe_fallback():
    result = parse_json_safely("not json", fallback={"queries": []})

    assert result.invalid_llm_output is True
    assert result.error_type == "invalid_json"
    assert result.data == {"queries": []}


def test_generic_llm_client_uses_mocked_openai_compatible_response(monkeypatch):
    def fake_post(url, headers, json, timeout):
        assert url == "https://example.test/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "demo-model"
        return FakeResponse(
            {
                "choices": [
                    {"message": {"content": '{"supported": true, "confidence": 0.9}'}}
                ]
            }
        )

    monkeypatch.setattr("lit_screening.llm_client.requests.post", fake_post)
    client = GenericLLMClient(
        provider_name="test",
        api_key_env_var="TEST_API_KEY",
        base_url="https://example.test",
        model="demo-model",
        api_key="test-key",
    )

    result = client.chat_json("system", "user")

    assert result.invalid_llm_output is False
    assert result.data["supported"] is True
    assert result.data["confidence"] == 0.9
