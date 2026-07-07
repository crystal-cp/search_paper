"""Configuration values for the literature-screening pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Runtime configuration for retrieval and output generation."""

    providers: list[str] = field(default_factory=lambda: ["openalex", "semantic_scholar"])
    max_per_query: int = 10
    from_year: int | None = None
    output_dir: str = "outputs"
    cache_dir: str = "data/cache"
    use_cache: bool = True
    request_timeout: float = 8.0
    request_retries: int = 1
    rate_limit_sleep: float = 1.0
    llm_backend: str = "none"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_timeout: float = 30.0
    use_query_families: bool = False
