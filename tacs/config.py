from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TACS_",
        extra="ignore",
    )

    # LLM
    llm_backend: str = "ollama"
    llm_max_tokens: int = 4096
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Generation
    seed: int = 42
    conversation_count: int = 50
    min_tool_calls: int = 3
    min_distinct_tools: int = 2
    max_turns: int = 10
    max_retries: int = 50

    # Paths
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    output_dir: Path = Path("output")

    # Logging
    log_level: str = "INFO"

    # Memory
    corpus_memory_enabled: bool = True
    memory_top_k: int = 5


config = Config()
