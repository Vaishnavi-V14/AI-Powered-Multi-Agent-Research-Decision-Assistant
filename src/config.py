"""Centralised configuration loaded from environment / .env file."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Web Search
    tavily_api_key: str = Field(default="", description="Tavily API key")

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server base URL"
    )
    ollama_model: str = Field(
        default="llama3.2:3b", description="Ollama model identifier"
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./data/chroma", description="ChromaDB persistence directory"
    )

    # SQLite
    database_url: str = Field(
        default="./data/sessions.db", description="SQLite database file path"
    )

    # Pipeline
    max_retry: int = Field(
        default=3, description="Maximum critic retry attempts before giving up"
    )
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()
