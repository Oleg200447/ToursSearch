from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "tour-search-service"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"

    db_host: str = "tours_db"
    db_port: int = 5432
    db_name: str = "tours_baseline"
    db_user: str = "admin"
    db_password: str = "admin"
    db_min_pool_size: int = 1
    db_max_pool_size: int = 10
    db_command_timeout_seconds: int = 30

    embedder_url: str = "http://travel_embedder:7997/embeddings"
    embedder_model: str = "/app/model"
    embedder_timeout_seconds: float = 10.0

    sql_limit_candidates: int = 100
    response_top_k: int = 30

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        normalized = value.upper().strip()
        if normalized not in allowed:
            raise ValueError(f"Unsupported log level: {value}")
        return normalized

    @model_validator(mode="after")
    def validate_result_limits(self) -> "Settings":
        if self.sql_limit_candidates <= 0:
            raise ValueError("sql_limit_candidates must be greater than 0")
        if self.response_top_k <= 0:
            raise ValueError("response_top_k must be greater than 0")
        if self.response_top_k > self.sql_limit_candidates:
            raise ValueError("response_top_k cannot be greater than sql_limit_candidates")
        return self

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
