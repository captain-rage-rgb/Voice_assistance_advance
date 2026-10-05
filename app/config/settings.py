"""
Configuration management using pydantic-settings.
Loads environment variables cleanly with fallback defaults.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Real-Time Voice Computer Agent"
    app_env: str = "development"
    log_level: str = "INFO"


# Singleton instance for application-wide use
settings = Settings()
