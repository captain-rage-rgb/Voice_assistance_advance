"""
Tests for application configuration and initialization.
"""

from app.config.settings import Settings


def test_default_settings():
    """Verify that default settings load with expected attributes and values."""
    settings = Settings()
    assert settings.app_name == "Real-Time Voice Computer Agent"
    assert settings.app_env in {"development", "testing", "production"}
    assert settings.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def test_settings_custom_env(monkeypatch):
    """Verify that environment variables cleanly override defaults."""
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings()
    assert settings.app_env == "testing"
    assert settings.log_level == "DEBUG"
