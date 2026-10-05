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

    # Audio & Microphone Configuration
    audio_sample_rate: int = 16000  # 16kHz standard for speech recognition and WebRTC VAD
    audio_channels: int = 1         # Mono audio
    audio_chunk_duration_ms: int = 30  # 10, 20, or 30 ms required by WebRTC VAD
    audio_device_index: int | None = None  # None uses system default input device
    audio_queue_max_size: int = 100        # Bounded queue to avoid unbounded memory growth

    # Voice Activity Detection (VAD) & Endpointing Configuration
    vad_aggressiveness: int = 2     # 0 (least aggressive / filters least) to 3 (most aggressive)
    vad_speech_start_frames: int = 3  # Consecutive speech frames (~90ms) to trigger SPEECH_STARTED
    vad_speech_end_frames: int = 25   # Consecutive silence frames (~750ms) to trigger SPEECH_STOPPED


# Singleton instance for application-wide use
settings = Settings()
