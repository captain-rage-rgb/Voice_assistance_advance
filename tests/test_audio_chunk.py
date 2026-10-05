"""
Unit tests for AudioChunk data model.
"""

from datetime import datetime
from app.audio.chunk import AudioChunk


def test_audio_chunk_properties():
    """Verify AudioChunk length, sample count calculation, and metadata."""
    # 16kHz, 16-bit mono, 30ms -> 480 samples -> 960 bytes
    sample_rate = 16000
    duration_ms = 30
    channels = 1
    sample_count = int(sample_rate * duration_ms / 1000)
    raw_data = b"\x00" * (sample_count * 2 * channels)

    chunk = AudioChunk(
        data=raw_data,
        sample_rate=sample_rate,
        channels=channels,
        duration_ms=duration_ms,
    )

    assert len(chunk) == 960
    assert chunk.sample_count == 480
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    assert chunk.duration_ms == 30
    assert isinstance(chunk.timestamp, datetime)
