"""
Unit tests for VoiceActivityDetector and endpointing logic.
Tests silence detection, speech onset, breath pauses, and speech ending without requiring a physical mic.
"""

import numpy as np
from app.audio.chunk import AudioChunk
from app.audio.vad import VADState, VoiceActivityDetector


def _generate_sine_chunk(freq: float = 440.0, sample_rate: int = 16000, duration_ms: int = 30) -> AudioChunk:
    """Generate a 16-bit PCM chunk representing a synthetic vocal/speech frequency."""
    t = np.linspace(0, duration_ms / 1000.0, int(sample_rate * duration_ms / 1000.0), endpoint=False)
    # Scaled sine wave in int16 range
    waveform = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return AudioChunk(
        data=waveform.tobytes(),
        sample_rate=sample_rate,
        channels=1,
        duration_ms=duration_ms,
    )


def _generate_silence_chunk(sample_rate: int = 16000, duration_ms: int = 30) -> AudioChunk:
    """Generate a 16-bit PCM chunk containing complete silence."""
    sample_count = int(sample_rate * duration_ms / 1000.0)
    return AudioChunk(
        data=b"\x00" * (sample_count * 2),
        sample_rate=sample_rate,
        channels=1,
        duration_ms=duration_ms,
    )


def test_vad_silence():
    """Verify that pure silence does not trigger speech state."""
    vad = VoiceActivityDetector(aggressiveness=2, speech_start_frames=3, speech_end_frames=5)
    silence = _generate_silence_chunk()

    for _ in range(10):
        is_speech, transition = vad.process_chunk(silence)
        assert not is_speech
        assert transition is None

    assert vad.state == VADState.SILENCE


def test_vad_speech_start_and_end_transitions():
    """Verify speech onset threshold and endpointing silence threshold transitions."""
    speech_start_thresh = 3
    speech_end_thresh = 5
    vad = VoiceActivityDetector(
        aggressiveness=1,
        speech_start_frames=speech_start_thresh,
        speech_end_frames=speech_end_thresh,
    )

    speech_chunk = _generate_sine_chunk(freq=300.0)
    silence_chunk = _generate_silence_chunk()

    # Step 1: Feed fewer speech chunks than the start threshold (should not trigger yet)
    for _ in range(speech_start_thresh - 1):
        _, transition = vad.process_chunk(speech_chunk)
        assert transition is None
        assert vad.state == VADState.SILENCE

    # Step 2: Feed the threshold chunk -> should transition to SPEAKING
    _, transition = vad.process_chunk(speech_chunk)
    assert transition == "SPEECH_STARTED"
    assert vad.state == VADState.SPEAKING

    # Step 3: Natural short pause (fewer silence chunks than endpoint threshold)
    for _ in range(speech_end_thresh - 1):
        _, transition = vad.process_chunk(silence_chunk)
        assert transition is None
        assert vad.state == VADState.SPEAKING  # Still speaking!

    # Step 4: Resume speaking after pause
    _, transition = vad.process_chunk(speech_chunk)
    assert transition is None
    assert vad.state == VADState.SPEAKING

    # Step 5: Full endpointing silence. Feed silence chunks until transition occurs.
    transition = None
    for _ in range(speech_end_thresh + 10):
        _, transition = vad.process_chunk(silence_chunk)
        if transition == "SPEECH_STOPPED":
            break

    assert transition == "SPEECH_STOPPED"
    assert vad.state == VADState.SILENCE


def test_vad_reset():
    """Verify reset returns state to SILENCE and clears counters."""
    vad = VoiceActivityDetector(aggressiveness=1, speech_start_frames=1)
    speech = _generate_sine_chunk()

    vad.process_chunk(speech)
    assert vad.state == VADState.SPEAKING

    vad.reset()
    assert vad.state == VADState.SILENCE
