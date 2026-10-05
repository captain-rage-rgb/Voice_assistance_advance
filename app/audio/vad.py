"""
Voice Activity Detection (VAD) and Endpointing using WebRTC VAD.
Processes continuous audio frames and manages speech state transitions.
"""

from enum import Enum
import logging
from typing import Optional
import webrtcvad
from app.audio.chunk import AudioChunk


class VADState(str, Enum):
    """Current activity state determined by VAD."""

    SILENCE = "SILENCE"
    SPEAKING = "SPEAKING"


class VoiceActivityDetector:
    """
    Lightweight Voice Activity Detector with configurable endpointing.
    Distinguishes between continuous speech, short breath pauses, and speech completion.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        speech_start_frames: int = 3,
        speech_end_frames: int = 25,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Args:
            aggressiveness: WebRTC VAD aggressiveness mode (0 to 3).
            speech_start_frames: Consecutive positive frames needed to enter SPEAKING state.
            speech_end_frames: Consecutive silence frames needed to exit SPEAKING state (endpointing).
        """
        self.aggressiveness = aggressiveness
        self.speech_start_frames = max(1, speech_start_frames)
        self.speech_end_frames = max(1, speech_end_frames)
        self.logger = logger or logging.getLogger("voice_agent.audio.vad")

        self._vad = webrtcvad.Vad(self.aggressiveness)
        self._state = VADState.SILENCE
        self._consecutive_speech = 0
        self._consecutive_silence = 0

    @property
    def state(self) -> VADState:
        """Return the current speaking state."""
        return self._state

    def reset(self) -> None:
        """Reset internal speech/silence counters to baseline SILENCE state."""
        self._state = VADState.SILENCE
        self._consecutive_speech = 0
        self._consecutive_silence = 0

    def process_chunk(self, chunk: AudioChunk) -> tuple[bool, Optional[str]]:
        """
        Evaluate an AudioChunk for voice activity and apply endpointing state logic.

        Args:
            chunk: The AudioChunk to analyze. Must be 16-bit mono PCM at 8000, 16000, 32000, or 48000 Hz,
                   and duration must be 10, 20, or 30 ms.

        Returns:
            A tuple of (is_speech_frame, transition_event_name):
            - is_speech_frame: Boolean result for this single chunk.
            - transition_event_name: 'SPEECH_STARTED' when transitioning from SILENCE -> SPEAKING,
                                     'SPEECH_STOPPED' when transitioning from SPEAKING -> SILENCE,
                                     or None if state did not change.
        """
        try:
            is_speech = self._vad.is_speech(chunk.data, chunk.sample_rate)
        except Exception as exc:
            self.logger.warning("VAD evaluation error on chunk: %s", exc)
            return False, None

        transition: Optional[str] = None

        if self._state == VADState.SILENCE:
            if is_speech:
                self._consecutive_speech += 1
                if self._consecutive_speech >= self.speech_start_frames:
                    self._state = VADState.SPEAKING
                    self._consecutive_silence = 0
                    transition = "SPEECH_STARTED"
            else:
                self._consecutive_speech = 0

        elif self._state == VADState.SPEAKING:
            if not is_speech:
                self._consecutive_silence += 1
                if self._consecutive_silence >= self.speech_end_frames:
                    self._state = VADState.SILENCE
                    self._consecutive_speech = 0
                    transition = "SPEECH_STOPPED"
            else:
                # Still speaking, reset consecutive silence count
                self._consecutive_silence = 0

        return is_speech, transition
