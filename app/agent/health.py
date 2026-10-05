"""
Internal Health Monitor and Subsystem Diagnostics (Phase 10).
Maintains lightweight health and availability state across all major subsystems
without transforming the overlay into a heavy dashboard.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, Optional


class SubsystemStatus(str, Enum):
    """Health states of an individual subsystem."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


@dataclass
class SubsystemHealth:
    """Diagnostic health snapshot of an individual subsystem."""

    name: str
    status: SubsystemStatus = SubsystemStatus.HEALTHY
    last_error: Optional[str] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """
    Central lightweight health monitor tracking subsystem availability:
    microphone, audio playback/TTS, STT/streaming, computer executor,
    overlay, and active run cancellation state.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("voice_agent.health")
        self._subsystems: Dict[str, SubsystemHealth] = {
            "microphone": SubsystemHealth(name="microphone"),
            "audio_playback": SubsystemHealth(name="audio_playback"),
            "streaming_stt": SubsystemHealth(name="streaming_stt"),
            "intent_detector": SubsystemHealth(name="intent_detector"),
            "executor": SubsystemHealth(name="executor"),
            "verifier": SubsystemHealth(name="verifier"),
            "overlay": SubsystemHealth(name="overlay"),
        }

    def record_healthy(self, subsystem: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Mark a subsystem as healthy and operational."""
        if subsystem in self._subsystems:
            sub = self._subsystems[subsystem]
            sub.status = SubsystemStatus.HEALTHY
            sub.last_error = None
            sub.last_updated = datetime.now(timezone.utc)
            if details:
                sub.details.update(details)

    def record_failure(self, subsystem: str, error: str, degraded: bool = False) -> None:
        """Record an error in a subsystem without crashing the entire agent."""
        if subsystem not in self._subsystems:
            self._subsystems[subsystem] = SubsystemHealth(name=subsystem)

        sub = self._subsystems[subsystem]
        sub.status = SubsystemStatus.DEGRADED if degraded else SubsystemStatus.ERROR
        sub.last_error = error
        sub.last_updated = datetime.now(timezone.utc)
        self.logger.warning("SUBSYSTEM HEALTH WARNING: [%s] -> %s (error: %s)", subsystem, sub.status.value, error)

    def get_status(self, subsystem: str) -> SubsystemStatus:
        """Get the current health status of a specific subsystem."""
        sub = self._subsystems.get(subsystem)
        return sub.status if sub else SubsystemStatus.UNAVAILABLE

    def get_summary(self) -> Dict[str, Any]:
        """Produce a diagnostic snapshot of all subsystem states."""
        return {
            name: {
                "status": sub.status.value,
                "last_error": sub.last_error,
                "last_updated": sub.last_updated.isoformat(),
            }
            for name, sub in self._subsystems.items()
        }
