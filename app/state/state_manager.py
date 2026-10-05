"""
Agent state model tracking lifecycle, execution status, and task context.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from app.agent.action import Action


class AgentLifecycleState(str, Enum):
    """Overall agent lifecycle states."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class AgentState:
    """Represents current runtime state."""

    lifecycle_state: AgentLifecycleState = AgentLifecycleState.IDLE
    current_task: Optional[str] = None
    current_action: Optional[Action] = None
    pending_actions_count: int = 0
    is_execution_active: bool = False
    cancellation_requested: bool = False
    last_error: Optional[str] = None

    # Audio & Voice Activity state
    microphone_active: bool = False
    voice_activity_active: bool = False
    audio_error: Optional[str] = None


class StateManager:
    """Manages agent state transitions and query operations."""

    def __init__(self) -> None:
        self._state = AgentState()

    @property
    def current(self) -> AgentState:
        """Returns the current state snapshot."""
        return self._state

    def set_lifecycle(self, state: AgentLifecycleState) -> None:
        """Update overall agent lifecycle state."""
        self._state.lifecycle_state = state

    def set_current_action(self, action: Optional[Action]) -> None:
        """Update currently executing action."""
        self._state.current_action = action
        self._state.is_execution_active = action is not None

    def set_pending_count(self, count: int) -> None:
        """Update count of pending actions in queue."""
        self._state.pending_actions_count = max(0, count)

    def request_cancellation(self) -> None:
        """Record that cancellation has been requested."""
        self._state.cancellation_requested = True

    def clear_cancellation(self) -> None:
        """Reset cancellation flag."""
        self._state.cancellation_requested = False

    def record_error(self, error: str) -> None:
        """Record the latest error message."""
        self._state.last_error = error

    def set_microphone_active(self, active: bool) -> None:
        """Update microphone streaming state."""
        self._state.microphone_active = active
        if not active:
            self._state.voice_activity_active = False

    def set_voice_activity_active(self, active: bool) -> None:
        """Update voice activity state."""
        self._state.voice_activity_active = active

    def set_audio_error(self, error: Optional[str]) -> None:
        """Record or clear audio error state."""
        self._state.audio_error = error

    def reset_task(self, task_name: Optional[str] = None) -> None:
        """Set or clear the active high-level task."""
        self._state.current_task = task_name
