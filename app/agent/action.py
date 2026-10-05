"""
Action models representing executable intent units.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class ActionStatus(str, Enum):
    """Lifecycle status of an action."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Action:
    """Represents a discrete action unit to be processed by the executor."""

    action_type: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    status: ActionStatus = ActionStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    result: Optional[Any] = None

    def mark_running(self) -> None:
        """Mark action as currently running."""
        self.status = ActionStatus.RUNNING

    def mark_completed(self, result: Optional[Any] = None) -> None:
        """Mark action as successfully completed."""
        self.status = ActionStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str) -> None:
        """Mark action as failed with error details."""
        self.status = ActionStatus.FAILED
        self.error_message = error

    def mark_cancelled(self) -> None:
        """Mark action as cancelled."""
        self.status = ActionStatus.CANCELLED
