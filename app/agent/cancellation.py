"""
Unified Cancellation Context for agent execution runs.
Provides a thread-safe, async-friendly cancellation token and lifecycle management
for active requests, streaming operations, and multi-step action sequences.
"""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Callable, List, Optional
import uuid


class CancellationToken:
    """
    Unified cancellation token associated with an active request/run.
    Allows registering cooperative cancellation callbacks and checking cancellation state.
    """

    def __init__(self, run_id: Optional[str] = None, logger: Optional[logging.Logger] = None) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())
        self.created_at: datetime = datetime.now(timezone.utc)
        self.logger = logger or logging.getLogger("voice_agent.cancellation")

        self._cancelled_event = asyncio.Event()
        self._reason: Optional[str] = None
        self._callbacks: List[Callable[[], None]] = []
        self._lock = asyncio.Lock()

    @property
    def is_cancelled(self) -> bool:
        """Return True if cancellation has been requested."""
        return self._cancelled_event.is_set()

    @property
    def reason(self) -> Optional[str]:
        """Return reason for cancellation if cancelled."""
        return self._reason

    def check_cancelled(self) -> None:
        """
        Raise asyncio.CancelledError if cancellation has been requested.
        Useful at cooperative async loop boundaries.
        """
        if self.is_cancelled:
            raise asyncio.CancelledError(f"Operation cancelled (run_id={self.run_id}): {self._reason}")

    async def cancel(self, reason: str = "User interruption") -> bool:
        """
        Trigger cancellation for this token.
        Safe to call multiple times (subsequent calls are no-ops).
        Returns True if this call initiated the cancellation, False if already cancelled.
        """
        async with self._lock:
            if self._cancelled_event.is_set():
                return False

            self._reason = reason
            self._cancelled_event.set()
            self.logger.info("CANCELLATION REQUESTED for run_id=%s: %s", self.run_id, reason)

            # Fire registered synchronous or cooperative callbacks
            for callback in self._callbacks:
                try:
                    callback()
                except Exception as exc:
                    self.logger.warning("Error running cancellation callback: %s", exc)

            return True

    def register_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to run immediately when cancellation is triggered."""
        if self.is_cancelled:
            try:
                callback()
            except Exception:
                pass
            return
        self._callbacks.append(callback)

    async def wait(self) -> None:
        """Wait asynchronously until cancellation is requested."""
        await self._cancelled_event.wait()
