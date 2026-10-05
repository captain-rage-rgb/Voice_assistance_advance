"""
Unified Run Manager (Phases 9 & 10).
Authoritative coordinator for active run lifecycles, run ID generation,
cancellation tokens, task context, stale-result protection, and resource cleanup.
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Dict, List, Optional
import uuid
from app.agent.cancellation import CancellationToken
from app.agent.context import TaskContext


class RunLifecycleState(str, Enum):
    """Lifecycle states of an individual agent run."""

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


class RunManager:
    """
    Authoritative coordinator for all run lifecycles.
    Enforces that only one authoritative run is active at a time,
    provides centralized stale-result rejection, and prevents resource leaks.
    """

    def __init__(
        self,
        max_run_history: int = 15,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.max_run_history = max_run_history
        self.logger = logger or logging.getLogger("voice_agent.run_manager")

        self._active_run_id: Optional[str] = None
        self._active_token: Optional[CancellationToken] = None
        self._tokens: Dict[str, CancellationToken] = {}
        self._contexts: Dict[str, TaskContext] = {}
        self._states: Dict[str, RunLifecycleState] = {}
        self._run_history: List[str] = []

        self._lock = asyncio.Lock()
        self._generation: int = 0

    @property
    def active_run_id(self) -> Optional[str]:
        """Return the current active run ID."""
        return self._active_run_id

    @property
    def active_token(self) -> Optional[CancellationToken]:
        """Return the current active cancellation token."""
        return self._active_token

    @property
    def active_generation(self) -> int:
        """Return the current monotonic generation number."""
        return self._generation

    async def get_or_create_active_token(self) -> CancellationToken:
        """Retrieve the current active token or initialize a new active run."""
        async with self._lock:
            if self._active_token is None or self._active_token.is_cancelled:
                return await self._create_new_run_locked()
            return self._active_token

    async def create_new_run(self, original_intent: str = "") -> CancellationToken:
        """Explicitly create and register a new authoritative active run."""
        async with self._lock:
            return await self._create_new_run_locked(original_intent)

    async def _create_new_run_locked(self, original_intent: str = "") -> CancellationToken:
        """Internal helper to create a run while holding the lock."""
        # If an existing run is still active, cancel and mark superseded
        if self._active_token and not self._active_token.is_cancelled:
            old_id = self._active_run_id
            self.logger.info("SUPERSEDING RUN: Old run %s superseded by new run", old_id)
            await self._active_token.cancel(reason="Superseded by new run")
            if old_id in self._states:
                self._states[old_id] = RunLifecycleState.SUPERSEDED

        self._generation += 1
        new_token = CancellationToken(logger=self.logger)
        run_id = new_token.run_id

        self._active_run_id = run_id
        self._active_token = new_token
        self._tokens[run_id] = new_token
        self._states[run_id] = RunLifecycleState.ACTIVE

        # Initialize structured context
        context = TaskContext(run_id=run_id, original_intent=original_intent)
        self._contexts[run_id] = context
        self._run_history.append(run_id)

        self._prune_history_locked()
        self.logger.info("RUN CREATED: Generation %d -> Run ID %s", self._generation, run_id)
        return new_token

    def is_run_active(self, run_id: Optional[str]) -> bool:
        """Central validation: check if a run_id is currently the active, uncancelled run."""
        if not run_id:
            return False
        if run_id != self._active_run_id:
            return False
        token = self._tokens.get(run_id)
        if token and token.is_cancelled:
            return False
        state = self._states.get(run_id)
        return state in (RunLifecycleState.ACTIVE, RunLifecycleState.PAUSED)

    def is_run_cancelled(self, run_id: Optional[str]) -> bool:
        """Check if a specific run was cancelled or superseded."""
        if not run_id:
            return False
        token = self._tokens.get(run_id)
        if token and token.is_cancelled:
            return True
        state = self._states.get(run_id)
        return state in (RunLifecycleState.CANCELLED, RunLifecycleState.SUPERSEDED)

    def get_context(self, run_id: Optional[str] = None) -> TaskContext:
        """Get the TaskContext for a specific run ID or the active run."""
        target_id = run_id or self._active_run_id
        if target_id and target_id in self._contexts:
            return self._contexts[target_id]
        # Return fallback or create if needed
        fallback_id = target_id or "default_run"
        if fallback_id not in self._contexts:
            self._contexts[fallback_id] = TaskContext(run_id=fallback_id)
        return self._contexts[fallback_id]

    async def cancel_active_run(self, reason: str = "Cancellation requested") -> None:
        """Cancel the currently active run."""
        async with self._lock:
            if self._active_token and not self._active_token.is_cancelled:
                run_id = self._active_run_id
                self.logger.info("CANCELLING ACTIVE RUN %s: %s", run_id, reason)
                await self._active_token.cancel(reason=reason)
                if run_id in self._states:
                    self._states[run_id] = RunLifecycleState.CANCELLED

    async def pause_active_run(self) -> bool:
        """Pause the currently active run without discarding task context."""
        async with self._lock:
            if self._active_run_id and self._states.get(self._active_run_id) == RunLifecycleState.ACTIVE:
                self._states[self._active_run_id] = RunLifecycleState.PAUSED
                context = self.get_context(self._active_run_id)
                context.pause()
                self.logger.info("RUN PAUSED: %s", self._active_run_id)
                return True
            return False

    async def resume_active_run(self) -> Optional[str]:
        """Resume the paused active run, returning the milestone resumed from."""
        async with self._lock:
            if self._active_run_id and self._states.get(self._active_run_id) == RunLifecycleState.PAUSED:
                self._states[self._active_run_id] = RunLifecycleState.ACTIVE
                context = self.get_context(self._active_run_id)
                milestone = context.resume()
                self.logger.info("RUN RESUMED: %s (from milestone: %s)", self._active_run_id, milestone)
                return milestone
            return None

    def mark_completed(self, run_id: Optional[str] = None) -> None:
        """Mark a run as successfully completed."""
        target_id = run_id or self._active_run_id
        if target_id and target_id in self._states:
            self._states[target_id] = RunLifecycleState.COMPLETED
            self.logger.info("RUN COMPLETED: %s", target_id)

    def mark_failed(self, run_id: Optional[str] = None, error: Optional[str] = None) -> None:
        """Mark a run as failed."""
        target_id = run_id or self._active_run_id
        if target_id and target_id in self._states:
            self._states[target_id] = RunLifecycleState.FAILED
            self.logger.info("RUN FAILED: %s (Error: %s)", target_id, error)

    def _prune_history_locked(self) -> None:
        """Evict older run data when history limit is exceeded to prevent memory leaks."""
        while len(self._run_history) > self.max_run_history:
            old_run_id = self._run_history.pop(0)
            if old_run_id != self._active_run_id:
                self._tokens.pop(old_run_id, None)
                self._contexts.pop(old_run_id, None)
                self._states.pop(old_run_id, None)

    async def shutdown(self) -> None:
        """Cleanly cancel active run and release all references."""
        async with self._lock:
            if self._active_token and not self._active_token.is_cancelled:
                await self._active_token.cancel(reason="System shutting down")
            self._active_token = None
            self._active_run_id = None
            self._tokens.clear()
            self._contexts.clear()
            self._states.clear()
            self._run_history.clear()
            self.logger.info("RUN MANAGER SHUTDOWN COMPLETE")
