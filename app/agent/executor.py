"""
Executor interface and mock implementation for Phase 2.
Decouples action execution logic from the core runtime.
"""

import asyncio
from typing import Any, Protocol
from app.agent.action import Action


class ActionExecutor(Protocol):
    """Protocol defining the executor interface."""

    async def execute(self, action: Action) -> Any:
        """
        Execute an action asynchronously.

        Args:
            action: The Action instance to execute.

        Returns:
            The result of the action execution.

        Raises:
            Exception: If action execution fails.
            asyncio.CancelledError: If action was interrupted/cancelled during execution.
        """
        ...


class MockExecutor:
    """
    Mock executor implementation for testing and development.
    Simulates execution delays, successes, and intentional failures
    without performing real system automation.
    """

    def __init__(self, default_delay: float = 0.05) -> None:
        self.default_delay = default_delay
        self.executed_actions: list[Action] = []

    async def execute(self, action: Action) -> str:
        """
        Simulate action execution. Supports cooperative cancellation via asyncio.sleep.
        """
        # Cooperative cancellation check: asyncio.sleep raises CancelledError if task cancelled
        await asyncio.sleep(self.default_delay)

        # Allow tests to trigger simulated failures via special argument
        if action.arguments.get("simulate_failure"):
            error_reason = action.arguments.get("failure_reason", "Simulated failure error")
            raise RuntimeError(error_reason)

        self.executed_actions.append(action)
        return f"SIMULATED: executed '{action.action_type}' with args {action.arguments}"
