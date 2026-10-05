"""
Tests for MockExecutor.
"""

import asyncio
import pytest
from app.agent.action import Action
from app.agent.executor import MockExecutor


def test_mock_executor_success():
    """Verify standard execution and simulation delay."""

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        action = Action("open_app", arguments={"application": "brave"})
        result = await executor.execute(action)

        assert "SIMULATED" in result
        assert "brave" in result
        assert len(executor.executed_actions) == 1

    asyncio.run(_runner())


def test_mock_executor_simulated_failure():
    """Verify that failure flag raises an exception as expected."""

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        action = Action(
            "failing_task",
            arguments={"simulate_failure": True, "failure_reason": "Device unavailable"},
        )
        with pytest.raises(RuntimeError, match="Device unavailable"):
            await executor.execute(action)

    asyncio.run(_runner())
