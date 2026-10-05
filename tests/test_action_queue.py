"""
Tests for Action model and ActionQueue.
"""

import asyncio
from app.agent.action import Action, ActionStatus
from app.agent.action_queue import ActionQueue


def test_action_status_transitions():
    """Verify action lifecycle state mutations."""
    action = Action(action_type="open_app", arguments={"name": "notepad"})
    assert action.status == ActionStatus.PENDING

    action.mark_running()
    assert action.status == ActionStatus.RUNNING

    action.mark_completed(result="done")
    assert action.status == ActionStatus.COMPLETED
    assert action.result == "done"

    action.mark_failed("Test error")
    assert action.status == ActionStatus.FAILED
    assert action.error_message == "Test error"

    action.mark_cancelled()
    assert action.status == ActionStatus.CANCELLED


def test_action_queue_enqueue_dequeue():
    """Verify basic enqueue, dequeue, and count operations."""

    async def _runner():
        queue = ActionQueue()
        a1 = Action("task1")
        a2 = Action("task2")

        await queue.enqueue(a1)
        await queue.enqueue(a2)
        assert queue.pending_count == 2

        next_action = await queue.get_next()
        assert next_action.action_type == "task1"
        assert queue.pending_count == 1

        next_action2 = await queue.get_next()
        assert next_action2.action_type == "task2"
        assert queue.pending_count == 0

    asyncio.run(_runner())


def test_action_queue_priority_ordering():
    """Verify that higher priority items are returned first."""

    async def _runner():
        queue = ActionQueue()
        low = Action("low_priority", priority=1)
        high = Action("high_priority", priority=10)

        await queue.enqueue(low)
        await queue.enqueue(high)

        # Inspect pending list
        pending = queue.get_pending_actions()
        assert pending[0].action_type == "high_priority"

        first = await queue.get_next()
        assert first.action_type == "high_priority"

    asyncio.run(_runner())


def test_action_queue_cancellation_and_clear():
    """Verify individual cancellation and bulk clearing of pending items."""

    async def _runner():
        queue = ActionQueue()
        a1 = Action("task1")
        a2 = Action("task2")

        await queue.enqueue(a1)
        await queue.enqueue(a2)

        # Cancel specific action
        cancelled = await queue.cancel_action(a1.action_id)
        assert cancelled is not None
        assert cancelled.status == ActionStatus.CANCELLED
        assert queue.pending_count == 1

        # Clear remaining
        cleared = await queue.clear_all()
        assert len(cleared) == 1
        assert cleared[0].action_id == a2.action_id
        assert cleared[0].status == ActionStatus.CANCELLED
        assert queue.pending_count == 0

    asyncio.run(_runner())
