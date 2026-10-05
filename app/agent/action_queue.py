"""
Asynchronous, cancellable action queue supporting inspection, priority, and clearing.
"""

import asyncio
from typing import List, Optional
from app.agent.action import Action, ActionStatus


class ActionQueue:
    """
    Asynchronous action queue designed for event-driven agent workflows.
    Allows enqueuing, priority sorting, inspection, and clearing of pending actions.
    """

    def __init__(self) -> None:
        self._pending: List[Action] = []
        self._condition = asyncio.Condition()

    async def enqueue(self, action: Action) -> None:
        """Add an action to the queue and re-sort by priority."""
        async with self._condition:
            self._pending.append(action)
            # Higher priority comes first; stable sort preserves FIFO arrival for equal priorities
            self._pending.sort(key=lambda a: a.priority, reverse=True)
            self._condition.notify()

    async def get_next(self) -> Action:
        """Retrieve the next highest-priority action, waiting if the queue is empty."""
        async with self._condition:
            while not self._pending:
                await self._condition.wait()
            return self._pending.pop(0)

    def get_pending_actions(self) -> List[Action]:
        """Inspect current pending actions without removing them."""
        return list(self._pending)

    @property
    def pending_count(self) -> int:
        """Return the number of currently pending actions."""
        return len(self._pending)

    async def cancel_action(self, action_id: str) -> Optional[Action]:
        """Cancel a specific pending action if found in the queue."""
        async with self._condition:
            for action in self._pending:
                if action.action_id == action_id and action.status == ActionStatus.PENDING:
                    action.mark_cancelled()
                    self._pending.remove(action)
                    return action
        return None

    async def clear_run(self, run_id: str) -> List[Action]:
        """
        Discard and cancel all pending actions belonging to a specific run_id.
        Prevents stale actions from old runs from executing.
        """
        async with self._condition:
            cancelled = []
            remaining = []
            for action in self._pending:
                if action.run_id == run_id and action.status == ActionStatus.PENDING:
                    action.mark_cancelled()
                    cancelled.append(action)
                else:
                    remaining.append(action)
            self._pending = remaining
            return cancelled

    async def clear_all(self) -> List[Action]:
        """
        Clear and cancel all pending actions.
        Returns the list of actions that were cancelled.
        """
        async with self._condition:
            cancelled = []
            for action in self._pending:
                if action.status == ActionStatus.PENDING:
                    action.mark_cancelled()
                    cancelled.append(action)
            self._pending.clear()
            return cancelled

