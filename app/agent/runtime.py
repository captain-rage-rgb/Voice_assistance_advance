"""
Core Agent Runtime orchestrating events, action queues, execution, and state.
"""

import asyncio
import logging
from typing import Optional
from app.agent.action import Action, ActionStatus
from app.agent.action_queue import ActionQueue
from app.agent.events import Event, EventBus, EventType
from app.agent.executor import ActionExecutor, MockExecutor
from app.state.state_manager import AgentLifecycleState, StateManager


class AgentRuntime:
    """
    Asynchronous Core Agent Runtime.
    Coordinates event dispatch, state transitions, action queueing,
    cooperative cancellation, and graceful shutdown.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        state_manager: Optional[StateManager] = None,
        action_queue: Optional[ActionQueue] = None,
        executor: Optional[ActionExecutor] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.state_manager = state_manager or StateManager()
        self.action_queue = action_queue or ActionQueue()
        self.executor = executor or MockExecutor()
        self.logger = logger or logging.getLogger("voice_agent.runtime")

        self._worker_task: Optional[asyncio.Task] = None
        self._current_executing_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Subscribe default internal handlers
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Register default runtime event handlers on the event bus."""
        self.event_bus.subscribe(EventType.INTENT_DETECTED, self._handle_intent_detected)
        self.event_bus.subscribe(EventType.USER_INTERRUPTED, self._handle_user_interrupted)

    async def start(self) -> None:
        """Start the agent runtime worker loop."""
        if self._worker_task and not self._worker_task.done():
            self.logger.warning("Runtime is already running.")
            return

        self.logger.info("AGENT STARTED: Starting runtime loop...")
        self.state_manager.set_lifecycle(AgentLifecycleState.RUNNING)
        self._stop_event.clear()

        # Launch the action queue processor as a background task within the current event loop
        self._worker_task = asyncio.create_task(self._process_queue())
        await self.event_bus.publish(Event(EventType.SYSTEM_STARTED))

    async def stop(self) -> None:
        """Gracefully shut down the agent runtime."""
        self.logger.info("SHUTDOWN REQUESTED: Stopping runtime...")
        self.state_manager.set_lifecycle(AgentLifecycleState.STOPPING)
        await self.event_bus.publish(Event(EventType.SYSTEM_STOPPING))

        # Signal stop event
        self._stop_event.set()

        # Cancel any active running execution task cooperatively
        if self._current_executing_task and not self._current_executing_task.done():
            self._current_executing_task.cancel()

        # Cancel all pending items remaining in queue
        cancelled_actions = await self.action_queue.clear_all()
        for action in cancelled_actions:
            self.logger.info("ACTION CANCELLED: Pending action %s cancelled during shutdown", action.action_id)
            await self.event_bus.publish(Event(EventType.ACTION_CANCELLED, {"action_id": action.action_id}))

        # Cancel worker loop if waiting
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        self.state_manager.set_current_action(None)
        self.state_manager.set_pending_count(0)
        self.state_manager.set_lifecycle(AgentLifecycleState.STOPPED)
        self.logger.info("AGENT STOPPED: Graceful shutdown complete.")
        await self.event_bus.publish(Event(EventType.SYSTEM_STOPPED))

    async def submit_action(self, action: Action) -> None:
        """Add an action directly to the queue and update state."""
        self.logger.info("ACTION QUEUED: %s (ID: %s)", action.action_type, action.action_id)
        await self.action_queue.enqueue(action)
        self.state_manager.set_pending_count(self.action_queue.pending_count)
        await self.event_bus.publish(
            Event(EventType.ACTION_QUEUED, {"action_id": action.action_id, "action_type": action.action_type})
        )

    async def interrupt(self, reason: str = "User requested cancellation") -> None:
        """
        Trigger an interruption. Cancels pending actions and cooperatively cancels
        the currently executing action.
        """
        self.logger.info("USER INTERRUPTED: %s", reason)
        self.state_manager.request_cancellation()
        await self.event_bus.publish(Event(EventType.USER_INTERRUPTED, {"reason": reason}))

    async def _handle_intent_detected(self, event: Event) -> None:
        """Convert an INTENT_DETECTED event into a queued Action."""
        intent_type = event.payload.get("intent") or event.payload.get("action_type") or "unknown_action"
        arguments = event.payload.get("arguments", {})
        priority = event.payload.get("priority", 0)

        action = Action(action_type=intent_type, arguments=arguments, priority=priority)
        self.logger.info("EVENT RECEIVED: Intent detected -> Created action %s", action.action_type)
        await self.event_bus.publish(Event(EventType.ACTION_CREATED, {"action_id": action.action_id}))
        await self.submit_action(action)

    async def _handle_user_interrupted(self, event: Event) -> None:
        """Handle USER_INTERRUPTED event by cancelling pending and active work."""
        # Cancel all pending actions in queue
        cancelled_actions = await self.action_queue.clear_all()
        for action in cancelled_actions:
            self.logger.info("ACTION CANCELLED: Pending action %s cancelled", action.action_id)
            await self.event_bus.publish(Event(EventType.ACTION_CANCELLED, {"action_id": action.action_id}))

        # Cancel currently executing task if active
        if self._current_executing_task and not self._current_executing_task.done():
            self._current_executing_task.cancel()

        self.state_manager.set_pending_count(0)
        self.state_manager.clear_cancellation()

    async def _process_queue(self) -> None:
        """Continuous worker loop processing actions sequentially from the queue."""
        while not self._stop_event.is_set():
            try:
                # Wait for next action with timeout so we periodically check stop_event
                try:
                    action = await asyncio.wait_for(self.action_queue.get_next(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if self._stop_event.is_set():
                    break

                # Execute action
                await self._execute_action(action)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("AGENT_ERROR: Unexpected error in queue worker: %s", str(e), exc_info=True)
                self.state_manager.record_error(str(e))
                await self.event_bus.publish(Event(EventType.AGENT_ERROR, {"error": str(e)}))

    async def _execute_action(self, action: Action) -> None:
        """Executes a single action with lifecycle events and error handling."""
        action.mark_running()
        self.state_manager.set_current_action(action)
        self.state_manager.set_pending_count(self.action_queue.pending_count)

        self.logger.info("ACTION STARTED: %s (ID: %s)", action.action_type, action.action_id)
        await self.event_bus.publish(Event(EventType.ACTION_STARTED, {"action_id": action.action_id}))

        # Track the active execution task so it can be cooperatively cancelled if interrupted
        self._current_executing_task = asyncio.create_task(self.executor.execute(action))

        try:
            result = await self._current_executing_task
            action.mark_completed(result)
            self.logger.info("ACTION COMPLETED: %s (Result: %s)", action.action_type, result)
            await self.event_bus.publish(
                Event(EventType.ACTION_COMPLETED, {"action_id": action.action_id, "result": result})
            )
        except asyncio.CancelledError:
            action.mark_cancelled()
            self.logger.info("ACTION CANCELLED: %s was cancelled during execution", action.action_type)
            await self.event_bus.publish(Event(EventType.ACTION_CANCELLED, {"action_id": action.action_id}))
        except Exception as exc:
            error_msg = str(exc)
            action.mark_failed(error_msg)
            self.logger.error("ACTION FAILED: %s failed: %s", action.action_type, error_msg)
            self.state_manager.record_error(error_msg)
            await self.event_bus.publish(
                Event(EventType.ACTION_FAILED, {"action_id": action.action_id, "error": error_msg})
            )
        finally:
            self._current_executing_task = None
            self.state_manager.set_current_action(None)
            self.state_manager.set_pending_count(self.action_queue.pending_count)
