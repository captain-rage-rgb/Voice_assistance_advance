"""
Tests for AgentRuntime: lifecycle, action processing, cancellation, and error handling.
"""

import asyncio
from app.agent.action import Action, ActionStatus
from app.agent.events import Event, EventType
from app.agent.executor import MockExecutor
from app.agent.runtime import AgentRuntime
from app.state.state_manager import AgentLifecycleState


def test_runtime_start_and_shutdown():
    """Verify runtime starts and shuts down cleanly."""

    async def _runner():
        runtime = AgentRuntime()
        assert runtime.state_manager.current.lifecycle_state == AgentLifecycleState.IDLE

        await runtime.start()
        assert runtime.state_manager.current.lifecycle_state == AgentLifecycleState.RUNNING

        await runtime.stop()
        assert runtime.state_manager.current.lifecycle_state == AgentLifecycleState.STOPPED

    asyncio.run(_runner())


def test_runtime_intent_event_to_action_execution():
    """Verify INTENT_DETECTED event causes action creation, queueing, and execution."""

    async def _runner():
        executor = MockExecutor(default_delay=0.02)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        completed_events = []

        async def on_completed(event: Event):
            completed_events.append(event)

        runtime.event_bus.subscribe(EventType.ACTION_COMPLETED, on_completed)

        # Publish simulated intent
        await runtime.event_bus.publish(
            Event(
                EventType.INTENT_DETECTED,
                payload={"intent": "open_app", "arguments": {"app": "brave"}},
            )
        )

        # Allow execution loop to process
        await asyncio.sleep(0.08)

        assert len(completed_events) == 1
        assert len(executor.executed_actions) == 1
        assert executor.executed_actions[0].action_type == "open_app"

        await runtime.stop()

    asyncio.run(_runner())


def test_runtime_action_failure_resilience():
    """Verify that a failing action does not crash the runtime or prevent further work."""

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        # Submit failing action
        failing_action = Action(
            "failing_task",
            arguments={"simulate_failure": True, "failure_reason": "Simulated hardware error"},
        )
        # Followed by normal action
        normal_action = Action("normal_task", arguments={"data": 123})

        await runtime.submit_action(failing_action)
        await runtime.submit_action(normal_action)

        # Wait for processing
        await asyncio.sleep(0.1)

        assert failing_action.status == ActionStatus.FAILED
        assert failing_action.error_message == "Simulated hardware error"
        assert normal_action.status == ActionStatus.COMPLETED

        # Runtime must still be running smoothly
        assert runtime.state_manager.current.lifecycle_state == AgentLifecycleState.RUNNING

        await runtime.stop()

    asyncio.run(_runner())


def test_runtime_interruption_cancels_pending_and_active():
    """Verify that interrupt cancels pending actions and actively running tasks."""

    async def _runner():
        executor = MockExecutor(default_delay=0.2)  # Long delay to catch in-flight
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        action1 = Action("long_task_1")
        action2 = Action("pending_task_2")

        await runtime.submit_action(action1)
        await runtime.submit_action(action2)

        # Wait briefly so action1 starts executing
        await asyncio.sleep(0.05)

        # Trigger interrupt
        await runtime.interrupt("User shouted stop")

        await asyncio.sleep(0.1)

        # action1 was in-flight and cancelled cooperatively
        assert action1.status == ActionStatus.CANCELLED
        # action2 was in queue and cancelled
        assert action2.status == ActionStatus.CANCELLED
        # Queue should now be empty
        assert runtime.action_queue.pending_count == 0

        await runtime.stop()

    asyncio.run(_runner())
