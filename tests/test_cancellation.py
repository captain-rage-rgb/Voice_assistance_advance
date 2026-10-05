"""
Focused acceptance and unit tests for Phase 6 Unified Interruption and Cancellation System.
Covers:
  - Test A: Stop during streaming (generation stops, no newly arriving actions execute)
  - Test B: Stop during execution (completed actions remain, current stops, subsequent do not start)
  - Test C: New command immediately after STOP (starts cleanly without state leakage)
  - Test D: Rapid repeated STOP presses (no crash, no duplicate error)
  - Test E: Stale async results rejected via run_id isolation
  - Test F: Speech / audio barge-in interruption semantics
"""

import asyncio
import pytest
from app.agent.action import Action, ActionStatus
from app.agent.cancellation import CancellationToken
from app.agent.events import Event, EventBus, EventType
from app.agent.executor import MockExecutor
from app.agent.overlay import FloatingOverlay, OverlayDisplayState
from app.agent.runtime import AgentRuntime
from app.state.state_manager import StateManager


def test_unified_cancellation_token_behavior():
    """Verify CancellationToken states, callbacks, and repeated calls."""

    async def _runner():
        token = CancellationToken(run_id="run-1")
        assert not token.is_cancelled

        callback_ran = []
        token.register_callback(lambda: callback_ran.append(True))

        # First cancel
        res1 = await token.cancel(reason="First cancel")
        assert res1 is True
        assert token.is_cancelled
        assert token.reason == "First cancel"
        assert len(callback_ran) == 1

        # Second cancel (should be safe no-op)
        res2 = await token.cancel(reason="Second cancel")
        assert res2 is False
        assert token.is_cancelled
        assert token.reason == "First cancel"

        with pytest.raises(asyncio.CancelledError):
            token.check_cancelled()

    asyncio.run(_runner())


def test_acceptance_test_a_stop_during_streaming():
    """
    Test A — Stop during streaming:
    Generation stops, no newly arriving actions execute, UI returns to idle/listening.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.1)
        runtime = AgentRuntime(executor=executor)
        overlay = FloatingOverlay(runtime=runtime)
        await runtime.start()

        # Simulate user speech that starts a run
        await runtime.event_bus.publish(
            Event(EventType.INTENT_DETECTED, {"intent": "step_1", "arguments": {}})
        )
        await asyncio.sleep(0.02)

        # User presses STOP on overlay
        await overlay.press_stop()
        assert overlay.display_state == OverlayDisplayState.STOPPED

        # Newly arriving intent from a cancelled stream arrives
        await runtime.event_bus.publish(
            Event(EventType.INTENT_DETECTED, {"intent": "step_2", "arguments": {}})
        )

        await asyncio.sleep(0.15)

        # Only step_1 could have run/cancelled, step_2 was either interrupted or rejected
        executed_types = [a.action_type for a in executor.executed_actions]
        assert "step_2" not in executed_types
        assert runtime.action_queue.pending_count == 0

        await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_b_stop_during_execution():
    """
    Test B — Stop during execution:
    In a sequence of 4 actions (1, 2, 3, 4), stop occurs after step 2:
    - Steps 1 and 2 completed
    - Steps 3 and 4 must not start
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.08)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        a1 = Action("step_1")
        a2 = Action("step_2")
        a3 = Action("step_3")
        a4 = Action("step_4")

        # Submit multi-action sequence
        await runtime.submit_action(a1)
        await runtime.submit_action(a2)
        await runtime.submit_action(a3)
        await runtime.submit_action(a4)

        # Wait for step 1 and step 2 to complete (~180ms)
        await asyncio.sleep(0.19)

        # Stop triggered
        await runtime.interrupt("User pressed STOP between step 2 and 3")

        await asyncio.sleep(0.15)

        assert a1.status == ActionStatus.COMPLETED
        assert a2.status == ActionStatus.COMPLETED
        assert a3.status == ActionStatus.CANCELLED
        assert a4.status == ActionStatus.CANCELLED
        assert [a.action_type for a in executor.executed_actions] == ["step_1", "step_2"]

        await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_c_new_command_after_stop():
    """
    Test C — New command after STOP:
    Press STOP and immediately start a new command.
    New command runs cleanly with a fresh run_id; no old output leaks.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.05)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        # Old command
        old_action = Action("old_task")
        await runtime.submit_action(old_action)
        old_run = runtime.active_run_id

        # Interrupt / STOP
        await runtime.interrupt("User cancelled old command")
        assert old_action.status == ActionStatus.CANCELLED

        # Immediately submit new command
        new_action = Action("new_task")
        await runtime.submit_action(new_action)
        new_run = runtime.active_run_id

        assert new_run != old_run

        await asyncio.sleep(0.1)

        assert new_action.status == ActionStatus.COMPLETED
        assert executor.executed_actions == [new_action]

        await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_d_rapid_repeated_stop():
    """
    Test D — Rapid repeated STOP:
    Press STOP repeatedly in rapid succession without crashes or duplicate errors.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.1)
        runtime = AgentRuntime(executor=executor)
        overlay = FloatingOverlay(runtime=runtime)
        await runtime.start()

        await runtime.submit_action(Action("busy_task"))

        # Rapid repeated STOP presses
        for _ in range(5):
            await overlay.press_stop()

        assert overlay.stop_count == 5
        assert overlay.display_state == OverlayDisplayState.STOPPED
        assert runtime.state_manager.current.lifecycle_state == runtime.state_manager.current.lifecycle_state.RUNNING

        await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_e_stale_async_result_rejection():
    """
    Test E — Stale async result rejection:
    An action carrying an old cancelled run_id is submitted or dequeued after a new run begins.
    It must be rejected/discarded and not executed.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.05)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        # Create run 1 and cancel it
        token1 = await runtime.get_or_create_run_token()
        run1_id = token1.run_id
        await runtime.interrupt("Cancel run 1")

        # Create run 2
        token2 = await runtime.get_or_create_run_token()
        run2_id = token2.run_id
        assert run1_id != run2_id

        # Stale action with run 1 arrives late
        stale_action = Action("stale_task", run_id=run1_id)
        await runtime.submit_action(stale_action)

        # Valid action for run 2 arrives
        valid_action = Action("valid_task", run_id=run2_id)
        await runtime.submit_action(valid_action)

        await asyncio.sleep(0.1)

        assert stale_action.status == ActionStatus.CANCELLED
        assert valid_action.status == ActionStatus.COMPLETED
        assert [a.action_type for a in executor.executed_actions] == ["valid_task"]

        await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_f_barge_in_interruption_event():
    """
    Test F — Barge-in / interruption via intent or speech:
    When an explicit INTERRUPT intent arrives during active work,
    the active action is cancelled and queue cleared immediately.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.2)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        long_task = Action("long_action")
        await runtime.submit_action(long_task)

        # Wait so it starts executing
        await asyncio.sleep(0.04)
        assert long_task.status == ActionStatus.RUNNING

        # Barge-in speech detected: INTERRUPT intent
        await runtime.event_bus.publish(
            Event(EventType.INTENT_DETECTED, {"intent": "INTERRUPT"})
        )

        await asyncio.sleep(0.15)

        assert long_task.status == ActionStatus.CANCELLED

        await runtime.stop()

    asyncio.run(_runner())
