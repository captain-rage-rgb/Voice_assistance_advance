"""
Phase 9 & 10 Acceptance Tests: Long-Running Tasks, Conversational Continuity,
Unified Run Management, Timeouts, Subsystem Hardening, and Resource Stability.
"""

import asyncio
import pytest
from app.agent.action import Action, ActionStatus
from app.agent.cancellation import CancellationToken
from app.agent.context import ExecutionOutcome, TaskContext, TaskMilestone
from app.agent.events import Event, EventBus, EventType
from app.agent.executor import MockExecutor
from app.agent.health import HealthMonitor, SubsystemStatus
from app.agent.overlay import FloatingOverlay, OverlayDisplayState
from app.agent.run_manager import RunLifecycleState, RunManager
from app.agent.runtime import AgentRuntime
from app.config.settings import settings


def test_acceptance_test_a_long_task():
    """
    Test A — Long task:
    Execute a simulated long multi-step task with milestones.
    Expected:
    * context history remains bounded
    * milestones update correctly
    * no resource leak occurs
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.005)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            token = await runtime.get_or_create_run_token()
            run_id = token.run_id

            milestone_names = [
                "document located",
                "document opened",
                "modification completed",
                "file saved",
                "result verified",
            ]

            # Submit 25 sequential actions across 5 milestones
            for i in range(25):
                action = Action(
                    action_type="type_text",
                    arguments={"text": f"step {i}"},
                    run_id=run_id,
                )
                await runtime.submit_action(action)
                if i % 5 == 0:
                    milestone_idx = i // 5
                    await runtime.record_milestone(name=milestone_names[milestone_idx])

            # Wait for execution to drain
            for _ in range(20):
                if runtime.action_queue.pending_count == 0 and not runtime.state_manager.current.is_execution_active:
                    break
                await asyncio.sleep(0.05)

            context = runtime.get_or_create_task_context(run_id)

            # Context must remain strictly bounded
            assert len(context.completed_actions) <= context.max_history_entries
            assert len(context.milestones) <= context.max_history_entries

            # Verify milestone recording
            recorded_names = [m.name for m in context.milestones]
            assert "document located" in recorded_names
            assert "result verified" in recorded_names

            # All actions completed, queue is empty
            assert runtime.action_queue.pending_count == 0
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_b_follow_up_command():
    """
    Test B — Follow-up command:
    Execute a task and then issue a context-dependent follow-up instruction.
    Expected:
    * reference resolution works
    * resulting action passes through validation and execution
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Command 1: Open browser and search
            a1 = Action("search", arguments={"query": "flights", "application": "browser"})
            await runtime.submit_action(a1)
            await asyncio.sleep(0.05)
            assert a1.status == ActionStatus.COMPLETED

            # Follow-up Command 2: Reference "do that again"
            await runtime.event_bus.publish(
                Event(
                    EventType.INTENT_DETECTED,
                    {"intent": "repeat", "arguments": {"reference": "do that again"}},
                )
            )
            await asyncio.sleep(0.05)

            # Follow-up Command 3: Reference "the browser"
            await runtime.event_bus.publish(
                Event(
                    EventType.INTENT_DETECTED,
                    {"intent": "open_application", "arguments": {"target": "the browser"}},
                )
            )
            await asyncio.sleep(0.05)

            # Executed actions should include the resolved search and open_application
            executed_types = [a.action_type for a in executor.executed_actions]
            assert "search" in executed_types
            assert "open_application" in executed_types

            # Ensure arguments were resolved to browser
            open_actions = [a for a in executor.executed_actions if a.action_type == "open_application"]
            assert len(open_actions) >= 1
            assert open_actions[0].arguments.get("application") == "browser"
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_c_interrupt_and_replace():
    """
    Test C — Interrupt and replace:
    Start task A, interrupt it, then immediately start task B.
    Expected:
    * task A cannot produce actions after replacement
    * task B runs normally
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.05)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            token_a = await runtime.get_or_create_run_token()
            run_a_id = token_a.run_id

            # Enqueue multiple actions for Task A
            a1 = Action("open_application", arguments={"application": "browser"}, run_id=run_a_id)
            a2 = Action("search", arguments={"query": "task A query"}, run_id=run_a_id)
            await runtime.submit_action(a1)
            await runtime.submit_action(a2)

            await asyncio.sleep(0.02)

            # Start Task B, superseding Task A
            token_b = await runtime.start_new_command("search", {"query": "task B query"})
            run_b_id = token_b.run_id

            assert runtime.is_run_cancelled(run_a_id)
            assert not runtime.is_run_cancelled(run_b_id)

            await asyncio.sleep(0.12)

            # Task A actions must not execute further
            assert a2.status == ActionStatus.CANCELLED

            # Task B executed
            executed_queries = [a.arguments.get("query") for a in executor.executed_actions]
            assert "task B query" in executed_queries
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_d_tts_barge_in():
    """
    Test D — TTS barge-in:
    Simulate agent progress/TTS speaking while a new user command arrives.
    Expected:
    * speech status yields to new command
    * new command is not lost
    * no stale callback affects new task
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        overlay = FloatingOverlay(runtime=runtime)
        await runtime.start()

        try:
            # Simulate agent progress update / TTS narration
            await runtime.report_progress("Searching for files across directories...")
            assert overlay.current_action_label == "Searching for files across directories..."

            # User barges in with new command
            token = await runtime.start_new_command("open_application", {"application": "notepad"})
            await asyncio.sleep(0.05)

            # New command executed cleanly
            assert any(a.arguments.get("application") == "notepad" for a in executor.executed_actions)
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_e_shutdown_during_execution():
    """
    Test E — Shutdown during execution:
    Close application while a task is actively executing.
    Expected:
    * active task is cancelled
    * executor/resources released cleanly
    * application exits cleanly without hanging
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.2)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        a1 = Action("open_application", arguments={"application": "browser"})
        await runtime.submit_action(a1)

        # Allow execution to begin
        await asyncio.sleep(0.03)
        assert a1.status == ActionStatus.RUNNING

        # Trigger graceful shutdown while task is in flight
        await runtime.stop()

        assert a1.status == ActionStatus.CANCELLED
        assert runtime.state_manager.current.lifecycle_state.value == "STOPPED"
        assert runtime.active_run_id is None

    asyncio.run(_runner())


def test_acceptance_test_f_subsystem_failure():
    """
    Test F — Subsystem failure:
    Inject failure into executor.
    Expected:
    * no uncontrolled crash
    * error recorded in HealthMonitor and StateManager
    * agent remains alive and responsive
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Action with hardware/execution failure
            failing_action = Action(
                "open_application",
                arguments={"application": "calc", "simulate_failure": True, "failure_reason": "Device IO failure"},
            )
            await runtime.submit_action(failing_action)
            await asyncio.sleep(0.08)

            assert failing_action.status == ActionStatus.FAILED

            # Health monitor recorded failure
            assert runtime.health_monitor.get_status("executor") in (SubsystemStatus.ERROR, SubsystemStatus.DEGRADED)

            # Runtime must still be healthy and running
            assert runtime.state_manager.current.lifecycle_state.value == "RUNNING"

            # Subsequent normal action succeeds
            normal_action = Action("search", arguments={"query": "test"})
            await runtime.submit_action(normal_action)
            await asyncio.sleep(0.05)
            assert normal_action.status == ActionStatus.COMPLETED
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_g_stale_callback():
    """
    Test G — Stale callback:
    Deliver delayed action/callback from an old run.
    Expected:
    * stale callback is safely rejected
    * current run remains unaffected
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Run 1
            token_1 = await runtime.get_or_create_run_token()
            run_1_id = token_1.run_id

            # Start Run 2
            token_2 = await runtime.start_new_command("search", {"query": "run 2 query"})
            run_2_id = token_2.run_id

            # Deliver stale intent with Run 1 ID
            await runtime.event_bus.publish(
                Event(EventType.INTENT_DETECTED, {"intent": "click", "arguments": {}, "run_id": run_1_id})
            )
            await asyncio.sleep(0.05)

            # Stale run action should not have executed
            executed_run_ids = [a.run_id for a in executor.executed_actions]
            assert run_1_id not in executed_run_ids
            assert run_2_id in executed_run_ids
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_h_timeout():
    """
    Test H — Timeout:
    Force an action timeout using a custom slow executor.
    Expected:
    * timeout recorded
    * recovery/failure outcome recorded
    * system remains responsive
    """

    async def _runner():
        # Executor that delays longer than action timeout
        executor = MockExecutor(default_delay=0.15)
        runtime = AgentRuntime(executor=executor)

        # Set a very low timeout for the test
        original_timeout = settings.action_timeout_seconds
        settings.action_timeout_seconds = 0.05

        await runtime.start()

        try:
            action = Action("open_application", arguments={"application": "slow_app"})
            await runtime.submit_action(action)

            # Wait for timeout and recovery to finish
            for _ in range(15):
                if action.status == ActionStatus.FAILED:
                    break
                await asyncio.sleep(0.05)

            # Action should be marked TIMED_OUT or FAILED
            assert action.status == ActionStatus.FAILED
            assert "timed out" in (action.error_message or "").lower()

            # Context should reflect outcome
            context = runtime.get_or_create_task_context(action.run_id)
            assert len(context.completed_actions) >= 1
            assert context.completed_actions[-1].outcome in (ExecutionOutcome.TIMED_OUT, ExecutionOutcome.FAILED)
        finally:
            settings.action_timeout_seconds = original_timeout
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_i_memory_resource_stability():
    """
    Test I — Memory/resource stability:
    Run repeated tasks and cancellations.
    Expected:
    * no continuously growing queues
    * no accumulating background tasks
    * context history remains bounded
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.005)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Repeat 20 cycles of creation, execution, and cancellation
            for i in range(20):
                if i % 2 == 0:
                    token = await runtime.get_or_create_run_token()
                    a = Action("search", arguments={"query": f"query {i}"}, run_id=token.run_id)
                    await runtime.submit_action(a)
                    await asyncio.sleep(0.01)
                else:
                    await runtime.interrupt("Rapid interrupt test")

            await asyncio.sleep(0.05)

            # Queues should be completely empty
            assert runtime.action_queue.pending_count == 0

            # Run manager history must remain bounded
            assert len(runtime.run_manager._run_history) <= runtime.run_manager.max_run_history
            assert len(runtime.run_manager._tokens) <= runtime.run_manager.max_run_history
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_pause_and_resume_preserves_task_context():
    """Verify that pausing and resuming preserves task milestones and context."""

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            token = await runtime.get_or_create_run_token()
            run_id = token.run_id

            # Record milestone 1
            await runtime.record_milestone(name="setup complete")

            # Pause task
            await runtime.pause()
            context = runtime.get_or_create_task_context(run_id)
            assert context.is_paused is True

            # Action submitted while paused should wait in queue
            a = Action("search", arguments={"query": "paused query"}, run_id=run_id)
            await runtime.submit_action(a)
            await asyncio.sleep(0.03)

            # Not executed while paused
            assert a.status == ActionStatus.PENDING

            # Resume task
            await runtime.resume()
            assert context.is_paused is False
            assert context.resumed_from_milestone == "setup complete"

            # Wait for execution and verification to complete after resume
            for _ in range(10):
                if a.status == ActionStatus.COMPLETED:
                    break
                await asyncio.sleep(0.03)

            assert a.status == ActionStatus.COMPLETED
        finally:
            await runtime.stop()

    asyncio.run(_runner())
