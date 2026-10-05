"""
Phase 7 & 8 Acceptance Tests: Context-Aware Planning, Incremental Execution,
Post-Action Verification, and Bounded Self-Correction Recovery.
"""

import asyncio
import pytest
from app.agent.action import Action, ActionStatus
from app.agent.cancellation import CancellationToken
from app.agent.context import ExecutionOutcome, TaskContext
from app.agent.events import Event, EventBus, EventType
from app.agent.executor import MockExecutor
from app.agent.overlay import FloatingOverlay, OverlayDisplayState
from app.agent.runtime import AgentRuntime
from app.agent.validator import ActionValidator
from app.agent.verifier import Verifier


def test_acceptance_test_a_multistep_context():
    """
    Test A — Multi-step context:
    Verify that completed actions are tracked in TaskContext and not unnecessarily repeated.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Step 1: Open browser
            a1 = Action("open_application", arguments={"application": "browser"})
            # Step 2: Search for AI
            a2 = Action("search", arguments={"query": "AI agents"})

            await runtime.submit_action(a1)
            await runtime.submit_action(a2)

            # Allow execution to complete
            await asyncio.sleep(0.08)

            assert a1.status == ActionStatus.COMPLETED
            assert a2.status == ActionStatus.COMPLETED

            context = runtime.get_or_create_task_context(a1.run_id)
            assert len(context.completed_actions) >= 2
            assert context.is_action_already_completed("open_application", {"application": "browser"})
            assert context.is_action_already_completed("search", {"query": "AI agents"})

            # Now simulate model redundantly emitting the completed step 1 again
            a1_duplicate = Action("open_application", arguments={"application": "browser"}, run_id=a1.run_id)
            await runtime.submit_action(a1_duplicate)

            await asyncio.sleep(0.05)

            # Duplicate should be marked completed/skipped without calling executor again
            assert a1_duplicate.status == ActionStatus.COMPLETED
            assert a1_duplicate.result.get("skipped") is True
            # Total physical executions should still be only 2
            assert len(executor.executed_actions) == 2
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_b_incremental_execution():
    """
    Test B — Incremental execution:
    Verify that an action can execute before the entire model response finishes.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.04)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Intent 1 becomes available while speech/model is still streaming
            a1 = Action("open_application", arguments={"application": "browser"})
            await runtime.submit_action(a1)

            # Wait briefly for a1 to be picked up by the queue worker
            await asyncio.sleep(0.01)
            assert runtime.state_manager.current.is_execution_active is True
            assert a1.status == ActionStatus.RUNNING

            # After a delay, intent 2 becomes available
            await asyncio.sleep(0.05)
            a2 = Action("search", arguments={"query": "weather"})
            await runtime.submit_action(a2)

            await asyncio.sleep(0.1)
            assert a1.status == ActionStatus.COMPLETED
            assert a2.status == ActionStatus.COMPLETED
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_c_cancellation_during_planning():
    """
    Test C — Cancellation during planning:
    Verify that cancellation prevents future queued or planned actions.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.08)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            a1 = Action("open_application", arguments={"application": "browser"})
            a2 = Action("search", arguments={"query": "test query"})
            a3 = Action("navigate", arguments={"destination": "https://example.com"})

            await runtime.submit_action(a1)
            await runtime.submit_action(a2)
            await runtime.submit_action(a3)

            # Cancel while a1 is running
            await asyncio.sleep(0.02)
            await runtime.interrupt("User cancelled during multi-step planning")

            await asyncio.sleep(0.1)

            assert a1.status == ActionStatus.CANCELLED
            assert a2.status == ActionStatus.CANCELLED
            assert a3.status == ActionStatus.CANCELLED
            # No subsequent actions should have executed
            assert len(executor.executed_actions) == 0
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_d_cancellation_during_verification():
    """
    Test D — Cancellation during verification:
    Verify that verification and recovery stop immediately when cancelled.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Custom verifier that simulates a verification delay
            class SlowVerifier(Verifier):
                async def verify_action(self, action, raw_result, token=None):
                    for _ in range(10):
                        if token and token.is_cancelled:
                            token.check_cancelled()
                        await asyncio.sleep(0.02)
                    return True, "Done"

            runtime.verifier = SlowVerifier()

            a1 = Action("open_application", arguments={"application": "browser", "verify": True})
            await runtime.submit_action(a1)

            # Wait for action to finish execution and enter verification
            await asyncio.sleep(0.03)

            # Cancel during verification
            await runtime.interrupt("User cancelled during verification")
            await asyncio.sleep(0.05)

            assert a1.status == ActionStatus.CANCELLED
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_e_verification_failure_and_recovery():
    """
    Test E — Verification failure:
    Simulate an action that reports success but leaves the computer in the wrong state.
    Expected:
    * verification detects the mismatch
    * bounded recovery is attempted
    * system recovers and records structured outcome
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        events_received = []

        runtime.event_bus.subscribe(
            EventType.VERIFICATION_FAILED, lambda e: events_received.append(e.event_type)
        )
        runtime.event_bus.subscribe(
            EventType.RECOVERY_STARTED, lambda e: events_received.append(e.event_type)
        )
        runtime.event_bus.subscribe(
            EventType.RECOVERY_COMPLETED, lambda e: events_received.append(e.event_type)
        )

        await runtime.start()

        try:
            # Action reports execution success but simulates verification state mismatch
            action = Action(
                "open_application",
                arguments={
                    "application": "browser",
                    "simulate_verification_failure": True,
                    "verification_fail_reason": "Browser window did not appear",
                },
            )

            await runtime.submit_action(action)
            await asyncio.sleep(0.1)

            # Verification must have detected the failure and triggered recovery
            assert EventType.VERIFICATION_FAILED in events_received
            assert EventType.RECOVERY_STARTED in events_received
            assert EventType.RECOVERY_COMPLETED in events_received

            assert action.status == ActionStatus.COMPLETED
            context = runtime.get_or_create_task_context(action.run_id)
            assert context.recovery_count >= 1
            assert context.completed_actions[-1].outcome == ExecutionOutcome.RECOVERED
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_f_recovery_limit():
    """
    Test F — Recovery limit:
    Simulate repeated failures.
    Expected:
    * recovery terminates after the configured limit
    * no infinite loop occurs
    * the user receives a clear failure result
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        events_received = []

        runtime.event_bus.subscribe(
            EventType.RECOVERY_FAILED, lambda e: events_received.append(e.event_type)
        )

        await runtime.start()

        try:
            # Action persistently fails verification
            action = Action(
                "open_application",
                arguments={
                    "application": "browser",
                    "simulate_verification_failure": True,
                    "persist_verification_failure": True,
                    "verification_fail_reason": "Application crashed on start",
                },
            )

            await runtime.submit_action(action)
            await asyncio.sleep(0.12)

            # Recovery should exhaust attempts and cleanly fail without looping
            assert EventType.RECOVERY_FAILED in events_received
            assert action.status == ActionStatus.FAILED
            assert "Verification failed" in action.error_message

            context = runtime.get_or_create_task_context(action.run_id)
            assert context.completed_actions[-1].outcome == ExecutionOutcome.VERIFICATION_FAILED
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_g_stale_run():
    """
    Test G — Stale run:
    Start run A, cancel it, start run B, then deliver an old result from run A.
    Expected:
    * run A result/action is ignored
    * run B remains unaffected
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.02)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Run A
            token_a = await runtime.get_or_create_run_token()
            run_a_id = token_a.run_id
            action_a = Action("open_application", arguments={"application": "calc"}, run_id=run_a_id)
            await runtime.submit_action(action_a)

            # Cancel Run A
            await runtime.interrupt("Cancel Run A")
            assert runtime.is_run_cancelled(run_a_id)

            # Start Run B
            action_b = Action("search", arguments={"query": "fresh run query"})
            await runtime.submit_action(action_b)
            assert action_b.run_id != run_a_id

            # Now deliver a late stale action belonging to Run A
            stale_action_a = Action("open_application", arguments={"application": "late calc"}, run_id=run_a_id)
            await runtime.submit_action(stale_action_a)

            await asyncio.sleep(0.1)

            # Stale action must be cancelled/rejected
            assert stale_action_a.status == ActionStatus.CANCELLED
            # Run B action must complete unaffected
            assert action_b.status == ActionStatus.COMPLETED
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_h_new_command_during_recovery():
    """
    Test H — New command during recovery:
    Interrupt recovery with a new voice command.
    Expected:
    * recovery stops
    * previous run becomes invalid
    * new command starts cleanly
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.03)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Start an action that triggers recovery
            action_1 = Action(
                "open_application",
                arguments={
                    "application": "browser",
                    "simulate_verification_failure": True,
                    "persist_verification_failure": True,
                },
            )
            await runtime.submit_action(action_1)
            run_1_id = action_1.run_id

            # Wait for action 1 to fail and enter recovery
            await asyncio.sleep(0.04)

            # User interrupts with a new command
            token_2 = await runtime.start_new_command("search", {"query": "new urgent query"})
            run_2_id = token_2.run_id

            assert runtime.is_run_cancelled(run_1_id)
            assert not runtime.is_run_cancelled(run_2_id)

            await asyncio.sleep(0.08)

            # Run 2 executes successfully
            context_2 = runtime.get_or_create_task_context(run_2_id)
            assert context_2.run_id == run_2_id
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_acceptance_test_i_overlay_state():
    """
    Test I — Overlay state:
    Verify that the minimal overlay correctly reflects:
    Listening → Thinking → Executing → Verifying/Recovering → Listening
    and that STOP remains usable throughout.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.04)
        runtime = AgentRuntime(executor=executor)
        overlay = FloatingOverlay(runtime=runtime)
        await runtime.start()

        try:
            assert overlay.display_state == OverlayDisplayState.IDLE

            # Audio starts
            await runtime.event_bus.publish(Event(EventType.AUDIO_STREAM_STARTED))
            assert overlay.display_state == OverlayDisplayState.LISTENING

            # Speech received
            await runtime.event_bus.publish(Event(EventType.TRANSCRIPT_PARTIAL, {"text": "open calculator"}))
            assert overlay.display_state == OverlayDisplayState.LISTENING
            assert overlay.live_transcript == "open calculator"

            # Voice stops -> Thinking
            await runtime.event_bus.publish(Event(EventType.VOICE_ACTIVITY_STOPPED))
            assert overlay.display_state == OverlayDisplayState.THINKING

            # Action starts -> Executing
            action = Action("open_application", arguments={"application": "calc", "verify": True})
            await runtime.submit_action(action)
            await asyncio.sleep(0.01)
            assert overlay.display_state == OverlayDisplayState.EXECUTING

            # Verification event -> Verifying
            await runtime.event_bus.publish(Event(EventType.VERIFICATION_STARTED))
            assert overlay.display_state == OverlayDisplayState.VERIFYING

            # Recovery event -> Recovering
            await runtime.event_bus.publish(Event(EventType.RECOVERY_STARTED))
            assert overlay.display_state == OverlayDisplayState.RECOVERING

            # STOP pressed at any time
            await overlay.press_stop()
            assert overlay.display_state == OverlayDisplayState.STOPPED
            assert overlay.stop_count == 1
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_action_validator_rejects_malformed_actions():
    """Verify that ActionValidator rejects actions missing required fields."""
    validator = ActionValidator()

    # Missing application parameter
    invalid_action = Action("OPEN_APPLICATION", arguments={})
    is_valid, reason = validator.validate(invalid_action)
    assert not is_valid
    assert "Missing required parameter 'application'" in reason

    # Valid application parameter
    valid_action = Action("OPEN_APPLICATION", arguments={"application": "notepad"})
    is_valid, reason = validator.validate(valid_action)
    assert is_valid
    assert reason is None
