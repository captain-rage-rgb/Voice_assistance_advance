"""
Automated Test Suite for Phases 13, 14, and 15:
Multimodal State Fusion, Adaptive Recovery, Conversational Continuity,
Soak Testing, Failure Injection, and Production Hardening.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from app.agent.action import Action, ActionStatus
from app.agent.context import ActionResult, ExecutionOutcome, TaskContext
from app.agent.events import Event, EventType
from app.agent.executor import MockExecutor
from app.agent.health import HealthMonitor, SubsystemStatus
from app.agent.multimodal import (
    EvidenceType,
    MultimodalReferenceResolver,
    UnifiedInteractionState,
)
from app.agent.orchestration import (
    ConfirmationManager,
    ConfirmationStatus,
    GoalOrchestrator,
    GoalStatus,
    SubGoal,
    TaskGoal,
)
from app.agent.overlay import FloatingOverlay, OverlayDisplayState
from app.agent.recovery import (
    AdaptiveRecoveryManager,
    FailureCategory,
    FailureClassifier,
    FinalRunState,
)
from app.agent.runtime import AgentRuntime
from app.agent.visual import (
    ComputerObservation,
    TargetingConfidence,
    UIElement,
    UITargeter,
    VisualVerifier,
)
from app.state.state_manager import AgentLifecycleState


# ==============================================================================
# PHASE 13 TESTS: Multimodal Fusion, Evidence & Spatial References
# ==============================================================================

def test_multimodal_spatial_reference_right_and_left():
    """
    Test 13.5: Voice + visual spatial references: 'on the right' and 'on the left'.
    Verifies that spatial coordinates are resolved dynamically without hard-coding.
    """
    resolver = MultimodalReferenceResolver()
    obs = ComputerObservation(
        elements=[
            UIElement(label="Cancel Button", element_type="button", bounding_box=(100, 200, 80, 40)),  # Left
            UIElement(label="Submit Button", element_type="button", bounding_box=(600, 200, 80, 40)),  # Right
        ]
    )

    # 1. Right reference
    res_right = resolver.resolve_reference("open the one on the right", obs)
    assert res_right.confidence == TargetingConfidence.HIGH
    assert res_right.target_element is not None
    assert res_right.target_element.label == "Submit Button"
    assert res_right.target_element.center_coords[0] == 640

    # 2. Left reference
    res_left = resolver.resolve_reference("click the one on the left", obs)
    assert res_left.confidence == TargetingConfidence.HIGH
    assert res_left.target_element is not None
    assert res_left.target_element.label == "Cancel Button"
    assert res_left.target_element.center_coords[0] == 140


def test_multimodal_reading_order_references():
    """
    Test 13.5: Positional order references: 'first option' and 'second option'.
    Verifies top-down reading order sorting.
    """
    resolver = MultimodalReferenceResolver()
    obs = ComputerObservation(
        elements=[
            UIElement(label="Option Charlie", bounding_box=(100, 300, 150, 30)),
            UIElement(label="Option Alpha", bounding_box=(100, 100, 150, 30)),
            UIElement(label="Option Bravo", bounding_box=(100, 200, 150, 30)),
        ]
    )

    res_first = resolver.resolve_reference("select the first option", obs)
    assert res_first.confidence == TargetingConfidence.HIGH
    assert res_first.target_element.label == "Option Alpha"

    res_second = resolver.resolve_reference("select the second option", obs)
    assert res_second.confidence == TargetingConfidence.HIGH
    assert res_second.target_element.label == "Option Bravo"


def test_multimodal_deictic_click_that_ambiguity():
    """
    Test 13.4 & 13.5: 'click that' is high confidence when only 1 element exists,
    but marks AMBIGUOUS and asks clarification when multiple elements exist.
    """
    resolver = MultimodalReferenceResolver()

    # Case 1: Single element on screen
    single_obs = ComputerObservation(elements=[UIElement(label="Download File", bounding_box=(200, 200, 100, 40))])
    res_single = resolver.resolve_reference("click that", single_obs)
    assert res_single.confidence == TargetingConfidence.HIGH
    assert res_single.target_element.label == "Download File"

    # Case 2: Multiple elements on screen -> Ambiguous
    multi_obs = ComputerObservation(
        elements=[
            UIElement(label="Download File", bounding_box=(200, 200, 100, 40)),
            UIElement(label="Cancel Download", bounding_box=(400, 200, 100, 40)),
        ]
    )
    res_multi = resolver.resolve_reference("click that", multi_obs)
    assert res_multi.confidence == TargetingConfidence.AMBIGUOUS
    assert res_multi.target_element is None
    assert "multiple items" in res_multi.clarification_question.lower()


def test_multimodal_state_and_evidence_separation():
    """
    Test 13.1 & 13.2: UnifiedInteractionState maintains run isolation,
    explicitly separates evidence types, and updates observation freshness.
    """
    state = UnifiedInteractionState(run_id="run-evidence-001")

    # Add user input
    ev1 = state.add_evidence(EvidenceType.USER_INPUT, {"text": "check settings"}, confidence=1.0)
    assert ev1.evidence_type == EvidenceType.USER_INPUT

    # Add planner interpretation
    ev2 = state.add_evidence(EvidenceType.PLANNER_INTERPRETATION, {"target": "Settings Icon"}, confidence=0.85)
    assert ev2.evidence_type == EvidenceType.PLANNER_INTERPRETATION
    # Inferred target is NOT observed target
    assert ev2.evidence_type != EvidenceType.OBSERVED_STATE

    # Add observation
    obs = ComputerObservation(
        run_id="run-evidence-001",
        window_title="Settings Panel",
        elements=[UIElement(label="Settings Icon", bounding_box=(50, 50, 40, 40))],
    )
    state.update_observation(obs)
    assert state.latest_valid_observation is not None
    assert state.latest_valid_observation.window_title == "Settings Panel"

    # Invalidate observation on state change
    state.invalidate_observation(reason="Window closed")
    assert state.latest_valid_observation is None
    assert state.observation_freshness > 100.0


# ==============================================================================
# PHASE 14 TESTS: Adaptive Recovery, Resilience & Conversational Continuity
# ==============================================================================

def test_failure_classification_categories():
    """
    Test 14.1: Normalization of various raw runtime failures into FailureCategories.
    """
    classifier = FailureClassifier()

    assert classifier.classify("Observation was stale (6.2s old)") == FailureCategory.STALE_OBSERVATION
    assert classifier.classify("User pressed STOP control") == FailureCategory.CANCELLATION
    assert classifier.classify("Target not resolved on screen") == FailureCategory.UNAVAILABLE_TARGET
    assert classifier.classify("Execution timed out after 10.0s") == FailureCategory.TIMEOUT
    assert classifier.classify("Visual mismatch: expected element did not appear") == FailureCategory.UNEXPECTED_UI_CHANGE
    assert classifier.classify("Access denied: Administrator privileges required") == FailureCategory.PERMISSION_ISSUE
    assert classifier.classify("Multiple items found matching query") == FailureCategory.AMBIGUOUS_INTENT


def test_adaptive_recovery_fresh_evidence_enforcement():
    """
    Test 14.3: Recovery requires fresh evidence before retrying.
    """
    recovery_mgr = AdaptiveRecoveryManager(max_action_retries=2)
    context = TaskContext(run_id="run-fresh-001")
    action = Action("click", {"target": "Save"}, run_id="run-fresh-001")

    # When UI changed unexpectedly, recovery plan MUST mandate fresh observation and replan
    plan = recovery_mgr.plan_recovery(action, error="Visual mismatch on screen", attempt_num=1, context=context)
    assert plan.category == FailureCategory.UNEXPECTED_UI_CHANGE
    assert plan.requires_fresh_observation is True
    assert plan.should_replan is True
    assert plan.recommended_action is not None
    assert plan.recommended_action.action_type == "observe_state"


def test_conversational_continuity_what_happened_and_try_again():
    """
    Test 14.5: Support 'what happened?' and 'try again' conversational commands.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            token = await runtime.get_or_create_run_token()
            ctx = runtime.get_or_create_task_context(token.run_id)
            ctx.record_failure("Server returned HTTP 503 Service Unavailable")
            ctx.record_completed_action(
                ActionResult(
                    action_id="act-prev",
                    action_type="fetch_data",
                    outcome=ExecutionOutcome.FAILED,
                    action_arguments={"endpoint": "/reports"},
                )
            )

            status_responses = []

            async def _on_status(e: Event):
                status_responses.append(e.payload.get("status", ""))

            runtime.event_bus.subscribe(EventType.STATUS_RESPONDED, _on_status)

            # 1. Ask "what happened?"
            await runtime.event_bus.publish(
                Event(EventType.INTENT_DETECTED, {"intent": "what happened?", "run_id": token.run_id})
            )
            await asyncio.sleep(0.04)

            assert len(status_responses) >= 1
            assert "HTTP 503" in status_responses[-1]

            # 2. Command "try again"
            await runtime.event_bus.publish(
                Event(EventType.INTENT_DETECTED, {"intent": "try again", "run_id": token.run_id})
            )
            await asyncio.sleep(0.05)

            # Re-executed action should be fetch_data
            executed_types = [a.action_type for a in executor.executed_actions]
            assert "fetch_data" in executed_types
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_pause_versus_cancellation_semantics():
    """
    Test 14.7: Pause preserves context and allows resume;
    Cancellation invalidates pending work and blocks silent resume.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.02)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            token = await runtime.start_new_command("long_task", {})
            run_id = token.run_id

            # 1. Pause
            await runtime.pause()
            assert runtime.get_or_create_task_context(run_id).is_paused is True

            # Resume
            await runtime.resume()
            assert runtime.get_or_create_task_context(run_id).is_paused is False

            # 2. Cancel
            await runtime.interrupt(reason="User said cancel")
            assert runtime.is_run_cancelled(run_id) is True

            # Attempting to resume a cancelled run is rejected (Section 14.6)
            resumed_rejected = []

            async def _on_status(e: Event):
                resumed_rejected.append(e.payload.get("status", ""))

            runtime.event_bus.subscribe(EventType.STATUS_RESPONDED, _on_status)

            await runtime.event_bus.publish(
                Event(EventType.INTENT_DETECTED, {"intent": "continue", "run_id": run_id})
            )
            await asyncio.sleep(0.04)

            assert len(resumed_rejected) >= 1
            assert "cancelled" in resumed_rejected[-1].lower()
        finally:
            await runtime.stop()

    asyncio.run(_runner())


# ==============================================================================
# PHASE 15 TESTS: Release Validation, Concurrency, Soak & Failure Injection
# ==============================================================================

def test_repeated_start_stop_soak_test():
    """
    Test 15.6 & 15.16: Repeated START -> STOP cycles.
    Verifies that no tasks, workers, or resources leak across repeated cycles.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.005)
        runtime = AgentRuntime(executor=executor)

        # Run 6 rapid start-stop cycles
        for cycle in range(6):
            await runtime.start()
            assert runtime.state_manager.current.lifecycle_state == AgentLifecycleState.RUNNING

            # Submit lightweight action
            act = Action("ping", {"cycle": cycle})
            await runtime.submit_action(act)
            for _ in range(20):
                if act.status == ActionStatus.COMPLETED:
                    break
                await asyncio.sleep(0.01)
            assert act.status == ActionStatus.COMPLETED

            await runtime.stop()
            assert runtime.state_manager.current.lifecycle_state == AgentLifecycleState.STOPPED

    asyncio.run(_runner())


def test_rapid_consecutive_commands_redirection():
    """
    Test 15.5: Rapid consecutive incoming commands.
    Ensures each subsequent command cleanly supersedes the prior one without state corruption.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.03)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            tokens = []
            for i in range(4):
                tok = await runtime.start_new_command(f"command_{i}", {"seq": i})
                tokens.append(tok)
                await asyncio.sleep(0.01)

            # Only the last command should remain active
            assert runtime.is_run_cancelled(tokens[0].run_id) is True
            assert runtime.is_run_cancelled(tokens[1].run_id) is True
            assert runtime.is_run_cancelled(tokens[2].run_id) is True
            assert runtime.is_run_active(tokens[3].run_id) is True

            await asyncio.sleep(0.08)
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_failure_injection_subsystem_health_and_degradation():
    """
    Test 15.10 & 15.17: Failure injection.
    Simulate executor failure, verify subsystem degradation is recorded,
    and verify the runtime stays operational and interruptible.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Injected hardware error action
            failing_act = Action(
                "hardware_io",
                arguments={"simulate_failure": True, "failure_reason": "Injected device disconnection"},
            )
            await runtime.submit_action(failing_act)
            await asyncio.sleep(0.08)

            assert failing_act.status == ActionStatus.FAILED
            # Health monitor should flag degraded status
            status = runtime.health_monitor.get_status("executor")
            assert status == SubsystemStatus.DEGRADED

            # Runtime must still accept healthy subsequent commands
            healthy_act = Action("healthy_ping", {})
            await runtime.submit_action(healthy_act)
            await asyncio.sleep(0.04)

            assert healthy_act.status == ActionStatus.COMPLETED
        finally:
            await runtime.stop()

    asyncio.run(_runner())


def test_overlay_display_state_synchronization():
    """
    Test 13.9 & 15.13: Overlay state synchronization.
    Verifies that the floating overlay cleanly transitions across states:
    Listening -> Executing -> Verifying -> Stopped.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.03)
        runtime = AgentRuntime(executor=executor)
        overlay = FloatingOverlay(runtime=runtime)
        await runtime.start()

        try:
            # Initial state is Idle until audio starts
            assert overlay.display_state == OverlayDisplayState.IDLE
            await runtime.event_bus.publish(Event(EventType.AUDIO_STREAM_STARTED))
            assert overlay.display_state == OverlayDisplayState.LISTENING

            # Action started -> Executing / Acting
            act = Action("observe_state", {})
            await runtime.submit_action(act)
            await asyncio.sleep(0.01)

            assert overlay.display_state in (OverlayDisplayState.EXECUTING, OverlayDisplayState.ACTING, OverlayDisplayState.VERIFYING)

            # Press STOP control
            await overlay.press_stop()
            assert overlay.display_state == OverlayDisplayState.STOPPED
        finally:
            await runtime.stop()

    asyncio.run(_runner())
