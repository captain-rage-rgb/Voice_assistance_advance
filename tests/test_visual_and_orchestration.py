"""
Automated Tests for Phase 11 & Phase 12 (Tests A through L).
Covers:
  - Test A: Visual target selection
  - Test B: Ambiguous target blocks and requests clarification
  - Test C: Stale observation rejected
  - Test D: Visual verification recognizes UI state change
  - Test E: Visual mismatch triggers replanning
  - Test F: Multi-goal task with dependency prerequisites
  - Test G: Sub-goal failure prevents dependent execution
  - Test H: Dynamic replanning on changed UI state
  - Test I: Unified cancellation during confirmation and execution
  - Test J: Confirmation invalidation across runs (Run A cannot authorize Run B)
  - Test K: Status request returns concise summary without chain-of-thought
  - Test L: Voice redirection supersedes active task with new run ID
"""

import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from app.agent.action import Action, ActionStatus
from app.agent.context import ActionResult, ExecutionOutcome, TaskContext
from app.agent.events import Event, EventType
from app.agent.executor import MockExecutor
from app.agent.orchestration import (
    ConfirmationManager,
    ConfirmationStatus,
    GoalOrchestrator,
    GoalStatus,
    SubGoal,
    TaskGoal,
)
from app.agent.overlay import FloatingOverlay, OverlayDisplayState
from app.agent.runtime import AgentRuntime
from app.agent.visual import (
    ComputerObservation,
    TargetingConfidence,
    UIElement,
    UITargeter,
    VisualVerifier,
)


# ==============================================================================
# TEST A — Visual Target Selection
# ==============================================================================
def test_test_a_visual_target_selection():
    """
    Test A: Given an observation containing multiple UI elements,
    verify the correct target is selected and coordinate targeting succeeds.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Create a rich observation with multiple elements
            obs = ComputerObservation(
                window_title="Settings Window",
                active_application="SystemSettings",
                elements=[
                    UIElement(label="Search Bar", element_type="input", bounding_box=(100, 50, 200, 30), source="accessibility"),
                    UIElement(label="Settings Icon", element_type="icon", bounding_box=(500, 200, 50, 50), source="accessibility"),
                    UIElement(label="Cancel Button", element_type="button", bounding_box=(300, 400, 80, 30), source="accessibility"),
                    UIElement(label="Apply Button", element_type="button", bounding_box=(400, 400, 80, 30), source="accessibility"),
                ],
            )

            token = await runtime.get_or_create_run_token()
            ctx = runtime.get_or_create_task_context(token.run_id)
            ctx.record_visual_observation(obs)

            # Issue action targeting the settings icon
            action = Action(
                action_type="click",
                arguments={"target": "Settings Icon"},
                run_id=token.run_id,
            )

            target_found_events = []
            runtime.event_bus.subscribe(
                EventType.VISUAL_TARGET_FOUND,
                lambda e: target_found_events.append(e) or asyncio.sleep(0),
            )

            await runtime.submit_action(action)
            await asyncio.sleep(0.08)

            assert action.status == ActionStatus.COMPLETED
            # Target coordinates derived from bounding box (500 + 25, 200 + 25) = (525, 225)
            assert action.arguments.get("x") == 525
            assert action.arguments.get("y") == 225
            assert len(target_found_events) >= 1
            assert target_found_events[0].payload["label"] == "Settings Icon"
        finally:
            await runtime.stop()

    asyncio.run(_runner())


# ==============================================================================
# TEST B — Ambiguous Target Handling
# ==============================================================================
def test_test_b_ambiguous_target_blocks_and_requests_clarification():
    """
    Test B: Given two equally plausible targets,
    verify that execution is blocked and clarification is requested.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Observation with ambiguous competing items
            obs = ComputerObservation(
                window_title="File Manager",
                active_application="Explorer",
                elements=[
                    UIElement(label="Settings (User Profile)", element_type="button", bounding_box=(100, 100, 150, 30), source="accessibility"),
                    UIElement(label="Settings (System Admin)", element_type="button", bounding_box=(100, 150, 150, 30), source="accessibility"),
                    UIElement(label="Help", element_type="button", bounding_box=(100, 200, 80, 30), source="accessibility"),
                ],
            )

            token = await runtime.get_or_create_run_token()
            ctx = runtime.get_or_create_task_context(token.run_id)
            ctx.record_visual_observation(obs)

            clarification_events = []
            ambiguous_events = []
            runtime.event_bus.subscribe(
                EventType.CLARIFICATION_REQUESTED,
                lambda e: clarification_events.append(e) or asyncio.sleep(0),
            )
            runtime.event_bus.subscribe(
                EventType.VISUAL_TARGET_AMBIGUOUS,
                lambda e: ambiguous_events.append(e) or asyncio.sleep(0),
            )

            action = Action(
                action_type="click",
                arguments={"target": "Settings"},
                run_id=token.run_id,
            )

            await runtime.submit_action(action)
            await asyncio.sleep(0.08)

            # Execution must be blocked
            assert action.status == ActionStatus.FAILED
            assert len(ambiguous_events) >= 1
            assert len(clarification_events) >= 1
            assert "multiple items" in clarification_events[0].payload["question"].lower()
            # Executor should NOT have executed the ambiguous click
            assert len(executor.executed_actions) == 0
        finally:
            await runtime.stop()

    asyncio.run(_runner())


# ==============================================================================
# TEST C — Stale Observation Rejection
# ==============================================================================
def test_test_c_stale_observation_rejected():
    """
    Test C: Use an outdated observation.
    Expected: Action is rejected and fresh observation is required.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Observation timestamped 10 seconds ago (freshness limit is 5.0s)
            stale_time = datetime.now(timezone.utc) - timedelta(seconds=10)
            obs = ComputerObservation(
                timestamp=stale_time,
                elements=[
                    UIElement(label="Download Button", element_type="button", bounding_box=(200, 200, 100, 40)),
                ],
            )

            token = await runtime.get_or_create_run_token()
            ctx = runtime.get_or_create_task_context(token.run_id)
            ctx.record_visual_observation(obs)

            stale_events = []
            runtime.event_bus.subscribe(
                EventType.VISUAL_OBSERVATION_STALE,
                lambda e: stale_events.append(e) or asyncio.sleep(0),
            )

            action = Action(
                action_type="click",
                arguments={"target": "Download Button"},
                run_id=token.run_id,
            )

            await runtime.submit_action(action)
            await asyncio.sleep(0.08)

            assert action.status == ActionStatus.FAILED
            assert len(stale_events) >= 1
            assert len(executor.executed_actions) == 0
        finally:
            await runtime.stop()

    asyncio.run(_runner())


# ==============================================================================
# TEST D — Visual Verification
# ==============================================================================
def test_test_d_visual_verification_detects_ui_change():
    """
    Test D: Simulate a successful UI change.
    Expected: Verification recognizes the resulting state.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            obs_before = ComputerObservation(
                window_title="Web Browser",
                elements=[UIElement(label="Open Dialog Button", bounding_box=(50, 50, 100, 30))],
            )
            obs_after = ComputerObservation(
                window_title="Settings Dialog",
                elements=[
                    UIElement(label="Settings Dialog Window", element_type="dialog", bounding_box=(100, 100, 400, 300)),
                    UIElement(label="Save Preferences", element_type="button", bounding_box=(200, 250, 100, 30)),
                ],
            )

            token = await runtime.get_or_create_run_token()
            ctx = runtime.get_or_create_task_context(token.run_id)
            ctx.record_visual_observation(obs_before)

            action = Action(
                action_type="click",
                arguments={
                    "x": 100,
                    "y": 65,
                    "expected_change": {"expected_element_appeared": "Settings Dialog Window"},
                    "post_observation": obs_after,
                },
                run_id=token.run_id,
            )

            await runtime.submit_action(action)
            await asyncio.sleep(0.08)

            assert action.status == ActionStatus.COMPLETED
            assert len(executor.executed_actions) == 1
        finally:
            await runtime.stop()

    asyncio.run(_runner())


# ==============================================================================
# TEST E — Visual Mismatch & Replanning
# ==============================================================================
def test_test_e_visual_mismatch_triggers_replanning():
    """
    Test E: Simulate an unexpected UI state.
    Expected: System triggers dynamic replanning rather than blindly replaying original action.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            replan_started = []
            replan_completed = []
            mismatch_events = []

            runtime.event_bus.subscribe(
                EventType.VISUAL_VERIFICATION_FAILED,
                lambda e: mismatch_events.append(e) or asyncio.sleep(0),
            )
            runtime.event_bus.subscribe(
                EventType.REPLANNING_STARTED,
                lambda e: replan_started.append(e) or asyncio.sleep(0),
            )
            runtime.event_bus.subscribe(
                EventType.REPLANNING_COMPLETED,
                lambda e: replan_completed.append(e) or asyncio.sleep(0),
            )

            token = await runtime.get_or_create_run_token()

            # Submit action simulating visual mismatch
            action = Action(
                action_type="click",
                arguments={"x": 50, "y": 50, "simulate_visual_mismatch": True},
                run_id=token.run_id,
            )

            await runtime.submit_action(action)
            await asyncio.sleep(0.08)

            assert len(mismatch_events) >= 1
            assert len(replan_started) >= 1
            assert len(replan_completed) >= 1
        finally:
            await runtime.stop()

    asyncio.run(_runner())


# ==============================================================================
# TEST F — Multi-Goal Task & Prerequisites
# ==============================================================================
def test_test_f_multi_goal_task_hierarchy_and_prerequisites():
    """
    Test F: Create a root task with multiple dependent sub-goals.
    Expected:
      * Prerequisites are respected.
      * Completed sub-goals are not repeated.
      * Root completion occurs only after all required sub-goals finish.
    """
    orchestrator = GoalOrchestrator()
    goal = TaskGoal(description="Locate, open, and check total in report", run_id="run-100")

    # Step 1: Locate report
    sg1 = goal.add_subgoal("Locate report", subgoal_id="sg-1")
    # Step 2: Open report (requires sg-1)
    sg2 = goal.add_subgoal("Open report", prerequisites=["sg-1"], subgoal_id="sg-2")
    # Step 3: Extract total (requires sg-2)
    sg3 = goal.add_subgoal("Extract total", prerequisites=["sg-2"], subgoal_id="sg-3")

    orchestrator.register_goal(goal)

    # Initially, next executable sub-goal must be sg-1
    next_sg = goal.get_next_executable_subgoal()
    assert next_sg is not None
    assert next_sg.subgoal_id == "sg-1"

    # sg-2 and sg-3 cannot execute yet
    assert not goal.all_prerequisites_met(sg2)
    assert not goal.all_prerequisites_met(sg3)
    assert not goal.is_completed()

    # Complete sg-1
    res1 = ActionResult(action_id="act-1", action_type="search", outcome=ExecutionOutcome.COMPLETED)
    orchestrator.advance_subgoal_completion(goal, sg1, res1)
    assert sg1.status == GoalStatus.COMPLETED

    # Now sg-2 is executable, but sg-3 is still blocked
    next_sg = goal.get_next_executable_subgoal()
    assert next_sg is not None
    assert next_sg.subgoal_id == "sg-2"
    assert not goal.all_prerequisites_met(sg3)

    # Complete sg-2
    res2 = ActionResult(action_id="act-2", action_type="open", outcome=ExecutionOutcome.COMPLETED)
    orchestrator.advance_subgoal_completion(goal, sg2, res2)
    assert sg2.status == GoalStatus.COMPLETED

    # Now sg-3 is executable
    next_sg = goal.get_next_executable_subgoal()
    assert next_sg is not None
    assert next_sg.subgoal_id == "sg-3"

    # Complete sg-3
    res3 = ActionResult(action_id="act-3", action_type="extract", outcome=ExecutionOutcome.COMPLETED)
    is_finished = orchestrator.advance_subgoal_completion(goal, sg3, res3)
    assert is_finished is True
    assert goal.is_completed() is True
    assert goal.status == GoalStatus.COMPLETED


# ==============================================================================
# TEST G — Sub-Goal Failure & Bounded Recovery
# ==============================================================================
def test_test_g_subgoal_failure_blocks_dependent_goals():
    """
    Test G: Force one sub-goal to fail.
    Expected: Dependent goals do not execute incorrectly; task enters controlled failure.
    """
    orchestrator = GoalOrchestrator()
    goal = TaskGoal(description="Download and process dataset", run_id="run-101")
    sg1 = goal.add_subgoal("Download dataset", subgoal_id="sg-dl")
    sg2 = goal.add_subgoal("Process dataset", prerequisites=["sg-dl"], subgoal_id="sg-proc")
    orchestrator.register_goal(goal)

    # Simulate failure on sub-goal 1
    fail_res = ActionResult(
        action_id="act-dl",
        action_type="download",
        outcome=ExecutionOutcome.FAILED,
        error="Network timeout downloading dataset",
    )
    orchestrator.advance_subgoal_completion(goal, sg1, fail_res)

    assert sg1.status == GoalStatus.FAILED
    # Dependent sub-goal 2 must NOT be executable
    assert not goal.all_prerequisites_met(sg2)
    assert goal.get_next_executable_subgoal() is None
    assert not goal.is_completed()


# ==============================================================================
# TEST H — Dynamic Replanning on State Change
# ==============================================================================
def test_test_h_dynamic_replanning_adapts_subgoal():
    """
    Test H: Change simulated UI state between actions.
    Expected: Sub-goal is replanned with fresh actions and stale plan is discarded.
    """
    orchestrator = GoalOrchestrator()
    goal = TaskGoal(description="Submit payment form", run_id="run-102")
    sg = goal.add_subgoal("Click submit", subgoal_id="sg-submit", actions=[Action("click", {"target": "Old Submit Button"})])
    orchestrator.register_goal(goal)

    # UI changed; submit button moved or changed ID
    new_actions = [
        Action("scroll_down", {"pixels": 100}),
        Action("click", {"target": "New Floating Submit Button"}),
    ]
    orchestrator.replan_subgoal(goal, sg, new_actions, reason="Submit button relocated below fold")

    assert len(sg.actions) == 2
    assert sg.actions[0].action_type == "scroll_down"
    assert sg.actions[1].arguments["target"] == "New Floating Submit Button"
    assert sg.metadata["replanned_reason"] == "Submit button relocated below fold"


# ==============================================================================
# TEST I — Unified Cancellation During Confirmation & Execution
# ==============================================================================
def test_test_i_cancellation_during_confirmation():
    """
    Test I: Cancel during visual analysis or confirmation.
    Expected: Active work terminates safely and no later actions execute.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.1)
        runtime = AgentRuntime(executor=executor)
        overlay = FloatingOverlay(runtime=runtime)
        await runtime.start()

        try:
            token = await runtime.get_or_create_run_token()

            # Submit consequential action requiring confirmation
            action = Action(
                action_type="delete",
                arguments={"file": "critical_database.db", "requires_confirmation": True},
                run_id=token.run_id,
            )

            req_received = asyncio.Event()

            async def _on_conf_req(e: Event):
                req_received.set()

            runtime.event_bus.subscribe(EventType.CONFIRMATION_REQUESTED, _on_conf_req)
            await runtime.submit_action(action)

            # Wait for confirmation request
            await asyncio.wait_for(req_received.wait(), timeout=1.0)
            assert overlay.display_state == OverlayDisplayState.WAITING_FOR_CONFIRMATION

            # User presses STOP control on overlay
            await overlay.press_stop()
            await asyncio.sleep(0.08)

            assert action.status == ActionStatus.CANCELLED
            assert overlay.display_state in (OverlayDisplayState.STOPPED, OverlayDisplayState.LISTENING)
            assert len(executor.executed_actions) == 0
        finally:
            await runtime.stop()

    asyncio.run(_runner())


# ==============================================================================
# TEST J — Confirmation Invalidation Across Runs
# ==============================================================================
def test_test_j_confirmation_invalidation_across_runs():
    """
    Test J: Create a confirmation for run A, cancel run A, start run B.
    Expected: Run A's confirmation cannot authorize anything in run B.
    """
    mgr = ConfirmationManager()
    action = Action("delete_file", {"target": "test.txt"}, run_id="run-A")
    req = mgr.request_confirmation(run_id="run-A", action=action, prompt="Confirm delete?")

    assert req.status == ConfirmationStatus.PENDING

    # Invalidate Run A (e.g. cancelled or superseded)
    mgr.invalidate_run("run-A")
    assert req.status == ConfirmationStatus.CANCELLED

    # Attempt to resolve this confirmation from Run B
    success, reason = mgr.resolve_confirmation(
        confirmation_id=req.confirmation_id,
        approved=True,
        caller_run_id="run-B",
    )
    assert success is False
    assert "not current run" in reason or "invalid" in reason.lower()
    assert req.status != ConfirmationStatus.APPROVED


# ==============================================================================
# TEST K — Status Request Without Chain-of-Thought
# ==============================================================================
def test_test_k_status_request_returns_concise_summary():
    """
    Test K: During a task, ask for status.
    Expected: Concise current-state response without internal reasoning.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.01)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            token = await runtime.get_or_create_run_token()
            goal = TaskGoal(description="Generate quarterly sales report", run_id=token.run_id)
            sg1 = goal.add_subgoal("Locate sales spreadsheet", subgoal_id="sg-sales-1")
            sg1.status = GoalStatus.ACTIVE
            goal.add_subgoal("Extract total revenue", subgoal_id="sg-sales-2")
            runtime.orchestrator.register_goal(goal)

            status_responses = []

            async def _on_status(e: Event):
                status_responses.append(e.payload.get("status"))

            runtime.event_bus.subscribe(EventType.STATUS_RESPONDED, _on_status)

            # User asks "What are you doing?"
            await runtime.event_bus.publish(
                Event(EventType.INTENT_DETECTED, {"intent": "what are you doing?", "run_id": token.run_id})
            )
            await asyncio.sleep(0.05)

            assert len(status_responses) == 1
            summary = status_responses[0]
            assert "Locate sales spreadsheet" in summary
            # Verify no CoT leaked
            assert "Thought:" not in summary
            assert "Plan:" not in summary
        finally:
            await runtime.stop()

    asyncio.run(_runner())


# ==============================================================================
# TEST L — Voice Redirection Supersedes Active Task
# ==============================================================================
def test_test_l_voice_redirection_supersedes_active_task():
    """
    Test L: Start task A and issue a conflicting new voice command.
    Expected: Task A is safely superseded, Task B starts with new run ID,
    and stale Task A actions cannot execute.
    """

    async def _runner():
        executor = MockExecutor(default_delay=0.05)
        runtime = AgentRuntime(executor=executor)
        await runtime.start()

        try:
            # Start Task A
            token_a = await runtime.start_new_command("browse_web", {"url": "https://example.com"})
            run_id_a = token_a.run_id

            # Issue conflicting new command (Voice barge-in redirection)
            await asyncio.sleep(0.01)
            token_b = await runtime.start_new_command("open_calculator", {})
            run_id_b = token_b.run_id

            assert run_id_a != run_id_b
            assert runtime.is_run_cancelled(run_id_a) is True
            assert runtime.is_run_active(run_id_b) is True

            # Attempt to queue an old stale action belonging to Run A
            stale_action = Action("stale_click", {"target": "old_page"}, run_id=run_id_a)
            await runtime.submit_action(stale_action)
            await asyncio.sleep(0.08)

            assert stale_action.status == ActionStatus.CANCELLED
            # Verify executor only executed the fresh action or run B
            executed_runs = {a.run_id for a in executor.executed_actions}
            assert run_id_a not in executed_runs or len(executor.executed_actions) <= 2
        finally:
            await runtime.stop()

    asyncio.run(_runner())
