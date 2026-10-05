"""
Tests for StateManager and AgentState.
"""

from app.agent.action import Action
from app.state.state_manager import AgentLifecycleState, StateManager


def test_state_manager_initial_state():
    """Verify default initial state of the state manager."""
    sm = StateManager()
    assert sm.current.lifecycle_state == AgentLifecycleState.IDLE
    assert sm.current.current_action is None
    assert sm.current.pending_actions_count == 0
    assert not sm.current.is_execution_active
    assert not sm.current.cancellation_requested
    assert sm.current.last_error is None


def test_state_manager_transitions():
    """Verify state updates for lifecycle, action, and cancellation."""
    sm = StateManager()

    sm.set_lifecycle(AgentLifecycleState.RUNNING)
    assert sm.current.lifecycle_state == AgentLifecycleState.RUNNING

    action = Action("test_action")
    sm.set_current_action(action)
    assert sm.current.current_action == action
    assert sm.current.is_execution_active is True

    sm.set_pending_count(3)
    assert sm.current.pending_actions_count == 3

    sm.request_cancellation()
    assert sm.current.cancellation_requested is True

    sm.clear_cancellation()
    assert sm.current.cancellation_requested is False

    sm.record_error("Some error")
    assert sm.current.last_error == "Some error"

    sm.set_current_action(None)
    assert sm.current.current_action is None
    assert sm.current.is_execution_active is False
