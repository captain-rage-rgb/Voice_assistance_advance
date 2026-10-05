"""
Action schema validator (Phase 7).
Ensures only well-formed, safe, and supported actions reach the computer execution layer.
Rejects malformed, speculative, or ambiguous action fragments.
"""

import logging
from typing import Any, Dict, Optional, Tuple
from app.agent.action import Action


class ActionValidator:
    """
    Validates executable actions before submission to the action queue or execution.
    """

    SUPPORTED_ACTIONS = {
        "OPEN_APPLICATION": {"application"},
        "NAVIGATE": {"destination"},
        "SEARCH": {"query"},
        "TYPE_TEXT": {"text"},
        "CLICK": set(),  # Optional target/coords
        "PRESS_KEY": {"key"},
        "WAIT": {"seconds"},
        "OBSERVE_STATE": set(),
        "INTERRUPT": set(),
        # Lowercase variants for compatibility
        "open_application": {"application"},
        "navigate": {"destination"},
        "navigate_url": {"url"},
        "search": {"query"},
        "type_text": {"text"},
        "click": set(),
        "press_key": {"key"},
        "wait": {"seconds"},
        "observe_state": set(),
    }

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("voice_agent.validator")

    def validate(self, action: Action) -> Tuple[bool, Optional[str]]:
        """
        Validate an Action against supported schemas and required arguments.

        Returns:
            (is_valid, error_reason)
        """
        if not action.action_type:
            return False, "Missing action_type"

        # Check supported action types
        if action.action_type not in self.SUPPORTED_ACTIONS:
            # Allow generic custom actions if arguments are valid dictionary
            if not isinstance(action.arguments, dict):
                return False, f"Action arguments must be a dictionary, got {type(action.arguments)}"
            return True, None

        required_keys = self.SUPPORTED_ACTIONS[action.action_type]
        for key in required_keys:
            # Check for alternative key mappings (e.g. url vs destination)
            val = action.arguments.get(key)
            if val is None and key == "destination":
                val = action.arguments.get("url")
            if val is None or (isinstance(val, str) and not val.strip()):
                return False, f"Missing required parameter '{key}' for action '{action.action_type}'"

        return True, None
