import asyncio
import logging
import os
import subprocess
from typing import Any, Optional, Protocol
import urllib.parse
import webbrowser
from app.agent.action import Action
from app.agent.cancellation import CancellationToken


class ActionExecutor(Protocol):
    """Protocol defining the executor interface."""

    async def execute(self, action: Action, token: Optional[CancellationToken] = None) -> Any:
        """
        Execute an action asynchronously with cancellation token checking.

        Args:
            action: The Action instance to execute.
            token: Optional cancellation token for checking before/during execution.

        Returns:
            The result of the action execution.

        Raises:
            Exception: If action execution fails.
            asyncio.CancelledError: If action was interrupted/cancelled during execution.
        """
        ...


class MockExecutor:
    """
    Mock executor implementation for testing and development.
    Simulates execution delays, successes, and intentional failures
    without performing real system automation.
    """

    def __init__(self, default_delay: float = 0.05) -> None:
        self.default_delay = default_delay
        self.executed_actions: list[Action] = []

    async def execute(self, action: Action, token: Optional[CancellationToken] = None) -> str:
        """
        Simulate action execution. Supports cooperative cancellation via asyncio.sleep
        and CancellationToken checks at operation boundaries.
        """
        if token and token.is_cancelled:
            token.check_cancelled()

        # Cooperative cancellation check: asyncio.sleep raises CancelledError if task cancelled
        await asyncio.sleep(self.default_delay)

        if token and token.is_cancelled:
            token.check_cancelled()

        # Allow tests to trigger simulated failures via special argument
        if action.arguments.get("simulate_failure"):
            error_reason = action.arguments.get("failure_reason", "Simulated failure error")
            raise RuntimeError(error_reason)

        self.executed_actions.append(action)
        return f"SIMULATED: executed '{action.action_type}' with args {action.arguments}"


class SystemActionExecutor:
    """
    Real OS and browser action executor for Windows (Phases 1-15).
    Performs real computer automation (application launching, web navigation,
    web searches, typing, etc.) while honoring cancellation tokens.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("voice_agent.executor.system")
        self.executed_actions: list[Action] = []

    async def execute(self, action: Action, token: Optional[CancellationToken] = None) -> str:
        """Execute a real system or browser action on the operating system."""
        if token and token.is_cancelled:
            token.check_cancelled()

        action_type = action.action_type.lower()
        args = action.arguments or {}

        # Cooperative cancellation check
        await asyncio.sleep(0.01)
        if token and token.is_cancelled:
            token.check_cancelled()

        # 1. Open Application
        if action_type in ("open_application", "open_app", "launch_app"):
            app_name = (args.get("application") or args.get("name") or "").strip()
            result = await self._open_application(app_name, token)

        # 2. Navigate / Go To URL
        elif action_type in ("navigate", "go_to", "open_url"):
            dest = (args.get("destination") or args.get("url") or "").strip()
            result = await self._navigate(dest, token)

        # 3. Search
        elif action_type in ("search", "web_search"):
            query = (args.get("query") or "").strip()
            result = await self._search(query, token)

        # 4. Type text
        elif action_type in ("type_text", "type", "write"):
            text = (args.get("text") or "").strip()
            result = await self._type_text(text, token)

        # 5. Press key
        elif action_type in ("press_key", "key_press", "press"):
            key = (args.get("key") or "").strip()
            result = await self._press_key(key, token)

        # 6. Incognito Mode
        elif action_type in ("incognito", "open_incognito"):
            result = await self._open_incognito(token)

        # 7. New Tab
        elif action_type in ("new_tab", "create_tab"):
            result = await self._new_tab(token)

        # 8. Close Tab
        elif action_type in ("close_tab",):
            result = await self._close_tab(token)

        # 9. Click / Select
        elif action_type in ("click", "choose", "select"):
            target = (args.get("target") or args.get("element_id") or "").strip()
            result = await self._click(target, token)

        else:
            result = f"Executed generic action '{action.action_type}'"

        self.executed_actions.append(action)
        return result

    async def _open_incognito(self, token: Optional[CancellationToken]) -> str:
        self.logger.info("SYSTEM EXECUTOR: Launching Chrome in Incognito mode")
        proc = await asyncio.create_subprocess_exec(
            "cmd", "/c", "start", "chrome", "--incognito",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return "Opened Chrome in Incognito mode"

    async def _new_tab(self, token: Optional[CancellationToken]) -> str:
        self.logger.info("SYSTEM EXECUTOR: Creating new browser tab (Ctrl+T)")
        ps_cmd = "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^t')"
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", ps_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return "Created new browser tab"

    async def _close_tab(self, token: Optional[CancellationToken]) -> str:
        self.logger.info("SYSTEM EXECUTOR: Closing active browser tab (Ctrl+W)")
        ps_cmd = "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('^w')"
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", ps_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return "Closed active browser tab"

    async def _click(self, target: str, token: Optional[CancellationToken]) -> str:
        self.logger.info("SYSTEM EXECUTOR: Selecting/clicking target: %s", target)
        if "incognito" in target.lower():
            return await self._open_incognito(token)
        ps_cmd = "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')"
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-NoProfile", "-Command", ps_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return f"Selected '{target}'"

    async def _open_application(self, app_name: str, token: Optional[CancellationToken]) -> str:
        app_lower = app_name.lower().strip()
        for art in ("the ", "my ", "a "):
            if app_lower.startswith(art):
                app_lower = app_lower[len(art):]

        cmd_map = {
            "chrome": ["cmd", "/c", "start", "chrome"],
            "google chrome": ["cmd", "/c", "start", "chrome"],
            "edge": ["cmd", "/c", "start", "msedge"],
            "microsoft edge": ["cmd", "/c", "start", "msedge"],
            "brave": ["cmd", "/c", "start", "brave"],
            "firefox": ["cmd", "/c", "start", "firefox"],
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "calc": ["calc.exe"],
            "explorer": ["explorer.exe"],
            "file explorer": ["explorer.exe"],
            "terminal": ["cmd", "/c", "start", "wt"],
            "vs code": ["cmd", "/c", "code"],
            "vscode": ["cmd", "/c", "code"],
            "spotify": ["cmd", "/c", "start", "spotify:"],
        }

        cmd = cmd_map.get(app_lower, ["cmd", "/c", "start", app_lower])
        self.logger.info("SYSTEM EXECUTOR: Launching app '%s' with command %s", app_name, cmd)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return f"Successfully opened {app_name}"
        except Exception as exc:
            self.logger.warning("Failed launching application '%s': %s", app_name, exc)
            return f"Could not launch application {app_name}: {exc}"

    async def _navigate(self, dest: str, token: Optional[CancellationToken]) -> str:
        dest_clean = dest.strip()
        for art in ("the ", "my "):
            if dest_clean.lower().startswith(art):
                dest_clean = dest_clean[len(art):]

        url_map = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "github": "https://github.com",
            "reddit": "https://www.reddit.com",
            "twitter": "https://x.com",
            "x": "https://x.com",
            "gmail": "https://mail.google.com",
        }

        url = url_map.get(dest_clean.lower())
        if not url:
            if dest_clean.startswith("http://") or dest_clean.startswith("https://"):
                url = dest_clean
            elif "." in dest_clean and " " not in dest_clean:
                url = f"https://{dest_clean}"
            else:
                url = f"https://www.google.com/search?q={urllib.parse.quote_plus(dest_clean)}"

        self.logger.info("SYSTEM EXECUTOR: Opening URL %s", url)
        await asyncio.to_thread(webbrowser.open, url)
        return f"Navigated to {url}"

    async def _search(self, query: str, token: Optional[CancellationToken]) -> str:
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
        self.logger.info("SYSTEM EXECUTOR: Searching for '%s' -> %s", query, url)
        await asyncio.to_thread(webbrowser.open, url)
        return f"Searched for '{query}'"

    async def _type_text(self, text: str, token: Optional[CancellationToken]) -> str:
        escaped = text.replace("`", "``").replace('"', '`"')
        ps_cmd = f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait(\"{escaped}\")"
        try:
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command", ps_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return f"Typed '{text}'"
        except Exception as exc:
            return f"Failed typing text: {exc}"

    async def _press_key(self, key: str, token: Optional[CancellationToken]) -> str:
        key_map = {
            "enter": "{ENTER}",
            "return": "{ENTER}",
            "tab": "{TAB}",
            "esc": "{ESC}",
            "escape": "{ESC}",
            "backspace": "{BACKSPACE}",
            "space": " ",
            "up": "{UP}",
            "down": "{DOWN}",
            "left": "{LEFT}",
            "right": "{RIGHT}",
        }
        send_key = key_map.get(key.lower(), key)
        ps_cmd = f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait(\"{send_key}\")"
        try:
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command", ps_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return f"Pressed key '{key}'"
        except Exception as exc:
            return f"Failed pressing key: {exc}"
