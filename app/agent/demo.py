"""
Demonstration scenario for Phase 2 simulated asynchronous Agent Runtime.
Demonstrates:
  1. Runtime startup
  2. Simulated INTENT_DETECTED event creating and queueing an action
  3. Action execution through MockExecutor
  4. Interruption and cancellation of pending and active actions
  5. Graceful shutdown
"""

import asyncio
from app.agent.events import Event, EventType
from app.agent.executor import MockExecutor
from app.agent.runtime import AgentRuntime
from app.config import settings
from app.utils import setup_logging


async def run_demo() -> None:
    """Run the Phase 2 simulated scenario."""
    logger = setup_logging(settings.log_level)
    logger.info("=== Starting Phase 2 Agent Runtime Demonstration ===")

    # Initialize mock executor with a 150ms simulated delay for testing
    executor = MockExecutor(default_delay=0.15)
    runtime = AgentRuntime(executor=executor, logger=logger)

    await runtime.start()

    logger.info("\n--- Scenario 1: Successful Intent Detection & Action Execution ---")
    # Simulate an intent detected event: "open_application: brave"
    await runtime.event_bus.publish(
        Event(
            EventType.INTENT_DETECTED,
            payload={
                "intent": "open_application",
                "arguments": {"application": "brave"},
            },
        )
    )

    # Allow time for action execution to complete
    await asyncio.sleep(0.3)

    logger.info("\n--- Scenario 2: Interruption / Cancellation ---")
    # Queue a slower action: "open_application: powerpoint"
    await runtime.event_bus.publish(
        Event(
            EventType.INTENT_DETECTED,
            payload={
                "intent": "open_application",
                "arguments": {"application": "powerpoint"},
            },
        )
    )

    # Queue a second pending action
    await runtime.event_bus.publish(
        Event(
            EventType.INTENT_DETECTED,
            payload={
                "intent": "navigate_url",
                "arguments": {"url": "https://example.com"},
            },
        )
    )

    # Briefly wait so first action begins running
    await asyncio.sleep(0.05)

    # Simulate user interruption ("Stop / Cancel")
    logger.info(">>> User says: 'Stop! Cancel that.'")
    await runtime.interrupt(reason="User said 'Stop'")

    # Allow cancellation propagation
    await asyncio.sleep(0.2)

    logger.info("\n--- Scenario 3: Graceful Shutdown ---")
    await runtime.stop()

    logger.info("=== Phase 2 Demonstration Finished Successfully ===")


def main() -> int:
    """CLI entry point for demo."""
    asyncio.run(run_demo())
    return 0


if __name__ == "__main__":
    main()
