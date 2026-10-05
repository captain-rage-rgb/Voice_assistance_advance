"""
Phase 3 Live Microphone and Audio Pipeline Demonstration.
Demonstrates:
  1. Starting the AudioPipeline and capturing real microphone input
  2. Streaming audio chunks through the bounded queue
  3. Real-time Voice Activity Detection (VAD) and Endpointing
  4. Emitting VOICE_ACTIVITY_STARTED and VOICE_ACTIVITY_STOPPED events
  5. Graceful shutdown after a demo listening period
"""

import asyncio
import sys
from app.agent.events import Event, EventBus, EventType
from app.audio.pipeline import AudioPipeline
from app.config import settings
from app.state.state_manager import StateManager
from app.utils import setup_logging


async def run_audio_demo(duration_seconds: int = 10) -> None:
    """Run live microphone audio pipeline demonstration."""
    logger = setup_logging(settings.log_level)
    logger.info("=== Starting Phase 3 Audio Pipeline Live Demonstration ===")

    event_bus = EventBus()
    state_manager = StateManager()

    # Subscribe to audio events for display
    async def log_event(evt: Event):
        if evt.event_type == EventType.VOICE_ACTIVITY_STARTED:
            logger.info(">>> [EVENT] VOICE_ACTIVITY_STARTED: User started speaking")
        elif evt.event_type == EventType.VOICE_ACTIVITY_STOPPED:
            logger.info("<<< [EVENT] VOICE_ACTIVITY_STOPPED: User stopped speaking")
        elif evt.event_type == EventType.AUDIO_ERROR:
            logger.error("!!! [EVENT] AUDIO_ERROR: %s", evt.payload.get("error"))

    event_bus.subscribe(EventType.VOICE_ACTIVITY_STARTED, log_event)
    event_bus.subscribe(EventType.VOICE_ACTIVITY_STOPPED, log_event)
    event_bus.subscribe(EventType.AUDIO_ERROR, log_event)

    pipeline = AudioPipeline(
        event_bus=event_bus,
        state_manager=state_manager,
        config=settings,
        logger=logger,
    )

    try:
        await pipeline.start()
        logger.info(
            "Listening to microphone for %d seconds... (Speak to test voice activity)",
            duration_seconds,
        )

        for remaining in range(duration_seconds, 0, -1):
            await asyncio.sleep(1)

    except Exception as exc:
        logger.error("Failed during audio demonstration: %s", exc)
    finally:
        logger.info("Shutting down audio pipeline...")
        await pipeline.stop()
        logger.info("=== Phase 3 Demonstration Finished Successfully ===")


def main() -> int:
    """CLI entry point."""
    duration = 5
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            pass
    asyncio.run(run_audio_demo(duration_seconds=duration))
    return 0


if __name__ == "__main__":
    main()
