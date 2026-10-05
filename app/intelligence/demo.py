"""
Phase 4 Demonstration Scenario for Streaming Speech-to-Text.
Demonstrates:
  1. Live microphone capture feeding streaming audio chunks
  2. STTManager integration (using live Deepgram if STT_API_KEY is configured,
     or MockStreamingSTT with realistic speech progression if no key is present)
  3. Real-time partial transcript revisions and deduplication
  4. Finalized transcript event dispatching
  5. Clean, leak-free async shutdown
"""

import asyncio
import sys
from app.agent.events import Event, EventBus, EventType
from app.audio.pipeline import AudioPipeline
from app.config import settings
from app.intelligence.mock_stt import MockStreamingSTT
from app.intelligence.stt_manager import STTManager
from app.intelligence.transcript import Transcript
from app.state.state_manager import StateManager
from app.utils import setup_logging


async def run_stt_demo(duration_seconds: int = 6) -> None:
    """Run STT pipeline demonstration."""
    logger = setup_logging(settings.log_level)
    logger.info("=== Starting Phase 4 Streaming Speech-to-Text Demonstration ===")

    event_bus = EventBus()
    state_manager = StateManager()

    # Event listener for live transcript output
    async def log_transcript_event(evt: Event):
        if evt.event_type == EventType.TRANSCRIPT_PARTIAL:
            text = evt.payload.get("text")
            latency = evt.payload.get("latency_ms")
            logger.info("  [PARTIAL] %s (latency: %s ms)", text, f"{latency:.1f}" if latency else "N/A")
        elif evt.event_type == EventType.TRANSCRIPT_FINAL:
            text = evt.payload.get("text")
            conf = evt.payload.get("confidence")
            logger.info(">>> [FINAL]   %s (confidence: %s)", text, f"{conf:.2f}" if conf else "N/A")
        elif evt.event_type == EventType.STT_ERROR:
            logger.error("!!! [STT_ERROR] %s", evt.payload.get("error"))

    event_bus.subscribe(EventType.TRANSCRIPT_PARTIAL, log_transcript_event)
    event_bus.subscribe(EventType.TRANSCRIPT_FINAL, log_transcript_event)
    event_bus.subscribe(EventType.STT_ERROR, log_transcript_event)

    # Determine provider: Deepgram if real key provided, otherwise realistic Mock provider
    has_real_key = bool(settings.stt_api_key and not settings.stt_api_key.startswith("your_"))
    mock_provider = None

    if has_real_key:
        logger.info("Real Deepgram API key detected. Using live streaming STT.")
        stt_provider = None  # STTManager will construct DeepgramStreamingSTT
    else:
        logger.info("No live Deepgram key provided. Using MockStreamingSTT with realistic speech simulation.")
        mock_provider = MockStreamingSTT()
        stt_provider = mock_provider

    # Set up AudioPipeline
    pipeline = AudioPipeline(
        event_bus=event_bus,
        state_manager=state_manager,
        config=settings,
        logger=logger,
    )

    # Set up STTManager consuming chunks from pipeline.audio_queue
    stt_manager = STTManager(
        audio_queue=pipeline.audio_queue,
        stt_provider=stt_provider,
        event_bus=event_bus,
        state_manager=state_manager,
        config=settings,
        logger=logger,
    )

    try:
        await pipeline.start()
        await stt_manager.start()

        if mock_provider:
            # Inject realistic transcript progression for demonstration
            logger.info("Simulating speech recognition progression while user is speaking...")
            await asyncio.sleep(0.5)
            await mock_provider.push_transcript(Transcript(text="Open", is_final=False))
            await asyncio.sleep(0.3)
            await mock_provider.push_transcript(Transcript(text="Open", is_final=False))  # Deduped
            await asyncio.sleep(0.3)
            await mock_provider.push_transcript(Transcript(text="Open Chrome", is_final=False))
            await asyncio.sleep(0.4)
            await mock_provider.push_transcript(Transcript(text="Open Chrome and", is_final=False))
            await asyncio.sleep(0.4)
            await mock_provider.push_transcript(Transcript(text="Open Chrome and go to YouTube", is_final=False))
            await asyncio.sleep(0.5)
            await mock_provider.push_transcript(
                Transcript(text="Open Chrome and go to YouTube", is_final=True, confidence=0.97)
            )

        logger.info("Listening for remaining %d seconds...", duration_seconds)
        await asyncio.sleep(duration_seconds)

    except Exception as exc:
        logger.error("Error during STT demo: %s", exc)
    finally:
        logger.info("Shutting down STT and Audio pipeline...")
        await stt_manager.stop()
        await pipeline.stop()
        logger.info("=== Phase 4 Demonstration Finished Successfully ===")


def main() -> int:
    """CLI entry point."""
    duration = 3
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            pass
    asyncio.run(run_stt_demo(duration_seconds=duration))
    return 0


if __name__ == "__main__":
    main()
