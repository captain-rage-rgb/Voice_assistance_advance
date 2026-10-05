"""
Application entry point for the Real-Time Voice Computer Agent (Phases 1-15).
Supports:
  1. Interactive Live Voice & Terminal Console:
     python main.py
  2. End-to-End Multimodal & Voice Demo:
     python main.py --demo
  3. Audio Pipeline (Mic & VAD) Demo:
     python main.py --audio-demo
  4. Streaming Speech-to-Text Demo:
     python main.py --stt-demo
  5. Run Automated Test Suite:
     python main.py --test
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

from app.agent.action import Action
from app.agent.events import Event, EventBus, EventType
from app.agent.executor import MockExecutor
from app.agent.overlay import FloatingOverlay, OverlayDisplayState
from app.agent.runtime import AgentRuntime
from app.agent.visual import ComputerObservation, UIElement
from app.audio.pipeline import AudioPipeline
from app.config import settings
from app.intelligence.deepgram_stt import DeepgramStreamingSTT
from app.intelligence.intent_detector import IncrementalIntentDetector
from app.intelligence.mock_stt import MockStreamingSTT
from app.intelligence.stt_manager import STTManager
from app.intelligence.transcript import Transcript
from app.state.state_manager import StateManager
from app.utils import setup_logging


BANNER = r"""
========================================================================
   __   ______  __ _____ ___   ___   ___  ___ _____ _____ _____ 
   \ \ / / __ \/ _/  _  \_ \ / _ \ / _ \/ __/_   _// ___//__  / 
    \ V / /_/ / / /  / / // // /_\ // /_\\__ \  / / / __/   / /  
     \_/\____/_/_/  /_/_//_/ \____//_/ \_\___/ /_/  \___/  /_/   
          REAL-TIME STREAMING VOICE COMPUTER AGENT (Phases 1-15)
========================================================================
"""


async def run_interactive() -> None:
    """Launch the interactive voice and terminal session."""
    logger = setup_logging(settings.log_level)
    print(BANNER)
    print("Initializing Agent subsystems...")

    event_bus = EventBus()
    state_manager = StateManager()
    runtime = AgentRuntime(event_bus=event_bus, state_manager=state_manager)
    overlay = FloatingOverlay(runtime=runtime, event_bus=event_bus)
    intent_detector = IncrementalIntentDetector(logger=logger)

    # Subscribe to feedback events for terminal display
    async def display_event(evt: Event) -> None:
        p = evt.payload
        if evt.event_type == EventType.TRANSCRIPT_PARTIAL:
            print(f"\r[TRANSCRIPT Partial] {p.get('text', '')}", end="", flush=True)
        elif evt.event_type == EventType.TRANSCRIPT_FINAL:
            print(f"\n[TRANSCRIPT Final]   {p.get('text', '')}")
        elif evt.event_type == EventType.INTENT_DETECTED:
            print(f"[*] INTENT DETECTED: {p.get('intent')} with args={p.get('arguments')}")
        elif evt.event_type == EventType.ACTION_STARTED:
            print(f" -> [EXEC] Starting action: {p.get('action_type', '')} (ID: {p.get('action_id', '')[:8]})")
        elif evt.event_type == EventType.ACTION_COMPLETED:
            print(f" <- [DONE] Action completed: {p.get('action_id', '')[:8]} (Verified={p.get('verified', False)})")
        elif evt.event_type == EventType.ACTION_FAILED:
            print(f" <!> [FAIL] Action failed: {p.get('error', 'Unknown')}")
        elif evt.event_type == EventType.USER_INTERRUPTED:
            print(f"\n[!] INTERRUPTED / STOPPED: {p.get('reason', 'User interrupted')}")
        elif evt.event_type == EventType.STATUS_RESPONDED:
            print(f"[STATUS] {p.get('status', '')}")
        elif evt.event_type == EventType.CLARIFICATION_REQUESTED:
            print(f"[CLARIFY] {p.get('question', '')}")
        elif evt.event_type == EventType.CONFIRMATION_REQUESTED:
            print(f"\n[CONFIRMATION REQUIRED] {p.get('prompt', '')} (Say 'yes' or 'no')")

    for etype in [
        EventType.TRANSCRIPT_PARTIAL,
        EventType.TRANSCRIPT_FINAL,
        EventType.INTENT_DETECTED,
        EventType.ACTION_STARTED,
        EventType.ACTION_COMPLETED,
        EventType.ACTION_FAILED,
        EventType.USER_INTERRUPTED,
        EventType.STATUS_RESPONDED,
        EventType.CLARIFICATION_REQUESTED,
        EventType.CONFIRMATION_REQUESTED,
    ]:
        event_bus.subscribe(etype, display_event)

    await runtime.start()

    # Attempt to start audio pipeline and STT if hardware is available
    audio_pipeline: Optional[AudioPipeline] = None
    stt_manager: Optional[STTManager] = None
    mic_active = False

    try:
        audio_pipeline = AudioPipeline(
            event_bus=event_bus,
            state_manager=state_manager,
            config=settings,
            logger=logger,
        )
        await audio_pipeline.start()
        mic_active = True

        has_deepgram = bool(settings.stt_api_key and not settings.stt_api_key.startswith("your_"))
        stt_provider = None if has_deepgram else MockStreamingSTT()
        stt_manager = STTManager(
            audio_queue=audio_pipeline.create_subscriber_queue(),
            stt_provider=stt_provider,
            event_bus=event_bus,
            state_manager=state_manager,
            intent_detector=intent_detector,
            config=settings,
            logger=logger,
        )
        await stt_manager.start()
        provider_name = "Deepgram Live STT" if has_deepgram else "Mock STT (Set STT_API_KEY for Deepgram)"
        print(f"[+] Microphone Capture: ACTIVE | STT Provider: {provider_name}")
    except Exception as exc:
        print(f"[-] Hardware audio pipeline inactive ({exc}). Console keyboard input ready.")

    print("\n------------------------------------------------------------------------")
    print("READY! You can speak into your microphone or type commands below.")
    print("Quick Commands:")
    print("  - 'open chrome and go to youtube and search for AI agents'")
    print("  - 'click submit button' | 'type hello world' | 'press enter'")
    print("  - 'what are you doing?' (status query) | 'what happened?'")
    print("  - 'pause' | 'resume' | 'stop' (barge-in cancellation)")
    print("  - 'demo' (run automated sequence) | 'help' | 'exit'")
    print("------------------------------------------------------------------------\n")

    loop = asyncio.get_running_loop()

    while True:
        try:
            user_input = await asyncio.to_thread(input, "voice-agent> ")
            cmd = user_input.strip()
            if not cmd:
                continue

            lower_cmd = cmd.lower()
            if lower_cmd in ("exit", "quit", "q"):
                print("\nExiting Voice Agent...")
                break

            if lower_cmd == "help":
                print("\nCommands Supported:")
                print("  Natural Voice Commands:")
                print("    'open <application>' (e.g. open chrome, open vscode, open notepad)")
                print("    'go to <website>' (e.g. go to youtube, go to google)")
                print("    'search for <query>' (e.g. search for AI agents)")
                print("    'type <text>' | 'press <key>' | 'click <target>'")
                print("  Control & Conversational:")
                print("    'stop' / 'cancel'  - Immediate unified cancellation / barge-in")
                print("    'pause' / 'resume' - Suspend or continue task")
                print("    'what are you doing?' - Status check")
                print("    'yes' / 'no'       - Resolve pending confirmation gate")
                print("    'demo'             - Run automated demo scenario")
                print("    'clear'            - Clear terminal screen\n")
                continue

            if lower_cmd == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                print(BANNER)
                continue

            if lower_cmd in ("stop", "cancel", "actually stop"):
                await overlay.press_stop()
                continue

            if lower_cmd in ("pause", "wait"):
                await runtime.pause()
                continue

            if lower_cmd in ("resume", "continue"):
                await runtime.resume()
                continue

            if lower_cmd == "demo":
                await run_integrated_demo(runtime, event_bus)
                continue

            # Process typed input as live speech
            await event_bus.publish(Event(EventType.TRANSCRIPT_FINAL, {"text": cmd, "confidence": 1.0}))
            intents = intent_detector.process_transcript(cmd, is_final=True)

            if intents:
                for detected in intents:
                    await event_bus.publish(
                        Event(
                            EventType.INTENT_DETECTED,
                            {
                                "intent": detected.intent_type.value,
                                "arguments": detected.parameters,
                                "source_text": detected.source_text,
                                "confidence": detected.confidence,
                            },
                        )
                    )
            else:
                # Fallback to direct action or conversational query
                await event_bus.publish(
                    Event(
                        EventType.INTENT_DETECTED,
                        {"intent": cmd, "arguments": {"raw": cmd}},
                    )
                )

            # Allow brief time for async processing to output
            await asyncio.sleep(0.15)

        except (KeyboardInterrupt, EOFError):
            print("\nReceived stop signal...")
            break
        except Exception as err:
            print(f"[Error] {err}")

    # Graceful shutdown
    print("\nShutting down subsystems cleanly...")
    if stt_manager:
        await stt_manager.stop()
    if audio_pipeline:
        await audio_pipeline.stop()
    await runtime.stop()
    print("Agent shut down successfully.")


async def run_integrated_demo(runtime: Optional[AgentRuntime] = None, event_bus: Optional[EventBus] = None) -> None:
    """Run an automated end-to-end demonstration showing incremental execution & resilience."""
    print("\n" + "=" * 65)
    print("=== STARTING REAL-TIME AGENT DEMONSTRATION SCENARIO ===")
    print("=" * 65)

    own_runtime = runtime is None
    if own_runtime:
        event_bus = EventBus()
        state_manager = StateManager()
        runtime = AgentRuntime(event_bus=event_bus, state_manager=state_manager)
        await runtime.start()

    detector = IncrementalIntentDetector()

    print("\n[Step 1] User begins speaking incrementally:")
    speech_fragments = [
        ("Open Chrome", False),
        ("Open Chrome and go to YouTube", False),
        ("Open Chrome and go to YouTube and search for AI agents", True),
    ]

    for fragment, is_final in speech_fragments:
        print(f"  Live speech stream: \"{fragment}\" (final={is_final})")
        await asyncio.sleep(0.4)
        intents = detector.process_transcript(fragment, is_final=is_final)
        for it in intents:
            print(f"    --> Incremental intent extracted: [{it.intent_type.value}] args={it.parameters}")
            await event_bus.publish(
                Event(
                    EventType.INTENT_DETECTED,
                    {"intent": it.intent_type.value, "arguments": it.parameters},
                )
            )

    await asyncio.sleep(1.0)

    print("\n[Step 2] Multimodal Visual Targeting & Verification:")
    sample_obs = ComputerObservation(
        window_title="Google Chrome - YouTube",
        elements=[
            UIElement(element_id="search_box", label="Search", element_type="textbox", bounding_box=(200, 50, 400, 30)),
            UIElement(element_id="submit_btn", label="Search button", element_type="button", bounding_box=(610, 50, 50, 30)),
        ],
    )
    runtime.get_or_create_task_context(runtime.active_run_id).latest_visual_observation = sample_obs
    print("  Visual observation fused: Detected search textbox and submit button.")

    target = runtime.targeter.resolve_target("Search button", sample_obs)
    elem_id = target.target_element.element_id if target.target_element else "None"
    print(f"  Targeted element: {elem_id} (Confidence: {target.confidence.value})")

    print("\n[Step 3] Conversational Status Query:")
    print("  User asks: 'What are you doing?'")
    await event_bus.publish(
        Event(EventType.INTENT_DETECTED, {"intent": "STATUS_QUERY", "arguments": {"is_status_query": True}})
    )
    await asyncio.sleep(0.5)

    print("\n[Step 4] Barge-in Interruption Showcase:")
    print("  User interrupts: 'Stop!'")
    await runtime.interrupt(reason="User said 'stop'")
    await asyncio.sleep(0.5)

    print("\n" + "=" * 65)
    print("=== DEMONSTRATION COMPLETE ===")
    print("=" * 65 + "\n")

    if own_runtime:
        await runtime.stop()


def main() -> int:
    """CLI entry point with flag dispatch."""
    parser = argparse.ArgumentParser(
        description="Real-Time Streaming Voice Computer Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py             # Launch live voice & interactive terminal console
  python main.py --demo      # Run end-to-end multimodal demonstration
  python main.py --audio-demo# Run Phase 3 audio capture and VAD demo
  python main.py --stt-demo  # Run Phase 4 streaming STT demo
  python main.py --test      # Run full pytest suite
        """,
    )
    parser.add_argument("--demo", action="store_true", help="Run automated end-to-end agent demonstration")
    parser.add_argument("--audio-demo", action="store_true", help="Run microphone capture & VAD demo")
    parser.add_argument("--stt-demo", action="store_true", help="Run streaming speech-to-text demo")
    parser.add_argument("--test", action="store_true", help="Run automated pytest suite")

    args = parser.parse_args()

    if args.test:
        import pytest
        return pytest.main(["-v", "tests"])

    if args.audio_demo:
        from app.audio.demo import run_audio_demo
        asyncio.run(run_audio_demo(duration_seconds=5))
        return 0

    if args.stt_demo:
        from app.intelligence.demo import run_stt_demo
        asyncio.run(run_stt_demo(duration_seconds=5))
        return 0

    if args.demo:
        asyncio.run(run_integrated_demo())
        return 0

    # Default: interactive console session
    asyncio.run(run_interactive())
    return 0


if __name__ == "__main__":
    sys.exit(main())
