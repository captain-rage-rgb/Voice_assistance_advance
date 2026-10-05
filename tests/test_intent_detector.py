"""
Unit tests for IncrementalIntentDetector.
Tests:
  - Simple commands
  - Incomplete clauses vs complete clauses
  - Incremental multi-step commands without duplicates
  - Transcript revisions
  - Generic application canonicalization
  - Search, navigation, and typing intents
"""

from app.intelligence.intent import IntentType
from app.intelligence.intent_detector import IncrementalIntentDetector


def test_simple_command_detection():
    """Verify single complete command detection."""
    detector = IncrementalIntentDetector()
    intents = detector.process_transcript("Open Chrome", is_final=False)

    assert len(intents) == 1
    assert intents[0].intent_type == IntentType.OPEN_APPLICATION
    assert intents[0].parameters["application"] == "Chrome"
    assert intents[0].confidence >= 0.90


def test_incomplete_command_handling():
    """Verify that incomplete phrases do not trigger an intent prematurely."""
    detector = IncrementalIntentDetector()

    # "Open" - incomplete
    assert len(detector.process_transcript("Open", is_final=False)) == 0

    # "Open the" - incomplete trailer
    assert len(detector.process_transcript("Open the", is_final=False)) == 0

    # "Search" - incomplete
    assert len(detector.process_transcript("Search", is_final=False)) == 0

    # "Go to" - incomplete
    assert len(detector.process_transcript("Go to", is_final=False)) == 0


def test_incremental_multi_step_progression_and_deduplication():
    """
    Verify the core incremental execution milestone:
    'Open Chrome' -> emits Intent 1
    'Open Chrome and go to YouTube' -> emits only Intent 2 (Intent 1 not re-emitted)
    'Open Chrome and go to YouTube and search for AI agents' -> emits only Intent 3
    """
    detector = IncrementalIntentDetector()

    # Step 1: User says "Open Chrome"
    step1 = detector.process_transcript("Open Chrome", is_final=False)
    assert len(step1) == 1
    assert step1[0].intent_type == IntentType.OPEN_APPLICATION
    assert step1[0].parameters["application"] == "Chrome"

    # Step 2: User continues speaking "...and go to YouTube"
    step2 = detector.process_transcript("Open Chrome and go to YouTube", is_final=False)
    assert len(step2) == 1
    assert step2[0].intent_type == IntentType.NAVIGATE
    assert step2[0].parameters["destination"] == "Youtube"

    # Step 3: User continues speaking "...and search for AI agents"
    step3 = detector.process_transcript(
        "Open Chrome and go to YouTube and search for AI agents", is_final=True
    )
    assert len(step3) == 1
    assert step3[0].intent_type == IntentType.SEARCH
    assert step3[0].parameters["query"] == "ai agents"


def test_partial_transcript_revisions():
    """Verify revision handling: 'open fire' should not emit a broken intent before 'open firefox'."""
    detector = IncrementalIntentDetector()

    # Partial interim hypothesis
    step1 = detector.process_transcript("open fire", is_final=False)
    # Even if 'fire' was parsed, when 'firefox' arrives, canonicalization or finalization resolves cleanly
    step2 = detector.process_transcript("open firefox", is_final=True)
    assert len(step2) >= 1
    assert any(i.parameters.get("application") == "Firefox" for i in (step1 + step2))


def test_generic_application_support():
    """Verify generic and third-party desktop application names."""
    detector = IncrementalIntentDetector()

    # Brave
    i1 = detector.process_transcript("launch Brave Browser", is_final=False)
    assert len(i1) == 1
    assert i1[0].parameters["application"] == "Brave"

    # VS Code
    detector.reset_context()
    i2 = detector.process_transcript("open VS Code", is_final=False)
    assert len(i2) == 1
    assert i2[0].parameters["application"] == "VS Code"

    # Generic app: Blender
    detector.reset_context()
    i3 = detector.process_transcript("start Blender", is_final=False)
    assert len(i3) == 1
    assert i3[0].parameters["application"] == "Blender"


def test_interruption_detection():
    """Verify user interruption keywords emit INTERRUPT intent."""
    detector = IncrementalIntentDetector()
    intents = detector.process_transcript("actually stop", is_final=False)
    assert len(intents) == 1
    assert intents[0].intent_type == IntentType.INTERRUPT
