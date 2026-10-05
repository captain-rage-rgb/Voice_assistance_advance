# Real-Time Voice Computer Agent

A real-time, streaming, interruptible voice-controlled computer agent inspired by a Jarvis-style desktop assistant.

## Core Differentiator

**The system is designed to execute recognized intents incrementally while the user is still speaking, rather than waiting for the complete utterance.**

Instead of recording a complete turn, stopping, running speech-to-text, querying an LLM, and only then acting, this architecture continuously listens to streaming audio, receives partial transcripts, recognizes complete sub-intents as they emerge, and begins executing deterministic computer actions in real time.

---

## Planned Capabilities (Future Phases)

* **Streaming Speech Recognition & VAD:** Continuous microphone audio capture with streaming partial and final transcripts.
* **Incremental Intent Detection:** Early intent resolution on partial transcripts to dispatch actions while speech is ongoing.
* **LLM Reasoning & Tool Calling:** Structured tool generation separating reasoning (*what to do*) from execution (*how to do it*).
* **Deterministic Computer Control:** Application management, browser automation (Playwright), window management, file operations, keyboard, and mouse automation on Windows.
* **Action Queue & Execution Engine:** Priority queueing, ordering, concurrency, and error handling.
* **First-Class Interruption & Cancellation:** Fast reaction to cancellation phrases ("Stop", "Cancel", "Wait", "Actually, open Firefox instead") to abort running and pending actions.
* **State Management & Safety:** Full awareness of active application, execution state, pending actions, with safety validation checks before running critical operations.

> **Note:** These are planned capabilities for upcoming phases and are **not** implemented in Phase 1.

---

## Current Status

**Phase 7 & 8 — Context-Aware Planning, Incremental Execution, Verification & Bounded Recovery (Complete)**

The agent acts as a continuous, context-aware computer operator with post-action verification and bounded recovery:
* **Lightweight Task Context (`app/agent/context.py`)**: Bounded execution context tracking current run ID, original intent, current task objective, completed actions with structured outcomes (`ExecutionOutcome`, `ActionResult`), and observations between meaningful steps without unbounded memory growth.
* **Context-Aware Deduplication (`app/agent/context.py`)**: Prevents re-executing already completed actions when streaming or multi-step models emit similar steps later.
* **Pre-Execution Action Validation (`app/agent/validator.py`)**: Strict schema, parameter, and safety validation ensuring malformed or unsupported action fragments never reach OS-level executors.
* **Post-Action Verification & Self-Correction Engine (`app/agent/verifier.py`)**: Verifies actual computer state changes following meaningful actions rather than fabricating success based on raw API return codes.
* **Bounded Self-Correction Recovery (`app/agent/runtime.py`)**: Automatic, bounded recovery loop when verification fails, adapting recovery attempts up to configured step and run limits, with cancellation always taking precedence.
* **Clarification Handling**: Concise clarification requests for ambiguous input, cleanly yielding control back to listening mode.
* **Overlay Lifecycle Feedback (`app/agent/overlay.py`)**: Minimal top overlay states (`Listening`, `Thinking`, `Executing`, `Verifying`, `Recovering`, `Stopped`) with immediate, non-blocking STOP button responsiveness throughout all phases.

---

## Getting Started

### Prerequisites

* Python 3.10+ (tested on Python 3.12 on Windows)

### 1. Set Up Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` (optional for defaults):

```powershell
copy .env.example .env
```

### 4. Run the Application

```powershell
python main.py
```

### 5. Run the Test Suite

```powershell
pytest
```

---

## Project Structure

```text
Voice_assistance_advance/
│
├── app/
│   ├── __init__.py
│   ├── audio/            # Streaming audio capture & VAD (future)
│   ├── agent/            # Action queue & execution engine (future)
│   ├── intelligence/     # STT & LLM intent detection (future)
│   ├── tools/            # Deterministic computer/browser tools (future)
│   ├── state/            # Execution & application state tracking (future)
│   ├── safety/           # Policy checks & confirmation guards (future)
│   ├── config/           # Pydantic settings & configuration
│   └── utils/            # Logging and general utilities
│
├── tests/                # Unit tests
│   ├── test_config.py
│   └── test_logging.py
│
├── docs/                 # Architecture notes & documentation
│   └── README.md
│
├── .env.example          # Environment variable template
├── .gitignore            # Git exclusion rules
├── pyproject.toml        # Project metadata & build tool configuration
├── requirements.txt      # Minimal pinned dependencies
├── README.md             # Project documentation
└── main.py               # Minimal application entry point
```
