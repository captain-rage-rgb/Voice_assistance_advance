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

**Phase 6 — Unified Interruption and Cancellation System (Complete)**

The agent's interruption and cancellation infrastructure is hardened with end-to-end propagation and race protection:
* **Unified Cancellation Primitive (`app/agent/cancellation.py`)**: Thread-safe, async-friendly `CancellationToken` tracking active requests/runs (`run_id`), supporting registered cooperative cancellation callbacks and idempotency (safe for repeated triggers).
* **Run Isolation & Stale Action Rejection (`app/agent/action.py`, `app/agent/action_queue.py`)**: `Action` models carry `run_id`. Stale actions from cancelled runs are automatically rejected upon submission, and `ActionQueue.clear_run(run_id)` discards queued actions without corrupting unrelated requests.
* **Cancellation-Aware Executor (`app/agent/executor.py`)**: `ActionExecutor` checks token cancellation at operation boundaries and supports cooperative async task cancellation.
* **Runtime Interruption Coordination (`app/agent/runtime.py`)**: Immediate interruption upon `interrupt()` or `INTERRUPT` intent events, cooperatively cancelling in-flight tasks, clearing pending queues, and discarding stale async results.
* **Minimal Floating Overlay Controller (`app/agent/overlay.py`)**: Minimal display state tracking (`Listening`, `Thinking`, `Executing`, `Stopped`) with an immediate, non-blocking STOP control wired to the unified cancellation token.

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
