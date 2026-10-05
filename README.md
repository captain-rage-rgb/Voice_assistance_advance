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

**Phase 9 & 10 — Long-Running Tasks, Conversational Continuity, Production Hardening & Reliability (Complete)**

The agent is hardened for extended real-world use with long-running task support and resilient lifecycle controls:
* **Unified Run Manager (`app/agent/run_manager.py`)**: Authoritative coordinator managing active run lifecycles, generation tracking, central stale-result protection, run replacement, and history eviction to prevent memory leaks.
* **Persistent Task Context & Milestones (`app/agent/context.py`)**: Structured `TaskMilestone` tracking for multi-stage operations, enabling recovery from the latest valid milestone without replaying from the beginning.
* **Conversational Continuity & Reference Resolution (`app/agent/context.py`, `app/agent/runtime.py`)**: Seamless follow-up commands resolving anaphoric references ("that", "it", "the browser", "do that again", "the cheapest one") against bounded conversation context without bypassing action validation.
* **Pause versus Cancellation (`app/agent/runtime.py`, `app/agent/run_manager.py`)**: Clean distinction between `STOP/CANCEL` (authoritative termination) and `PAUSE/WAIT` (temporarily pausing actions while retaining context; resumed with "continue").
* **Subsystem Error Boundaries & Health Monitoring (`app/agent/health.py`)**: Non-crashing error boundaries across all subsystems with diagnostic internal health reporting (`HealthMonitor`).
* **Bounded Timeouts**: Action execution (`settings.action_timeout_seconds`), verification (`settings.verification_timeout_seconds`), and task timeouts with bounded failure/recovery paths.
* **Graceful Shutdown**: Rapid, bounded cancellation of active runs, clearing of queues, stopping of worker loops, and release of all audio and model resources.
* **Overlay Lifecycle Hardening (`app/agent/overlay.py`)**: Minimal top overlay states (`Listening`, `Thinking`, `Executing`, `Verifying`, `Recovering`, `Paused`, `Stopped`) with always-responsive STOP control throughout.

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
