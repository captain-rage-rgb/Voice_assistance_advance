"""
Microphone capture abstraction and sounddevice implementation.
Provides an async stream interface while isolating internal PortAudio threading.
"""

import asyncio
import logging
from typing import Optional, Protocol
import sounddevice as sd
from app.audio.chunk import AudioChunk


class AudioCapture(Protocol):
    """Protocol defining audio capture interface."""

    async def start(self) -> None:
        """Start capturing audio."""
        ...

    async def stop(self) -> None:
        """Stop capturing audio."""
        ...

    @property
    def is_running(self) -> bool:
        """Check if audio capture is active."""
        ...


class MicrophoneCapture:
    """
    Continuous microphone stream capture using sounddevice (PortAudio).
    Captures 16-bit linear PCM audio in discrete chunks and transfers them
    thread-safely into an asyncio.Queue without blocking the event loop.
    """

    def __init__(
        self,
        output_queue: asyncio.Queue[AudioChunk],
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        device_index: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.device_index = device_index
        self.logger = logger or logging.getLogger("voice_agent.audio.capture")

        # Number of samples per chunk: e.g., 16000 * 30 / 1000 = 480 samples
        self.blocksize = int(self.sample_rate * self.chunk_duration_ms / 1000)

        self._stream: Optional[sd.RawInputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Return True if capture stream is currently active."""
        return self._is_running

    def _audio_callback(self, indata: bytes, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """
        Low-level PortAudio callback executing on an internal sounddevice audio thread.
        Thread-safely forwards raw PCM bytes to asyncio event loop.
        """
        if status:
            self.logger.warning("Audio input status flag: %s", status)

        if not self._is_running or not self._loop:
            return

        chunk = AudioChunk(
            data=bytes(indata),
            sample_rate=self.sample_rate,
            channels=self.channels,
            duration_ms=self.chunk_duration_ms,
        )

        def _enqueue():
            try:
                self.output_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                # Discard oldest chunk if queue is full to avoid memory leaks
                try:
                    self.output_queue.get_nowait()
                    self.output_queue.put_nowait(chunk)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

        # Thread-safe schedule onto asyncio loop
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(_enqueue)

    async def start(self) -> None:
        """Initialize and open the microphone stream."""
        if self._is_running:
            self.logger.warning("Microphone is already running.")
            return

        self._loop = asyncio.get_running_loop()

        try:
            self.logger.info(
                "Opening audio stream (device: %s, rate: %d Hz, chunk: %d ms, blocksize: %d)...",
                self.device_index if self.device_index is not None else "default",
                self.sample_rate,
                self.chunk_duration_ms,
                self.blocksize,
            )
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.blocksize,
                device=self.device_index,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_running = True
            self.logger.info("Microphone stream successfully started.")
        except Exception as exc:
            self._is_running = False
            self.logger.error("Failed to start microphone stream: %s", exc)
            raise RuntimeError(f"Microphone initialization failed: {exc}") from exc

    async def stop(self) -> None:
        """Stop and close the microphone stream."""
        if not self._is_running:
            return

        self.logger.info("Stopping microphone stream...")
        self._is_running = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                self.logger.warning("Error closing audio stream: %s", exc)
            finally:
                self._stream = None

        self.logger.info("Microphone stream stopped.")
