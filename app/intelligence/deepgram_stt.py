"""
Deepgram streaming speech-to-text implementation using native WebSockets.
Streams 16-bit linear PCM audio and receives real-time interim and final transcripts.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import AsyncIterator, Optional
import websockets
from app.audio.chunk import AudioChunk
from app.intelligence.transcript import Transcript


class DeepgramStreamingSTT:
    """
    Real-time streaming speech recognition provider using Deepgram's WebSocket API.
    Handles connection management, audio chunk serialization, and transcript parsing.
    """

    BASE_URL = "wss://api.deepgram.com/v1/listen"

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        language: str = "en-US",
        sample_rate: int = 16000,
        channels: int = 1,
        endpointing_ms: int = 500,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.channels = channels
        self.endpointing_ms = endpointing_ms
        self.logger = logger or logging.getLogger("voice_agent.intelligence.deepgram")

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._is_connected = False
        self._connect_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Return True if WebSocket connection is open and active."""
        return self._is_connected and self._ws is not None and self._ws.open

    def _build_ws_url(self) -> str:
        """Construct Deepgram streaming WebSocket URL with query parameters."""
        query_params = [
            f"model={self.model}",
            f"language={self.language}",
            f"sample_rate={self.sample_rate}",
            f"channels={self.channels}",
            "encoding=linear16",
            "interim_results=true",
            f"endpointing={self.endpointing_ms}",
            "punctuate=true",
        ]
        return f"{self.BASE_URL}?{'&'.join(query_params)}"

    async def connect(self) -> None:
        """Connect to Deepgram streaming WebSocket endpoint."""
        async with self._connect_lock:
            if self.is_connected:
                return

            if not self.api_key or self.api_key.startswith("your_"):
                raise ValueError("Valid Deepgram API key is required to connect.")

            url = self._build_ws_url()
            headers = {"Authorization": f"Token {self.api_key}"}

            self.logger.info("Connecting to Deepgram streaming STT (%s, %s)...", self.model, self.language)
            try:
                self._ws = await websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                )
                self._is_connected = True
                self.logger.info("Deepgram streaming STT connection established.")
            except Exception as exc:
                self._is_connected = False
                self._ws = None
                self.logger.error("Deepgram connection failed: %s", exc)
                raise ConnectionError(f"Failed to connect to Deepgram: {exc}") from exc

    async def send_audio(self, chunk: AudioChunk) -> None:
        """Send raw linear PCM audio bytes to Deepgram."""
        if not self.is_connected or not self._ws:
            return

        try:
            await self._ws.send(chunk.data)
        except Exception as exc:
            self.logger.warning("Failed sending audio chunk to Deepgram: %s", exc)
            self._is_connected = False
            raise

    async def receive_transcripts(self) -> AsyncIterator[Transcript]:
        """
        Asynchronously receive and parse Deepgram WebSocket messages into Transcript models.
        """
        if not self._ws:
            return

        try:
            async for raw_message in self._ws:
                if isinstance(raw_message, bytes):
                    continue

                try:
                    payload = json.loads(raw_message)
                except json.JSONDecodeError:
                    continue

                # Deepgram response schema check
                channel = payload.get("channel", {})
                alternatives = channel.get("alternatives", [])
                if not alternatives:
                    continue

                top_alt = alternatives[0]
                transcript_text = top_alt.get("transcript", "")
                if not transcript_text:
                    continue

                is_final = bool(payload.get("is_final", False))
                confidence = top_alt.get("confidence")
                # speech_final indicates utterance completion if present
                segment_id = payload.get("metadata", {}).get("request_id")

                yield Transcript(
                    text=transcript_text,
                    is_final=is_final,
                    confidence=confidence,
                    segment_id=segment_id,
                    timestamp=datetime.now(timezone.utc),
                )

        except websockets.exceptions.ConnectionClosed as exc:
            self.logger.info("Deepgram WebSocket closed: code=%s, reason=%s", exc.code, exc.reason)
        except Exception as exc:
            self.logger.error("Deepgram receiver error: %s", exc)
            raise
        finally:
            self._is_connected = False

    async def disconnect(self) -> None:
        """Send CloseStream message and cleanly close WebSocket."""
        async with self._connect_lock:
            if not self._ws:
                self._is_connected = False
                return

            self.logger.info("Disconnecting Deepgram STT stream...")
            self._is_connected = False
            try:
                # Tell Deepgram to finish processing
                if self._ws.open:
                    await self._ws.send(json.dumps({"type": "CloseStream"}))
                    await asyncio.sleep(0.05)
                    await self._ws.close()
            except Exception as exc:
                self.logger.warning("Error during Deepgram disconnect: %s", exc)
            finally:
                self._ws = None
            self.logger.info("Deepgram STT disconnected.")
