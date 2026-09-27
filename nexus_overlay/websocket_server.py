"""
NEXUS AI v4.0 - Ring Overlay WebSocket Server
Hardware: Intel i3 7th Gen - 12GB RAM - Ollama + Groq API

WebSocket server running in the NEXUS core process. The overlay (separate
pywebview process) connects as a client.

Message schema (core -> overlay) - intentionally minimal:
  {"type": "state", "value": "idle"|"listening"|"thinking"|"speaking"|"error"}
  {"type": "audio", "source": "mic"|"tts", "level": 0.0-1.0, "bins": [0.0-1.0, ...]}

The overlay is a passive visual surface: it only receives state + audio and
never sends control frames back, so the schema stays tiny and the core ReAct
loop is never touched by inbound traffic.

Threading model:
  * The server (bind + dispatch) runs on a dedicated daemon thread with its own
    asyncio event loop.
  * Sends from the core / ReAct thread are enqueued with
    loop.call_soon_threadsafe(...) and drained by a dispatch task on the
    server loop. This is the only safe way to touch an asyncio.Queue from a
    thread that is not its owner; a bare put_nowait from the ReAct thread can
    set a Future's result on the wrong thread ("Future attached to a different
    loop").
  * If no overlay is connected yet the frame is buffered/dropped - the core
    never blocks or crashes. On (re)connect the server re-pushes current state.
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Optional, Set, Callable
from functools import lru_cache

from websockets.asyncio.server import serve as ws_serve
from websockets.exceptions import ConnectionClosed

from nexus_overlay.config import get_overlay_config

logger = logging.getLogger("nexus.overlay.ws")

# Single source of truth for the core->overlay "state" value. No other states
# are ever sent, keeping the wire schema minimal.
VALID_STATES = ("idle", "listening", "thinking", "speaking", "error")


@dataclass
class OverlayMessage:
    """A frame sent to the overlay. Schema is intentionally minimal (state|audio)."""
    type: str  # "state" | "audio"
    value: Optional[str] = None       # state: idle|listening|thinking|speaking|error
    source: Optional[str] = None      # audio: mic|tts
    level: Optional[float] = None     # audio: 0.0-1.0
    bins: Optional[list] = None       # audio: list of 0.0-1.0 floats

    def to_json(self) -> str:
        if self.type == "state":
            return json.dumps({"type": "state", "value": self.value})
        if self.type == "audio":
            return json.dumps({
                "type": "audio",
                "source": self.source,
                "level": self.level,
                "bins": self.bins,
            })
        return json.dumps({"type": self.type})


class OverlayWebSocketServer:
    """WebSocket server for overlay communication (runs inside the NEXUS core)."""

    # Max pending frames buffered ahead of a slow/disconnected overlay. Bounded so
    # a stuck overlay can't grow RAM unbounded on an i3.
    _QUEUE_MAXSIZE = 256

    def __init__(self):
        self._config = get_overlay_config()
        self._server = None
        self._clients: Set = set()
        self._clients_lock = threading.Lock()
        self._running = False
        self._stopped = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        # Created inside the server's own event loop (see _run_server) so the
        # asyncio.Queue is always owned by the loop that drains it.
        self._send_queue: Optional[asyncio.Queue] = None
        # State debouncing (anti-flapping - never flicker < ~150ms in a state)
        self._last_state: Optional[str] = None
        self._last_state_time: float = 0.0
        # Audio frame rate limiting
        self._last_audio_send: float = 0.0
        self._audio_send_interval: float = 1.0 / 30.0  # Max 30 fps for audio frames
        # Last known state - re-pushed to a (re)connecting overlay so it never
        # lands on a stale frame.
        self._current_state: str = "idle"
# Diagnostics callbacks (optional, for logging/debugging only)
        self._state_change_callback: Optional[Callable[[str], None]] = None
        self._audio_frame_callback: Optional[Callable[[str, float, list], None]] = None
        self._input_callback: Optional[Callable[[str], None]] = None

    def set_state_callback(self, callback: Callable[[str], None]) -> None:
        self._state_change_callback = callback

    def set_audio_callback(self, callback: Callable[[str, float, list], None]) -> None:
        """Set callback for audio frames (for logging/debugging only)."""
        self._audio_frame_callback = callback

    # - Enqueue (called from the core / ReAct thread) ----------------------------

    def _enqueue(self, message: str) -> bool:
        """
        Enqueue a frame for broadcast to all connected overlays.

        Non-blocking and safe to call from any thread (including the ReAct loop
        thread). Uses call_soon_threadsafe to touch the asyncio.Queue only from
        its owning loop. Returns False if there is nowhere to enqueue yet
        (server not up / loop not running) - the message is dropped, never
        blocking the caller.
        """
        loop = self._loop
        if loop is None or not loop.is_running() or self._send_queue is None:
            return False
        queue_ref = self._send_queue

        def _put() -> None:
            try:
                queue_ref.put_nowait(message)
            except asyncio.QueueFull:
                pass  # backpressure: drop the tail frame (audio is rate-limited)

        try:
            loop.call_soon_threadsafe(_put)
            return True
        except RuntimeError:
            return False  # loop closed while scheduling

    def _normalize_state(self, state: str) -> str:
        """Validate/normalize a state value; returns '' if invalid."""
        if state not in VALID_STATES:
            logger.warning("Ignoring invalid state '%s' (expected %s)", state, VALID_STATES)
            return ""
        return state

    def send_state(self, state: str) -> None:
        """
        Push a state change to the overlay (non-blocking, debounced).

        Debouncing: the same state is not sent twice, and a transition lasting
        less than state_debounce_ms (default 150ms) is suppressed so the ring
        never flickers between modes.
        """
        state = self._normalize_state(state)
        if not state:
            return

        now = time.time()
        # Never send the same state twice in a row.
        if state == self._last_state:
            return
        # Suppress rapid flapping below the debounce window.
        if now - self._last_state_time < self._config.state_debounce_ms / 1000.0:
            return

        self._last_state = state
        self._last_state_time = now
        self._current_state = state
        self._enqueue(OverlayMessage(type="state", value=state).to_json())

        if self._state_change_callback:
            try:
                self._state_change_callback(state)
            except Exception:
                pass

    def send_audio(self, source: str, level: float, bins: list) -> None:
        """
        Push an audio frame to the overlay (non-blocking, rate-limited to 30 FPS).

        Args:
            source: "mic" or "tts"
            level:  Overall RMS level 0.0-1.0
            bins:   FFT magnitude bins (already 0.0-1.0), truncated to config.fft_bins
        """
        if source not in ("mic", "tts"):
            logger.warning("Ignoring audio from unknown source '%s'", source)
            return

        now = time.time()
        if now - self._last_audio_send < self._audio_send_interval:
            return  # rate limit; the ring would skip frames anyway
        self._last_audio_send = now

        level = max(0.0, min(1.0, float(level)))
        bins = [max(0.0, min(1.0, float(b))) for b in bins[: self._config.fft_bins]]

        self._enqueue(OverlayMessage(
            type="audio", source=source, level=level, bins=bins,
        ).to_json())

        if self._audio_frame_callback:
            try:
                self._audio_frame_callback(source, level, bins)
            except Exception:
                pass

    @property
    def connected_clients(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    @property
    def is_running(self) -> bool:
        return self._running and not self._stopped

    # - Lifecycle ---------------------------------------------------------------

    def start(self) -> bool:
        """Start the WebSocket server on a background thread. Returns True on success."""
        if self._running:
            return True
        self._stopped = False
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True,
                                        name="nexus-overlay-ws")
        self._thread.start()
        # Give the server thread a moment to bind the socket so very early sends
        # (e.g. initial IDLE state) are buffered instead of dropped. Still
        # non-blocking for the core: this is a fixed 150ms sleep, not a wait on
        # the overlay connecting.
        time.sleep(0.15)
        alive = self._thread.is_alive()
        if alive:
            logger.info("Overlay WebSocket server on ws://%s:%d%s",
                        self._config.ws_host, self._config.ws_port, self._config.ws_path)
        else:
            logger.error("Overlay WebSocket server thread died during startup")
        return alive

    def stop(self) -> None:
        """Stop the WebSocket server and release the socket."""
        if not self._running:
            return
        self._stopped = True
        self._running = False
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._do_stop(), loop)
                future.result(timeout=3.0)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._clients_lock:
            self._clients.clear()
        logger.info("Overlay WebSocket server stopped")

    async def _do_stop(self) -> None:
        """Runs on the server loop: close the socket then stop the loop."""
        try:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
        except Exception:
            pass
        # Stop the loop that is running _serve(); serve_forever() will exit.
        if self._loop is not None:
            self._loop.stop()

    def _run_server(self) -> None:
        """Background thread: own event loop + bind + serve + dispatch."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # The queue must be created on the loop that owns/destroys it.
        self._send_queue = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except Exception as e:  # pragma: no cover - defensive
            if not self._stopped:
                logger.error("Overlay WS server error: %s", e)
        finally:
            self._loop = None
            self._send_queue = None
            try:
                loop.close()
            except Exception:
                pass

    async def _serve(self) -> None:
        """Bind the socket and run the connection handler + message dispatcher."""
        self._server = await ws_serve(
            self._handle_connection,
            self._config.ws_host,
            self._config.ws_port,
            # The overlay is a thin passive client; we don't need the server's
            # keepalive ping/pong chatter consuming CPU on an i3.
            ping_interval=None,
            ping_timeout=None,
        )
# Server is bound & accepting. (start() already slept briefly.)
        dispatch = asyncio.create_task(self._drain_queue())
        try:
            await self._server.serve_forever()
        finally:
            dispatch.cancel()
            try:
                await dispatch
            except (asyncio.CancelledError, Exception):
                pass

    async def _handle_connection(self, websocket) -> None:
        """Accept an overlay client: re-sync state, then handle incoming messages."""
        # Check path (overlay connects to /nexus/overlay)
        path = getattr(websocket.request, 'path', '/')
        if path != self._config.ws_path and path != '/':
            logger.debug("Overlay connection rejected: wrong path %s", path)
            await websocket.close()
            return
            
        with self._clients_lock:
            self._clients.add(websocket)
        logger.info("Overlay connected (%d client(s))", len(self._clients))

        # Re-push the current state so a late/reconnecting overlay never
        # shows a stale frame (e.g. still "thinking" after the agent finished).
        self._enqueue(OverlayMessage(type="state", value=self._current_state).to_json())

        try:
            async for message in websocket:
                await self._handle_incoming_message(websocket, message)
        except ConnectionClosed:
            pass
        except Exception as e:
            logger.debug("Overlay connection error: %s", e)
        finally:
            with self._clients_lock:
                self._clients.discard(websocket)
            logger.info("Overlay disconnected (%d client(s) left)", len(self._clients))

    async def _handle_incoming_message(self, websocket, message: str) -> None:
        """Handle incoming message from overlay (text input, voice input, etc.)."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Overlay sent invalid JSON: %s", message[:100])
            return

        msg_type = data.get("type")

        if msg_type == "input":
            text = data.get("text", "").strip()
            if text:
                logger.debug("Overlay text input: %s", text[:80])
                # Notify integration to process text and await response
                if self._input_callback:
                    try:
                        response = await self._input_callback(text)
                        if response:
                            await self._send_to_client(websocket, {
                                "type": "response",
                                "text": response
                            })
                    except Exception as e:
                        logger.error("Input callback error: %s", e)

        elif msg_type == "voice_input":
            # Voice input is handled by the wake-word daemon, not here
            # The overlay sends base64 audio which the core would need to process
            logger.debug("Overlay voice input received (not handled by WS server)")

        elif msg_type == "get_config":
            # Client requesting config - already sent on connect via injection
            pass

        elif msg_type == "ping":
            # Heartbeat - respond with pong
            self._enqueue(json.dumps({"type": "pong", "ts": data.get("ts")}))

        else:
            logger.debug("Overlay unknown message type: %s", msg_type)

    async def _send_to_client(self, websocket, message: dict) -> None:
        """Send a message to a specific client."""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.debug("Failed to send to client: %s", e)

    def set_input_callback(self, callback: Callable[[str], Awaitable[str]]) -> None:
        """Set async callback for text input from overlay. Returns response."""
        self._input_callback = callback

    async def _drain_queue(self) -> None:
        """Background task: pull enqueued frames and broadcast to all clients."""
        while self._running and not self._stopped:
            try:
                message = await asyncio.wait_for(self._send_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if message is None:
                break
            await self._broadcast(message)

    async def _broadcast(self, message: str) -> None:
        """Send a frame to every connected overlay; reap dead connections."""
        if not self._clients:
            # No overlay connected - frame is just dropped here (already buffered/
            # dropped on queue overflow). Core never blocks on this.
            return
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(message)
            except (ConnectionClosed, Exception):
                dead.append(ws)
        if dead:
            with self._clients_lock:
                for ws in dead:
                    self._clients.discard(ws)


@lru_cache(maxsize=1)
def get_ws_server() -> OverlayWebSocketServer:
    """Get the singleton OverlayWebSocketServer instance."""
    return OverlayWebSocketServer()
