"""
NEXUS AI v4.0 — Ring Overlay Core Integration
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Integrates the ring overlay with NEXUS core components:
- AgentOrchestrator: state transitions (idle -> thinking -> speaking)
- WakeWordDaemon: mic audio tap + listening state + voice input transcription
- TTSEngine: TTS audio tap + speaking state

All integration is non-blocking and follows the existing callback architecture.
The overlay schema is intentionally minimal — only "state" and "audio" messages
are pushed to the overlay.
"""

import asyncio
import logging
import threading
import time
from typing import Optional, Dict, Any
from functools import lru_cache

from nexus_overlay.config import get_overlay_config
from nexus_overlay.websocket_server import get_ws_server
from nexus_overlay.audio_tap import get_audio_tap_manager

logger = logging.getLogger("nexus.overlay.integration")


class OverlayIntegration:
    """
    Central integration point for the ring overlay.

    Wires together:
    - WebSocket server (core -> overlay)
    - Audio tap manager (mic/TTS -> FFT -> overlay)
    - Orchestrator state transitions
    - Wake word daemon (mic tap + listening/transcribing states)
    - TTS engine (speaking state + audio tap)

    All state pushes go through ws_server.send_state()/send_audio() which are
    non-blocking (thread-safe enqueue). The core ReAct loop is never blocked.
    """

    def __init__(self):
        self._config = get_overlay_config()
        self._ws_server = get_ws_server()
        self._audio_tap = get_audio_tap_manager()
        self._orchestrator = None
        self._wake_word_daemon = None
        self._tts_engine = None
        self._current_state = "idle"
        self._state_lock = threading.Lock()
        self._monitor_thread = None
        self._monitor_running = False

    def initialize(self, orchestrator=None, wake_word_daemon=None,
                   tts_engine=None) -> bool:
        """
        Initialize integration with core components.

        Called after components are created but before they are started.
        Note: the wake-word recorder may not exist yet at this point (it is
        created lazily inside WakeWordDaemon.start()), so the mic-frame tap
        is deferred to start() where it is hooked lazily.
        """
        self._orchestrator = orchestrator
        self._wake_word_daemon = wake_word_daemon
        self._tts_engine = tts_engine

        if wake_word_daemon:
            self._audio_tap.set_wake_word_daemon(wake_word_daemon)
            self._hook_wake_word_callback()

        if tts_engine:
            self._audio_tap.set_tts_engine(tts_engine)
            self._hook_tts_engine()

        # Wire orchestrator state changes to overlay
        if orchestrator:
            self._hook_orchestrator()

        # Wire websocket input callback to process text from overlay
        self._ws_server.set_input_callback(self._process_text_input)

        logger.info("Overlay integration initialized")
        return True

    def start(self) -> bool:
        """Start all overlay services."""
        success = True

        if not self._ws_server.start():
            logger.warning("Failed to start WebSocket server")
            success = False

        if not self._audio_tap.start():
            logger.warning("Failed to start audio tap manager")
            success = False

        # Lazily hook mic capture now that the daemon thread may have started
        # and created its recorder.
        if self._wake_word_daemon:
            self._audio_tap.set_wake_word_daemon(self._wake_word_daemon)
            self._hook_mic_capture()

        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="nexus-overlay-monitor",
        )
        self._monitor_thread.start()

        self.set_state("idle")
        logger.info("Overlay integration started")
        return success

    def stop(self) -> None:
        """Stop all overlay services."""
        self._monitor_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        self._audio_tap.stop()
        self._ws_server.stop()
        logger.info("Overlay integration stopped")

    def set_state(self, state: str) -> None:
        """Set overlay state (thread-safe). Delegated to ws_server.send_state."""
        with self._state_lock:
            if state == self._current_state:
                return
            self._current_state = state
        self._ws_server.send_state(state)
        logger.debug("Overlay state: %s", state)

    
    def _hook_orchestrator(self) -> None:
        """
        Hook into orchestrator to send state updates to overlay.
        
        The orchestrator runs the ReAct loop. We want to send:
        - "thinking" when the agent starts reasoning
        - "speaking" when TTS starts (already handled by TTS hook)
        - "idle" when the cycle completes
        """
        if not self._orchestrator:
            return

        # We'll patch the orchestrator's run method to inject state updates
        original_run = self._orchestrator.run

        async def wrapped_run(user_input: str, session_context: Optional[Dict[str, Any]] = None):
            # Signal thinking state when processing starts
            self.set_state("thinking")
            try:
                result = await original_run(user_input, session_context)
                # On success, return to idle
                self.set_state("idle")
                return result
            except Exception as e:
                logger.error("Orchestrator run error: %s", e)
                self.set_state("error")
                raise

        self._orchestrator.run = wrapped_run
        logger.debug("Orchestrator hooked for overlay state updates")

    def _hook_wake_word_callback(self) -> None:
        """
        Hook the wake-word daemon event callback to translate wake-word
        events into overlay state transitions. The recorder is NOT tapped
        here (it may not exist yet); mic-frame tapping happens in
        _hook_mic_capture called from start().
        """
        if not self._wake_word_daemon:
            return

        original_callback = self._wake_word_daemon._callback

        def combined_callback(event):
            # Translate wake-word events to ring states:
            #   WAKE_DETECTED -> listening (mic is live)
            #   TRANSCRIBING  -> thinking (Whisper is processing)
            #   The orchestrator will take over as thinking/speaking next.
            from nexus_audio.wake_word import WakeWordState

            if event.type == "wake":
                self.set_state("listening")
            elif event.type == "transcription":
                text = getattr(event, "text", "").strip()
                if text:
                    logger.info("Wake word transcription: %s", text[:80])
                    self._process_text_input(text)
            elif event.type == "state_change":
                if event.state == WakeWordState.IDLE:
                    self.set_state("idle")
                elif event.state == WakeWordState.WAKE_DETECTED:
                    self.set_state("listening")
                elif event.state == WakeWordState.TRANSCRIBING:
                    self.set_state("thinking")
                elif event.state == WakeWordState.ERROR:
                    self.set_state("error")

            if original_callback:
                try:
                    original_callback(event)
                except Exception as e:
                    logger.debug("Original wake word callback error: %s", e)

        self._wake_word_daemon.set_callback(combined_callback)
        logger.debug("Wake word callback hooked for overlay state updates")


    def _hook_mic_capture(self) -> None:
        """
        Hook the wake-word daemon's microphone recorder to push audio frames
        to the overlay's audio tap.

        The wake-word daemon creates its PvRecorder lazily in start().
        We wrap the recorder's read() method so every frame read for wake-word
        detection is also pushed to the audio tap (which computes FFT bins
        and sends them to the overlay over WebSocket).

        This is the mic -> overlay audio path. No new microphone stream is
        opened; we tap the existing one used by the wake-word engine.
        """
        if not self._wake_word_daemon:
            return

        daemon = self._wake_word_daemon

        # The recorder is created lazily in daemon.start(). Poll for it.
        recorder = None
        for _ in range(30):  # up to 3 seconds (100ms polling)
            recorder = getattr(daemon, "_recorder", None)
            if recorder is not None:
                break
            time.sleep(0.1)

        if recorder is None:
            logger.warning("Overlay: wake-word recorder not found, mic tap skipped")
            return

        if not hasattr(recorder, "read"):
            logger.warning("Overlay: recorder has no read() method, mic tap skipped")
            return

        # Save original read method
        original_read = recorder.read

        def tapped_read():
            """Wrapped read that pushes frames to the audio tap."""
            try:
                frame = original_read()
                if frame:
                    self._audio_tap.push_mic_frame(frame)
                return frame
            except Exception as e:
                logger.debug("Overlay: mic frame tap error: %s", e)
                try:
                    return original_read()
                except Exception:
                    return None

        # Replace the recorder's read method with our wrapped version
        recorder.read = tapped_read
        logger.debug("Overlay: mic capture hooked (recorder.read wrapped)")


    def _hook_tts_engine(self) -> None:

        """
        Install a TTS audio-tap callback into the TTS engine so every spoken
        utterance pushes PCM bins to the overlay (speaking state is set by
        _broadcast_orchestrator_cycle when TTS starts).

        NOTE: the TTS engine's say() already reads a module-level
        `_tts_audio_tap_callback` (tts_engine.py:253:
        `if pcm_data and _tts_audio_tap_callback:`). We patch that attribute on
        the engine instance so decoded PCM flows into the audio tap. Wired after
        tts_engine is created and before start().
        """

        if not self._tts_engine:
            return
        self._tts_engine._tts_audio_tap_callback = self._tts_frame_pushed


    def _tts_frame_pushed(self, pcm_data: bytes) -> None:
        """Called back by the TTS engine each time an utterance's PCM is ready."""
        self._broadcast_audio("tts", pcm_data)


    def _monitor_loop(self) -> None:

        """
        Keep-alive monitor: re-broadcast the overlay's current state every ~5s so a
        late-reconnecting overlay never shows a stale frame (e.g. stuck on
        listening after the wake-word daemon reset to IDLE).
        """

        while self._monitor_running:
            try:
                self._ws_server.send_state(self._current_state)
            except Exception as e:
                logger.debug("Monitor state push error: %s", e)
            time.sleep(5.0)


    async def _process_text_input(self, text: str) -> str:

        """
        Route transcribed (or typed) text through the orchestrator. The
        orchestrator drives the thinking->speaking cycle; the overlay just
        reflects whatever the orchestrator reports.
        Returns the final response text.
        """

        if not self._orchestrator:
            return ""
        try:
            # Orchestrator uses run() method
            result = await self._orchestrator.run(text)
            return result.get("final_response", "")
        except Exception as e:
            logger.warning("Overlay: submit text to orchestrator failed: %s", e)
            self._ws_server.send_state("error")
            return ""

import enum as _enum
from functools import lru_cache as _lru_cache

class OverlayState(_enum.Enum):
    """Enum of the five ring states — single source of truth for state values."""
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"

@_lru_cache(maxsize=1)
def get_overlay_integration() -> OverlayIntegration:
    """Get or create the OverlayIntegration singleton."""
    return OverlayIntegration()



