"""
NEXUS AI v4.0 — Porcupine wake-word + Whisper transcription daemon.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Runs on Thread 4 (Audio/Wake Word Thread) as a dedicated daemon.
Listens for wake words via Porcupine, then records audio and
transcribes via faster-whisper. Non-blocking — communicates with
the UI and agent threads via queues.

Performance (i3 7th Gen):
  - Porcupine frame processing: ~2-5ms per 512-sample frame
  - Whisper base transcription (5s audio): ~1-3 seconds on CPU
  - Total voice → text latency: ~2-5 seconds
"""

import asyncio
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from functools import lru_cache

from nexus_config.settings import get_settings, APP_ROOT
from nexus_brain.service_lifecycle import ServiceBase

logger = logging.getLogger("nexus.wake_word")


# ─── Wake Word State ─────────────────────────────────────────────────────────

class WakeWordState(str, Enum):
    """States of the wake word daemon."""
    IDLE = "idle"                           # Listening for wake word
    WAKE_DETECTED = "wake_detected"         # Wake word heard, listening
    TRANSCRIBING = "transcribing"           # Processing audio via Whisper
    SPEAKING = "speaking"                   # NEXUS AI is speaking (silence detection off)
    DISABLED = "disabled"                   # Wake word disabled in settings
    ERROR = "error"                         # Something went wrong


@dataclass
class WakeWordEvent:
    """
    Event emitted by the wake word daemon.
    
    Fields:
        type: "wake" | "transcription" | "error" | "state_change"
        text: Transcribed text (for "transcription" events).
        state: Current daemon state.
        error: Error message (for "error" events).
        confidence: Wake word detection confidence.
    """
    type: str  # "wake" | "transcription" | "error" | "state_change"
    text: str = ""
    state: WakeWordState = WakeWordState.IDLE
    error: str = ""
    confidence: float = 0.0


# ─── Wake Word Daemon ───────────────────────────────────────────────────────

class WakeWordDaemon(ServiceBase):
    """
    Background daemon for wake-word-triggered voice input with lifecycle management.
    
    Architecture:
      - Thread 1 (this class): Audio capture + Porcupine detection loop
      - On wake: record audio → queue for Whisper → transcribe → emit result
    
    Usage:
        daemon = get_wake_word_daemon()
        async with daemon.lifespan():
            daemon.set_callback(lambda event: print(event.text))
            # daemon runs in background
    """
    
    def __init__(self):
        super().__init__("wake_word")
        self._settings = get_settings()
        
        # Callback for transcription results
        self._callback: Optional[Callable[[WakeWordEvent], None]] = None
        
        # Porcupine and recorder instances
        self._porcupine = None
        self._recorder = None
        self._whisper_model = None
        
        # Control flags
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Current state
        self._state = WakeWordState.DISABLED
        self._state_lock = threading.Lock()
        
        # Audio buffer (for recording after wake)
        self._audio_buffer: List[int] = []
        self._recording = False
        
        # Queues for async transcription
        self._transcription_queue: queue.Queue = queue.Queue()
        
        # Result event
        self._wake_detected_event = threading.Event()
        
        # Check if Porcupine key is available
        if not self._settings.PORCUPINE_ACCESS_KEY:
            logger.warning("No Porcupine access key. Wake word disabled. Set PORCUPINE_ACCESS_KEY in .env")
            self._state = WakeWordState.DISABLED
        elif not self._settings.ENABLE_WAKE_WORD:
            self._state = WakeWordState.DISABLED
        else:
            self._state = WakeWordState.IDLE
    
    def set_callback(self, callback: Callable[[WakeWordEvent], None]) -> None:
        """
        Set the callback for transcription results.
        
        Args:
            callback: Called with WakeWordEvent when wake word is detected
                     or transcription completes.
        """
        self._callback = callback
    
    # ── Service Lifecycle ───────────────────────────────────────────────────
    
    async def _start(self) -> None:
        """Start wake word daemon."""
        if self._state == WakeWordState.DISABLED:
            return
        
        if self._running:
            return
        
        try:
            self._init_porcupine()
        except Exception as e:
            logger.error("Failed to initialize Porcupine: %s", e)
            self._set_state(WakeWordState.ERROR)
            self._emit_event(WakeWordEvent(
                type="error",
                error=f"Porcupine init failed: {e}",
                state=WakeWordState.ERROR,
            ))
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="nexus-wake-word",
        )
        self._thread.start()
        
        # Start transcription worker thread
        threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="nexus-whisper-worker",
        ).start()
        
        logger.info("Wake word daemon started (words: %s)", self._settings.WAKE_WORDS)
    
    async def _stop(self) -> None:
        """Stop wake word daemon."""
        self._running = False
        self._wake_detected_event.set()  # Unblock any waiting
        
        if self._recorder:
            try:
                self._recorder.delete()
            except Exception:
                pass
            self._recorder = None
        
        if self._porcupine:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None
        
        self._set_state(WakeWordState.DISABLED)
        logger.info("Wake word daemon stopped.")
    
    async def _health_check(self) -> tuple[bool, str, Dict[str, Any]]:
        """Check wake word health."""
        if self._state == WakeWordState.DISABLED:
            return False, "Wake word disabled", {"enabled": False, "porcupine_key": False}
        
        if not self._settings.PORCUPINE_ACCESS_KEY:
            return False, "No Porcupine access key", {"porcupine_key": False}
        
        if not self._running:
            return False, "Daemon not running", {"running": False}
        
        # Check Porcupine import
        try:
            import pvporcupine
            return True, "Wake word healthy", {
                "running": True,
                "porcupine_key": True,
                "state": self._state.value,
            }
        except ImportError:
            return False, "pvporcupine not installed", {"pvporcupine": False}
        except Exception as e:
            return False, f"Health check error: {e}", {"running": self._running}
    
    # ── Public API (legacy sync methods for compatibility) ──────────────────
    
    def start(self) -> bool:
        """Start the wake word detection daemon (sync wrapper)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't await in running loop, start in background
                asyncio.create_task(self._start())
                return True
            else:
                return loop.run_until_complete(self._start()) is not False
        except Exception:
            return False
    
    def stop(self) -> None:
        """Stop the wake word daemon (sync wrapper)."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._stop())
            else:
                loop.run_until_complete(self._stop())
        except Exception:
            pass
    
    def trigger_manual(self) -> None:
        """
        Manually trigger voice input (hotkey fallback).
        
        Called when the user presses Ctrl+Shift+N or clicks the mic button.
        This bypasses wake word detection and starts recording immediately.
        """
        if self._state == WakeWordState.DISABLED:
            return
        
        logger.info("Manual wake triggered (hotkey)")
        self._start_recording()
    
    def is_listening(self) -> bool:
        """Check if the daemon is actively listening for wake words."""
        with self._state_lock:
            return self._state == WakeWordState.IDLE and self._running
    
    def get_state(self) -> WakeWordState:
        """Get the current daemon state."""
        with self._state_lock:
            return self._state
    
    # ── Internal ──────────────────────────────────────────────────────────
    
    def _set_state(self, state: WakeWordState) -> None:
        """Update daemon state and emit state change event."""
        with self._state_lock:
            old_state = self._state
            self._state = state
        
        if old_state != state:
            self._emit_event(WakeWordEvent(
                type="state_change",
                state=state,
            ))
    
    def _emit_event(self, event: WakeWordEvent) -> None:
        """Emit an event to the registered callback."""
        if self._callback:
            try:
                self._callback(event)
            except Exception as e:
                logger.error("Wake word callback error: %s", e)
    
    def _init_porcupine(self) -> None:
        """Initialize Porcupine wake word engine with configured keywords."""
        import pvporcupine
        
        keyword_paths = None
        sensitivities = [self._settings.WAKE_WORD_SENSITIVITY] * len(self._settings.WAKE_WORDS)
        
        self._porcupine = pvporcupine.create(
            access_key=self._settings.PORCUPINE_ACCESS_KEY,
            keywords=self._settings.WAKE_WORDS,
            sensitivities=sensitivities,
        )
        
        # Initialize recorder
        import pvrecorder
        self._recorder = pvrecorder.PvRecorder(
            frame_length=512,
            device_index=-1,  # Default mic
        )
        logger.debug("Porcupine initialized (frame_length=%d)", self._porcupine.frame_length)
    
    def _detection_loop(self) -> None:
        """
        Main detection loop — runs on background thread.
        
        Continuously reads audio frames and checks for wake words.
        On detection: records audio, queues for transcription.
        """
        if not self._porcupine or not self._recorder:
            return
        
        try:
            self._recorder.start()
            self._set_state(WakeWordState.IDLE)
            
            while self._running:
                # Read audio frame (32ms at 16kHz)
                frame = self._recorder.read()
                
                if self._recording:
                    # Collect audio for transcription
                    self._audio_buffer.extend(frame)
                    
                    # Check recording duration
                    elapsed = (time.time() - self._record_start_time)
                    if elapsed >= self._settings.WHISPER_RECORD_SECONDS:
                        self._stop_recording_and_transcribe()
                    continue
                
                # Check for wake word
                result = self._porcupine.process(frame)
                
                if result >= 0:
                    keyword_index = result
                    keyword = self._settings.WAKE_WORDS[keyword_index]
                    confidence = 0.95  # Porcupine doesn't give confidence scores
                    
                    logger.info("Wake word detected: '%s' (idx=%d)", keyword, keyword_index)
                    self._emit_event(WakeWordEvent(
                        type="wake",
                        text=keyword,
                        state=WakeWordState.WAKE_DETECTED,
                        confidence=confidence,
                    ))
                    
                    # Start recording
                    self._start_recording()
                
                # Small sleep to prevent busy-waiting on weak hardware
                if not self._recording:
                    time.sleep(0.005)  # 5ms = ~200Hz polling, fine for voice
        
        except Exception as e:
            logger.error("Wake word detection loop error: %s", e)
            self._set_state(WakeWordState.ERROR)
            self._emit_event(WakeWordEvent(
                type="error",
                error=f"Detection loop: {e}",
                state=WakeWordState.ERROR,
            ))
        finally:
            try:
                self._recorder.stop()
            except Exception:
                pass
    
    def _start_recording(self) -> None:
        """Start recording audio after wake word detection."""
        self._audio_buffer = []
        self._recording = True
        self._record_start_time = time.time()
        self._set_state(WakeWordState.WAKE_DETECTED)
        
        # Start a timeout thread (in case recording hangs)
        threading.Timer(
            self._settings.WHISPER_RECORD_SECONDS + 2.0,
            self._recording_timeout,
        ).start()
    
    def _recording_timeout(self) -> None:
        """Timeout handler — forces transcription if recording is stuck."""
        if self._recording:
            logger.warning("Recording timeout — forcing transcription")
            self._stop_recording_and_transcribe()
    
    def _stop_recording_and_transcribe(self) -> None:
        """Stop recording and queue audio for Whisper transcription."""
        if not self._recording:
            return
        
        self._recording = False
        self._set_state(WakeWordState.TRANSCRIBING)
        
        # Convert buffer to bytes
        audio_data = bytes(self._audio_buffer)
        
        # Queue for transcription
        self._transcription_queue.put(audio_data)
        
        logger.debug("Recording stopped (%d frames, %.1f seconds)",
            len(self._audio_buffer),
            len(self._audio_buffer) / 16000 if self._audio_buffer else 0)
    
    def _transcription_worker(self) -> None:
        """
        Background thread for Whisper transcription.
        
        Processes audio from the transcription queue. Whisper runs on CPU
        (int8 quantized) which takes ~1-3 seconds for 5s of audio on i3.
        """
        # Lazy-load Whisper model
        model = None
        
        while self._running:
            try:
                audio_data = self._transcription_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if audio_data is None:
                break
            
            try:
                if model is None:
                    model = self._load_whisper_model()
                
                if model is None:
                    self._emit_event(WakeWordEvent(
                        type="error",
                        error="Whisper model failed to load",
                        state=WakeWordState.ERROR,
                    ))
                    self._set_state(WakeWordState.IDLE)
                    continue
                
                # Transcribe
                logger.debug("Transcribing %d bytes of audio...", len(audio_data))
                segments, info = model.transcribe(
                    audio_data,
                    beam_size=1,               # Fastest (vs 5 for best quality)
                    best_of=1,
                    temperature=0.0,
                    vad_filter=True,            # Filter out silence
                    vad_parameters={"threshold": 0.5, "min_speech_duration_ms": 250},
                )
                
                text = " ".join(seg.text for seg in segments).strip()
                
                if text:
                    logger.info("Transcription: '%s'", text)
                    self._emit_event(WakeWordEvent(
                        type="transcription",
                        text=text,
                        state=WakeWordState.IDLE,
                    ))
                else:
                    logger.debug("No speech detected in audio")
                
                self._set_state(WakeWordState.IDLE)
                
            except Exception as e:
                logger.error("Transcription error: %s", e)
                self._emit_event(WakeWordEvent(
                    type="error",
                    error=f"Transcription: {e}",
                    state=WakeWordState.IDLE,
                ))
                self._set_state(WakeWordState.IDLE)
    
    def _load_whisper_model(self):
        """
        Lazy-load faster-whisper model.
        
        Model loading takes ~2-3 seconds for 'base' on i3.
        Only loads when first wake word is detected.
        """
        try:
            from faster_whisper import WhisperModel
            
            model_size = self._settings.WHISPER_MODEL_SIZE
            compute_type = self._settings.WHISPER_COMPUTE_TYPE
            device = self._settings.WHISPER_DEVICE
            
            logger.info("Loading Whisper model '%s' (%s, %s)...", model_size, device, compute_type)
            
            model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=2,  # i3 has 2 physical cores
                num_workers=1,
            )
            
            logger.info("Whisper model loaded.")
            return model
        
        except ImportError:
            logger.error("faster-whisper not installed. Install: pip install faster-whisper")
            return None
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)
            return None


@lru_cache(maxsize=1)
def get_wake_word_daemon() -> WakeWordDaemon:
    """
    Return the singleton WakeWordDaemon instance.
    
    Returns:
        WakeWordDaemon: The singleton daemon instance.
    """
    return WakeWordDaemon()
