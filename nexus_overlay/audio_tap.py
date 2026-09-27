"""
NEXUS AI v4.0 — Ring Overlay Audio Tap Manager
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Taps into existing audio pipelines without opening new streams:
- Mic: Taps the Porcupine recorder's frame buffer (already running for wake word)
- TTS: Taps pygame mixer's playback buffer (already playing TTS)

Computes FFT bins on the core side and pushes to WebSocket server.
No new audio devices opened, no additional latency added to pipelines.
"""

import logging
import threading
import time
from collections import deque
from typing import Optional, Callable, List
from functools import lru_cache

import numpy as np

from nexus_overlay.config import get_overlay_config
from nexus_overlay.websocket_server import get_ws_server

logger = logging.getLogger("nexus.overlay.audio")


class AudioTapManager:
    """
    Manages audio taps for mic and TTS streams.
    
    Taps existing audio pipelines:
    - Mic: Reads from Porcupine recorder's frame buffer (via wake word daemon)
    - TTS: Monitors pygame mixer playback (via TTSEngine)
    
    Computes FFT bins and sends to overlay via WebSocket.
    Runs on a dedicated thread to avoid blocking audio pipelines.
    """
    
    def __init__(self):
        self._config = get_overlay_config()
        self._ws_server = get_ws_server()
        
        # Audio buffers (ring buffers for rolling FFT)
        self._mic_buffer: deque = deque(maxlen=int(self._config.sample_rate * self._config.audio_buffer_seconds))
        self._tts_buffer: deque = deque(maxlen=int(self._config.sample_rate * self._config.audio_buffer_seconds))
        
        # Buffer locks
        self._mic_lock = threading.Lock()
        self._tts_lock = threading.Lock()
        
        # FFT state
        self._fft_size = self._config.fft_size
        self._bins = self._config.fft_bins
        
        # Window function for FFT
        self._window = np.hanning(self._fft_size)
        
        # Control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Source references (set by core during init)
        self._wake_word_daemon = None
        self._tts_engine = None
        
        # Last sent values for change detection
        self._last_mic_level = 0.0
        self._last_tts_level = 0.0
        self._last_mic_bins: Optional[List[float]] = None
        self._last_tts_bins: Optional[List[float]] = None
        
        # Callbacks for integration
        self._mic_frame_callback: Optional[Callable[[bytes], None]] = None
        self._tts_frame_callback: Optional[Callable[[bytes], None]] = None
    
    def set_wake_word_daemon(self, daemon) -> None:
        """Set the wake word daemon to tap for mic audio."""
        self._wake_word_daemon = daemon
        # If daemon has a recorder, we can tap its frame buffer
        if hasattr(daemon, '_recorder') and daemon._recorder:
            logger.debug("Wake word daemon recorder found for mic tap")
    
    def set_tts_engine(self, engine) -> None:
        """Set the TTS engine to tap for playback audio."""
        self._tts_engine = engine
        logger.debug("TTS engine set for playback tap")
    
    def set_mic_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for raw mic frames (for external processing)."""
        self._mic_frame_callback = callback
    
    def set_tts_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for raw TTS frames (for external processing)."""
        self._tts_frame_callback = callback
    
    def push_mic_frame(self, frame: bytes) -> None:
        """
        Push a mic audio frame (called from wake word detection loop).
        
                Frame is 512 samples at 16kHz = 32ms of 16-bit PCM.
        """
        # Buffer frames unconditionally — the processing thread may not have
        # started yet when early mic frames arrive, but they must not be dropped.
        with self._mic_lock:
            # Convert bytes to int16 array and append
            if len(frame) >= 2:
                samples = np.frombuffer(frame, dtype=np.int16)
                self._mic_buffer.extend(samples.astype(np.float32) / 32768.0)
        
        # Call external callback if set
        if self._mic_frame_callback:
            try:
                self._mic_frame_callback(frame)
            except Exception:
                pass
    
    def push_tts_data(self, audio_data: bytes) -> None:
        """
        Push TTS audio data (called when TTS plays).
        
                Audio data is MP3 decoded to PCM by pygame.
        """
        # Buffer unconditionally — frames may arrive before the processing
        # thread starts (e.g. early TTS) and must not be dropped.
        with self._tts_lock:
            if len(audio_data) >= 2:
                samples = np.frombuffer(audio_data, dtype=np.int16)
                self._tts_buffer.extend(samples.astype(np.float32) / 32768.0)
        
        if self._tts_frame_callback:
            try:
                self._tts_frame_callback(audio_data)
            except Exception:
                pass
    
    def start(self) -> bool:
        """Start the audio processing thread."""
        if self._running:
            return True
        
        self._running = True
        self._thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="nexus-overlay-audio",
        )
        self._thread.start()
        logger.info("Audio tap manager started")
        return True
    
    def stop(self) -> None:
        """Stop the audio processing thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Audio tap manager stopped")
    
    def _processing_loop(self) -> None:
        """Main audio processing loop (runs on dedicated thread)."""
        frame_interval = 1.0 / 30.0  # 30 FPS for audio frames
        
        while self._running:
            start = time.time()
            
            # Process mic buffer
            self._process_source("mic")
            
            # Process TTS buffer  
            self._process_source("tts")
            
            # Sleep to maintain frame rate
            elapsed = time.time() - start
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)
    
    def _process_source(self, source: str) -> None:
        """Process audio buffer for a source and send FFT bins."""
        if source == "mic":
            buffer = self._mic_buffer
            lock = self._mic_lock
            last_level = self._last_mic_level
            last_bins = self._last_mic_bins
        else:
            buffer = self._tts_buffer
            lock = self._tts_lock
            last_level = self._last_tts_level
            last_bins = self._last_tts_bins
        
        # Get buffer snapshot
        with lock:
            if len(buffer) < self._fft_size:
                return
            # Take the most recent FFT_SIZE samples
            audio_data = np.array(list(buffer)[-self._fft_size:], dtype=np.float32)
        
        # Compute FFT
        level, bins = self._compute_fft(audio_data)
        
        # Only send if changed significantly (reduce WebSocket traffic)
        if self._should_send(source, level, bins, last_level, last_bins):
            self._ws_server.send_audio(source, level, bins)
            
            # Update last sent
            if source == "mic":
                self._last_mic_level = level
                self._last_mic_bins = bins
            else:
                self._last_tts_level = level
                self._last_tts_bins = bins
    
    def _compute_fft(self, audio_data: np.ndarray) -> tuple:
        """
        Compute FFT magnitude bins from audio data.
        
        Returns:
            (overall_level, bins_list)
        """
        # Apply window
        windowed = audio_data * self._window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        
        # Normalize to 0.0-1.0
        max_mag = np.max(magnitude)
        if max_mag > 0:
            magnitude = magnitude / max_mag
        
        # Overall level (RMS of windowed signal)
        level = float(np.sqrt(np.mean(windowed ** 2)))
        level = min(1.0, level * 3.0)  # Scale for visibility
        
        # Downsample to target bin count
        if len(magnitude) > self._bins:
            # Average adjacent bins
            indices = np.linspace(0, len(magnitude) - 1, self._bins, dtype=int)
            bins = magnitude[indices]
        else:
            bins = magnitude[:self._bins]
        
        # Pad if needed
        if len(bins) < self._bins:
            bins = np.pad(bins, (0, self._bins - len(bins)), mode='constant')
        
        return level, bins.tolist()
    
    def _should_send(self, source: str, level: float, bins: List[float],
                     last_level: float, last_bins: Optional[List[float]]) -> bool:
        """Check if audio frame should be sent (change detection)."""
        # Always send if level changed significantly
        if abs(level - last_level) > 0.05:
            return True
        
        # Check bins change
        if last_bins is not None:
            bin_diff = sum(abs(b - lb) for b, lb in zip(bins, last_bins))
            if bin_diff > 0.5:  # Threshold for total bin change
                return True
        
        # Always send if active (non-silent)
        if level > 0.02:
            return True
        
        return False


@lru_cache(maxsize=1)
def get_audio_tap_manager() -> AudioTapManager:
    """Get the singleton AudioTapManager instance."""
    return AudioTapManager()