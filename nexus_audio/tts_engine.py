"""
NEXUS AI v4.0 — edge-tts multi-profile speech engine.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Text-to-speech using edge-tts (Microsoft Edge's TTS engine) with
pygame mixer for non-blocking playback. Supports 5 distinct voice
profiles for different interaction modes.

Memory-optimized: Temp MP3 files are cleaned within 30 seconds of
playback. pygame mixer runs on a dedicated audio sub-system.

Audio tap integration: Decodes MP3 to PCM for real-time FFT analysis
by the ring overlay.
"""

import asyncio
import logging
import os
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

from nexus_config.settings import get_settings, APP_ROOT
from nexus_brain.service_lifecycle import ServiceBase

logger = logging.getLogger("nexus.tts")


# ─── Voice Profiles ──────────────────────────────────────────────────────────

class VoiceProfile(str, Enum):
    """
    TTS voice profiles for different interaction modes.
    
    NEXUS_PRIME:    Default — authoritative, clear, professional (default female en-US)
    NEXUS_ALERT:    Urgent — faster, higher pitch for warnings and errors
    NEXUS_CASUAL:   Friendly — warm tone for greetings and casual chat
    NEXUS_TECHNICAL: Precise — slower, clear enunciation for code and technical content
    NEXUS_FEMALE:   Alternative — a different voice option
    """
    NEXUS_PRIME = "en-US-AriaNeural"
    NEXUS_ALERT = "en-US-GuyNeural"
    NEXUS_CASUAL = "en-US-JennyNeural"
    NEXUS_TECHNICAL = "en-US-EricNeural"
    NEXUS_FEMALE = "en-GB-SoniaNeural"
    
    @classmethod
    def from_name(cls, name: str) -> "VoiceProfile":
        """Get VoiceProfile from profile name (e.g., 'NEXUS_PRIME')."""
        try:
            return cls[name]
        except KeyError:
            return cls.NEXUS_PRIME


# Profile → style override map
VOICE_STYLES: Dict[VoiceProfile, Dict[str, str]] = {
    VoiceProfile.NEXUS_PRIME: {"style": "newscast-formal", "rate": "+10%"},
    VoiceProfile.NEXUS_ALERT: {"style": "angry", "rate": "+20%"},
    VoiceProfile.NEXUS_CASUAL: {"style": "cheerful", "rate": "+5%"},
    VoiceProfile.NEXUS_TECHNICAL: {"style": "narration-professional", "rate": "-10%"},
    VoiceProfile.NEXUS_FEMALE: {"style": "gentle", "rate": "+0%"},
}


# ─── TTS Engine ──────────────────────────────────────────────────────────────

MAX_TTS_TEXT_LENGTH = 1000  # Characters — longer text is truncated with warning
MAX_TEMP_AGE = 30  # Seconds — temp files cleaned after this time

# Audio tap callback (set by overlay integration)
_tts_audio_tap_callback: Optional[callable] = None


def set_tts_audio_tap_callback(callback: callable) -> None:
    """Set the callback for TTS audio data (called with PCM bytes)."""
    global _tts_audio_tap_callback
    _tts_audio_tap_callback = callback


def _decode_mp3_to_pcm(mp3_path: str) -> Optional[bytes]:
    """
    Decode MP3 file to raw PCM bytes (16-bit, mono, 16kHz).
    
    Uses pydub with ffmpeg if available, falls back to None.
    """
    try:
        from pydub import AudioSegment
        
        # Load MP3
        audio = AudioSegment.from_mp3(mp3_path)
        
        # Convert to 16kHz mono 16-bit PCM
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Return raw bytes
        return audio.raw_data
        
    except Exception as e:
        logger.debug("MP3 decode failed (ffmpeg may not be available): %s", e)
        return None


class TTSEngine(ServiceBase):
    """
    Non-blocking text-to-speech engine with lifecycle management.
    
    Uses edge-tts for online synthesis and pygame.mixer for playback.
    Both operations are async-compatible via thread pool execution.
    
    Usage:
        tts = get_tts_engine()
        async with tts.lifespan():
            await tts.say("Hello, I am NEXUS AI")
            await tts.say("Warning!", profile=VoiceProfile.NEXUS_ALERT)
            await tts.stop()
    """
    
    def __init__(self):
        super().__init__("tts")
        self._settings = get_settings()
        self._current_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pygame_initialized = False
        
        # Track temp files for cleanup
        self._temp_files: List[str] = []
        self._cleanup_thread: Optional[threading.Thread] = None
    
    def _init_pygame(self) -> None:
        """Initialize pygame mixer for audio playback."""
        try:
            import pygame
            pygame.mixer.pre_init(
                frequency=self._settings.AUDIO_SAMPLE_RATE,
                size=-16,
                channels=1,
                buffer=self._settings.AUDIO_BUFFER_SIZE,
            )
            pygame.mixer.init()
            self._pygame_initialized = True
            logger.debug("pygame mixer initialized (rate=%dHz, buffer=%d)",
                self._settings.AUDIO_SAMPLE_RATE, self._settings.AUDIO_BUFFER_SIZE)
        except Exception as e:
            self._pygame_initialized = False
            logger.warning("pygame mixer init failed: %s. TTS disabled.", e)
    
    # ── Service Lifecycle ───────────────────────────────────────────────────
    
    async def _start(self) -> None:
        """Start TTS engine."""
        self._init_pygame()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="nexus-tts-cleanup",
        )
        self._cleanup_thread.start()
        
        logger.info("TTSEngine started (profile: %s)", self._settings.DEFAULT_TTS_VOICE)
    
    async def _stop(self) -> None:
        """Stop TTS engine."""
        self._stop_current()
        
        # Stop cleanup thread
        if self._cleanup_thread:
            # The thread is daemon, so it will exit when main thread exits
            pass
        
        # Clean up temp files
        for temp_path in self._temp_files:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        self._temp_files.clear()
        
        # Quit pygame
        try:
            import pygame
            pygame.mixer.quit()
        except Exception:
            pass
        
        logger.info("TTSEngine stopped")
    
    async def _health_check(self) -> tuple[bool, str, Dict[str, Any]]:
        """Check TTS health."""
        if not self._settings.ENABLE_TTS:
            return False, "TTS disabled in settings", {"enabled": False}
        
        if not self._pygame_initialized:
            return False, "pygame mixer not initialized", {"pygame": False}
        
        # Check edge-tts import
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return True, "TTS healthy", {
                "pygame": True,
                "edge_tts": True,
                "available_voices": len(voices),
            }
        except Exception as e:
            return False, f"edge-tts error: {e}", {"pygame": True, "edge_tts": False}
    
    # ── Public API ───────────────────────────────────────────────────────────
    
    async def say(
        self,
        text: str,
        profile: Optional[VoiceProfile] = None,
        wait: bool = False,
    ) -> bool:
        """
        Speak the given text using the specified voice profile.
        
        This is async-compatible but the actual synthesis and playback
        happen on a background thread to never block the caller.
        
        Args:
            text: Text to speak. Truncated to MAX_TTS_TEXT_LENGTH.
            profile: Voice profile. Defaults to settings.DEFAULT_TTS_VOICE.
            wait: If True, blocks until playback completes.
        
        Returns:
            True if speech was queued/played successfully.
        """
        if not self._settings.ENABLE_TTS or not self._pygame_initialized:
            return False
        
        if not text or not text.strip():
            return False
        
        # Truncate long text
        if len(text) > MAX_TTS_TEXT_LENGTH:
            logger.warning("TTS text truncated from %d to %d chars", len(text), MAX_TTS_TEXT_LENGTH)
            text = text[:MAX_TTS_TEXT_LENGTH] + "... [truncated]"
        
        profile = profile or VoiceProfile.from_name(self._settings.DEFAULT_TTS_VOICE)
        
        # Stop any currently playing speech
        self._stop_current()
        self._stop_event.clear()
        
        # Start synthesis + playback on background thread
        thread = threading.Thread(
            target=self._synthesize_and_play,
            args=(text, profile),
            daemon=True,
            name="nexus-tts-playback",
        )
        thread.start()
        
        self._current_thread = thread
        
        if wait:
            thread.join()
        
        return True
    
    def _synthesize_and_play(self, text: str, profile: VoiceProfile) -> None:
        """
        Synthesize speech and play it (runs on background thread).
        
        Uses edge-tts for synthesis → writes temp MP3 → decodes to PCM for audio tap → plays via pygame.
        """
        if self._stop_event.is_set():
            return
        
        temp_path = None
        try:
            # Generate TTS audio via edge-tts
            import edge_tts
            
            voice = profile.value
            style = VOICE_STYLES.get(profile, {})
            rate = style.get("rate", self._settings.TTS_SPEAKING_RATE)
            
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
            
            # Write to temp file
            tmp = tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False,
                dir=str(APP_ROOT / "temp"),
            )
            temp_path = tmp.name
            tmp.close()
            
            # Run the async communicate in a new event loop
            async def _synthesize():
                await communicate.save(temp_path)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_synthesize())
            loop.close()
            
            if self._stop_event.is_set():
                self._cleanup_file(temp_path)
                return
            
            # Track for cleanup
            with self._lock:
                self._temp_files.append(temp_path)
            
            # Decode MP3 to PCM for audio tap (non-blocking)
            pcm_data = _decode_mp3_to_pcm(temp_path)
            if pcm_data and _tts_audio_tap_callback:
                try:
                    _tts_audio_tap_callback(pcm_data)
                except Exception as e:
                    logger.debug("TTS audio tap callback error: %s", e)
            
            # Play via pygame
            self._play_mp3(temp_path)
            
        except ImportError:
            logger.error("edge-tts not installed. Install with: pip install edge-tts")
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
        finally:
            if temp_path:
                # Schedule cleanup after 30 seconds
                threading.Timer(MAX_TEMP_AGE, self._cleanup_file, args=[temp_path]).start()
    
    def _play_mp3(self, file_path: str) -> None:
        """
        Play an MP3 file using pygame mixer.
        
        Blocks until playback finishes or stop is requested.
        """
        if not self._pygame_initialized:
            return
        
        try:
            import pygame
            
            if self._stop_event.is_set():
                return
            
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish (checking stop event)
            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.1)
        
        except pygame.error as e:
            logger.debug("pygame playback error: %s", e)
        except Exception as e:
            logger.warning("Audio playback error: %s", e)
    
    def _stop_current(self) -> None:
        """Stop any currently playing audio."""
        try:
            import pygame
            if self._pygame_initialized:
                pygame.mixer.music.stop()
        except Exception:
            pass
        
        self._stop_event.set()
        self._current_thread = None
    
    async def stop(self) -> None:
        """Stop current speech and clear queue."""
        self._stop_current()
    
    def _cleanup_file(self, file_path: Optional[str]) -> None:
        """Remove a temp file if it exists."""
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                with self._lock:
                    if file_path in self._temp_files:
                        self._temp_files.remove(file_path)
            except Exception:
                pass
    
    def _cleanup_loop(self) -> None:
        """Periodic cleanup of old temp files."""
        while True:
            time.sleep(60)
            try:
                now = time.time()
                temp_dir = APP_ROOT / "temp"
                if temp_dir.exists():
                    for f in temp_dir.glob("*.mp3"):
                        try:
                            age = now - f.stat().st_mtime
                            if age > MAX_TEMP_AGE:
                                f.unlink()
                        except Exception:
                            pass
            except Exception:
                pass
    
    def is_speaking(self) -> bool:
        """Check if audio is currently playing."""
        if self._pygame_initialized:
            try:
                import pygame
                return pygame.mixer.music.get_busy()
            except Exception:
                pass
        return False


@lru_cache(maxsize=1)
def get_tts_engine() -> TTSEngine:
    """
    Return the singleton TTSEngine instance.
    
    Returns:
        TTSEngine: The singleton TTS engine instance.
    """
    return TTSEngine()
