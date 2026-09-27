"""
NEXUS AI v4.0 — Audio subsystem.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides voice input (wake word + Whisper) and voice output (edge-tts + pygame).
Fully asynchronous — never blocks the UI thread.
"""

from nexus_audio.tts_engine import get_tts_engine, TTSEngine, VoiceProfile
from nexus_audio.wake_word import get_wake_word_daemon, WakeWordDaemon, WakeWordState
from nexus_audio.audio_utils import (
    get_platform_audio_backend, enumerate_audio_devices,
    test_audio_output, test_microphone, suppress_alsa_errors, get_system_volume,
)
from nexus_audio.audio_processor import (
    calculate_rms, is_silence, detect_speech_segments,
    normalize_audio, trim_silence,
)
from nexus_audio.capture import get_audio_capture, AudioCapture, AudioConfig

__all__ = [
    "get_tts_engine", "TTSEngine", "VoiceProfile",
    "get_wake_word_daemon", "WakeWordDaemon", "WakeWordState",
    "get_platform_audio_backend", "enumerate_audio_devices",
    "test_audio_output", "test_microphone", "suppress_alsa_errors", "get_system_volume",
    "calculate_rms", "is_silence", "detect_speech_segments",
    "normalize_audio", "trim_silence",
    "get_audio_capture", "AudioCapture", "AudioConfig",
]
