"""
NEXUS AI v4.0 — Unified Audio Capture Service with VAD endpointing.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Single audio owner: one input stream (sounddevice) feeding a thread-safe queue.
Wake-word and push-to-talk STT both read from this service.
Never opens two streams.
"""

import asyncio
import logging
import queue
import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any
from functools import lru_cache

from nexus_config.settings import get_settings, APP_ROOT
from nexus_brain.service_lifecycle import ServiceBase

logger = logging.getLogger("nexus.audio.capture")


# ─── VAD (Voice Activity Detection) ───────────────────────────────────────────

try:
    import webrtcvad
    _WEBRTCAVAILABLE = True
except ImportError:
    _WEBRTCAVAILABLE = False
    logger.warning("webrtcvad not installed, using energy-based VAD fallback")


class VadDetector:
    """Voice Activity Detection using webrtcvad (preferred) or energy fallback."""
    
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.bytes_per_frame = self.frame_size * 2  # 16-bit mono
        
        if _WEBRTCAVAILABLE:
            self._vad = webrtcvad.Vad(2)  # Aggressiveness 0-3
            self._mode = "webrtcvad"
        else:
            self._vad = None
            self._mode = "energy"
            self._energy_threshold = 0.015
        
        logger.info(f"VAD initialized: {self._mode} (sr={sample_rate}, frame={frame_duration_ms}ms)")
    
    def is_speech(self, frame_bytes: bytes) -> bool:
        """Return True if frame contains speech."""
        if len(frame_bytes) < self.bytes_per_frame:
            return False
        
        frame = frame_bytes[:self.bytes_per_frame]
        
        if self._mode == "webrtcvad" and self._vad:
            try:
                return self._vad.is_speech(frame, self.sample_rate)
            except Exception:
                pass
        
        # Energy-based fallback
        return self._energy_vad(frame)
    
    def _energy_vad(self, frame_bytes: bytes) -> bool:
        """Energy-based VAD fallback."""
        try:
            # Convert to numpy int16
            samples = np.frombuffer(frame_bytes, dtype=np.int16)
            if len(samples) == 0:
                return False
            
            # RMS energy
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            normalized = rms / 32768.0  # Normalize to 0-1
            
            return normalized > self._energy_threshold
        except Exception:
            return False


# ─── Audio Capture Service ────────────────────────────────────────────────────

@dataclass
class AudioConfig:
    """Audio capture configuration."""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    blocksize: int = 512  # frames per callback
    device_index: Optional[int] = None  # None = default
    vad_frame_ms: int = 30
    silence_timeout_ms: int = 700
    min_speech_ms: int = 300
    max_duration_s: float = 30.0
    pre_speech_buffer_ms: int = 300  # Keep audio before speech starts


class AudioCapture(ServiceBase):
    """
    Single audio capture service for the entire application.
    
    Provides:
    - One input stream (sounddevice) at 16kHz mono
    - Thread-safe frame queue for consumers (wake-word, STT)
    - VAD endpointing for push-to-talk
    - Device enumeration and selection
    """
    
    def __init__(self):
        super().__init__("audio_capture")
        self._settings = get_settings()
        self._config = AudioConfig(
            sample_rate=self._settings.AUDIO_SAMPLE_RATE,
            blocksize=self._settings.AUDIO_BUFFER_SIZE,
        )
        
        # Stream state
        self._stream = None
        self._running = False
        self._stream_thread: Optional[threading.Thread] = None
        
        # Frame queue for consumers
        self._frame_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # VAD
        self._vad = VadDetector(
            sample_rate=self._config.sample_rate,
            frame_duration_ms=self._config.vad_frame_ms,
        )
        
        # Capture session state (for push-to-talk)
        self._capture_session: Optional["_CaptureSession"] = None
        self._session_lock = threading.Lock()
        
        # Device info
        self._device_info: Optional[Dict] = None
    
    # ── Service Lifecycle ───────────────────────────────────────────────────
    
    async def _start(self) -> None:
        """Start audio capture stream."""
        self._start_stream()
        logger.info("AudioCapture started (device=%s)", self._device_info)
    
    async def _stop(self) -> None:
        """Stop audio capture stream."""
        self._stop_stream()
        logger.info("AudioCapture stopped")
    
    async def _health_check(self) -> tuple[bool, str, Dict[str, Any]]:
        """Check audio capture health."""
        if not self._running:
            return False, "Not running", {"running": False}
        
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return True, "Healthy", {
                "running": True,
                "device": self._device_info,
                "sample_rate": self._config.sample_rate,
                "queue_size": self._frame_queue.qsize(),
            }
        except Exception as e:
            return False, f"Health check failed: {e}", {"running": self._running}
    
    # ── Stream Management ───────────────────────────────────────────────────
    
    def _start_stream(self) -> None:
        """Start the sounddevice input stream."""
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed. Install: pip install sounddevice")
            raise
        
        # Select device
        device_idx = self._config.device_index
        if device_idx is None:
            device_idx = sd.default.device[0]  # Default input
        
        # Get device info
        try:
            info = sd.query_devices(device_idx, "input")
            self._device_info = {
                "index": device_idx,
                "name": info["name"],
                "max_input_channels": info["max_input_channels"],
                "default_samplerate": info["default_samplerate"],
            }
        except Exception as e:
            logger.warning(f"Could not query device {device_idx}: {e}")
            self._device_info = {"index": device_idx}
        
        # Check if device supports our sample rate
        try:
            sd.check_input_settings(
                device=device_idx,
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype=self._config.dtype,
            )
        except Exception as e:
            logger.warning(f"Device {device_idx} may not support {self._config.sample_rate}Hz: {e}")
        
        # Start stream
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio callback status: {status}")
            if self._running:
                try:
                    # Copy frame data (indata is numpy array)
                    frame_bytes = indata.tobytes()
                    self._frame_queue.put_nowait(frame_bytes)
                except queue.Full:
                    # Drop oldest frame if queue full
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame_bytes)
                    except queue.Empty:
                        pass
        
        self._stream = sd.InputStream(
            device=device_idx,
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype=self._config.dtype,
            blocksize=self._config.blocksize,
            callback=audio_callback,
        )
        
        self._stream.start()
        self._running = True
        logger.info(f"Audio stream started on device {device_idx} ({self._config.sample_rate}Hz)")
    
    def _stop_stream(self) -> None:
        """Stop the sounddevice input stream."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Stream close error: {e}")
            self._stream = None
        
        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
    
    # ── Public API ──────────────────────────────────────────────────────────
    
    def get_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        """Get next audio frame (blocking with timeout)."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_frames_generator(self):
        """Generator yielding audio frames."""
        while self._running:
            frame = self.get_frame(0.1)
            if frame:
                yield frame
    
    def capture_once(
        self,
        callback: Callable[[str], None],
        max_duration: float = 30.0,
        silence_timeout: float = 0.7,
        min_speech_duration: float = 0.3,
    ) -> None:
        """
        Capture one utterance with VAD endpointing (push-to-talk).
        
        Runs in background thread, calls callback(transcript) when done.
        """
        with self._session_lock:
            if self._capture_session and self._capture_session.active:
                logger.warning("Capture session already active")
                return
            
            session = _CaptureSession(
                capture=self,
                callback=callback,
                max_duration=max_duration,
                silence_timeout=silence_timeout,
                min_speech_duration=min_speech_duration,
                vad=self._vad,
            )
            self._capture_session = session
            session.start()
    
    def stop_capture(self) -> None:
        """Stop current capture session."""
        with self._session_lock:
            if self._capture_session:
                self._capture_session.stop()
                self._capture_session = None
    
    def is_capturing(self) -> bool:
        with self._session_lock:
            return self._capture_session is not None and self._capture_session.active
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """List available audio input devices."""
        try:
            import sounddevice as sd
            devices = []
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    devices.append({
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["max_input_channels"],
                        "sample_rate": int(dev["default_samplerate"]),
                        "is_default": i == sd.default.device[0],
                    })
            return devices
        except Exception as e:
            logger.error(f"Device enumeration failed: {e}")
            return []
    
    def set_device(self, device_index: int) -> bool:
        """Change input device (restarts stream)."""
        if self._running:
            self._stop_stream()
        self._config.device_index = device_index
        try:
            self._start_stream()
            return True
        except Exception as e:
            logger.error(f"Failed to set device {device_index}: {e}")
            return False


# ─── Capture Session (Push-to-Talk) ──────────────────────────────────────────

class _CaptureSession:
    """Single push-to-talk capture session with VAD endpointing."""
    
    def __init__(
        self,
        capture: AudioCapture,
        callback: Callable[[str], None],
        max_duration: float,
        silence_timeout: float,
        min_speech_duration: float,
        vad: VadDetector,
    ):
        self.capture = capture
        self.callback = callback
        self.max_duration = max_duration
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self.vad = vad
        
        self.active = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Audio buffering
        self._audio_buffer: List[bytes] = []
        self._pre_speech_buffer: List[bytes] = []
        self._max_pre_speech_frames = int(
            capture._config.sample_rate * capture._config.pre_speech_buffer_ms / 1000
        ) // capture._config.blocksize
        
        # State
        self._speech_started = False
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0
        self._session_start_time = 0.0
    
    def start(self) -> None:
        self.active = True
        self._stop_event.clear()
        self._session_start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name="nexus-capture-session")
        self._thread.start()
    
    def stop(self) -> None:
        self.active = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _run(self) -> None:
        """Main capture loop with VAD endpointing."""
        try:
            logger.info("Capture session started (push-to-talk)")
            
            # Notify UI we're listening
            # (UI state is set by caller before calling capture_once)
            
            while self.active and not self._stop_event.is_set():
                # Check max duration
                if time.time() - self._session_start_time > self.max_duration:
                    logger.info("Max duration reached, finalizing")
                    break
                
                # Get frame
                frame = self.capture.get_frame(timeout=0.1)
                if frame is None:
                    continue
                
                is_speech = self.vad.is_speech(frame)
                now = time.time()
                
                if is_speech:
                    if not self._speech_started:
                        # Speech just started
                        self._speech_started = True
                        self._speech_start_time = now
                        self._last_speech_time = now
                        # Include pre-speech buffer
                        self._audio_buffer.extend(self._pre_speech_buffer)
                        self._pre_speech_buffer.clear()
                        logger.debug("Speech detected, starting recording")
                    else:
                        self._last_speech_time = now
                    
                    self._audio_buffer.append(frame)
                
                else:
                    # Silence
                    if self._speech_started:
                        self._audio_buffer.append(frame)
                        
                        # Check silence timeout
                        if now - self._last_speech_time >= self.silence_timeout:
                            speech_duration = self._last_speech_time - self._speech_start_time
                            if speech_duration >= self.min_speech_duration:
                                logger.info(f"Speech ended after {speech_duration:.1f}s, transcribing...")
                                self._finalize_and_transcribe()
                            else:
                                logger.debug(f"Speech too short ({speech_duration:.1f}s), discarding")
                                self._reset_speech_state()
                            break
                    else:
                        # Still waiting for speech - maintain pre-speech buffer
                        self._pre_speech_buffer.append(frame)
                        if len(self._pre_speech_buffer) > self._max_pre_speech_frames:
                            self._pre_speech_buffer.pop(0)
            
        except Exception as e:
            logger.error(f"Capture session error: {e}")
            if self.callback:
                try:
                    self.callback("")
                except Exception:
                    pass
        finally:
            self.active = False
    
    def _reset_speech_state(self) -> None:
        self._speech_started = False
        self._audio_buffer.clear()
        self._pre_speech_buffer.clear()
    
    def _finalize_and_transcribe(self) -> None:
        """Convert buffered audio to WAV and transcribe."""
        if not self._audio_buffer:
            if self.callback:
                self.callback("")
            return
        
        # Concatenate frames
        audio_data = b"".join(self._audio_buffer)
        
        # Run transcription in background to not block
        def transcribe_thread():
            try:
                text = self._transcribe_audio(audio_data)
                if text and text.strip():
                    logger.info(f"Transcription: '{text[:100]}'")
                else:
                    logger.debug("Empty transcription")
                    text = ""
                
                if self.callback:
                    self.callback(text)
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                if self.callback:
                    self.callback("")
        
        threading.Thread(target=transcribe_thread, daemon=True, name="nexus-transcribe").start()
    
    def _transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio using Groq Whisper (online) or faster-whisper (offline)."""
        # Try Groq first (online, fast)
        text = self._transcribe_groq(audio_data)
        if text:
            return text
        
        # Fallback to local faster-whisper
        return self._transcribe_local(audio_data)
    
    def _transcribe_groq(self, audio_data: bytes) -> Optional[str]:
        """Transcribe via Groq Whisper API."""
        try:
            from groq import Groq
            import io
            
            settings = get_settings()
            if not settings.GROQ_API_KEY:
                return None
            
            client = Groq(api_key=settings.GROQ_API_KEY)
            
            # Create WAV in memory
            wav_buffer = io.BytesIO()
            self._write_wav(wav_buffer, audio_data)
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"
            
            response = client.audio.transcriptions.create(
                file=wav_buffer,
                model="whisper-large-v3-turbo",  # or "whisper-large-v3"
                language="en",
                response_format="text",
                temperature=0.0,
            )
            
            return response.strip() if response else None
            
        except Exception as e:
            logger.debug(f"Groq transcription failed: {e}")
            return None
    
    def _transcribe_local(self, audio_data: bytes) -> str:
        """Transcribe using local faster-whisper."""
        try:
            from faster_whisper import WhisperModel
            import io
            import tempfile
            import os
            
            settings = get_settings()
            model_size = settings.WHISPER_MODEL_SIZE
            
            # Load model (cached)
            if not hasattr(_CaptureSession, "_whisper_model"):
                logger.info(f"Loading local Whisper model: {model_size}")
                _CaptureSession._whisper_model = WhisperModel(
                    model_size_or_path=model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=2,
                )
            
            model = _CaptureSession._whisper_model
            
            # Write to temp file (faster-whisper needs file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                self._write_wav(f, audio_data)
                temp_path = f.name
            
            try:
                segments, info = model.transcribe(
                    temp_path,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters={"threshold": 0.5, "min_speech_duration_ms": 250},
                    language="en",
                )
                
                text = " ".join(seg.text for seg in segments).strip()
                return text
            finally:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Local transcription failed: {e}")
            return ""
    
    @staticmethod
    def _write_wav(file_obj, audio_data: bytes) -> None:
        """Write PCM16 mono audio data as WAV."""
        import wave
        
        with wave.open(file_obj, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio_data)


# ─── Singleton Accessor ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_audio_capture() -> AudioCapture:
    """Get the singleton AudioCapture instance."""
    return AudioCapture()


# ─── Selftest ─────────────────────────────────────────────────────────────────

def selftest() -> bool:
    """Test audio capture: list devices, record 3s, print RMS, transcribe."""
    print("=== NEXUS Audio Capture Selftest ===")
    
    capture = get_audio_capture()
    
    # List devices
    devices = capture.list_devices()
    print(f"\nInput devices ({len(devices)}):")
    for d in devices:
        default = " (DEFAULT)" if d["is_default"] else ""
        print(f"  [{d['index']}] {d['name']} - {d['channels']}ch @ {d['sample_rate']}Hz{default}")
    
    if not devices:
        print("No input devices found!")
        return False
    
    # Start capture
    import asyncio
    asyncio.run(capture._start())
    
    print("\nRecording 3 seconds... (speak now)")
    frames = []
    start = time.time()
    while time.time() - start < 3.0:
        frame = capture.get_frame(0.5)
        if frame:
            frames.append(frame)
    
    # Calculate RMS
    if frames:
        all_audio = b"".join(frames)
        samples = np.frombuffer(all_audio, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        peak = np.max(np.abs(samples))
        print(f"\nAudio stats: {len(frames)} frames, {len(all_audio)} bytes")
        print(f"  Peak: {peak} ({peak/32768*100:.1f}%)")
        print(f"  RMS: {rms:.1f} ({rms/32768*100:.2f}%)")
    
    # Test transcription
    print("\nTranscribing...")
    # Create a temporary session to access transcription methods
    from nexus_config.settings import get_settings
    session = _CaptureSession(
        capture=capture,
        callback=lambda x: None,
        max_duration=30.0,
        silence_timeout=0.7,
        min_speech_duration=0.3,
        vad=capture._vad,
    )
    text = session._transcribe_groq(b"".join(frames)) if frames else ""
    if not text and frames:
        text = session._transcribe_local(b"".join(frames))
    
    print(f"Transcription: '{text}'")
    
    asyncio.run(capture._stop())
    print("\nSelftest complete.")
    return True


if __name__ == "__main__":
    selftest()