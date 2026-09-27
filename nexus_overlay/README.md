# NEXUS Ring Overlay

A transparent, frameless, always-on-top, audio-reactive circular HUD overlay for NEXUS AI. Runs as a separate pywebview process communicating with the NEXUS core via local WebSocket.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEXUS Core Process                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Orchestrator│  │WakeWordDaemon│  │  TTSEngine  │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────┐               │
│  │         OverlayIntegration                   │               │
│  │  ┌─────────────┐  ┌─────────────────────┐  │               │
│  │  │ WebSocket   │  │ AudioTapManager     │  │               │
│  │  │ Server      │  │ (FFT → bins)        │  │               │
│  │  └──────┬──────┘  └──────────┬──────────┘  │               │
│  └─────────┼────────────────────┼─────────────┘               │
└────────────┼────────────────────┼─────────────────────────────┘
           │ WebSocket            │ PCM frames
           ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Overlay Process (pywebview)                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  overlay.html (Canvas 2D - Ring Visualization)             ││
│  │  - Concentric counter-rotating rings                       ││
│  │  - Radial frequency bars (FFT bins)                        ││
│  │  - Particle bursts on state transitions                    ││
│  │  - Idle breathing pulse                                     ││
│  │  - Radar sweep wedge (thinking)                            ││
│  │  - Error jitter/pulse                                       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Components

| File | Purpose |
|------|---------|
| `overlay_process.py` | Pywebview entry point - frameless transparent window |
| `websocket_server.py` | WebSocket server (runs in core process) |
| `core_integration.py` | Wires orchestrator, wake word, TTS to overlay |
| `audio_tap.py` | Taps mic/TTS → computes FFT → sends bins |
| `config.py` | Centralized configuration (colors, sizes, behavior) |
| `assets/overlay.html` | Canvas 2D ring visualization (JS) |

## Message Schema

**Core → Overlay (WebSocket):**
```json
// State change
{"type": "state", "value": "idle"|"listening"|"thinking"|"speaking"|"error"}

// Audio frame (30 FPS max)
{"type": "audio", "source": "mic"|"tts", "level": 0.0-1.0, "bins": [0.0-1.0, ...]}
```

## Standalone Visual Testing

Run the overlay without the full NEXUS core:

```bash
# From project root
python -m nexus_overlay.overlay_process

# With custom position
python -m nexus_overlay.overlay_process --x 100 --y 100

# With debug logging
python -m nexus_overlay.overlay_process --debug
```

**What you'll see:**
- A 240×240 transparent frameless window in the bottom-right corner
- Cyan "idle" ring with breathing pulse
- No audio reactivity (no core connected)

**To test with simulated audio/state:**
```bash
# Terminal 1: Start WebSocket server (core side)
python -c "
from nexus_overlay.websocket_server import OverlayWebSocketServer
import time
server = OverlayWebSocketServer()
server.start()
print('Server running on ws://127.0.0.1:8765/nexus/overlay')
print('Press Ctrl+C to stop')
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()
"

# Terminal 2: Run overlay (connects to server above)
python -m nexus_overlay.overlay_process --debug

# Terminal 3: Send test messages
python -c "
import asyncio
import websockets
import json
import time

async def test():
    uri = 'ws://127.0.0.1:8765/nexus/overlay'
    async with websockets.connect(uri) as ws:
        states = ['idle', 'listening', 'thinking', 'speaking', 'error']
        for state in states:
            await ws.send(json.dumps({'type': 'state', 'value': state}))
            print(f'Sent: {state}')
            time.sleep(2)
        # Send audio frames
        for i in range(30):
            await ws.send(json.dumps({
                'type': 'audio',
                'source': 'mic',
                'level': 0.5 + 0.3 * (i % 10) / 10,
                'bins': [0.1 + 0.8 * ((j + i) % 10) / 10 for j in range(40)]
            }))
            time.sleep(0.033)

asyncio.run(test())
"
```

## Configuration

All visual parameters live in `config.py` (`OverlayConfig` dataclass). Persisted to `~/.nexus_ai/overlay_config.json`.

Key settings:
- **Window**: `window_width`, `window_height`, `always_on_top`, `frameless`, `transparent`
- **Rings**: `ring_count`, `ring_multipliers`, `ring_line_widths`, `ring_dash_patterns`, `ring_rotation_speeds`
- **Bars**: `bar_count`, `bar_min_length`, `bar_max_length`
- **Particles**: `particle_burst_count`, `particle_base_speed`, `particle_decay_min/max`
- **States**: `states` dict with `color`, `accent`, `speed` per state
- **Audio**: `fft_size`, `fft_bins`, `sample_rate`
- **Debounce**: `state_debounce_ms` (default 150ms)

Override via `.env`:
```bash
OVERLAY_WS_PORT=8765
OVERLAY_WINDOW_WIDTH=240
OVERLAY_WINDOW_HEIGHT=240
```

## Integration with NEXUS Core

In `main.py` boot sequence:
```python
from nexus_overlay.core_integration import get_overlay_integration
from nexus_audio.wake_word import get_wake_word_daemon
from nexus_audio.tts_engine import get_tts_engine

overlay = get_overlay_integration()
overlay.initialize(
    orchestrator=orchestrator,
    wake_word_daemon=get_wake_word_daemon() if settings.ENABLE_WAKE_WORD else None,
    tts_engine=get_tts_engine() if settings.ENABLE_TTS else None,
)
overlay.start()
```

On shutdown:
```python
overlay.stop()
```

## Audio Taps

**Mic:** Hooks `WakeWordDaemon._recorder.read()` — receives 512-sample frames (32ms @ 16kHz) from Porcupine's recorder.

**TTS:** TTSEngine decodes synthesized MP3 to PCM via `pydub` (requires ffmpeg) and pushes full utterance to `AudioTapManager.push_tts_data()`. Falls back gracefully if ffmpeg unavailable.

Both feed rolling buffers → FFT computed at 30 FPS → bins sent over WebSocket.

## Performance Notes

- **Target hardware**: Intel i3 7th Gen, 12GB RAM
- **Overlay process**: ~15-30 MB RAM (WebView2)
- **WebSocket server**: Negligible (asyncio + queue)
- **Audio tap**: ~5 MB, 30 FPS FFT on 1024-sample windows
- **Canvas animation**: 60 FPS target, adaptive particle count

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Overlay window doesn't appear | Check pywebview installed: `pip install pywebview` |
| WebSocket connection refused | Ensure core process starts WebSocket server before overlay |
| No audio reactivity | Verify ffmpeg installed for TTS tap: `winget install ffmpeg` |
| High CPU | Reduce `UI_PARTICLE_COUNT` in settings, lower `target_fps` in config |
| Window not transparent | Windows: Requires WebView2 runtime (included in Win10/11) |

## Dependencies

- `pywebview>=5.0` — Native webview (WebView2 on Windows)
- `websockets>=12.0` — WebSocket server
- `screeninfo>=0.8` — Monitor detection for positioning
- `numpy>=1.24` — FFT computation
- `pydub>=0.25` — MP3→PCM decode (optional, needs ffmpeg)