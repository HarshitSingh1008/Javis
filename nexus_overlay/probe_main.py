"""
DIAGNOSTIC HARNESS — throwaway, not part of the deliverable.
Walks through main.py's boot sequence step-by-step and reports where it dies.
Run:  python nexus_overlay/probe_main.py
"""
import sys, io, os, logging, traceback
from pathlib import Path

# Mirror main.py's UTF-8 plumbing for Windows console
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("probe")


def run():
    log.info("=== STEP 1: import settings ===")
    try:
        from nexus_config.settings import get_settings, validate_on_boot, APP_ROOT
        s = get_settings()
        log.info("OK settings from %s", APP_ROOT / ".env" if (APP_ROOT / ".env").exists() else APP_ROOT)
        log.info("  ENABLE_WAKE_WORD=%s ENABLE_TTS=%s OVERLAY_WS_PORT=%s",
                 getattr(s, "ENABLE_WAKE_WORD", "?"),
                 getattr(s, "ENABLE_TTS", "?"),
                 getattr(s, "OVERLAY_WS_PORT", "?"))
    except Exception as e:
        log.error("STEP 1 FAILED: %s", e); traceback.print_exc(); return 1

    log.info("=== STEP 2: import overlay packages ===")
    try:
        import nexus_overlay
        from nexus_overlay.config import get_overlay_config
        from nexus_overlay.websocket_server import get_ws_server
        from nexus_overlay.audio_tap import get_audio_tap_manager
        from nexus_overlay.core_integration import get_overlay_integration, OverlayState
        from nexus_overlay.overlay_process import OverlayProcess
        log.info("OK nexus_overlay imports; OverlayState=%s", [v.name for v in OverlayState])
    except Exception as e:
        log.error("STEP 2 FAILED: %s", e); traceback.print_exc(); return 1

    log.info("=== STEP 3: overlay config check ===")
    try:
        c = get_overlay_config()
        log.info("OK config ws_host=%s port=%s path=%s states=%s",
                 c.ws_host, c.ws_port, c.ws_path, list(c.states.keys()))
    except Exception as e:
        log.error("STEP 3 FAILED: %s", e); traceback.print_exc(); return 1

    log.info("=== STEP 4: WS server standalone smoke (not started) ===")
    try:
        ws = get_ws_server()
        ws.send_state("idle")
        ws.send_state("listening")
        ws.send_state("idle")   # dedup, no-op
        ws.send_audio("mic", 0.5, list(range(80)))
        ws.send_audio("tts", 0.3, list(range(40)))
        log.info("OK ws server smoke; last_state=%s clients=%d running=%s",
                 ws._last_state, ws.connected_clients, ws.is_running)
    except Exception as e:
        log.error("STEP 4 FAILED: %s", e); traceback.print_exc(); return 1

    log.info("=== STEP 5: audio tap create ===")
    try:
        at = get_audio_tap_manager()
        log.info("OK audio tap created; running=%s", at._running)
    except Exception as e:
        log.error("STEP 5 FAILED: %s", e); traceback.print_exc(); return 1

    log.info("=== STEP 6: core integration create+init (no real components) ===")
    try:
        oi = get_overlay_integration()
        oi.initialize()
        log.info("OK core integration initialized (no-op without components)")
    except Exception as e:
        log.error("STEP 6 FAILED: %s", e); traceback.print_exc(); return 1

    log.info("=== STEP 7: simulate set_state (WS server not started -> no-op enqueue) ===")
    try:
        oi.set_state("idle")
        oi.set_state("listening")
        oi.set_state("idle")   # dedup
        log.info("OK set_state smoke; current=%s", oi._current_state)
    except Exception as e:
        log.error("STEP 7 FAILED: %s", e); traceback.print_exc(); return 1

    log.info("=== STEP 8: overlay process can be constructed (not started) ===")
    try:
        ov = OverlayProcess()
        log.info("OK OverlayProcess constructed; config=%s", type(ov._config).__name__)
    except Exception as e:
        log.error("STEP 8 FAILED: %s", e); traceback.print_exc(); return 1

    log.info("=== ALL STEPS PASSED — main.py boot should get to UI launch ===")
    log.info("NOTE: real failure may be at UI launch (pywebview, subprocess) which this harness does not execute.")
    return 0


if __name__ == "__main__":
    rc = run()
    log.info("PROBE EXIT CODE: %d", rc)
    sys.exit(rc)
