"""
NEXUS AI v4.0 — Enterprise Autonomous Desktop Omni-Agent
Hardware: Intel i3 7th Gen · 12GB RAM · UHD 620 · Ollama + Groq API

Hardened boot sequence:
1. Parse command-line arguments
2. Initialize configuration and logging
3. Create runtime directories
4. Run health check
5. Show onboarding if first run
6. Initialize tool registry
7. Initialize memory subsystem
8. Initialize audio subsystem
9. Initialize agent orchestrator
10. Start the UI (GUI or headless)
11. Set up crash handler
12. Enter main loop
"""

import sys
import io
import os
import logging
import signal
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

# Fix Unicode encoding for Windows console
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ─── Boot Constants ──────────────────────────────────────────────────────────

NEXUS_VERSION = "4.0.0"
LOG_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"


def main():
    """
    NEXUS AI boot sequence.
    
    Called when running `python main.py` or `nexus run`.
    """
    # ─── Step 0: Parse CLI arguments ─────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser(
        description=f"NEXUS AI v{NEXUS_VERSION} — Autonomous Desktop Omni-Agent"
    )
    parser.add_argument("--ui-mode", choices=["overlay", "hud", "headless"], default="overlay",
                       help="UI mode: overlay (ring HUD), hud (CustomTkinter JARVIS), headless (CLI)")
    parser.add_argument("--headless", action="store_true",
                       help="Run without GUI (CLI/text mode) - alias for --ui-mode headless")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose debug logging")
    parser.add_argument("--health-only", action="store_true",
                       help="Run health check and exit")
    parser.add_argument("--onboarding", action="store_true",
                       help="Force show onboarding wizard")
    parser.add_argument("--selftest", action="store_true",
                       help="Run audio capture selftest and exit")
    args = parser.parse_args()
    
    # Handle --headless alias
    if args.headless:
        args.ui_mode = "headless"
    
    # ─── Step 1: Configure Logging ───────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("nexus.boot")
    logger.info("NEXUS AI v%s boot sequence starting...", NEXUS_VERSION)
    logger.info("Platform: %s | Python: %s", sys.platform, sys.version.split()[0])
    logger.info("UI mode: %s", args.ui_mode)
    
    # ─── Step 2: Initialize Configuration ────────────────────────────────────
    from nexus_config.settings import get_settings, validate_on_boot, APP_ROOT
    from nexus_config.audit_logger import get_audit_logger
    
    settings = get_settings()
    logger.info("Configuration loaded from %s", APP_ROOT / ".env")
    
    # ─── Step 3: Create Runtime Directories ──────────────────────────────────
    for d in ["temp", "crash_reports", "metrics", "synthesized_tools", "workflows"]:
        (APP_ROOT / d).mkdir(parents=True, exist_ok=True)
    logger.info("Runtime directories verified")
    
    # ─── Step 4: Run Health Check ────────────────────────────────────────────
    from nexus_config.health_check import run_health_check, print_health_report
    
    health = run_health_check()
    
    if args.health_only:
        print_health_report(health)
        sys.exit(0 if health.ok else 1)
    
    if not health.ok:
        logger.warning("Health check found issues (%d warnings, %d errors)",
                       len(health.warnings), len(health.errors))
    
    if health.degraded_subsystems:
        logger.info("Degraded subsystems: %s", ", ".join(health.degraded_subsystems))
    
    # ─── Step 5: Show Onboarding if First Run ────────────────────────────────
    from nexus_ui.onboarding import OnboardingWizard
    
    wizard = OnboardingWizard()
    if args.onboarding or wizard.is_first_run():
        logger.info("First run detected — showing onboarding wizard")
        wizard.run_gui()
    
    # ─── Step 6: Set Up Crash Handler ─────────────────────────────────────────
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions by writing a crash report."""
        from nexus_config.crash_reporter import write_crash_report
        write_crash_report(exc_type, exc_value, exc_traceback)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = global_exception_handler
    
    # ─── Step 7: Initialize Billing & Licensing ─────────────────────────────
    logger.info("Initializing billing and licensing subsystem...")
    try:
        from nexus_billing.license_manager import get_license_manager
        lm = get_license_manager()
        current_tier = lm.current_tier()
        logger.info("License loaded — current tier: %s", current_tier)
    except Exception as e:
        logger.warning("Billing subsystem initialization skipped: %s", e)
    
    # ─── Step 8: Initialize Tool Registry ────────────────────────────────────
    logger.info("Initializing tool registry...")
    try:
        from nexus_tools.registry import get_tool_registry
        registry = get_tool_registry()
        count = registry.load_builtin_tools()
        logger.info("Tool registry initialized with %d tools", count)
    except Exception as e:
        logger.error("Tool registry initialization failed: %s", e)
        sys.exit(1)
    
    # ─── Step 9: Initialize Memory Subsystem ────────────────────────────────
    logger.info("Initializing memory subsystem...")
    try:
        from nexus_memory.vector_store import get_vector_store
        store = get_vector_store()
        store.heartbeat()
        logger.info("Memory subsystem ready (ChromaDB)")
    except Exception as e:
        logger.warning("Memory subsystem initialized with in-memory fallback: %s", e)
    
    # ─── Step 10: Initialize Audio Subsystem ─────────────────────────────────
    if settings.ENABLE_WAKE_WORD or settings.ENABLE_TTS:
        logger.info("Initializing audio subsystem...")
        logger.info("Audio subsystem ready (on-demand initialization)")
    
    # ─── Step 11: Initialize Agent Orchestrator ──────────────────────────────
    logger.info("Initializing agent orchestrator...")
    try:
        from nexus_brain.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        
        # Set up tool executor (uses registry)
        async def tool_executor(tool_name: str, tool_input: Dict[str, Any], is_ui: bool) -> str:
            try:
                if tool_name == "__list_tools__":
                    return registry.format_for_prompt()
                result = await registry.execute(tool_name, tool_input, is_ui)
                return result
            except Exception as e:
                logger.error("Tool execution failed for '%s': %s", tool_name, e)
                return f"Error executing {tool_name}: {str(e)}"
        
        orchestrator.set_tool_executor(tool_executor)
        logger.info("Agent orchestrator ready")
    except Exception as e:
        logger.error("Agent orchestrator initialization failed: %s", e)
        sys.exit(1)
    
    # ─── Step 11b: Initialize Ring Overlay Integration ───────────────────────
    logger.info("Initializing ring overlay integration...")
    try:
        from nexus_overlay.core_integration import get_overlay_integration
        from nexus_audio.wake_word import get_wake_word_daemon
        from nexus_audio.tts_engine import get_tts_engine
        
        overlay_integration = get_overlay_integration()
        
        wake_word_daemon = get_wake_word_daemon() if settings.ENABLE_WAKE_WORD else None
        tts_engine = get_tts_engine() if settings.ENABLE_TTS else None
        
        overlay_integration.initialize(
            orchestrator=orchestrator,
            wake_word_daemon=wake_word_daemon,
            tts_engine=tts_engine,
        )
        
        if wake_word_daemon:
            try:
                wake_word_daemon.start()
                logger.info("Wake word daemon started")
            except Exception as e:
                logger.warning("Failed to start wake word daemon: %s", e)
        
        overlay_integration.start()
        logger.info("Ring overlay integration ready")
    except Exception as e:
        logger.warning("Ring overlay initialization failed (continuing without): %s", e)
        overlay_integration = None
    
    # ─── Shutdown handler ───
    def _shutdown() -> None:
        """Clean up overlay services on exit."""
        try:
            from nexus_overlay.websocket_server import get_ws_server
            from nexus_overlay.audio_tap import get_audio_tap_manager
            ws = get_ws_server()
            if ws.is_running:
                ws.stop()
            audio = get_audio_tap_manager()
            audio.stop()
        except Exception as e:
            logging.getLogger("nexus.boot").debug("Shutdown cleanup error: %s", e)
    
    # ─── Step 12: Handle Ctrl+C Gracefully ──────────────────────────────────
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received — cleaning up...")
        _shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # ─── Step 13: Print Boot Greeting ────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  NEXUS AI v4.0 — Autonomous Desktop Omni-Agent         ║")
    print("║  Hardware: i3 7th Gen · 12GB RAM · Ollama + Groq API   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    if health.warnings:
        print(f"  ({len(health.warnings)} warnings — run 'nexus health' for details)")
    
    # ─── Handle selftest ─────────────────────────────────────────────────────
    if args.selftest:
        from nexus_audio.capture import selftest
        ok = selftest()
        sys.exit(0 if ok else 1)
    
    # ─── Step 14: Start UI ──────────────────────────────────────────────────
    logger.info("Starting UI (%s mode)...", args.ui_mode)
    
    # Get or create persistent event loop for the orchestrator
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if args.ui_mode == "overlay":
        _run_overlay_ui(orchestrator, overlay_integration)
    elif args.ui_mode == "hud":
        _run_hud_ui(orchestrator, loop)
    else:
        _run_headless_ui(orchestrator, loop)
    
    # ─── Step 15: Shutdown ──────────────────────────────────────────────────
    _shutdown()
    logger.info("NEXUS AI shutdown complete.")


def _run_overlay_ui(orchestrator, overlay_integration):
    """Run the ring overlay as primary UI in a separate process."""
    import subprocess
    import sys
    import logging
    
    logger = logging.getLogger("nexus.boot")
    
    overlay_proc = subprocess.Popen([
        sys.executable, "-m", "nexus_overlay.overlay_process"
    ])
    
    logger.info("Overlay process started (PID: %d)", overlay_proc.pid)
    
    try:
        overlay_proc.wait()
    except KeyboardInterrupt:
        overlay_proc.terminate()
        overlay_proc.wait(timeout=2)


def _run_hud_ui(orchestrator, loop: asyncio.AbstractEventLoop):
    """Run the CustomTkinter JARVIS HUD with proper async integration."""
    from nexus_ui.custom_hud import build_hud, run_hud
    from nexus_brain.events import get_event_bus
    
    hud = build_hud()
    
    # Create async queues for event bus
    output_queue = asyncio.Queue(maxsize=500)
    progress_queue = asyncio.Queue(maxsize=200)
    
    # Wire orchestrator queues to event bus
    orchestrator.set_queues(output_queue, progress_queue)
    
    # Wire HUD input callback to submit via orchestrator on the persistent loop
    def on_user_input(text: str) -> None:
        logger = logging.getLogger("nexus.boot")
        logger.info("User input received (%d chars)", len(text))
        
        # Schedule orchestrator.run on the persistent event loop
        async def process():
            try:
                result = await orchestrator.run(text)
                return result
            except Exception as e:
                logger.error("Orchestrator error: %s", e)
                return {"final_response": f"Error: {e}"}
        
        # Run on persistent loop (thread-safe)
        future = asyncio.run_coroutine_threadsafe(process(), loop)
        try:
            result = future.result(timeout=60.0)
        except Exception as e:
            logger.error("Orchestrator future error: %s", e)
            result = {"final_response": f"Error: {e}"}
        
        # The orchestrator now emits typed events directly to the event bus
        # which the HUD subscribes to. No need to push to output_queue manually.
    
    hud.set_input_callback(on_user_input)
    
    # Set initial status
    hud.set_status("Ready")
    
    # Start the UI (this blocks)
    run_hud(hud, headless=False)
    
    # Cleanup
    try:
        loop.call_soon_threadsafe(loop.stop)
    except Exception:
        pass


def _run_headless_ui(orchestrator, loop: asyncio.AbstractEventLoop):
    """Run in headless CLI mode using persistent event loop."""
    logger = logging.getLogger("nexus.boot")
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  JARVIS v4.0 — Autonomous Desktop Omni-Agent           ║")
    print("║  Hardware: i3 7th Gen · 12GB RAM · Ollama + Groq API   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("▶  ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye, Sir.")
            break
        
        if user_input.lower() in ("exit", "quit"):
            break
        
        try:
            future = asyncio.run_coroutine_threadsafe(orchestrator.run(user_input), loop)
            result = future.result(timeout=60.0)
            
            if result and result.get("final_response"):
                print(result["final_response"])
            else:
                print("No response was returned.")
        except Exception as e:
            logger.error("Orchestrator error: %s", e)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()