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
    parser.add_argument("--headless", action="store_true",
                       help="Run without GUI (CLI/text mode)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose debug logging")
    parser.add_argument("--health-only", action="store_true",
                       help="Run health check and exit")
    parser.add_argument("--onboarding", action="store_true",
                       help="Force show onboarding wizard")
    args = parser.parse_args()
    
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
    
    # ─── Step 2: Initialize Configuration ────────────────────────────────────
    from nexus_config.settings import get_settings, validate_on_boot, APP_ROOT
    from nexus_config.audit_logger import get_audit_logger
    
    settings = get_settings()
    logger.info("Configuration loaded from %s", APP_ROOT / ".env")
    
    # ─── Step 3: Create Runtime Directories ──────────────────────────────────
    # The settings module already creates required directories at import time.
    # Ensure temp and crash_reports exist.
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
    
    # Print degraded subsystems
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
        # Call the original excepthook
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
        logger.warning("Tool registry initialization failed: %s", e)
    
    # ─── Step 9: Initialize Memory Subsystem ────────────────────────────────
    logger.info("Initializing memory subsystem...")
    try:
        from nexus_memory.vector_store import get_vector_store
        store = get_vector_store()
        # Test connectivity
        store.heartbeat()
        logger.info("Memory subsystem ready (ChromaDB)")
    except Exception as e:
        logger.warning("Memory subsystem initialized with in-memory fallback: %s", e)
    
    # ─── Step 10: Initialize Audio Subsystem ─────────────────────────────────
    if settings.ENABLE_WAKE_WORD or settings.ENABLE_TTS:
        logger.info("Initializing audio subsystem...")
        # Audio is initialized on-demand by the wake word / TTS modules
        logger.info("Audio subsystem ready (on-demand initialization)")
    
    # ─── Step 11: Initialize Agent Orchestrator ──────────────────────────────
    logger.info("Initializing agent orchestrator...")
    try:
        from nexus_brain.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        
        # Set up tool executor
        async def tool_executor(tool_name: str, tool_input: Dict[str, Any], is_ui: bool) -> str:
            """Execute a tool from the registry."""
            try:
                if tool_name == "__list_tools__":
                    return registry.format_for_prompt()
                
                result = await registry.execute(tool_name, tool_input)
                return result
            except Exception as e:
                logger.error("Tool execution failed for '%s': %s", tool_name, e)
                return f"Error executing {tool_name}: {str(e)}"
        
        orchestrator.set_tool_executor(tool_executor)
        logger.info("Agent orchestrator ready")
    except Exception as e:
        logger.error("Agent orchestrator initialization failed: %s", e)
        sys.exit(1)
    
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
    
    # ─── Step 14: Start UI ──────────────────────────────────────────────────
    logger.info("Starting UI (%s mode)...", "headless" if args.headless else "GUI")
    
    from nexus_ui.custom_hud import build_hud, run_hud
    hud = build_hud()
    
    # Wire orchestrator to HUD
    def on_user_input(text: str) -> None:
        """Callback when user submits input to the HUD."""
        logger.info("User input received (%d chars)", len(text))
        try:
            # Queue the orchestrator to run
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def process():
                result = await orchestrator.run(text)
                return result
            
            result = loop.run_until_complete(process())
            loop.close()
            
            if result:
                hud.output_queue.put(str(result))
        except Exception as e:
            logger.error("Orchestrator error: %s", e)
            hud.set_status(f"Error: {e}")
    
    hud.set_input_callback(on_user_input)
    
    # Set status
    hud.set_status("Ready")
    
    # Start the UI
    run_hud(hud, headless=args.headless)
    
    # ─── Step 15: Shutdown ──────────────────────────────────────────────────
    _shutdown()
    
    logger.info("NEXUS AI shutdown complete.")


def _shutdown():
    """Graceful shutdown of all subsystems."""
    logger = logging.getLogger("nexus.boot")
    
    # Save session context
    try:
        from nexus_memory.session_context import get_session_context
        ctx = get_session_context()
        ctx.save()
        logger.info("Session context saved")
    except Exception as e:
        logger.debug("Session context save skipped: %s", e)
    
    # Flush metrics
    try:
        from nexus_config.metrics import get_metrics_collector
        metrics = get_metrics_collector()
        metrics.shutdown()
        logger.info("Metrics flushed")
    except Exception as e:
        logger.debug("Metrics flush skipped: %s", e)
    
    # Flush audit log
    try:
        from nexus_config.audit_logger import get_audit_logger
        audit = get_audit_logger()
        audit._auto_flush_loop()  # Final flush
        logger.info("Audit log flushed")
    except Exception:
        pass


if __name__ == "__main__":
    main()