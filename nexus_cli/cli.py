"""
NEXUS AI v4.0 — Click CLI: nexus run, health, install, config, workflow, memory, logs.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Command-line interface for NEXUS AI management and monitoring.
Provides: run, health, install, workflow, memory, logs, config commands.
"""

import sys
import logging
from typing import Optional

logger = logging.getLogger("nexus.cli")


def cli() -> None:
    """
    Main CLI entry point. Dispatches to subcommands.
    
    Usage:
        nexus run          Start NEXUS AI agent
        nexus health       Run system health check
        nexus install      Install a plugin
        nexus workflow     Manage workflow macros
        nexus memory       Query agent memory
        nexus logs         View audit logs
        nexus config       View/edit configuration
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NEXUS AI v4.0 — Autonomous Desktop Omni-Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nexus run              Start the agent (GUI mode)
  nexus health           Check system health
  nexus install plugin   Install a marketplace plugin
  nexus workflow list    List compiled workflows
  nexus memory query     Query agent memory
  nexus logs             Show recent audit log entries
  nexus config show      Display current configuration
        """,
    )
    
    parser.add_argument("command", nargs="?", default="run", help="Subcommand to execute")
    parser.add_argument("args", nargs="*", help="Additional arguments for subcommand")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    command_map = {
        "run": _cmd_run,
        "health": _cmd_health,
        "install": _cmd_install,
        "workflow": _cmd_workflow,
        "memory": _cmd_memory,
        "logs": _cmd_logs,
        "config": _cmd_config,
    }
    
    cmd_func = command_map.get(args.command)
    if cmd_func:
        cmd_func(args.args, headless=args.headless)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


def _cmd_run(args: list, headless: bool = False) -> None:
    """Start the NEXUS AI agent."""
    from nexus_ui.custom_hud import build_hud, run_hud
    hud = build_hud()
    run_hud(hud, headless=headless)


def _cmd_health(args: list, headless: bool = False) -> None:
    """Run system health check."""
    from nexus_config.health_check import run_health_check, print_health_report
    result = run_health_check()
    print_health_report(result)
    sys.exit(0 if result.ok else 1)


def _cmd_install(args: list, headless: bool = False) -> None:
    """Install a plugin."""
    if not args:
        print("Usage: nexus install <plugin_name>")
        sys.exit(1)
    plugin_name = args[0]
    from nexus_cli.installer import PluginInstaller
    installer = PluginInstaller()
    installer.install(plugin_name)


def _cmd_workflow(args: list, headless: bool = False) -> None:
    """Manage workflow macros."""
    if not args or args[0] == "list":
        from pathlib import Path
        from nexus_config.settings import APP_ROOT
        wf_dir = APP_ROOT / "workflows"
        if wf_dir.exists():
            workflows = list(wf_dir.glob("*.nexflow.json"))
            if workflows:
                print(f"Compiled workflows ({len(workflows)}):")
                for wf in workflows:
                    print(f"  - {wf.stem}")
            else:
                print("No compiled workflows found.")
        else:
            print("No workflows directory found.")
    else:
        print("Usage: nexus workflow [list]")


def _cmd_memory(args: list, headless: bool = False) -> None:
    """Query agent memory."""
    query = " ".join(args) if args else "latest tasks"
    try:
        from nexus_memory.vector_store import get_vector_store
        store = get_vector_store()
        results = store.query_memories(query, n_results=5)
        if results:
            print(f"Memory results for: {query}")
            for i, r in enumerate(results[:5]):
                print(f"  {i+1}. {str(r)[:100]}...")
        else:
            print("No memory results found.")
    except Exception as e:
        print(f"Memory query failed: {e}")


def _cmd_logs(args: list, headless: bool = False) -> None:
    """View recent audit log entries."""
    n = 10
    if args:
        try:
            n = int(args[0])
        except ValueError:
            pass
    
    try:
        from nexus_config.audit_logger import get_audit_logger
        entries = get_audit_logger().get_recent_entries(n)
        if entries:
            print(f"Recent audit log entries ({len(entries)}):")
            for entry in entries:
                ts = entry.get("timestamp", "")[11:19] if len(entry.get("timestamp", "")) >= 19 else ""
                et = entry.get("event_type", "UNKNOWN")
                fn = entry.get("function_name", "unknown")
                ok = "✓" if entry.get("success", True) else "✗"
                print(f"  [{ts}] {et:20s} {fn:25s} {ok}")
        else:
            print("No audit log entries found.")
    except Exception as e:
        print(f"Failed to read audit log: {e}")


def _cmd_config(args: list, headless: bool = False) -> None:
    """View/edit configuration."""
    sub = args[0] if args else "show"
    
    if sub == "show":
        from nexus_config.settings import get_settings
        settings = get_settings()
        print("NEXUS AI Configuration:")
        print(f"  Version: {settings.app_version}")
        print(f"  Platform: {settings.platform_name}")
        print(f"  Ollama URL: {settings.OLLAMA_BASE_URL}")
        print(f"  Primary Model: {settings.OLLAMA_MODEL}")
        print(f"  Intent Model: {settings.OLLAMA_INTENT_MODEL}")
        print(f"  Theme: {settings.UI_THEME}")
        print(f"  Groq Fallback: {'***set***' if settings.GROQ_API_KEY else 'not set (local only)'}")
    else:
        print("Usage: nexus config [show]")


if __name__ == "__main__":
    cli()