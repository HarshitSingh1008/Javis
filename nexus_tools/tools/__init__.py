"""
NEXUS AI v4.0 — Tools sub-package init.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Exports all 22 tool functions for dynamic registration by the ToolRegistry.
Each tool is imported by its tXX_module name for clean namespace access.
"""

from nexus_tools.tools.t01_system_command import run_system_command
from nexus_tools.tools.t02_file_manager import file_manager
from nexus_tools.tools.t03_web_search import web_search
from nexus_tools.tools.t04_web_fetch import web_fetch
from nexus_tools.tools.t05_browser_ghost import browser_ghost
from nexus_tools.tools.t06_desktop_automation import desktop_automation
from nexus_tools.tools.t07_python_interpreter import python_interpreter
from nexus_tools.tools.t08_document_builder import document_builder
from nexus_tools.tools.t09_pdf_reader import pdf_reader
from nexus_tools.tools.t10_window_manager import window_manager
from nexus_tools.tools.t11_clipboard_manager import clipboard_manager
from nexus_tools.tools.t12_local_vector_db import local_vector_db
from nexus_tools.tools.t13_system_monitor import system_monitor
from nexus_tools.tools.t14_code_editor_control import code_editor_control
from nexus_tools.tools.t15_email_client import email_client
from nexus_tools.tools.t16_workflow_macro import workflow_macro
from nexus_tools.tools.t17_image_processor import image_processor
from nexus_tools.tools.t18_data_analyzer import data_analyzer
from nexus_tools.tools.t19_process_manager import process_manager
from nexus_tools.tools.t20_notification_sender import notification_sender
from nexus_tools.tools.t21_calendar_manager import calendar_manager
from nexus_tools.tools.t22_git_operations import git_operations

# ─── Tool Registration List ─────────────────────────────────────────────────
# Used by the ToolRegistry to dynamically register all tools at boot.
ALL_TOOLS = [
    run_system_command,
    file_manager,
    web_search,
    web_fetch,
    browser_ghost,
    desktop_automation,
    python_interpreter,
    document_builder,
    pdf_reader,
    window_manager,
    clipboard_manager,
    local_vector_db,
    system_monitor,
    code_editor_control,
    email_client,
    workflow_macro,
    image_processor,
    data_analyzer,
    process_manager,
    notification_sender,
    calendar_manager,
    git_operations,
]

__all__ = ["ALL_TOOLS"]
# Add tool names (handle both regular functions and StructuredTool objects)
for tool in ALL_TOOLS:
    name = getattr(tool, "__name__", None) or getattr(tool, "name", None)
    if name:
        __all__.append(name)