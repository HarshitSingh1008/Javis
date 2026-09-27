"""
NEXUS AI v4.0 — First-boot 4-screen setup wizard with API key validation.
Hardware: Intel i3 7th Gen · 12GB RAM · Groq API

Guides new users through:
1. Welcome & System Check
2. API Key Configuration (Groq, Porcupine)
3. Theme Selection & Finish
"""

import logging
import sys
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from nexus_config.settings import get_settings, validate_on_boot, APP_ROOT
from nexus_ui.theme_engine import get_theme_engine, ThemeColors

logger = logging.getLogger("nexus.ui.onboarding")


class OnboardingWizard:
    """
    First-boot setup wizard.
    
    Walks the user through configuration screens:
    - Screen 1: Welcome + system status check
    - Screen 2: API key entry (Groq, Porcupine)
    - Screen 3: Theme selection + finish
    
    Each screen has next/back navigation and input validation.
    """
    
    def __init__(self):
        self._settings = get_settings()
        self._theme = get_theme_engine()
        self._current_screen: int = 0
        self._total_screens: int = 3
        self._config: Dict[str, Any] = {}
        self._on_complete: Optional[Callable[[], None]] = None
        
        # Load current settings as defaults
        self._config["GROQ_API_KEY"] = self._settings.GROQ_API_KEY or ""
        self._config["PORCUPINE_ACCESS_KEY"] = self._settings.PORCUPINE_ACCESS_KEY or ""
        self._config["UI_THEME"] = self._settings.UI_THEME
    
    def set_on_complete(self, callback: Callable[[], None]) -> None:
        """Set callback for when wizard completes."""
        self._on_complete = callback
    
    def is_first_run(self) -> bool:
        """
        Check if this is the first run (no .env file or no API keys).
        
        Returns:
            True if onboarding should be shown.
        """
        env_file = APP_ROOT / ".env"
        if not env_file.exists():
            return True
        
        boot = validate_on_boot()
        return len(boot.missing_keys) > 0
    
    def run_gui(self) -> None:
        """
        Run the onboarding wizard GUI.
        
        Creates a customtkinter window with 4 screens.
        Falls back to CLI mode if customtkinter is unavailable.
        """
        try:
            import customtkinter as ctk
            self._run_ctk_wizard()
        except ImportError:
            self._run_cli_wizard()
    
    def _run_ctk_wizard(self) -> None:
        """Run the onboarding wizard using customtkinter."""
        try:
            import customtkinter as ctk
        except ImportError:
            self._run_cli_wizard()
            return
        
        colors = self._theme.colors
        
        # Create window
        window = ctk.CTk()
        window.title("NEXUS AI v4.0 — Setup Wizard")
        window.geometry("600x500")
        window.configure(fg_color=colors.bg_primary)
        window.resizable(False, False)
        
        # Main container
        container = ctk.CTkFrame(window, fg_color=colors.bg_primary)
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            container, text="Welcome to NEXUS AI",
            font=self._theme.get_font("title", bold=True),
            text_color=colors.accent_primary,
        )
        title.pack(pady=(20, 10))
        
        # Screen content (will be swapped)
        content_frame = ctk.CTkFrame(container, fg_color=colors.bg_secondary)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self._window = window
        self._closed = False
        self._current_screen = 0
        self._fields = {}
        self._content_frame = content_frame
        self._title_label = title
        self._ctk = ctk
        window.protocol("WM_DELETE_WINDOW", self._close_window)
        
        # Button frame
        button_frame = ctk.CTkFrame(container, fg_color=colors.bg_primary)
        button_frame.pack(fill="x", pady=(10, 0))
        
        skip_button = ctk.CTkButton(
            button_frame, text="Skip Setup",
            command=self._close_window,
            fg_color=colors.bg_tertiary,
            hover_color=colors.bg_hover,
            text_color=colors.text_secondary,
            font=self._theme.get_font("small"),
        )
        skip_button.pack(side="left", padx=5)
        
        next_button = ctk.CTkButton(
            button_frame, text="Next →",
            command=self._next_screen,
            fg_color=colors.accent_primary,
            hover_color=colors.accent_secondary,
            text_color=colors.bg_primary,
            font=self._theme.get_font("small", bold=True),
        )
        next_button.pack(side="right", padx=5)
        
        # Progress dots
        progress_frame = ctk.CTkFrame(container, fg_color=colors.bg_primary, height=30)
        progress_frame.pack(fill="x", pady=(10, 0))
        
        progress_label = ctk.CTkLabel(
            progress_frame,
            text="Step 1 of 4",
            font=self._theme.get_font("small"),
            text_color=colors.text_secondary,
        )
        progress_label.pack()
        
        self._next_button = next_button
        self._progress_label = progress_label
        self._back_button = ctk.CTkButton(button_frame, text="Back", command=self._back_screen)
        self._back_button.pack(side="right", padx=5)
        self._render_screen()
        window.mainloop()

    def _capture_fields(self) -> None:
        for key, widget in self._fields.items():
            self._config[key] = widget.get().strip()

    def _next_screen(self) -> None:
        if self._closed:
            return
        self._capture_fields()
        if self._current_screen < self._total_screens - 1:
            self._current_screen += 1
            self._render_screen()
            return
        try:
            self._save_config()
        except OSError:
            self._progress_label.configure(text="Could not save settings. Please check file permissions and retry.")
            return
        self._close_window()
        if self._on_complete:
            self._on_complete()

    def _back_screen(self) -> None:
        if self._closed:
            return
        self._capture_fields()
        if self._current_screen > 0:
            self._current_screen -= 1
            self._render_screen()

    def _close_window(self) -> None:
        if self._closed:
            return
        self._closed = True
        # This wizard owns a separate Tk root. Cancel its pending events, but
        # leave registered commands for their owning widgets to delete during
        # destroy(). Calling root.after_cancel() on child-owned timers deletes
        # commands without updating the child's bookkeeping (double deletion).
        for callback in self._window.tk.call("after", "info"):
            self._window.tk.call("after", "cancel", callback)
        self._window.destroy()

    def _render_screen(self) -> None:
        ctk = self._ctk
        for child in self._content_frame.winfo_children():
            child.destroy()
        self._fields = {}
        titles = ["Welcome to NEXUS AI", "API Key Configuration", "Theme & Finish"]
        self._title_label.configure(text=titles[self._current_screen])
        self._progress_label.configure(text=f"Step {self._current_screen + 1} of {self._total_screens}")
        self._back_button.configure(state="disabled" if self._current_screen == 0 else "normal")
        self._next_button.configure(text="Finish & Launch" if self._current_screen == 2 else "Next →")

        def label(text):
            ctk.CTkLabel(self._content_frame, text=text, wraplength=480,
                         justify="left").pack(padx=20, pady=5)

        def entry(key, title, secret=False):
            label(title)
            widget = ctk.CTkEntry(self._content_frame, width=440, show="*" if secret else "")
            widget.insert(0, self._config.get(key, ""))
            widget.pack(padx=20, pady=3)
            self._fields[key] = widget

        if self._current_screen == 0:
            label("Set up your AI desktop agent. Choose Next to configure your API key and theme.")
            label(f"Platform: {sys.platform} | Python: {sys.version.split()[0]}")
            label("All processing uses Groq cloud models. A free API key is required.")
        elif self._current_screen == 1:
            entry("GROQ_API_KEY", "Groq API key (required)\nGet one free at: https://console.groq.com", secret=True)
            entry("PORCUPINE_ACCESS_KEY", "Porcupine access key (optional, wake word)", secret=True)
        else:
            label("Choose your theme. Finish saves your configuration and launches the application.")
            theme = ctk.CTkOptionMenu(self._content_frame,
                                      values=["dark_platinum", "midnight_blue", "void_black"])
            theme.set(self._config["UI_THEME"])
            theme.pack(pady=15)
            self._fields["UI_THEME"] = theme
            label("API keys are stored locally. You can rerun setup with --onboarding.")
    
    def _run_cli_wizard(self) -> None:
        """Run the onboarding wizard using CLI prompts."""
        print("\n" + "=" * 50)
        print("  NEXUS AI v4.0 — First-Time Setup")
        print("=" * 50)
        
        print("\nWelcome! Let's get you set up in a few quick steps.\n")
        
        # Step 1: Groq API Key
        print("[Step 1/3] API Key Configuration")
        print("-" * 40)
        print("Get a free Groq API key at: https://console.groq.com")
        current_key = self._config.get("GROQ_API_KEY", "")
        if current_key:
            print(f"  Current key: {current_key[:8]}... (already set)")
        else:
            key = input("  Enter Groq API Key (required): ").strip()
            if key:
                self._config["GROQ_API_KEY"] = key
        
        # Optional: Porcupine key
        porc_key = self._config.get("PORCUPINE_ACCESS_KEY", "")
        if not porc_key:
            key = input("  Enter Porcupine API Key (optional, for wake word): ").strip()
            if key:
                self._config["PORCUPINE_ACCESS_KEY"] = key
        
        # Step 2: Theme
        print("\n[Step 2/3] Theme Selection")
        print("-" * 40)
        print("  1) Dark Platinum (default) — Dark gray with cyan accents")
        print("  2) Midnight Blue — Deep blue with gold accents")
        print("  3) Void Black — True black with purple accents")
        theme_choice = input("  Choose theme [1]: ").strip()
        theme_map = {"1": "dark_platinum", "2": "midnight_blue", "3": "void_black"}
        self._config["UI_THEME"] = theme_map.get(theme_choice, "dark_platinum")
        
        # Step 3: Save
        self._save_config()
        
        print("\n[Step 3/3] Setup Complete!")
        print("-" * 40)
        print("  Configuration saved to ~/.nexus_ai/.env")
        print("  Models: deepseek-r1-distill-llama-70b (planner)")
        print("          qwen-2.5-coder-32b (automation)")
        print("          llama-3.1-8b-instant (voice)")
        print("  You can change these settings anytime.\n")
        
        if self._on_complete:
            self._on_complete()
    
    def _save_config(self) -> None:
        """Save configuration to .env file."""
        env_file = APP_ROOT / ".env"
        
        # Update only wizard-owned keys; retain other settings and comments.
        from dotenv import set_key
        import os

        env_file.parent.mkdir(parents=True, exist_ok=True)
        for key, value in self._config.items():
            set_key(str(env_file), key, str(value), encoding="utf-8")
            # Settings may already have been loaded into the process environment.
            os.environ[key] = str(value)
        get_settings.cache_clear()
        self._settings = get_settings()
        self._theme.set_theme(self._config["UI_THEME"])
        logger.info("Onboarding config saved to %s", env_file)
