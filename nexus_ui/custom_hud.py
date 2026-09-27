"""
NEXUS AI v4.0 — JARVIS HUD: Immersive Iron Man Style Interface
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

The main user interface with JARVIS-inspired design:
- Full-screen immersive layout with glass-morphism panels
- Zone 1 (Left): Particle system, arc reactor, system metrics, status rings
- Zone 2 (Center): Conversation log with streaming text, holographic styling
- Zone 3 (Right): DAG execution visualization with connection pulses
- Zone 4 (Bottom): Input bar with waveform, send button, mic toggle
- Overlay: Scanlines, connection pulses, reactive particles

Typed event bus architecture ensures the UI thread is NEVER blocked.
Only AssistantTextDelta/AssistantTextFinal/UIStateChange events render to chat.
"""

import logging
import queue
import re
import threading
import time
from typing import Optional, Dict, Any, Callable
from functools import lru_cache

from nexus_config.settings import get_settings
from nexus_ui.theme_engine import get_theme_engine, ThemeColors
from nexus_ui.animation_engine import (
    AnimationEngine, render_all, render_scanlines, render_connection_pulses
)
from nexus_ui.notification_manager import NotificationManager, get_notification_manager, Notification
from nexus_ui.dag_visualizer import DAGVisualizer
from nexus_brain.events import (
    get_event_bus,
    EventType,
    UIState,
    AssistantTextDelta,
    AssistantTextFinal,
    UserMessage,
    UIStateChange,
    ToolProgress,
)

logger = logging.getLogger("nexus.ui.hud")


# ─── Sanitizer for Leak Prevention (Bug B) ────────────────────────────────────

# Patterns that MUST NEVER appear in chat
_LEAK_PATTERNS = [
    (re.compile(r"```tool\s*\n.*?\n```", re.DOTALL), ""),
    (re.compile(r"TOOL CALL:\s*\w+"), ""),
    (re.compile(r"PARAMS:\s*\{.*?\}"), ""),
    (re.compile(r"<function=\w+>"), ""),
    (re.compile(r"\{\s*\"tool\"\s*:\s*\".*?\"\s*,\s*\"input\"\s*:\s*\{.*?\}\s*\}"), ""),
    (re.compile(r"Thought:|Action:|Observation:"), ""),
    (re.compile(r"\[Tool \w+ returned:.*?\]"), ""),
    (re.compile(r"\[Tool \w+ error:.*?\]"), ""),
]

def sanitize_for_chat(text: str) -> str:
    """Strip any leaked tool call patterns from text before rendering."""
    if not text:
        return ""
    result = text
    for pattern, replacement in _LEAK_PATTERNS:
        result = pattern.sub(replacement, result)
    # Clean up excessive whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()
    return result


def has_leaked_content(text: str) -> bool:
    """Check if text contains any leaked patterns (for testing)."""
    for pattern, _ in _LEAK_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ─── NexusHUD Class ──────────────────────────────────────────────────────────

class NexusHUD:
    """
    Main NEXUS AI JARVIS HUD window.
    
    Typed event bus architecture:
    - Subscribes to UI events: AssistantTextDelta, AssistantTextFinal, UserMessage, UIStateChange, ToolProgress
    - ToolCall, ToolResult, Trace, Error go to audit log ONLY (never to chat)
    
    All queue polling happens via root.after() timer callbacks.
    """
    
    def __init__(self):
        self._settings = get_settings()
        self._theme = get_theme_engine()
        self._animation = AnimationEngine()
        self._notifications = get_notification_manager()
        self._dag = DAGVisualizer()
        
        # Root window (will be set by run_hud)
        self.root = None
        self._main_canvas = None  # Main overlay canvas for scanlines/pulses
        
        # Event bus
        self._event_bus = get_event_bus()
        
        # UI state
        self._running = False
        self._input_callback: Optional[Callable[[str], None]] = None
        self._conversation_lines: list = []
        self._max_conversation_lines: int = 1000
        self._is_listening: bool = False
        self._waveform_active: bool = False
        self._current_assistant_message: str = ""  # Buffer for streaming
        
        # Widget references (set during _build_ui)
        self._conversation_text = None
        self._input_entry = None
        self._send_button = None
        self._mic_button = None
        self._status_label = None
        self._status_ring_canvas = None
        self._metrics_labels: Dict[str, object] = {}
        self._particle_canvas = None
        self._reactor_canvas = None
        self._waveform_canvas = None
        self._dag_canvas = None
        self._left_panel = None
        self._center_panel = None
        self._right_panel = None
        self._bottom_panel = None
        self._top_bar = None
        self._version_label = None
        self._time_label = None
        
        # Animation loop handle
        self._animation_after_id: Optional[str] = None
        self._poll_after_id: Optional[str] = None
        self._clock_after_id: Optional[str] = None
        
        # Voice capture state
        self._voice_capture_active = False
        self._voice_capture_thread: Optional[threading.Thread] = None
    
    def set_input_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for when user submits input (text or voice)."""
        self._input_callback = callback
    
    def submit_input(self, text: str) -> None:
        """Submit user input to the agent."""
        if not text or not text.strip():
            return
        
        if self._input_callback:
            self._input_callback(text.strip())
        
        # Clear input
        if self._input_entry:
            self._input_entry.delete(0, "end")
    
    def render_assistant_message(self, delta: str, is_final: bool = False) -> None:
        """
        Single sink for rendering assistant messages.
        Only called for AssistantTextDelta and AssistantTextFinal events.
        """
        if not self._conversation_text:
            return
        
        try:
            # Sanitize before rendering (defense in depth)
            clean_delta = sanitize_for_chat(delta)
            if not clean_delta and not is_final:
                return
            
            self._conversation_text.configure(state="normal")
            
            if is_final:
                # Final message - add with "assistant" tag
                self._conversation_text.insert("end", clean_delta, "assistant")
                self._conversation_text.insert("end", "\n\n")
                self._current_assistant_message = ""
            else:
                # Streaming delta
                self._conversation_text.insert("end", clean_delta, "assistant")
                self._current_assistant_message += clean_delta
            
            self._conversation_text.see("end")
            self._conversation_text.configure(state="disabled")
        except Exception as e:
            logger.debug(f"Render assistant message error: {e}")
    
    def render_user_message(self, text: str) -> None:
        """Render user message (typed or transcribed)."""
        if not self._conversation_text:
            return
        
        try:
            self._conversation_text.configure(state="normal")
            self._conversation_text.insert("end", f"{text}\n\n", "user")
            self._conversation_text.see("end")
            self._conversation_text.configure(state="disabled")
        except Exception as e:
            logger.debug(f"Render user message error: {e}")
    
    def set_status(self, status: str) -> None:
        """Update the status bar text."""
        if self._status_label:
            try:
                self._status_label.configure(text=status)
            except Exception:
                pass
    
    def set_ui_state(self, state: UIState, detail: str = "") -> None:
        """Update UI state (idle/listening/thinking/speaking/error)."""
        self._is_listening = (state == UIState.LISTENING)
        
        status_text = {
            UIState.IDLE: "✓ READY",
            UIState.LISTENING: "🎤 LISTENING...",
            UIState.THINKING: "🤔 THINKING...",
            UIState.SPEAKING: "🔊 SPEAKING...",
            UIState.ERROR: "✗ ERROR",
        }.get(state, state.value.upper())
        
        if detail:
            status_text += f" — {detail}"
        
        self.set_status(status_text)
        
        # Update mic button visual state
        if self._mic_button:
            try:
                colors = self._theme.colors
                if state == UIState.LISTENING:
                    self._mic_button.configure(
                        text="● LISTENING",
                        fg_color=colors.accent_error,
                        hover_color="#cc0000",
                    )
                    self._waveform_active = True
                elif state == UIState.SPEAKING:
                    self._mic_button.configure(
                        text="🔊 SPEAKING",
                        fg_color=colors.accent_gold,
                        hover_color=colors.accent_gold_dim,
                    )
                    self._waveform_active = True
                else:
                    self._mic_button.configure(
                        text="🎤 MIC",
                        fg_color=colors.accent_primary,
                        hover_color=colors.accent_secondary,
                    )
                    self._waveform_active = False
            except Exception:
                pass
        
        # Reactor intensity
        intensity_map = {
            UIState.IDLE: 0.3,
            UIState.LISTENING: 1.0,
            UIState.THINKING: 0.5,
            UIState.SPEAKING: 0.8,
            UIState.ERROR: 1.0,
        }
        self._animation.set_reactor_intensity(intensity_map.get(state, 0.5))
        
        # Reactive particles on state change
        if self._reactor_canvas and state in (UIState.LISTENING, UIState.ERROR):
            cx = self._reactor_canvas.winfo_width() // 2
            cy = self._reactor_canvas.winfo_height() // 2
            count = 20 if state == UIState.ERROR else 12
            self._animation.trigger_reactive_particles(cx, cy, count)
    
    def update_metrics(self, cpu: float, ram: float, disk: float) -> None:
        """Update the metrics display."""
        labels = {
            "cpu": f"CPU  {cpu:>5.1f}%",
            "ram": f"RAM  {ram:>6.0f}MB",
            "disk": f"DISK {disk:>5.1f}GB",
        }
        for key, text in labels.items():
            if key in self._metrics_labels:
                try:
                    self._metrics_labels[key].configure(text=text)
                except Exception:
                    pass
        
        # Update progress bars
        if "cpu_bar" in self._metrics_labels:
            try:
                self._metrics_labels["cpu_bar"].set(cpu / 100.0)
            except Exception:
                pass
        if "ram_bar" in self._metrics_labels:
            try:
                # Estimate RAM percentage (rough)
                self._metrics_labels["ram_bar"].set(min(ram / 12000.0, 1.0))
            except Exception:
                pass
    
    def _poll_event_bus(self) -> None:
        """Poll event bus queues for new events (called via root.after)."""
        if not self._running:
            return
        
        # The event bus pushes to our queues directly
        # We process them here
        
        # Process UI event queue (would be separate in production)
        # For now, we rely on the event bus calling our handlers directly
        # via the async event loop. This is a simplified version.
        
        # Notifications
        self._notifications.process_queue()
        
        # Schedule next poll
        if self._running and self.root:
            try:
                self._poll_after_id = self.root.after(50, self._poll_event_bus)
            except Exception:
                pass
    
    def start(self) -> None:
        """Start the UI main loop and subscribe to event bus."""
        self._running = True
        
        # Subscribe to typed events
        self._event_bus.subscribe(EventType.ASSISTANT_TEXT_DELTA, self._on_text_delta)
        self._event_bus.subscribe(EventType.ASSISTANT_TEXT_FINAL, self._on_text_final)
        self._event_bus.subscribe(EventType.USER_MESSAGE, self._on_user_message)
        self._event_bus.subscribe(EventType.UI_STATE_CHANGE, self._on_ui_state_change)
        self._event_bus.subscribe(EventType.TOOL_PROGRESS, self._on_tool_progress)
        
        if self.root:
            self._poll_after_id = self.root.after(100, self._poll_event_bus)
            self._animation_after_id = self.root.after(16, self._animation_loop)
            self._clock_after_id = self.root.after(1000, self._update_clock)
            self._notifications.set_toast_callback(self._show_toast)
    
    def stop(self) -> None:
        """Stop the UI main loop."""
        self._running = False
        if self._poll_after_id:
            try:
                self.root.after_cancel(self._poll_after_id)
            except Exception:
                pass
        if self._animation_after_id:
            try:
                self.root.after_cancel(self._animation_after_id)
            except Exception:
                pass
        if self._clock_after_id:
            try:
                self.root.after_cancel(self._clock_after_id)
            except Exception:
                pass
        
        # Stop voice capture if active
        self._stop_voice_capture()
    
    # ── Event Handlers ───────────────────────────────────────────────────────
    
    async def _on_text_delta(self, event: AssistantTextDelta) -> None:
        """Handle streaming text delta from response AI."""
        if self.root:
            # Schedule on UI thread
            self.root.after(0, lambda: self.render_assistant_message(event.content, is_final=False))
    
    async def _on_text_final(self, event: AssistantTextFinal) -> None:
        """Handle final text from response AI."""
        if self.root:
            self.root.after(0, lambda: self.render_assistant_message(event.content, is_final=True))
            # Ensure UI state returns to idle
            self.root.after(0, lambda: self.set_ui_state(UIState.IDLE))
    
    async def _on_user_message(self, event: UserMessage) -> None:
        """Handle user message (typed or voice)."""
        if self.root:
            self.root.after(0, lambda: self.render_user_message(event.content))
            # Show thinking state while agent processes
            self.root.after(0, lambda: self.set_ui_state(UIState.THINKING, "Processing..."))
    
    async def _on_ui_state_change(self, event: UIStateChange) -> None:
        """Handle UI state change."""
        if self.root:
            self.root.after(0, lambda: self.set_ui_state(event.state, event.detail))
    
    async def _on_tool_progress(self, event: ToolProgress) -> None:
        """Handle tool progress for DAG visualization."""
        if self._dag_canvas:
            self._dag.update_node_status(event.tool_name, event.status, 0, None)
            if event.status == "running":
                # Find node and pulse
                if event.tool_name in self._dag.nodes:
                    node = self._dag.nodes[event.tool_name]
                    cx = node.x + node.width / 2
                    cy = node.y + node.height / 2
                    for dep_id in node.depends_on:
                        if dep_id in self._dag.nodes:
                            dep = self._dag.nodes[dep_id]
                            dep_cx = dep.x + dep.width / 2
                            dep_cy = dep.y + dep.height / 2
                            self._animation.add_connection_pulse(dep_cx, dep_cy, cx, cy)
    
    # ── Voice Capture (Bug D Fix) ────────────────────────────────────────────
    
    def _start_voice_capture(self) -> None:
        """Start push-to-talk voice capture using shared AudioCapture service."""
        if self._voice_capture_active:
            return
        
        self._voice_capture_active = True
        
        def capture_thread():
            try:
                # Import here to avoid circular deps
                from nexus_audio.capture import get_audio_capture
                
                capture = get_audio_capture()
                
                # Start capture with VAD endpointing
                def on_transcript(text: str):
                    if text and text.strip():
                        # Submit to agent on UI thread
                        if self.root and self._input_callback:
                            self.root.after(0, lambda: self._input_callback(text.strip()))
                
                # This runs in background thread
                capture.capture_once(
                    callback=on_transcript,
                    max_duration=30.0,
                    silence_timeout=0.7,
                )
                
            except Exception as e:
                logger.error(f"Voice capture error: {e}")
                if self.root:
                    self.root.after(0, lambda: self.set_ui_state(UIState.ERROR, f"Mic error: {e}"))
            finally:
                self._voice_capture_active = False
                if self.root:
                    self.root.after(0, lambda: self.set_ui_state(UIState.IDLE))
        
        self._voice_capture_thread = threading.Thread(
            target=capture_thread,
            daemon=True,
            name="nexus-voice-capture",
        )
        self._voice_capture_thread.start()
    
    def _stop_voice_capture(self) -> None:
        """Stop voice capture."""
        self._voice_capture_active = False
        # The capture thread will exit on its own via timeout or callback
    
    def _on_mic_button(self) -> None:
        """Handle mic button click - toggle push-to-talk."""
        if self._voice_capture_active:
            self._stop_voice_capture()
            self.set_ui_state(UIState.IDLE)
        else:
            self.set_ui_state(UIState.LISTENING, "Speak now...")
            self._start_voice_capture()
    
    # ── Animation Loop ───────────────────────────────────────────────────────
    
    def _animation_loop(self) -> None:
        """Animation frame update loop with per-frame budget."""
        if not self._running:
            return
        
        frame_start = time.perf_counter()
        
        dt = 1.0 / 60
        self._animation.update(dt)
        
        # Render all animation layers
        render_all(self._animation)
        
        # Render DAG
        if self._dag_canvas:
            self._dag.render(self._dag_canvas)
        
        # Render scanlines and connection pulses on main canvas
        if self._main_canvas:
            render_scanlines(self._animation, self._main_canvas)
            render_connection_pulses(self._animation, self._main_canvas)
        
        # Frame budget: drop to 30fps if frame takes > 16ms, pause if > 33ms
        frame_time = (time.perf_counter() - frame_start) * 1000
        next_delay = 16
        if frame_time > 16:
            next_delay = 33  # ~30fps
        if frame_time > 33:
            next_delay = 100  # pause
        
        if self.root:
            self._animation_after_id = self.root.after(next_delay, self._animation_loop)
    
    def _update_clock(self) -> None:
        """Update the clock in the top bar."""
        if not self._running or not self._time_label:
            return
        
        try:
            import datetime
            now = datetime.datetime.now()
            time_str = now.strftime("%H:%M:%S")
            date_str = now.strftime("%a, %b %d")
            self._time_label.configure(text=f"{time_str}  |  {date_str}")
        except Exception:
            pass
        
        if self.root:
            self._clock_after_id = self.root.after(1000, self._update_clock)
    
    def _show_toast(self, notification: Notification) -> None:
        """Display an in-app toast notification."""
        if not self.root:
            return
        
        try:
            import customtkinter as ctk
            colors = self._theme.colors
            
            toast = ctk.CTkToplevel(self.root)
            toast.overrideredirect(True)
            toast.attributes("-topmost", True)
            toast.attributes("-alpha", 0.95)
            
            # Position at top-right
            self.root.update_idletasks()
            rx = self.root.winfo_x()
            ry = self.root.winfo_y()
            rw = self.root.winfo_width()
            toast.geometry(f"350x80+{rx + rw - 370}+{ry + 60}")
            
            frame = ctk.CTkFrame(toast, fg_color=colors.bg_glass, border_width=1, border_color=colors.border_gold)
            frame.pack(fill="both", expand=True, padx=2, pady=2)
            
            title_color = colors.accent_error if notification.priority.value in ("high", "critical") else colors.text_gold
            title_lbl = ctk.CTkLabel(
                frame, text=notification.title,
                font=self._theme.get_font("small", bold=True),
                text_color=title_color,
            )
            title_lbl.pack(anchor="w", padx=15, pady=(10, 0))
            
            msg_lbl = ctk.CTkLabel(
                frame, text=notification.message,
                font=self._theme.get_font("small"),
                text_color=colors.text_secondary,
                wraplength=320,
            )
            msg_lbl.pack(anchor="w", padx=15, pady=(0, 10))
            
            # Auto-dismiss
            def dismiss():
                try:
                    toast.destroy()
                except Exception:
                    pass
            
            toast.after(notification.timeout_ms, dismiss)
            
        except Exception as e:
            logger.debug(f"Toast error: {e}")


# ─── Build & Run Functions ──────────────────────────────────────────────────

def build_hud() -> NexusHUD:
    """Build and return the NEXUS AI JARVIS HUD instance."""
    return NexusHUD()


def run_hud(hud: Optional[NexusHUD] = None, headless: bool = False) -> None:
    """Run the NEXUS AI JARVIS HUD main loop."""
    if headless:
        _run_headless(hud or NexusHUD())
        return
    
    if hud is None:
        hud = build_hud()
    
    try:
        import customtkinter as ctk
        ctk.set_appearance_mode("dark")
        
        root = ctk.CTk()
        hud.root = root
        
        root.title("JARVIS")
        root.geometry(f"{hud._settings.WINDOW_WIDTH}x{hud._settings.WINDOW_HEIGHT}")
        root.configure(fg_color=hud._theme.colors.bg_primary)
        
        # Make window slightly transparent for glass effect (Windows only)
        try:
            root.attributes("-alpha", 0.98)
        except Exception:
            pass
        
        # Build UI
        _build_jarvis_ui(hud, root)
        
        # Start
        hud.start()
        root.mainloop()
    
    except ImportError:
        logger.warning("customtkinter not available, running headless")
        _run_headless(hud)
    except Exception as e:
        logger.error(f"HUD error: {e}")
        _run_headless(hud)


def _build_jarvis_ui(hud: NexusHUD, root: object) -> None:
    """Build the JARVIS-style CustomTkinter UI layout."""
    try:
        import customtkinter as ctk
        colors = hud._theme.colors
        
        # ─── Main Container ─────────────────────────────────────────────────
        main_frame = ctk.CTkFrame(root, fg_color=colors.bg_primary, border_width=0)
        main_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Main overlay canvas for scanlines and connection pulses
        hud._main_canvas = ctk.CTkCanvas(
            main_frame, bg=colors.bg_primary, highlightthickness=0, bd=0
        )
        hud._main_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        hud._animation.set_canvases(hud_canvas=hud._main_canvas)
        
        # Bind mouse for reactive particles
        def on_mouse_move(event):
            hud._animation.set_mouse_position(event.x, event.y)
        hud._main_canvas.bind("<Motion>", on_mouse_move)
        
        # ─── Top Bar ────────────────────────────────────────────────────────
        top_frame = ctk.CTkFrame(
            main_frame, fg_color=colors.bg_glass, border_width=1, border_color=colors.border_glass,
            height=48, corner_radius=0
        )
        top_frame.pack(fill="x", padx=0, pady=0)
        top_frame.pack_propagate(False)
        hud._top_bar = top_frame
        
        # Left side - JARVIS logo/title
        left_top = ctk.CTkFrame(top_frame, fg_color="transparent")
        left_top.pack(side="left", padx=20, pady=4)
        
        title_frame = ctk.CTkFrame(left_top, fg_color="transparent")
        title_frame.pack(side="left")
        
        jarvis_label = ctk.CTkLabel(
            title_frame, text="JARVIS",
            font=hud._theme.get_font("hero", bold=True),
            text_color=colors.text_gold,
        )
        jarvis_label.pack(side="left")
        
        version_label = ctk.CTkLabel(
            title_frame, text="  v4.0",
            font=hud._theme.get_font("small", bold=True),
            text_color=colors.text_dim,
        )
        version_label.pack(side="left", pady=(8, 0))
        hud._version_label = version_label
        
        # Status ring indicator
        status_ring_frame = ctk.CTkFrame(left_top, fg_color="transparent", width=48, height=48)
        status_ring_frame.pack(side="left", padx=15)
        status_ring_frame.pack_propagate(False)
        
        hud._status_ring_canvas = ctk.CTkCanvas(
            status_ring_frame, bg=colors.bg_glass, highlightthickness=0, width=40, height=40
        )
        hud._status_ring_canvas.pack(expand=True)
        hud._animation.set_canvases(particle_canvas=hud._status_ring_canvas)
        
        # Center - System status
        center_top = ctk.CTkFrame(top_frame, fg_color="transparent")
        center_top.pack(side="left", expand=True)
        
        hud._status_label = ctk.CTkLabel(
            center_top, text="INITIALIZING SYSTEMS...",
            font=hud._theme.get_font("normal", bold=True),
            text_color=colors.text_accent,
        )
        hud._status_label.pack(pady=8)
        
        # Right side - Clock
        right_top = ctk.CTkFrame(top_frame, fg_color="transparent")
        right_top.pack(side="right", padx=20, pady=4)
        
        hud._time_label = ctk.CTkLabel(
            right_top, text="00:00:00",
            font=hud._theme.get_mono_font("large"),
            text_color=colors.text_gold,
        )
        hud._time_label.pack()
        
        # ════════════════════════════════════════════════════════════════════
        # ZONE 4: BOTTOM PANEL - Input Bar with Waveform
        # ════════════════════════════════════════════════════════════════════
        bottom_frame = ctk.CTkFrame(
            main_frame, fg_color=colors.bg_glass, border_width=1, border_color=colors.border_glass,
            height=100, corner_radius=0
        )
        bottom_frame.pack(fill="x", padx=0, pady=0, side="bottom")
        bottom_frame.pack_propagate(False)
        hud._bottom_panel = bottom_frame
        
        # Waveform canvas (top of bottom panel)
        waveform_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent", height=30)
        waveform_frame.pack(fill="x", padx=20, pady=(8, 0))
        waveform_frame.pack_propagate(False)
        
        hud._waveform_canvas = ctk.CTkCanvas(
            waveform_frame, bg=colors.bg_primary, highlightthickness=0, bd=0, height=28
        )
        hud._waveform_canvas.pack(fill="both", expand=True)
        hud._animation.set_canvases(waveform_canvas=hud._waveform_canvas)
        
        # Input area
        input_frame = ctk.CTkFrame(bottom_frame, fg_color=colors.bg_tertiary, border_width=1, border_color=colors.border_focus, corner_radius=10)
        input_frame.pack(fill="x", padx=20, pady=(5, 12))
        
        # Mic button - NOW CONNECTED TO VOICE CAPTURE
        hud._mic_button = ctk.CTkButton(
            input_frame, text="🎤 MIC",
            command=hud._on_mic_button,
            fg_color=colors.accent_primary,
            hover_color=colors.accent_secondary,
            text_color=colors.bg_primary,
            font=hud._theme.get_font("small", bold=True),
            width=80,
            height=36,
            corner_radius=8,
        )
        hud._mic_button.pack(side="left", padx=(12, 8), pady=10)
        
        # Input entry
        hud._input_entry = ctk.CTkEntry(
            input_frame,
            font=hud._theme.get_font("normal"),
            placeholder_text="💬  Ask JARVIS anything...",
            fg_color=colors.bg_tertiary,
            text_color=colors.text_primary,
            border_width=0,
            height=36,
        )
        hud._input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=10)
        
        def on_send():
            text = hud._input_entry.get()
            hud.submit_input(text)
        
        hud._send_button = ctk.CTkButton(
            input_frame, text="SEND →",
            command=on_send,
            fg_color=colors.accent_gold,
            hover_color=colors.accent_gold_dim,
            text_color=colors.bg_primary,
            font=hud._theme.get_font("small", bold=True),
            width=90,
            height=36,
            corner_radius=8,
        )
        hud._send_button.pack(side="right", padx=(0, 12), pady=10)
        
        # Bind Enter key
        hud._input_entry.bind("<Return>", lambda e: on_send())
        
        # Focus input on startup
        hud._input_entry.focus_set()
        
        # ─── Content Area (3-column grid) ───────────────────────────────────
        content_frame = ctk.CTkFrame(main_frame, fg_color=colors.bg_primary)
        content_frame.pack(fill="both", expand=True, padx=4, pady=4)
        
        content_frame.grid_columnconfigure(0, weight=0, minsize=320)  # Left
        content_frame.grid_columnconfigure(1, weight=1, minsize=500)   # Center
        content_frame.grid_columnconfigure(2, weight=0, minsize=320)  # Right
        content_frame.grid_rowconfigure(0, weight=1)
        
        # ════════════════════════════════════════════════════════════════════
        # ZONE 1: LEFT PANEL - Arc Reactor, Particles, Metrics, Status Rings
        # ════════════════════════════════════════════════════════════════════
        left_frame = ctk.CTkFrame(
            content_frame, fg_color=colors.bg_glass, border_width=1, border_color=colors.border_glass,
            corner_radius=12
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=0)
        left_frame.grid_rowconfigure(1, weight=1)
        hud._left_panel = left_frame
        
        # Left panel header
        left_header = ctk.CTkFrame(left_frame, fg_color="transparent", height=40)
        left_header.pack(fill="x", padx=15, pady=(15, 5))
        left_header.pack_propagate(False)
        
        ctk.CTkLabel(
            left_header, text="CORE SYSTEMS",
            font=hud._theme.get_font("small", bold=True),
            text_color=colors.text_gold,
        ).pack(side="left")
        
        # Particle canvas (background)
        hud._particle_canvas = ctk.CTkCanvas(
            left_frame, bg=colors.bg_secondary, highlightthickness=0, bd=0
        )
        hud._particle_canvas.pack(fill="x", padx=10, pady=(0, 5))
        hud._animation.set_canvases(particle_canvas=hud._particle_canvas)
        
        # Arc Reactor canvas (centerpiece)
        reactor_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        reactor_frame.pack(pady=10)
        
        hud._reactor_canvas = ctk.CTkCanvas(
            reactor_frame, bg=colors.bg_secondary, highlightthickness=0,
            width=180, height=180
        )
        hud._reactor_canvas.pack()
        hud._animation.set_canvases(reactor_canvas=hud._reactor_canvas)
        
        # Reactor label
        ctk.CTkLabel(
            reactor_frame, text="ARC REACTOR",
            font=hud._theme.get_font("small", bold=True),
            text_color=colors.text_dim,
        ).pack(pady=(5, 0))
        
        # System metrics
        metrics_frame = ctk.CTkFrame(left_frame, fg_color=colors.bg_tertiary, border_width=1, border_color=colors.border_primary, corner_radius=8)
        metrics_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            metrics_frame, text="SYSTEM METRICS",
            font=hud._theme.get_font("small", bold=True),
            text_color=colors.text_gold,
        ).pack(anchor="w", padx=15, pady=(10, 5))
        
        for key, label in [("cpu", "CPU"), ("ram", "RAM"), ("disk", "DISK")]:
            metric_row = ctk.CTkFrame(metrics_frame, fg_color="transparent")
            metric_row.pack(fill="x", padx=15, pady=3)
            
            lbl = ctk.CTkLabel(
                metric_row, text=f"{label}: --",
                font=hud._theme.get_mono_font("normal"),
                text_color=colors.text_primary,
            )
            lbl.pack(side="left")
            hud._metrics_labels[key] = lbl
            
            # Progress bar for CPU/RAM
            if key in ("cpu", "ram"):
                bar = ctk.CTkProgressBar(
                    metric_row, width=100, height=6,
                    progress_color=colors.accent_primary if key == "cpu" else colors.accent_gold,
                    fg_color=colors.bg_primary,
                )
                bar.pack(side="right", padx=(10, 0))
                bar.set(0)
                hud._metrics_labels[f"{key}_bar"] = bar
        
        # Status rings (network, audio, etc.)
        rings_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        rings_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            rings_frame, text="SUBSYSTEM STATUS",
            font=hud._theme.get_font("small", bold=True),
            text_color=colors.text_gold,
        ).pack(anchor="w", pady=(0, 8))
        
        subsystems = [
            ("OLLAMA", "ollama", colors.accent_success),
            ("GROQ", "groq", colors.accent_primary),
            ("AUDIO", "audio", colors.accent_gold),
            ("MEMORY", "memory", colors.accent_info),
        ]
        
        for name, key, color in subsystems:
            row = ctk.CTkFrame(rings_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)
            
            ring_canvas = ctk.CTkCanvas(row, bg=colors.bg_glass, highlightthickness=0, width=20, height=20)
            ring_canvas.pack(side="left", padx=(0, 10))
            ring_canvas.create_oval(2, 2, 18, 18, outline=colors.border_primary, width=1, tags=f"ring_{key}")
            ring_canvas.create_oval(6, 6, 14, 14, fill=colors.dag_pending, outline="", tags=f"ring_dot_{key}")
            
            ctk.CTkLabel(
                row, text=name,
                font=hud._theme.get_font("small"),
                text_color=colors.text_secondary,
            ).pack(side="left")
            
            hud._metrics_labels[f"ring_{key}"] = ring_canvas
            hud._metrics_labels[f"ring_dot_{key}"] = (ring_canvas, f"ring_dot_{key}", color)
        
        # ════════════════════════════════════════════════════════════════════
        # ZONE 2: CENTER PANEL - Conversation Log (Holographic Style)
        # ════════════════════════════════════════════════════════════════════
        center_frame = ctk.CTkFrame(
            content_frame, fg_color=colors.bg_glass, border_width=1, border_color=colors.border_glass,
            corner_radius=12
        )
        center_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=0)
        center_frame.grid_rowconfigure(1, weight=1)
        hud._center_panel = center_frame
        
        # Center header
        center_header = ctk.CTkFrame(center_frame, fg_color="transparent", height=40)
        center_header.pack(fill="x", padx=20, pady=(15, 5))
        center_header.pack_propagate(False)
        
        ctk.CTkLabel(
            center_header, text="NEURAL INTERFACE",
            font=hud._theme.get_font("small", bold=True),
            text_color=colors.text_gold,
        ).pack(side="left")
        
        # Connection indicator
        conn_frame = ctk.CTkFrame(center_header, fg_color="transparent")
        conn_frame.pack(side="right")
        
        conn_canvas = ctk.CTkCanvas(conn_frame, bg=colors.bg_glass, highlightthickness=0, width=12, height=12)
        conn_canvas.pack(side="left", padx=(0, 8))
        conn_canvas.create_oval(2, 2, 10, 10, fill=colors.accent_success, outline="", tags="conn_dot")
        
        ctk.CTkLabel(
            conn_frame, text="LINK ESTABLISHED",
            font=hud._theme.get_font("small"),
            text_color=colors.accent_success,
        ).pack(side="left")
        
        # Conversation area with custom styling
        conv_container = ctk.CTkFrame(center_frame, fg_color=colors.bg_primary, border_width=1, border_color=colors.border_primary, corner_radius=8)
        conv_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        hud._conversation_text = ctk.CTkTextbox(
            conv_container,
            font=hud._theme.get_font("normal"),
            text_color=colors.text_primary,
            fg_color=colors.bg_primary,
            border_width=0,
            wrap="word",
            state="disabled",
            corner_radius=0,
        )
        hud._conversation_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure text tags for styling
        hud._conversation_text.tag_config("user", foreground=colors.text_gold)
        hud._conversation_text.tag_config("assistant", foreground=colors.text_primary)
        hud._conversation_text.tag_config("system", foreground=colors.text_accent)
        hud._conversation_text.tag_config("error", foreground=colors.accent_error)
        hud._conversation_text.tag_config("tool", foreground=colors.text_dim)
        
        # ════════════════════════════════════════════════════════════════════
        # ZONE 3: RIGHT PANEL - DAG Visualization
        # ════════════════════════════════════════════════════════════════════
        right_frame = ctk.CTkFrame(
            content_frame, fg_color=colors.bg_glass, border_width=1, border_color=colors.border_glass,
            corner_radius=12
        )
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(2, 0), pady=0)
        right_frame.grid_rowconfigure(1, weight=1)
        hud._right_panel = right_frame
        
        # Right header
        right_header = ctk.CTkFrame(right_frame, fg_color="transparent", height=40)
        right_header.pack(fill="x", padx=15, pady=(15, 5))
        right_header.pack_propagate(False)
        
        ctk.CTkLabel(
            right_header, text="EXECUTION MATRIX",
            font=hud._theme.get_font("small", bold=True),
            text_color=colors.text_gold,
        ).pack(side="left")
        
        # DAG canvas
        dag_container = ctk.CTkFrame(right_frame, fg_color=colors.bg_secondary, border_width=1, border_color=colors.border_primary, corner_radius=8)
        dag_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        hud._dag_canvas = ctk.CTkCanvas(
            dag_container, bg=colors.bg_secondary, highlightthickness=0, bd=0
        )
        hud._dag_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        hud._dag.set_canvas_size(280, 500)
        
        # Legend
        legend_frame = ctk.CTkFrame(right_frame, fg_color="transparent", height=60)
        legend_frame.pack(fill="x", padx=15, pady=(0, 15))
        legend_frame.pack_propagate(False)
        
        for status_name, color in [
            ("PENDING", colors.dag_pending),
            ("RUNNING", colors.dag_running),
            ("SUCCESS", colors.dag_success),
            ("FAILED", colors.dag_failed),
        ]:
            item = ctk.CTkFrame(legend_frame, fg_color="transparent")
            item.pack(side="left", padx=8)
            
            dot = ctk.CTkCanvas(item, bg=colors.bg_glass, highlightthickness=0, width=12, height=12)
            dot.pack(side="left", padx=(0, 4))
            dot.create_oval(2, 2, 10, 10, fill=color, outline="")
            
            ctk.CTkLabel(
                item, text=status_name,
                font=hud._theme.get_font("small"),
                text_color=colors.text_secondary,
            ).pack(side="left")
        
        # Set canvas references for animation engine
        hud._animation.set_canvases(
            particle_canvas=hud._particle_canvas,
            reactor_canvas=hud._reactor_canvas,
            waveform_canvas=hud._waveform_canvas,
            hud_canvas=hud._main_canvas,
        )
        
        # Initialize particle system
        hud._animation.reset_particles()
        
        # Bind window resize
        def on_resize(event):
            if hud._dag_canvas:
                hud._dag.set_canvas_size(event.width, event.height)
        root.bind("<Configure>", on_resize)
        
    except ImportError:
        raise


def _run_headless(hud: NexusHUD) -> None:
    """Run NEXUS AI in headless (CLI) mode."""
    import sys
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
        
        if hud._input_callback:
            hud._input_callback(user_input)