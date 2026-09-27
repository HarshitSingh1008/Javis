"""
NEXUS AI v4.0 — Advanced JARVIS Animation Engine
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides canvas-based animations optimized for i3 7th Gen:
- Multi-layer particle system (ambient, gold, reactive)
- Iron Man style arc reactor (rotating rings, pulsing core)
- Holographic scanline overlay
- Audio-responsive waveform with frequency bands
- Connection pulse animations for DAG
- Adaptive frame rate to maintain 30-60 FPS on low-power CPU
"""

import logging
import math
import random
import time
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

from nexus_config.settings import get_settings
from nexus_ui.theme_engine import get_theme_engine

logger = logging.getLogger("nexus.ui.animation")


# ─── Constants ───────────────────────────────────────────────────────────────

MAX_PARTICLES: int = 120
"""Maximum particles before performance degrades on i3."""

PARTICLE_SPEED: float = 0.4
"""Base particle movement speed in pixels per frame."""

ARC_REACTOR_RINGS: int = 4
"""Number of rotating rings in arc reactor."""

ARC_REACTOR_SEGMENTS: int = 32
"""Number of segments per ring."""

WAVEFORM_BARS: int = 64
"""Number of bars in the waveform display."""

SCANLINE_COUNT: int = 40
"""Number of horizontal scanlines."""

FRAME_BUDGET_MS: float = 16.67
"""Target frame budget for 60 FPS (16.67ms)."""

PARTICLE_TYPES = Enum("ParticleType", "AMBIENT GOLD REACTIVE CURSOR")


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class Particle:
    """A single ambient particle with type-specific behavior."""
    x: float
    y: float
    vx: float
    vy: float
    size: int
    alpha: float
    color: str
    life: float = 1.0
    max_life: float = 1.0
    ptype: str = "AMBIENT"
    angle: float = 0.0
    orbit_center_x: float = 0.0
    orbit_center_y: float = 0.0
    orbit_radius: float = 0.0
    orbit_speed: float = 0.0


@dataclass
class ArcReactorRing:
    """A single rotating ring in the arc reactor."""
    radius: float
    segments: int
    angle: float
    speed: float
    color: str
    alpha: float
    pulse_phase: float = 0.0
    glow_intensity: float = 1.0


@dataclass
class ArcReactorCore:
    """Central core of the arc reactor."""
    x: float
    y: float
    radius: float
    pulse: float = 0.0
    pulse_speed: float = 2.0
    color: str = "#00d4ff"
    glow_radius: float = 0.0


@dataclass
class Scanline:
    """Holographic scanline effect."""
    y: float
    speed: float
    alpha: float
    color: str
    thickness: int = 1


@dataclass
class WaveformBand:
    """Frequency band for waveform visualization."""
    index: int
    value: float = 0.0
    target: float = 0.0
    smoothing: float = 0.3
    color: str = "#00d4ff"
    is_gold: bool = False


@dataclass
class ConnectionPulse:
    """Animated pulse traveling along DAG connections."""
    x: float
    y: float
    target_x: float
    target_y: float
    progress: float = 0.0
    speed: float = 0.02
    color: str = "#00d4ff"
    size: float = 4.0


# ─── Animation Engine ────────────────────────────────────────────────────────

class AnimationEngine:
    """
    Manages all canvas-based UI animations for JARVIS-style HUD.
    
    Design notes for i3 7th Gen:
    - Particle count auto-scales based on measured frame time.
    - All animation state is pre-computed, not rendered per frame.
    - Uses simple physics (no collision detection) to keep math cheap.
    - Batched rendering to minimize canvas operations.
    """
    
    def __init__(self):
        self._settings = get_settings()
        self._theme = get_theme_engine()
        
        # Particle systems (multiple layers)
        self.particles_ambient: List[Particle] = []
        self.particles_gold: List[Particle] = []
        self.particles_reactive: List[Particle] = []
        self._particle_count: int = min(
            self._settings.UI_PARTICLE_COUNT, MAX_PARTICLES
        )
        
        # Arc reactor (Iron Man style)
        self.arc_reactor_rings: List[ArcReactorRing] = []
        self.arc_reactor_core: Optional[ArcReactorCore] = None
        self._init_arc_reactor()
        
        # Holographic scanlines
        self.scanlines: List[Scanline] = []
        self._init_scanlines()
        
        # Waveform (audio visualization)
        self.waveform_bands: List[WaveformBand] = []
        self._init_waveform()
        
        # Connection pulses (DAG visualization)
        self.connection_pulses: List[ConnectionPulse] = []
        
        # Performance tracking
        self._last_frame_time: float = 0.0
        self._frame_times: List[float] = []
        self._adaptive_count: int = self._particle_count
        self._frame_count: int = 0
        
        # Canvas references
        self._particle_canvas: Optional[object] = None
        self._reactor_canvas: Optional[object] = None
        self._waveform_canvas: Optional[object] = None
        self._hud_canvas: Optional[object] = None  # Main HUD canvas for scanlines
        
        # Mouse interaction
        self._mouse_x: float = 0.0
        self._mouse_y: float = 0.0
        self._mouse_active: bool = False
        
        # Reactor state
        self._reactor_active: bool = True
        self._reactor_intensity: float = 1.0
        
        # Initialize particles
        self.reset_particles()
    
    def set_canvases(
        self,
        particle_canvas: object = None,
        reactor_canvas: object = None,
        waveform_canvas: object = None,
        hud_canvas: object = None,
    ) -> None:
        """Set canvas references for rendering."""
        self._particle_canvas = particle_canvas
        self._reactor_canvas = reactor_canvas
        self._waveform_canvas = waveform_canvas
        self._hud_canvas = hud_canvas
    
    def set_mouse_position(self, x: float, y: float) -> None:
        """Update mouse position for reactive particles."""
        self._mouse_x = x
        self._mouse_y = y
        self._mouse_active = True
    
    def set_reactor_intensity(self, intensity: float) -> None:
        """Set arc reactor intensity (0.0 to 1.0)."""
        self._reactor_intensity = max(0.0, min(1.0, intensity))
    
    def trigger_reactive_particles(self, x: float, y: float, count: int = 8) -> None:
        """Spawn reactive particles at a position (e.g., on click)."""
        if not self._particle_canvas:
            return
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            particle = Particle(
                x=x,
                y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                size=random.randint(2, 5),
                alpha=random.uniform(0.6, 1.0),
                color=self._theme.colors.particle_gold if random.random() > 0.5 else self._theme.colors.particle_color,
                life=random.uniform(0.5, 1.5),
                max_life=1.5,
                ptype="REACTIVE",
            )
            self.particles_reactive.append(particle)
    
    def add_connection_pulse(self, x1: float, y1: float, x2: float, y2: float, color: str = None) -> None:
        """Add a pulse traveling along a connection line."""
        pulse_color = color or self._theme.colors.accent_primary
        self.connection_pulses.append(ConnectionPulse(
            x=x1, y=y1,
            target_x=x2, target_y=y2,
            color=pulse_color,
            size=random.uniform(3, 6),
            speed=random.uniform(0.015, 0.03),
        ))
    
    def update(self, dt: float) -> None:
        """
        Update all animation states for the current frame.
        
        Args:
            dt: Delta time in seconds since last frame.
        """
        self._frame_count += 1
        self._update_particles(dt)
        self._update_arc_reactor(dt)
        self._update_scanlines(dt)
        self._update_waveform(dt)
        self._update_connection_pulses(dt)
        self._update_performance(dt)
    
    def reset_particles(self) -> None:
        """Reinitialize all particle systems."""
        self.particles_ambient.clear()
        self.particles_gold.clear()
        self.particles_reactive.clear()
        
        ambient_count = int(self._particle_count * 0.6)
        gold_count = int(self._particle_count * 0.25)
        
        for _ in range(ambient_count):
            self._add_particle("AMBIENT")
        for _ in range(gold_count):
            self._add_particle("GOLD")
    
    def set_waveform_levels(self, levels: List[float]) -> None:
        """
        Set waveform bar levels from audio input.
        
        Args:
            levels: List of float values 0.0-1.0 for each band.
        """
        for i in range(min(len(levels), WAVEFORM_BANDS)):
            if i < len(self.waveform_bands):
                band = self.waveform_bands[i]
                target = min(1.0, max(0.0, levels[i]))
                band.target = target
    
    def _update_particles(self, dt: float) -> None:
        """Update all particle systems."""
        speed_mult = PARTICLE_SPEED * dt * 60
        
        # Update ambient particles
        for particle in self.particles_ambient[:]:
            particle.x += particle.vx * speed_mult
            particle.y += particle.vy * speed_mult
            particle.life -= dt * 0.08
            
            # Mouse attraction for ambient particles
            if self._mouse_active:
                dx = self._mouse_x - particle.x
                dy = self._mouse_y - particle.y
                dist = math.hypot(dx, dy)
                if dist < 150 and dist > 1:
                    force = (150 - dist) / 150 * 0.3
                    particle.vx += (dx / dist) * force * dt * 60
                    particle.vy += (dy / dist) * force * dt * 60
            
            if particle.life <= 0:
                self.particles_ambient.remove(particle)
                self._add_particle("AMBIENT")
        
        # Update gold particles (slower, more graceful)
        for particle in self.particles_gold[:]:
            particle.x += particle.vx * speed_mult * 0.6
            particle.y += particle.vy * speed_mult * 0.6
            particle.life -= dt * 0.05
            
            if particle.life <= 0:
                self.particles_gold.remove(particle)
                self._add_particle("GOLD")
        
        # Update reactive particles (short-lived, fast)
        for particle in self.particles_reactive[:]:
            particle.x += particle.vx * speed_mult * 1.5
            particle.y += particle.vy * speed_mult * 1.5
            particle.vx *= 0.98  # Friction
            particle.vy *= 0.98
            particle.life -= dt * 0.5
            
            if particle.life <= 0:
                self.particles_reactive.remove(particle)
    
    def _add_particle(self, ptype: str) -> None:
        """Add a new particle of specified type."""
        if not self._particle_canvas:
            return
        
        try:
            width = int(self._particle_canvas.winfo_width())
            height = int(self._particle_canvas.winfo_height())
        except Exception:
            width, height = 300, 600
        
        edge = random.choice(["left", "right", "top", "bottom"])
        if edge == "left":
            x, y = 0, random.randint(0, height)
            vx = random.uniform(0.1, 0.5)
            vy = random.uniform(-0.2, 0.2)
        elif edge == "right":
            x, y = width, random.randint(0, height)
            vx = random.uniform(-0.5, -0.1)
            vy = random.uniform(-0.2, 0.2)
        elif edge == "top":
            x, y = random.randint(0, width), 0
            vx = random.uniform(-0.2, 0.2)
            vy = random.uniform(0.1, 0.5)
        else:
            x, y = random.randint(0, width), height
            vx = random.uniform(-0.2, 0.2)
            vy = random.uniform(-0.5, -0.1)
        
        if ptype == "AMBIENT":
            particle = Particle(
                x=x, y=y, vx=vx, vy=vy,
                size=random.randint(1, 3),
                alpha=random.uniform(0.15, 0.5),
                color=self._theme.colors.particle_color,
                life=random.uniform(3.0, 8.0),
                max_life=8.0,
                ptype="AMBIENT",
            )
            self.particles_ambient.append(particle)
        elif ptype == "GOLD":
            particle = Particle(
                x=x, y=y, vx=vx * 0.5, vy=vy * 0.5,
                size=random.randint(1, 2),
                alpha=random.uniform(0.1, 0.4),
                color=self._theme.colors.particle_gold,
                life=random.uniform(5.0, 12.0),
                max_life=12.0,
                ptype="GOLD",
            )
            self.particles_gold.append(particle)
    
    def _init_arc_reactor(self) -> None:
        """Initialize Iron Man style arc reactor with multiple rotating rings."""
        colors = self._theme.colors
        
        # Core
        self.arc_reactor_core = ArcReactorCore(
            x=0, y=0, radius=20,
            color=colors.arc_reactor_color,
        )
        
        # Rings - inner to outer
        ring_configs = [
            (28, 16, 15, colors.arc_reactor_color, 0.6),
            (42, 24, -10, colors.arc_reactor_gold, 0.5),
            (56, 32, 7, colors.arc_reactor_color, 0.4),
            (70, 32, -5, colors.arc_reactor_gold, 0.3),
        ]
        
        for radius, segments, speed, color, alpha in ring_configs:
            self.arc_reactor_rings.append(ArcReactorRing(
                radius=radius,
                segments=segments,
                angle=random.uniform(0, 360),
                speed=speed,
                color=color,
                alpha=alpha,
                pulse_phase=random.uniform(0, 2 * math.pi),
            ))
    
    def _update_arc_reactor(self, dt: float) -> None:
        """Update arc reactor rotation and pulse."""
        if not self._reactor_active:
            return
        
        colors = self._theme.colors
        
        # Update core pulse
        core = self.arc_reactor_core
        core.pulse += dt * core.pulse_speed * self._reactor_intensity
        core.glow_radius = core.radius + math.sin(core.pulse * 3) * 8 * self._reactor_intensity
        
        # Alternate core color between cyan and gold
        if int(core.pulse) % 2 == 0:
            core.color = colors.arc_reactor_color
        else:
            core.color = colors.arc_reactor_gold
        
        # Update rings
        for ring in self.arc_reactor_rings:
            ring.angle += dt * ring.speed * self._reactor_intensity
            if ring.angle > 360:
                ring.angle -= 360
            
            ring.pulse_phase += dt * 2
            ring.glow_intensity = 0.5 + 0.5 * math.sin(ring.pulse_phase) * self._reactor_intensity
            
            # Alternate ring colors
            if int(ring.pulse_phase / math.pi) % 2 == 0:
                ring.color = colors.arc_reactor_color
            else:
                ring.color = colors.arc_reactor_gold
    
    def _init_scanlines(self) -> None:
        """Initialize holographic scanlines."""
        colors = self._theme.colors
        for i in range(SCANLINE_COUNT):
            self.scanlines.append(Scanline(
                y=random.uniform(0, 900),
                speed=random.uniform(0.5, 2.0),
                alpha=random.uniform(0.02, 0.08),
                color=colors.scanline_color,
                thickness=1 if random.random() > 0.7 else 2,
            ))
    
    def _update_scanlines(self, dt: float) -> None:
        """Update scanline positions."""
        for line in self.scanlines:
            line.y += line.speed * dt * 60
            if line.y > 900:
                line.y = -line.thickness
                line.speed = random.uniform(0.5, 2.0)
                line.alpha = random.uniform(0.02, 0.08)
    
    def _init_waveform(self) -> None:
        """Initialize waveform frequency bands."""
        colors = self._theme.colors
        for i in range(WAVEFORM_BARS):
            is_gold = (i % 8 == 0)  # Every 8th bar is gold
            self.waveform_bands.append(WaveformBand(
                index=i,
                color=colors.waveform_gold if is_gold else colors.waveform_color,
                is_gold=is_gold,
                smoothing=random.uniform(0.2, 0.4),
            ))
    
    def _update_waveform(self, dt: float) -> None:
        """Smooth waveform bands toward targets."""
        for band in self.waveform_bands:
            band.value += (band.target - band.value) * band.smoothing * dt * 60
            # Decay target toward zero
            band.target *= 0.95
    
    def _update_connection_pulses(self, dt: float) -> None:
        """Update connection pulses along DAG edges."""
        for pulse in self.connection_pulses[:]:
            pulse.progress += pulse.speed * dt * 60
            if pulse.progress >= 1.0:
                self.connection_pulses.remove(pulse)
            else:
                # Interpolate position
                pulse.x = pulse.x + (pulse.target_x - pulse.x) * pulse.speed * dt * 60 * 10
                pulse.y = pulse.y + (pulse.target_y - pulse.y) * pulse.speed * dt * 60 * 10
    
    def _update_performance(self, dt: float) -> None:
        """Track frame times and adjust particle count adaptively."""
        now = time.perf_counter()
        if self._last_frame_time > 0:
            frame_ms = (now - self._last_frame_time) * 1000
            self._frame_times.append(frame_ms)
            
            # Keep last 30 frame times
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            
            # Check if we need to reduce particles
            if len(self._frame_times) >= 10:
                avg_frame_ms = sum(self._frame_times[-10:]) / 10
                if avg_frame_ms > 33:  # Below 30 FPS
                    self._adaptive_count = max(20, self._adaptive_count - 8)
                    self._trim_particles()
                elif avg_frame_ms < 16:  # Above 60 FPS
                    target = min(self._particle_count, self._adaptive_count + 5)
                    self._adaptive_count = min(target, MAX_PARTICLES)
        
        self._last_frame_time = now
    
    def _trim_particles(self) -> None:
        """Trim particle arrays to adaptive count."""
        total_target = self._adaptive_count
        ambient_target = int(total_target * 0.6)
        gold_target = int(total_target * 0.25)
        
        if len(self.particles_ambient) > ambient_target:
            self.particles_ambient = self.particles_ambient[:ambient_target]
        if len(self.particles_gold) > gold_target:
            self.particles_gold = self.particles_gold[:gold_target]
    
    @property
    def current_particle_count(self) -> int:
        """Get current adaptive particle count."""
        return self._adaptive_count
    
    @property
    def current_fps(self) -> float:
        """Calculate current FPS from frame times."""
        if len(self._frame_times) < 2:
            return 60.0
        avg_ms = sum(self._frame_times[-10:]) / min(10, len(self._frame_times))
        return 1000.0 / avg_ms if avg_ms > 0 else 60.0


# ─── Rendering Functions ─────────────────────────────────────────────────────

def render_particles(engine: AnimationEngine, canvas: object) -> None:
    """Render all particle systems to canvas."""
    if not canvas:
        return
    
    try:
        canvas.delete("particles")
        
        # Render ambient particles
        for p in engine.particles_ambient:
            alpha_int = int(p.alpha * 255)
            color = _hex_with_alpha(p.color, alpha_int)
            canvas.create_oval(
                p.x - p.size, p.y - p.size,
                p.x + p.size, p.y + p.size,
                fill=color, outline="", tags="particles"
            )
        
        # Render gold particles
        for p in engine.particles_gold:
            alpha_int = int(p.alpha * 255)
            color = _hex_with_alpha(p.color, alpha_int)
            canvas.create_oval(
                p.x - p.size, p.y - p.size,
                p.x + p.size, p.y + p.size,
                fill=color, outline="", tags="particles"
            )
        
        # Render reactive particles (with glow)
        for p in engine.particles_reactive:
            alpha_int = int(p.alpha * 255)
            color = _hex_with_alpha(p.color, alpha_int)
            # Outer glow
            glow_size = p.size + 3
            glow_alpha = int(p.alpha * 100)
            glow_color = _hex_with_alpha(p.color, glow_alpha)
            canvas.create_oval(
                p.x - glow_size, p.y - glow_size,
                p.x + glow_size, p.y + glow_size,
                fill=glow_color, outline="", tags="particles"
            )
            # Core
            canvas.create_oval(
                p.x - p.size, p.y - p.size,
                p.x + p.size, p.y + p.size,
                fill=color, outline="", tags="particles"
            )
    except Exception as e:
        logger.debug(f"Particle render error: {e}")


def render_arc_reactor(engine: AnimationEngine, canvas: object) -> None:
    """Render Iron Man style arc reactor to canvas."""
    if not canvas or not engine.arc_reactor_core:
        return
    
    try:
        canvas.delete("reactor")
        colors = engine._theme.colors
        
        core = engine.arc_reactor_core
        cx = canvas.winfo_width() // 2
        cy = canvas.winfo_height() // 2
        
        # Update core position
        core.x = cx
        core.y = cy
        
        # Draw outer glow (multiple layers for bloom effect)
        for i in range(4):
            glow_r = core.glow_radius + i * 15
            alpha = int(30 * engine._reactor_intensity * (1 - i * 0.2))
            if alpha > 0:
                glow_color = _hex_with_alpha(core.color, alpha)
                canvas.create_oval(
                    cx - glow_r, cy - glow_r,
                    cx + glow_r, cy + glow_r,
                    fill=glow_color, outline="", tags="reactor"
                )
        
        # Draw rings
        for ring in engine.arc_reactor_rings:
            _draw_arc_ring(canvas, cx, cy, ring, engine._reactor_intensity)
        
        # Draw core
        core_r = core.radius
        # Core background
        canvas.create_oval(
            cx - core_r, cy - core_r,
            cx + core_r, cy + core_r,
            fill="#000814", outline=core.color, width=2, tags="reactor"
        )
        # Core inner glow
        inner_r = core_r - 4
        core_alpha = int(200 * engine._reactor_intensity)
        core_color = _hex_with_alpha(core.color, core_alpha)
        canvas.create_oval(
            cx - inner_r, cy - inner_r,
            cx + inner_r, cy + inner_r,
            fill=core_color, outline="", tags="reactor"
        )
        # Core center highlight
        highlight_r = inner_r // 2
        canvas.create_oval(
            cx - highlight_r, cy - highlight_r,
            cx + highlight_r, cy + highlight_r,
            fill="#ffffff", outline="", tags="reactor"
        )
        
        # Center crosshair lines
        line_len = core_r + 10
        canvas.create_line(
            cx - line_len, cy, cx + line_len, cy,
            fill=core.color, width=1, tags="reactor"
        )
        canvas.create_line(
            cx, cy - line_len, cx, cy + line_len,
            fill=core.color, width=1, tags="reactor"
        )
        
    except Exception as e:
        logger.debug(f"Reactor render error: {e}")


def _draw_arc_ring(canvas: object, cx: int, cy: int, ring: ArcReactorRing, intensity: float) -> None:
    """Draw a single arc reactor ring with segments."""
    r = ring.radius
    segments = ring.segments
    angle_offset = math.radians(ring.angle)
    segment_angle = 2 * math.pi / segments
    
    alpha = int(255 * ring.alpha * ring.glow_intensity * intensity)
    if alpha < 10:
        return
    
    color = _hex_with_alpha(ring.color, alpha)
    
    for i in range(segments):
        # Only draw some segments for dashed effect
        if random.random() > 0.3:
            continue
        
        a1 = i * segment_angle + angle_offset
        a2 = a1 + segment_angle * random.uniform(0.3, 0.8)
        
        x1 = cx + r * math.cos(a1)
        y1 = cy + r * math.sin(a1)
        x2 = cx + r * math.cos(a2)
        y2 = cy + r * math.sin(a2)
        
        canvas.create_line(
            x1, y1, x2, y2,
            fill=color, width=2, tags="reactor", capstyle="round"
        )
    
    # Outer ring outline
    outline_alpha = int(alpha * 0.3)
    outline_color = _hex_with_alpha(ring.color, outline_alpha)
    canvas.create_oval(
        cx - r, cy - r, cx + r, cy + r,
        outline=outline_color, width=1, tags="reactor"
    )


def render_scanlines(engine: AnimationEngine, canvas: object) -> None:
    """Render holographic scanline overlay."""
    if not canvas:
        return
    
    try:
        canvas.delete("scanlines")
        width = canvas.winfo_width()
        
        for line in engine.scanlines:
            if line.alpha <= 0:
                continue
            alpha_int = int(line.alpha * 255)
            color = _hex_with_alpha(line.color, alpha_int)
            y = int(line.y)
            canvas.create_line(
                0, y, width, y,
                fill=color, width=line.thickness, tags="scanlines"
            )
    except Exception as e:
        logger.debug(f"Scanline render error: {e}")


def render_waveform(engine: AnimationEngine, canvas: object) -> None:
    """Render audio waveform visualization."""
    if not canvas:
        return
    
    try:
        canvas.delete("waveform")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        center_y = height // 2
        
        bar_width = max(2, width // WAVEFORM_BARS - 1)
        
        for i, band in enumerate(engine.waveform_bands):
            bar_height = band.value * (height * 0.8)
            x = i * (bar_width + 1) + 2
            
            alpha = int(180 + 75 * band.value)
            color = _hex_with_alpha(band.color, alpha)
            
            # Main bar (upward)
            canvas.create_rectangle(
                x, center_y - bar_height,
                x + bar_width, center_y,
                fill=color, outline="", tags="waveform"
            )
            # Mirror bar (downward)
            canvas.create_rectangle(
                x, center_y,
                x + bar_width, center_y + bar_height,
                fill=color, outline="", tags="waveform"
            )
            
            # Peak indicator
            if band.value > 0.7:
                peak_alpha = int(255 * band.value)
                peak_color = _hex_with_alpha(band.color, peak_alpha)
                canvas.create_rectangle(
                    x, center_y - bar_height - 2,
                    x + bar_width, center_y - bar_height,
                    fill=peak_color, outline="", tags="waveform"
                )
                canvas.create_rectangle(
                    x, center_y + bar_height,
                    x + bar_width, center_y + bar_height + 2,
                    fill=peak_color, outline="", tags="waveform"
                )
    except Exception as e:
        logger.debug(f"Waveform render error: {e}")


def render_connection_pulses(engine: AnimationEngine, canvas: object) -> None:
    """Render animated pulses on DAG connections."""
    if not canvas or not engine.connection_pulses:
        return
    
    try:
        canvas.delete("conn_pulses")
        
        for pulse in engine.connection_pulses:
            alpha = int(255 * (1 - pulse.progress) * 0.8)
            if alpha < 10:
                continue
            color = _hex_with_alpha(pulse.color, alpha)
            size = pulse.size * (1 - pulse.progress * 0.5)
            
            canvas.create_oval(
                pulse.x - size, pulse.y - size,
                pulse.x + size, pulse.y + size,
                fill=color, outline="", tags="conn_pulses"
            )
            
            # Trail effect
            trail_size = size * 0.6
            trail_alpha = int(alpha * 0.4)
            trail_color = _hex_with_alpha(pulse.color, trail_alpha)
            canvas.create_oval(
                pulse.x - trail_size, pulse.y - trail_size,
                pulse.x + trail_size, pulse.y + trail_size,
                fill=trail_color, outline="", tags="conn_pulses"
            )
    except Exception as e:
        logger.debug(f"Connection pulse render error: {e}")


def _hex_with_alpha(hex_color: str, alpha: int) -> str:
    """Add alpha channel to hex color (returns tkinter-compatible format)."""
    # Tkinter doesn't support alpha in hex directly, so we blend with black
    # For actual alpha, we'd need to use PhotoImage or similar
    # This is a simplified version - in practice, we'd use a different approach
    alpha = max(0, min(255, alpha))
    if alpha >= 255:
        return hex_color
    # Blend with black background for simulated alpha
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r * alpha / 255)
    g = int(g * alpha / 255)
    b = int(b * alpha / 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def render_all(engine: AnimationEngine) -> None:
    """Render all animation layers to their respective canvases."""
    render_particles(engine, engine._particle_canvas)
    render_arc_reactor(engine, engine._reactor_canvas)
    render_scanlines(engine, engine._hud_canvas)
    render_waveform(engine, engine._waveform_canvas)
    render_connection_pulses(engine, engine._hud_canvas)