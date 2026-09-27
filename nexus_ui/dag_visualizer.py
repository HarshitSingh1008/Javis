"""
NEXUS AI v4.0 — JARVIS DAG Execution Visualization
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Displays a directed acyclic graph (DAG) of task execution nodes with
real-time status updates (pending/running/success/failed).
Futuristic JARVIS-style rendering with glow effects, animated connections,
and holographic node visualization.
"""

import logging
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from nexus_ui.theme_engine import get_theme_engine

logger = logging.getLogger("nexus.ui.dag")


class NodeStatus(Enum):
    """Status of a DAG execution node."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """A single node in the task execution DAG."""
    id: str
    label: str
    status: NodeStatus = NodeStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None
    x: float = 0.0
    y: float = 0.0
    width: float = 160.0
    height: float = 48.0
    
    # Animation state
    pulse_phase: float = 0.0
    glow_intensity: float = 0.0
    entry_animation: float = 0.0  # 0 to 1


@dataclass
class ConnectionLine:
    """Animated connection between nodes."""
    from_id: str
    to_id: str
    progress: float = 0.0  # For data flow animation
    particles: List[Dict] = field(default_factory=list)


class DAGVisualizer:
    """
    Real-time DAG execution visualization with JARVIS aesthetics.
    
    Features:
    - Layered node layout with smooth positioning
    - Glowing nodes with status-based colors
    - Animated data flow along connections
    - Particle effects on node activation
    - Holographic panel styling
    """
    
    def __init__(self):
        self._theme = get_theme_engine()
        self.nodes: Dict[str, DAGNode] = {}
        self.connections: List[ConnectionLine] = []
        self._canvas_width: int = 400
        self._canvas_height: int = 600
        self._node_spacing_x: float = 180.0
        self._node_spacing_y: float = 85.0
        self._padding: float = 30.0
        self._animation_time: float = 0.0
        
        # Particle effects for node transitions
        self._node_particles: Dict[str, List[Dict]] = {}
    
    def set_canvas_size(self, width: int, height: int) -> None:
        """Set the canvas dimensions for layout calculation."""
        self._canvas_width = max(300, width)
        self._canvas_height = max(400, height)
    
    def set_dag_plan(self, plan: Dict[str, Any]) -> None:
        """
        Load a DAG plan from the task planner.
        
        Args:
            plan: Dict with 'steps' list containing step definitions
                  with 'id', 'label', and 'depends_on' fields.
        """
        self.nodes.clear()
        self.connections.clear()
        self._node_particles.clear()
        
        steps = plan.get("steps", [])
        if not steps:
            return
        
        # Create nodes
        for step in steps:
            node_id = step.get("id", f"step_{len(self.nodes)}")
            self.nodes[node_id] = DAGNode(
                id=node_id,
                label=step.get("label", node_id),
                depends_on=step.get("depends_on", []),
                entry_animation=0.0,
            )
        
        # Create connections
        for node_id, node in self.nodes.items():
            for dep_id in node.depends_on:
                if dep_id in self.nodes:
                    self.connections.append(ConnectionLine(
                        from_id=dep_id,
                        to_id=node_id,
                    ))
        
        # Calculate layout
        self._calculate_layout()
        
        # Initialize entry animations
        for node in self.nodes.values():
            node.entry_animation = 0.0
    
    def update_node_status(
        self,
        node_id: str,
        status: NodeStatus,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """
        Update the status of a DAG node.
        
        Args:
            node_id: ID of the node to update.
            status: New status.
            duration_ms: Execution duration in milliseconds.
            error: Error message if failed.
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            old_status = node.status
            node.status = status
            node.duration_ms = duration_ms
            node.error = error
            
            # Trigger animations on status change
            if status == NodeStatus.RUNNING and old_status == NodeStatus.PENDING:
                node.pulse_phase = 0.0
                node.glow_intensity = 1.0
                self._spawn_node_particles(node_id, "activate")
            elif status == NodeStatus.SUCCESS and old_status == NodeStatus.RUNNING:
                node.glow_intensity = 0.5
                self._spawn_node_particles(node_id, "success")
            elif status == NodeStatus.FAILED and old_status == NodeStatus.RUNNING:
                node.glow_intensity = 1.0
                self._spawn_node_particles(node_id, "error")
            
            # Animate connection flow
            for conn in self.connections:
                if conn.to_id == node_id and status == NodeStatus.RUNNING:
                    conn.progress = 0.0
                    conn.particles = []
                    # Spawn flow particles
                    for _ in range(5):
                        conn.particles.append({
                            "pos": 0.0,
                            "speed": random.uniform(0.02, 0.04),
                            "size": random.uniform(2, 4),
                            "alpha": random.uniform(0.5, 1.0),
                        })
    
    def _spawn_node_particles(self, node_id: str, ptype: str) -> None:
        """Spawn particle effect at node."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        cx = node.x + node.width / 2
        cy = node.y + node.height / 2
        
        particles = []
        count = 12 if ptype == "activate" else 20
        colors = {
            "activate": [self._theme.colors.accent_primary, self._theme.colors.accent_gold],
            "success": [self._theme.colors.accent_success, self._theme.colors.accent_gold],
            "error": [self._theme.colors.accent_error, self._theme.colors.accent_warning],
        }
        palette = colors.get(ptype, [self._theme.colors.accent_primary])
        
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            particles.append({
                "x": cx, "y": cy,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": random.uniform(0.5, 1.5),
                "max_life": 1.5,
                "size": random.uniform(2, 5),
                "color": random.choice(palette),
                "alpha": random.uniform(0.6, 1.0),
            })
        
        self._node_particles[node_id] = particles
    
    def get_node_color(self, status: NodeStatus) -> str:
        """Get the color for a node based on its status."""
        colors = {
            NodeStatus.PENDING: self._theme.colors.dag_pending,
            NodeStatus.RUNNING: self._theme.colors.dag_running,
            NodeStatus.SUCCESS: self._theme.colors.dag_success,
            NodeStatus.FAILED: self._theme.colors.dag_failed,
            NodeStatus.SKIPPED: self._theme.colors.dag_pending,
        }
        return colors.get(status, self._theme.colors.dag_pending)
    
    def get_node_glow_color(self, status: NodeStatus) -> str:
        """Get the glow color for a node."""
        colors = {
            NodeStatus.PENDING: self._theme.colors.border_primary,
            NodeStatus.RUNNING: self._theme.colors.dag_glow,
            NodeStatus.SUCCESS: self._theme.colors.accent_success,
            NodeStatus.FAILED: self._theme.colors.accent_error,
            NodeStatus.SKIPPED: self._theme.colors.text_dim,
        }
        return colors.get(status, self._theme.colors.dag_glow)
    
    def update_animations(self, dt: float) -> None:
        """Update all animation states."""
        self._animation_time += dt
        
        # Update node entry animations
        for node in self.nodes.values():
            if node.entry_animation < 1.0:
                node.entry_animation = min(1.0, node.entry_animation + dt * 2)
            
            # Pulse for running nodes
            if node.status == NodeStatus.RUNNING:
                node.pulse_phase += dt * 4
                node.glow_intensity = 0.5 + 0.5 * math.sin(node.pulse_phase)
            elif node.glow_intensity > 0:
                node.glow_intensity = max(0, node.glow_intensity - dt * 0.5)
        
        # Update connection flow particles
        for conn in self.connections:
            for p in conn.particles[:]:
                p["pos"] += p["speed"]
                if p["pos"] >= 1.0:
                    conn.particles.remove(p)
        
        # Update node particles
        for node_id, particles in self._node_particles.items():
            for p in particles[:]:
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                p["vx"] *= 0.98
                p["vy"] *= 0.98
                p["life"] -= dt
                if p["life"] <= 0:
                    particles.remove(p)
            if not particles:
                del self._node_particles[node_id]
    
    def render(self, canvas: object) -> None:
        """
        Render the DAG on a tkinter Canvas with JARVIS styling.
        
        Args:
            canvas: tkinter Canvas widget to draw on.
        """
        try:
            canvas.delete("dag_all")
            
            # Draw connection lines first (behind nodes)
            for conn in self.connections:
                if conn.from_id in self.nodes and conn.to_id in self.nodes:
                    from_node = self.nodes[conn.from_id]
                    to_node = self.nodes[conn.to_id]
                    self._draw_connection(canvas, from_node, to_node, conn)
            
            # Draw nodes
            for node_id, node in self.nodes.items():
                self._draw_node(canvas, node)
            
            # Draw node particles on top
            for node_id, particles in self._node_particles.items():
                self._draw_particles(canvas, particles)
        
        except Exception as e:
            logger.debug(f"DAG render error: {e}")
    
    def _draw_node(self, canvas: object, node: DAGNode) -> None:
        """Draw a single DAG node with JARVIS styling."""
        # Apply entry animation
        entry_scale = node.entry_animation
        if entry_scale <= 0:
            return
        
        x = node.x
        y = node.y
        w = node.width * entry_scale
        h = node.height * entry_scale
        cx = x + node.width / 2
        cy = y + node.height / 2
        x = cx - w / 2
        y = cy - h / 2
        
        status = node.status
        base_color = self.get_node_color(status)
        glow_color = self.get_node_glow_color(status)
        text_color = self._theme.colors.text_primary
        
        # Glow layers (for running/active nodes)
        if node.glow_intensity > 0.1 and status in (NodeStatus.RUNNING, NodeStatus.SUCCESS, NodeStatus.FAILED):
            glow_r = int(node.glow_intensity * 8)
            for i in range(glow_r, 0, -1):
                alpha = int(30 * node.glow_intensity * (1 - i / glow_r))
                if alpha > 0:
                    gcolor = self._blend_color(glow_color, alpha)
                    canvas.create_rectangle(
                        x - i, y - i, x + w + i, y + h + i,
                        outline=gcolor, width=1, tags="dag_all"
                    )
                    # Rounded corners for glow
                    canvas.create_oval(
                        x - i - 4, y - i - 4, x - i + 4, y - i + 4,
                        outline=gcolor, width=1, tags="dag_all"
                    )
                    canvas.create_oval(
                        x + w + i - 4, y - i - 4, x + w + i + 4, y - i + 4,
                        outline=gcolor, width=1, tags="dag_all"
                    )
                    canvas.create_oval(
                        x - i - 4, y + h + i - 4, x - i + 4, y + h + i + 4,
                        outline=gcolor, width=1, tags="dag_all"
                    )
                    canvas.create_oval(
                        x + w + i - 4, y + h + i - 4, x + w + i + 4, y + h + i + 4,
                        outline=gcolor, width=1, tags="dag_all"
                    )
        
        # Node background panel
        bg_color = self._theme.colors.bg_tertiary
        if status == NodeStatus.RUNNING:
            # Subtle animated background for running
            bg_color = self._blend_color(base_color, 30)
        elif status == NodeStatus.SUCCESS:
            bg_color = self._blend_color(self._theme.colors.accent_success, 20)
        elif status == NodeStatus.FAILED:
            bg_color = self._blend_color(self._theme.colors.accent_error, 20)
        
        # Rounded rectangle background
        radius = 8
        self._draw_rounded_rect(canvas, x, y, w, h, radius, 
                                fill=bg_color, outline=base_color, width=2, tags="dag_all")
        
        # Top accent bar
        bar_h = 3
        self._draw_rounded_rect(canvas, x, y, w, bar_h, radius,
                                fill=base_color, outline="", tags="dag_all")
        # Only round top corners
        canvas.create_rectangle(x, y + bar_h - radius, x + w, y + bar_h, 
                                fill=base_color, outline="", tags="dag_all")
        
        # Status indicator (animated)
        dot_x = x + 14
        dot_y = y + h / 2
        dot_r = 6
        
        if status == NodeStatus.RUNNING:
            # Pulsing ring
            pulse_r = dot_r + math.sin(node.pulse_phase * 2) * 3
            pulse_alpha = int(100 * (0.5 + 0.5 * math.sin(node.pulse_phase)))
            pulse_color = self._blend_color(base_color, pulse_alpha)
            canvas.create_oval(
                dot_x - pulse_r, dot_y - pulse_r,
                dot_x + pulse_r, dot_y + pulse_r,
                outline=pulse_color, width=2, tags="dag_all"
            )
        
        # Status dot
        dot_color = base_color
        canvas.create_oval(
            dot_x - dot_r, dot_y - dot_r,
            dot_x + dot_r, dot_y + dot_r,
            fill=dot_color, outline="", tags="dag_all"
        )
        
        # Inner highlight for running/success
        if status in (NodeStatus.RUNNING, NodeStatus.SUCCESS):
            hl_r = dot_r - 2
            hl_color = self._blend_color("#ffffff", 180)
            canvas.create_oval(
                dot_x - hl_r, dot_y - hl_r,
                dot_x + hl_r, dot_y + hl_r,
                fill=hl_color, outline="", tags="dag_all"
            )
        
        # Label text
        label_x = x + 28
        label_y = y + h / 2
        label_text = node.label[:22]
        if len(node.label) > 22:
            label_text += "…"
        
        canvas.create_text(
            label_x, label_y,
            text=label_text,
            fill=text_color,
            anchor="w",
            font=self._theme.get_mono_font("small"),
            tags="dag_all",
        )
        
        # Duration text
        if node.duration_ms > 0:
            dur_x = x + w - 12
            dur_y = y + h + 14
            dur_text = f"{node.duration_ms:.0f}ms"
            canvas.create_text(
                dur_x, dur_y,
                text=dur_text,
                fill=self._theme.colors.text_dim,
                anchor="e",
                font=self._theme.get_mono_font("small"),
                tags="dag_all",
            )
        
        # Error indicator
        if node.error and status == NodeStatus.FAILED:
            err_x = x + w / 2
            err_y = y + h + 28
            canvas.create_text(
                err_x, err_y,
                text="⚠ " + node.error[:30],
                fill=self._theme.colors.accent_error,
                anchor="center",
                font=self._theme.get_font("small"),
                tags="dag_all",
            )
    
    def _draw_connection(self, canvas: object, from_node: DAGNode, to_node: DAGNode, conn: ConnectionLine) -> None:
        """Draw a futuristic connection between two nodes."""
        x1 = from_node.x + from_node.width
        y1 = from_node.y + from_node.height / 2
        x2 = to_node.x
        y2 = to_node.y + to_node.height / 2
        
        # Determine connection color based on source node status
        if from_node.status == NodeStatus.SUCCESS:
            line_color = self._theme.colors.dag_success
        elif from_node.status == NodeStatus.RUNNING:
            line_color = self._theme.colors.dag_running
        elif from_node.status == NodeStatus.FAILED:
            line_color = self._theme.colors.dag_failed
        else:
            line_color = self._theme.colors.border_primary
        
        # Curved path (bezier curve)
        ctrl_x = (x1 + x2) / 2
        ctrl_y = y1  # Straight horizontal with slight curve
        
        # Draw main line with multiple segments for curve
        segments = 20
        for i in range(segments):
            t1 = i / segments
            t2 = (i + 1) / segments
            
            # Quadratic bezier
            sx1 = (1 - t1)**2 * x1 + 2 * (1 - t1) * t1 * ctrl_x + t1**2 * x2
            sy1 = (1 - t1)**2 * y1 + 2 * (1 - t1) * t1 * ctrl_y + t1**2 * y2
            sx2 = (1 - t2)**2 * x1 + 2 * (1 - t2) * t2 * ctrl_x + t2**2 * x2
            sy2 = (1 - t2)**2 * y1 + 2 * (1 - t2) * t2 * ctrl_y + t2**2 * y2
            
            # Vary alpha along path
            alpha = int(150 * (1 - t1 * 0.3))
            seg_color = self._blend_color(line_color, alpha)
            
            canvas.create_line(
                sx1, sy1, sx2, sy2,
                fill=seg_color, width=2, tags="dag_all", capstyle="round", smooth=True
            )
        
        # Arrow head at destination
        self._draw_arrow_head(canvas, x2, y2, ctrl_x, ctrl_y, line_color)
        
        # Animated flow particles
        for p in conn.particles:
            t = p["pos"]
            px = (1 - t)**2 * x1 + 2 * (1 - t) * t * ctrl_x + t**2 * x2
            py = (1 - t)**2 * y1 + 2 * (1 - t) * t * ctrl_y + t**2 * y2
            
            size = p["size"] * p["alpha"]
            alpha = int(255 * p["alpha"])
            pcolor = self._blend_color(self._theme.colors.accent_primary, alpha)
            
            canvas.create_oval(
                px - size, py - size,
                px + size, py + size,
                fill=pcolor, outline="", tags="dag_all"
            )
            
            # Trail
            trail_t = max(0, t - 0.05)
            tx = (1 - trail_t)**2 * x1 + 2 * (1 - trail_t) * trail_t * ctrl_x + trail_t**2 * x2
            ty = (1 - trail_t)**2 * y1 + 2 * (1 - trail_t) * trail_t * ctrl_y + trail_t**2 * y2
            trail_size = size * 0.5
            trail_alpha = int(alpha * 0.3)
            trail_color = self._blend_color(self._theme.colors.accent_primary, trail_alpha)
            canvas.create_oval(
                tx - trail_size, ty - trail_size,
                tx + trail_size, ty + trail_size,
                fill=trail_color, outline="", tags="dag_all"
            )
    
    def _draw_arrow_head(self, canvas: object, x: float, y: float, ctrl_x: float, ctrl_y: float, color: str) -> None:
        """Draw arrow head at end of connection."""
        # Calculate angle from control point to end
        angle = math.atan2(y - ctrl_y, x - ctrl_x)
        
        arrow_size = 10
        # Arrow points
        ax1 = x - arrow_size * math.cos(angle - 0.4)
        ay1 = y - arrow_size * math.sin(angle - 0.4)
        ax2 = x - arrow_size * math.cos(angle + 0.4)
        ay2 = y - arrow_size * math.sin(angle + 0.4)
        
        canvas.create_polygon(
            x, y, ax1, ay1, ax2, ay2,
            fill=color, outline="", tags="dag_all"
        )
    
    def _draw_particles(self, canvas: object, particles: List[Dict]) -> None:
        """Draw particle effects."""
        for p in particles:
            alpha = int(255 * (p["life"] / p["max_life"]) * p.get("alpha", 1.0))
            if alpha < 10:
                continue
            color = self._blend_color(p["color"], alpha)
            size = p["size"] * (p["life"] / p["max_life"])
            
            canvas.create_oval(
                p["x"] - size, p["y"] - size,
                p["x"] + size, p["y"] + size,
                fill=color, outline="", tags="dag_all"
            )
    
    def _draw_rounded_rect(self, canvas: object, x: float, y: float, w: float, h: float, 
                           radius: float, **kwargs) -> None:
        """Draw a rounded rectangle on canvas."""
        r = min(radius, w/2, h/2)
        if r <= 1:
            canvas.create_rectangle(x, y, x + w, y + h, **kwargs)
            return
        
        # Main rectangles
        canvas.create_rectangle(x + r, y, x + w - r, y + h, **kwargs)
        canvas.create_rectangle(x, y + r, x + w, y + h - r, **kwargs)
        
        # Four corners
        canvas.create_arc(x, y, x + 2*r, y + 2*r, start=90, extent=90, **kwargs)
        canvas.create_arc(x + w - 2*r, y, x + w, y + 2*r, start=0, extent=90, **kwargs)
        canvas.create_arc(x, y + h - 2*r, x + 2*r, y + h, start=180, extent=90, **kwargs)
        canvas.create_arc(x + w - 2*r, y + h - 2*r, x + w, y + h, start=270, extent=90, **kwargs)
    
    def _blend_color(self, hex_color: str, alpha: int) -> str:
        """Blend hex color with black background for simulated alpha."""
        alpha = max(0, min(255, alpha))
        if alpha >= 255:
            return hex_color
        
        # Parse hex
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join(c*2 for c in hex_color)
        
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # Blend with black (0,0,0)
        r = int(r * alpha / 255)
        g = int(g * alpha / 255)
        b = int(b * alpha / 255)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _calculate_layout(self) -> None:
        """Calculate node positions using a simple layered layout."""
        if not self.nodes:
            return
        
        # Find root nodes (no dependencies)
        root_ids = [
            nid for nid, node in self.nodes.items()
            if not node.depends_on
        ]
        
        # Simple BFS layering
        layers: List[List[str]] = []
        visited: set = set()
        current_layer = root_ids
        
        while current_layer:
            layers.append(current_layer)
            visited.update(current_layer)
            
            next_layer = []
            for nid in current_layer:
                node = self.nodes[nid]
                # Find nodes that depend on this one
                for other_id, other_node in self.nodes.items():
                    if other_id not in visited and nid in other_node.depends_on:
                        if other_id not in next_layer:
                            next_layer.append(other_id)
            
            # Also add any unvisited nodes that have all deps satisfied
            for nid, node in self.nodes.items():
                if nid not in visited and nid not in next_layer:
                    if all(dep in visited for dep in node.depends_on):
                        next_layer.append(nid)
            
            current_layer = next_layer
        
        # Position nodes by layer
        start_y = self._padding
        for layer_idx, layer in enumerate(layers):
            count = len(layer)
            total_width = count * self._node_spacing_x
            start_x = (self._canvas_width - total_width) / 2
            
            for i, nid in enumerate(layer):
                if nid in self.nodes:
                    self.nodes[nid].x = start_x + i * self._node_spacing_x
                    self.nodes[nid].y = start_y
                    # Stagger entry animation
                    self.nodes[nid].entry_animation = 0.0
            
            start_y += self._node_spacing_y