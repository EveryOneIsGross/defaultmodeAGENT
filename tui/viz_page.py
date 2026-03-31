"""Visualization page for the Agent Manager TUI."""

from typing import Optional, List, Tuple
from collections import defaultdict
import numpy as np

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.events import Click
from textual.widgets import Label, Button, Select, Static, Checkbox

from rich.text import Text as RichText

from tui.shared import (
    VizNode, get_bot_caches, load_memory_cache,
    build_tfidf_vectors, reduce_dimensions, find_connections,
    _draw_line, tokenize,
)


class VizPage(Vertical):
    """Memory latent space visualization page with zoom and navigation."""

    BINDINGS = [
        Binding("w", "move_up", "Up", show=False),
        Binding("s", "move_down", "Down", show=False),
        Binding("a", "move_left", "Left", show=False),
        Binding("d", "move_right", "Right", show=False),
        Binding("up", "pan_up", "Pan Up", show=False),
        Binding("down", "pan_down", "Pan Down", show=False),
        Binding("left", "pan_left", "Pan Left", show=False),
        Binding("right", "pan_right", "Pan Right", show=False),
        Binding("enter", "select_node", "Select", show=False),
        Binding("+", "zoom_in", "Zoom+", show=False),
        Binding("=", "zoom_in", "Zoom+", show=False),
        Binding("-", "zoom_out", "Zoom-", show=False),
        Binding("f", "focus_selected", "Focus", show=False),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache: dict = {}
        self.loaded_bot: Optional[str] = None
        self.nodes: List[VizNode] = []
        self.method: str = "pca"
        self.selected_idx: int = -1
        self.connections: List[int] = []
        self.zoom: int = 1
        self.extended_neighbors: bool = False
        self.view_cx: float = 0.0
        self.view_cy: float = 0.0
        self._grid_w: int = 80
        self._grid_h: int = 40

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Select([], id="viz-bot-select", prompt="select bot"),
            Select([], id="viz-user-select", prompt="all users"),
            Checkbox("Global (all users)", id="viz-global"),
            Select([("PCA", "pca"), ("UMAP", "umap")], id="viz-method", value="pca"),
            Checkbox("Extended", id="viz-extended"),
            Button("Load", id="viz-load-btn"),
            Button("Refresh", id="viz-refresh-btn"),
            id="viz-controls",
        )
        yield Static("", id="viz-status")
        yield Horizontal(
            Vertical(
                Static("[dim]Load memories to visualize[/dim]", id="viz-content"),
                id="viz-canvas",
            ),
            Vertical(
                Static("[bold]Memory Details[/bold]", id="viz-detail-header"),
                ScrollableContainer(
                    Static("", id="viz-detail-content"),
                    id="viz-detail-scroll",
                ),
                Static("[bold]Connections[/bold]", id="viz-conn-header"),
                ScrollableContainer(
                    Vertical(id="viz-connections"),
                    id="viz-conn-scroll",
                ),
                id="viz-detail-panel",
            ),
            id="viz-main",
        )

    def on_mount(self):
        self._refresh_bot_list()

    def _refresh_bot_list(self):
        bots = [(b, b) for b in get_bot_caches()]
        sel = self.query_one("#viz-bot-select", Select)
        sel.set_options(bots)
        if bots and (not sel.value or sel.value == Select.BLANK):
            sel.value = bots[0][1]

    def _refresh_user_list(self):
        users = list(self.cache.get("user_memories", {}).keys())
        user_sel = self.query_one("#viz-user-select", Select)
        user_sel.set_options([("", "all users")] + [(u, u) for u in users])

    @on(Select.Changed, "#viz-method")
    def on_method_change(self, event: Select.Changed):
        if event.value and event.value != Select.BLANK:
            self.method = event.value

    @on(Checkbox.Changed, "#viz-global")
    def on_global_change(self, event: Checkbox.Changed):
        self.query_one("#viz-user-select", Select).disabled = event.value

    @on(Checkbox.Changed, "#viz-extended")
    def on_extended_change(self, event: Checkbox.Changed):
        self.extended_neighbors = event.value
        if self.nodes and 0 <= self.selected_idx < len(self.nodes):
            self._show_node_details(self.nodes[self.selected_idx])

    @on(Button.Pressed, "#viz-load-btn")
    def on_load(self):
        sel = self.query_one("#viz-bot-select", Select)
        if not sel.value or sel.value == Select.BLANK:
            return
        self.loaded_bot = sel.value
        self.cache = load_memory_cache(sel.value)
        if not self.cache:
            self.query_one("#viz-status", Static).update("[bold]failed to load[/bold]")
            return
        self._refresh_user_list()
        self._generate_viz()

    @on(Button.Pressed, "#viz-refresh-btn")
    def on_refresh(self):
        if self.cache:
            self._generate_viz()

    def _generate_viz(self):
        if not self.cache:
            return

        memories = self.cache.get("memories", [])
        user_mems = self.cache.get("user_memories", {})

        is_global = self.query_one("#viz-global", Checkbox).value
        user_sel = self.query_one("#viz-user-select", Select)
        user_id = None if is_global or not user_sel.value or user_sel.value == Select.BLANK else user_sel.value

        if user_id:
            memory_ids = [mid for mid in user_mems.get(user_id, [])
                         if mid < len(memories) and memories[mid] is not None]
        else:
            memory_ids = [i for i, m in enumerate(memories) if m is not None]

        if not memory_ids:
            self.query_one("#viz-status", Static).update("[dim]no memories to visualize[/dim]")
            self.nodes = []
            self._render_canvas()
            return

        max_nodes = 16000
        if len(memory_ids) > max_nodes:
            memory_ids = memory_ids[:max_nodes]
            self.query_one("#viz-status", Static).update(
                f"[bold]showing {max_nodes} (limited)[/bold]"
            )
        else:
            self.query_one("#viz-status", Static).update(
                f"visualizing {len(memory_ids)} memories"
            )

        vectors, terms, raw_mags = build_tfidf_vectors(self.cache, memory_ids)
        coords = reduce_dimensions(vectors, self.method)

        centroid = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
        max_dist = distances.max() if distances.max() > 0 else 1
        dist_scores = distances / max_dist

        max_mag = raw_mags.max() if raw_mags.max() > 0 else 1
        mag_scores = raw_mags / max_mag

        scores = 0.5 * dist_scores + 0.5 * mag_scores

        self.nodes = []
        for i, mid in enumerate(memory_ids):
            uid = next((u for u, mids in user_mems.items() if mid in mids), None)
            text = memories[mid] if mid < len(memories) else ""
            self.nodes.append(VizNode(
                mid=mid, x=coords[i, 0], y=coords[i, 1],
                text=text, user_id=uid, score=scores[i],
            ))

        self.selected_idx = 0 if self.nodes else -1
        self.connections = []
        self.zoom = 1
        if self.nodes:
            self.view_cx = float(np.mean([n.x for n in self.nodes]))
            self.view_cy = float(np.mean([n.y for n in self.nodes]))
        self._render_canvas()

        if self.nodes:
            self._show_node_details(self.nodes[0])

    def _render_canvas(self):
        content = self.query_one("#viz-content", Static)
        if not self.nodes:
            content.update("[dim]Load memories to visualize[/dim]")
            return

        try:
            canvas = self.query_one("#viz-canvas")
            cw = canvas.content_size.width
            ch = canvas.content_size.height
            grid_w = max(40, cw) if cw > 0 else 80
            grid_h = max(20, ch - 1) if ch > 1 else 40
        except Exception:
            grid_w, grid_h = 80, 40
        self._grid_w = grid_w
        self._grid_h = grid_h

        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        data_min_x, data_max_x = min(xs), max(xs)
        data_min_y, data_max_y = min(ys), max(ys)
        data_range_x = max(data_max_x - data_min_x, 0.001)
        data_range_y = max(data_max_y - data_min_y, 0.001)

        view_range_x = data_range_x / self.zoom
        view_range_y = data_range_y / self.zoom
        if self.zoom == 1:
            view_range_x *= 1.08
            view_range_y *= 1.08
        view_min_x = self.view_cx - view_range_x / 2
        view_min_y = self.view_cy - view_range_y / 2

        pad = 1
        for node in self.nodes:
            gx = int((node.x - view_min_x) / view_range_x * (grid_w - pad * 2)) + pad
            gy = int((node.y - view_min_y) / view_range_y * (grid_h - pad * 2)) + pad
            node.grid_x = gx
            node.grid_y = gy

        grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]
        node_set = set()

        visible = []
        for i, node in enumerate(self.nodes):
            if 0 <= node.grid_x < grid_w and 0 <= node.grid_y < grid_h:
                visible.append((i, node))
                node_set.add((node.grid_x, node.grid_y))

        if len(visible) < 5000:
            density = defaultdict(int)
            for _, node in visible:
                for ddx in range(-3, 4):
                    for ddy in range(-2, 3):
                        if ddx == 0 and ddy == 0:
                            continue
                        nx, ny = node.grid_x + ddx, node.grid_y + ddy
                        if 0 <= nx < grid_w and 0 <= ny < grid_h:
                            density[(nx, ny)] += 1
            for (x, y), count in density.items():
                if (x, y) not in node_set:
                    if count >= 8:
                        grid[y][x] = "▓"
                    elif count >= 4:
                        grid[y][x] = "▒"
                    elif count >= 2:
                        grid[y][x] = "░"

        if self.selected_idx >= 0 and self.connections:
            sel = self.nodes[self.selected_idx]
            for conn_mid in self.connections:
                for node in self.nodes:
                    if node.mid == conn_mid:
                        _draw_line(grid, sel.grid_x, sel.grid_y,
                                   node.grid_x, node.grid_y, grid_w, grid_h)
                        break

        for i, node in visible:
            gx, gy = node.grid_x, node.grid_y
            if i == self.selected_idx:
                grid[gy][gx] = "◉"
            elif node.mid in self.connections:
                grid[gy][gx] = "◎"
            elif node.score > 0.85:
                grid[gy][gx] = "◆"
            elif node.score > 0.7:
                grid[gy][gx] = "●"
            elif node.score > 0.5:
                grid[gy][gx] = "◐"
            elif node.score > 0.3:
                grid[gy][gx] = "○"
            elif node.score > 0.15:
                grid[gy][gx] = "·"
            else:
                grid[gy][gx] = "∘"

        lines = ["".join(row) for row in grid]
        ext = " EXT" if self.extended_neighbors else ""
        vis = len(visible)
        legend = f"◆hi ●md ◐mid ○lo ·dim ▓▒░density | ◉sel ◎conn ─│╲╱┼link | z:{self.zoom} v:{vis}/{len(self.nodes)}{ext} | WASD:nav +-/scroll:zoom arrows:pan"
        lines.append(legend)
        content.update("\n".join(lines))

    def _center_on_selected(self):
        """Center viewport on selected node without re-rendering."""
        if self.nodes and 0 <= self.selected_idx < len(self.nodes):
            self.view_cx = self.nodes[self.selected_idx].x
            self.view_cy = self.nodes[self.selected_idx].y

    def _focus_selected(self):
        self._center_on_selected()
        self._render_canvas()

    def _data_ranges(self) -> Tuple[float, float]:
        if not self.nodes:
            return (1.0, 1.0)
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        return (max(max(xs) - min(xs), 0.001), max(max(ys) - min(ys), 0.001))

    def _find_nearest(self, dx: int, dy: int):
        if not self.nodes or self.selected_idx < 0:
            return
        current = self.nodes[self.selected_idx]
        best_idx, best_dist = -1, float("inf")

        for i, node in enumerate(self.nodes):
            if i == self.selected_idx:
                continue
            dir_x = node.grid_x - current.grid_x
            dir_y = node.grid_y - current.grid_y
            if dx != 0 and (dx * dir_x <= 0):
                continue
            if dy != 0 and (dy * dir_y <= 0):
                continue
            dist = abs(dir_x) + abs(dir_y)
            if dist < best_dist:
                best_dist, best_idx = dist, i

        if best_idx >= 0:
            self.selected_idx = best_idx
            self._center_on_selected()
            self._render_canvas()
            self._show_node_details(self.nodes[best_idx])

    def action_move_up(self):
        self._find_nearest(0, -1)

    def action_move_down(self):
        self._find_nearest(0, 1)

    def action_move_left(self):
        self._find_nearest(-1, 0)

    def action_move_right(self):
        self._find_nearest(1, 0)

    def action_zoom_in(self):
        if self.zoom < 10:
            self.zoom += 1
            self._render_canvas()

    def action_zoom_out(self):
        if self.zoom > 1:
            self.zoom -= 1
            self._render_canvas()

    def action_focus_selected(self):
        self._focus_selected()

    def action_select_node(self):
        if self.nodes and 0 <= self.selected_idx < len(self.nodes):
            self._show_node_details(self.nodes[self.selected_idx], show_full=True)

    def action_pan_up(self):
        _, dr_y = self._data_ranges()
        self.view_cy -= dr_y / self.zoom * 0.15
        self._render_canvas()

    def action_pan_down(self):
        _, dr_y = self._data_ranges()
        self.view_cy += dr_y / self.zoom * 0.15
        self._render_canvas()

    def action_pan_left(self):
        dr_x, _ = self._data_ranges()
        self.view_cx -= dr_x / self.zoom * 0.15
        self._render_canvas()

    def action_pan_right(self):
        dr_x, _ = self._data_ranges()
        self.view_cx += dr_x / self.zoom * 0.15
        self._render_canvas()

    def on_click(self, event: Click) -> None:
        """Handle mouse clicks to select nodes."""
        if not self.nodes:
            return

        try:
            canvas = self.query_one("#viz-canvas")
        except Exception:
            return

        if not canvas.region.contains(event.screen_x, event.screen_y):
            return

        click_x = event.screen_x - canvas.region.x - 1
        click_y = event.screen_y - canvas.region.y - 1

        gw, gh = self._grid_w, self._grid_h
        best_idx = -1
        best_dist = float("inf")
        for i, node in enumerate(self.nodes):
            if not (0 <= node.grid_x < gw and 0 <= node.grid_y < gh):
                continue
            dist = abs(node.grid_x - click_x) + abs(node.grid_y - click_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        max_dist = max(3, self.zoom + 2)
        if best_idx >= 0 and best_dist <= max_dist:
            self.selected_idx = best_idx
            self._center_on_selected()
            self._render_canvas()
            self._show_node_details(self.nodes[best_idx])

    def on_mouse_scroll_up(self, event) -> None:
        """Zoom in on mouse scroll up over canvas."""
        try:
            canvas = self.query_one("#viz-canvas")
            if canvas.region.contains(event.screen_x, event.screen_y):
                if self.nodes and self.zoom < 10:
                    self.zoom += 1
                    self._render_canvas()
                event.stop()
        except Exception:
            pass

    def on_mouse_scroll_down(self, event) -> None:
        """Zoom out on mouse scroll down over canvas."""
        try:
            canvas = self.query_one("#viz-canvas")
            if canvas.region.contains(event.screen_x, event.screen_y):
                if self.nodes and self.zoom > 1:
                    self.zoom -= 1
                    self._render_canvas()
                event.stop()
        except Exception:
            pass

    def on_resize(self, event) -> None:
        """Re-render canvas when terminal resizes."""
        if self.nodes:
            self._render_canvas()

    def _show_node_details(self, node: VizNode, show_full: bool = False):
        text = node.text if show_full else (node.text[:16000] + "..." if len(node.text) > 16000 else node.text)
        header = RichText.from_markup(
            f"[bold]Memory #{node.mid}[/bold]\n"
            f"[dim]User: {node.user_id or 'unknown'}[/dim]\n"
            f"[dim]Score: {node.score:.2f}[/dim]\n\n"
        )
        header.append(text)
        self.query_one("#viz-detail-content", Static).update(header)

        top_k = 16 if self.extended_neighbors else 6
        connections = find_connections(self.cache, node.mid, top_k=top_k)
        self.connections = [c[0] for c in connections]
        self._render_canvas()

        conn_container = self.query_one("#viz-connections", Vertical)
        conn_container.remove_children()

        memories = self.cache.get("memories", [])
        for conn_mid, score, terms in connections:
            conn_text = memories[conn_mid] if conn_mid < len(memories) else ""
            conn_container.mount(
                Vertical(
                    Label(RichText.from_markup(f"[bold]#{conn_mid}[/bold] [dim]sim={score:.2f}[/dim]")),
                    Label(RichText.from_markup(f"[dim]shared: {', '.join(terms)}[/dim]")),
                    Static(RichText(conn_text)),
                    classes="viz-conn-item",
                )
            )
