"""Logs page for the Agent Manager TUI."""

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, Input, Button, TextArea, Select, Static

from tui.shared import PATHS


class LogsPage(Vertical):
    REFRESH_INTERVALS = [("Off", 0), ("5s", 5), ("10s", 10), ("30s", 30)]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.current_bot: Optional[str] = None
        self.last_size: int = 0
        self.all_entries: list[dict] = []
        self.auto_refresh_timer = None

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Select([], id="log-bot-select", prompt="select bot"),
            Input(placeholder="search keyword", id="log-search"),
            Select(self.REFRESH_INTERVALS, id="log-refresh-interval", value=0),
            Button("Refresh", id="log-refresh-btn"),
            id="log-controls",
        )
        yield Horizontal(
            Label("", id="log-status"),
            id="log-status-bar",
        )
        yield TextArea(id="log-viewer", read_only=True, soft_wrap=True)

    def on_mount(self):
        self._refresh_bot_list()

    def _refresh_bot_list(self):
        bots = []
        if PATHS.cache_dir.exists():
            for d in PATHS.cache_dir.iterdir():
                if d.is_dir() and (d / "logs").exists():
                    bots.append((d.name, d.name))
        sel = self.query_one("#log-bot-select", Select)
        sel.set_options(bots)
        if bots and (not sel.value or sel.value == Select.BLANK):
            sel.value = bots[0][1]
            self._load_logs(sel.value, full=True)

    @on(Select.Changed, "#log-bot-select")
    def on_bot_change(self, event: Select.Changed):
        if event.value and event.value != Select.BLANK:
            self.current_bot = event.value
            self._load_logs(event.value, full=True)

    @on(Select.Changed, "#log-refresh-interval")
    def on_interval_change(self, event: Select.Changed):
        if self.auto_refresh_timer:
            self.auto_refresh_timer.stop()
            self.auto_refresh_timer = None
        interval = event.value
        if interval and interval > 0:
            self.auto_refresh_timer = self.set_interval(interval, self._auto_refresh)

    def _auto_refresh(self):
        if self.current_bot:
            self._load_logs(self.current_bot, full=False)

    @on(Button.Pressed, "#log-refresh-btn")
    def on_refresh(self):
        sel = self.query_one("#log-bot-select", Select)
        if sel.value and sel.value != Select.BLANK:
            self._load_logs(sel.value, full=True)

    @on(Input.Changed, "#log-search")
    def on_search(self, event: Input.Changed):
        self._render_logs()

    def _load_logs(self, bot_name: str, full: bool = True):
        import json
        p = PATHS.bot_log(bot_name)
        if not p.exists():
            self.query_one("#log-status", Label).update(f"no logs at {p}")
            self.all_entries = []
            self._render_logs()
            return
        try:
            current_size = p.stat().st_size
            if not full and current_size == self.last_size:
                return
            self.last_size = current_size
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = p.read_text(encoding="utf-8", errors="replace")

            self.all_entries = []
            skipped: list[tuple[int, str]] = []

            for lineno, raw in enumerate(text.splitlines(), 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    skipped.append((lineno, str(e)))
                    continue
                if not isinstance(entry, dict):
                    skipped.append((lineno, f"expected object, got {type(entry).__name__}"))
                    continue
                self.all_entries.append(entry)

            # Inject skipped-line report as a synthetic entry at the top
            if skipped:
                detail = "; ".join(f"L{n}: {reason}" for n, reason in skipped[:10])
                if len(skipped) > 10:
                    detail += f" … +{len(skipped) - 10} more"
                self.all_entries.insert(0, {
                    "event": "import_warning",
                    "level": "WARNING",
                    "skipped_lines": len(skipped),
                    "detail": detail,
                })

            status = f"{len(self.all_entries)} entries from {p.name}"
            if skipped:
                status += f"  ⚠ {len(skipped)} line(s) skipped"
            self.query_one("#log-status", Label).update(status)
            self._render_logs()
        except Exception as ex:
            self.query_one("#log-status", Label).update(f"error: {ex}")

    def _render_logs(self):
        import json
        viewer = self.query_one("#log-viewer", TextArea)
        search = self.query_one("#log-search", Input).value.strip().lower()
        blocks = []

        SKIP = {"timestamp", "event", "level", "data", "created_at", "event_type", "id"}
        PREFIXES = {
            "user_message": ">>> ",
            "ai_response": "<<< ",
            "response": "<<< ",
            "error": "!!! ",
        }

        for e in self.all_entries:
            ts = e.get("timestamp", "")[:19] if e.get("timestamp") else ""
            evt = e.get("event", "?")
            level = e.get("level", "")

            lines = [f"[{ts}] {evt.upper()}" + (f" ({level})" if level else "")]

            for key, val in e.items():
                if key in SKIP or val is None:
                    continue

                if isinstance(val, dict):
                    val = json.dumps(val, ensure_ascii=False)
                elif isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                else:
                    val = str(val)

                if len(val) > 1000:
                    val = val[:1000] + "..."

                prefix = PREFIXES.get(key, "")
                label = key.replace("_", " ").title()
                lines.append(f"  {label}: {prefix}{val}")

            block = "\n".join(lines)

            if not search or search in block.lower():
                blocks.append(block)

        viewer.text = "\n\n".join(blocks[-300:])
        viewer.scroll_end(animate=False)
