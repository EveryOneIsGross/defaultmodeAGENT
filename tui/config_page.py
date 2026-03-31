"""Config page for the Agent Manager TUI - live edit per-bot config overrides."""

import json
from typing import Any, Optional, get_args, get_origin

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Label, Select, Button, Input, Switch, TabbedContent, TabPane

from tui.shared import PATHS, discover_bots

#import sys
#sys.path.insert(0, str(PATHS.root / "agent"))
from agent.bot_config import (
    ConversationConfig, PersonaConfig, DMNConfig,
    AttentionConfig, SpikeConfig, FileConfig, SearchConfig, SystemConfig,
)

# (tab_id, display_label, config_class, fields_to_skip)
SECTIONS: list[tuple[str, str, type, set]] = [
    ("conversation", "Conversation", ConversationConfig, set()),
    ("persona",      "Persona",      PersonaConfig,      set()),
    ("dmn",          "DMN",          DMNConfig,          {"modes", "dmn_api_type", "dmn_model"}),
    ("attention",    "Attention",    AttentionConfig,    {"stop_words"}),
    ("spike",        "Spike",        SpikeConfig,        set()),
    ("files",        "Files",        FileConfig,         {"allowed_extensions", "allowed_image_extensions"}),
    ("search",       "Search",       SearchConfig,       set()),
    ("system",       "System",       SystemConfig,       set()),
]

_SIMPLE = (int, float, str, bool)


def _resolve_type(annotation: Any) -> type | None:
    """Unwrap Optional[X] and return the inner type if it's simple, else None."""
    if annotation in _SIMPLE:
        return annotation
    for a in get_args(annotation):
        if a in _SIMPLE:
            return a
    return None


def _simple_fields(config_class: type) -> list[tuple[str, type, Any, str]]:
    """Return (name, type, default, description) for each simple-typed field."""
    instance_defaults = config_class().model_dump()
    result = []
    for name, fi in config_class.model_fields.items():
        t = _resolve_type(fi.annotation)
        if t is None:
            continue
        result.append((name, t, instance_defaults.get(name), fi.description or ""))
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_overrides(bot_name: str) -> dict:
    p = PATHS.cache_dir / bot_name / "config_overrides.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_overrides(bot_name: str, overrides: dict) -> bool:
    p = PATHS.cache_dir / bot_name / "config_overrides.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(overrides, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Page widget
# ---------------------------------------------------------------------------

class ConfigPage(Vertical):
    """Per-bot config override editor.

    Edits are written to cache/{bot}/config_overrides.json.
    Changes apply on the bot's next launch.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._bot: Optional[str] = None
        self._overrides: dict = {}

    def compose(self) -> ComposeResult:
        bots = [(b, b) for b in discover_bots()]
        yield Horizontal(
            Vertical(
                Select(bots, id="cfg-bot-select", prompt="select bot"),
                Button("Reload", id="cfg-reload-btn"),
                Label("", id="cfg-bot-status", classes="cfg-bot-status"),
                Label("[dim]applies on next launch[/dim]", classes="cfg-hint"),
                classes="cfg-sidebar",
            ),
            Vertical(
                TabbedContent(id="cfg-tabs"),
                Horizontal(
                    Button("Save overrides", variant="success", id="cfg-save-btn"),
                    Button("Clear overrides", variant="warning", id="cfg-clear-btn"),
                    Label("", id="cfg-status"),
                    classes="cfg-actions",
                ),
                classes="cfg-main",
            ),
            id="cfg-root",
        )

    async def on_mount(self):
        tc = self.query_one("#cfg-tabs", TabbedContent)
        for tab_id, label, _, _ in SECTIONS:
            await tc.add_pane(TabPane(
                label,
                ScrollableContainer(id=f"cfg-section-{tab_id}"),
                id=f"cfg-tab-{tab_id}",
            ))
        bots = discover_bots()
        if bots:
            self._bot = bots[0]
            self.query_one("#cfg-bot-select", Select).value = bots[0]
            self._load_bot(bots[0])

    # ------------------------------------------------------------------
    # Loading / population
    # ------------------------------------------------------------------

    def _load_bot(self, bot_name: str):
        self._bot = bot_name
        self._overrides = load_overrides(bot_name)
        self._populate_all_sections()
        override_count = sum(len(v) for v in self._overrides.values())
        status = f"[dim]{bot_name}[/dim]"
        if override_count:
            status += f"  [bold]({override_count} override{'s' if override_count != 1 else ''})[/bold]"
        self.query_one("#cfg-bot-status", Label).update(status)
        self.query_one("#cfg-status", Label).update("")

    def _populate_all_sections(self):
        for tab_id, _, config_class, skip in SECTIONS:
            self._populate_section(tab_id, config_class, skip)

    def _populate_section(self, tab_id: str, config_class: type, skip: set):
        container = self.query_one(f"#cfg-section-{tab_id}", ScrollableContainer)
        container.remove_children()
        section_overrides = self._overrides.get(tab_id, {})
        for name, ftype, default, desc in _simple_fields(config_class):
            if name in skip:
                continue
            current = section_overrides.get(name, default)
            container.mount(self._make_field_row(tab_id, name, ftype, current, default, desc))

    def _make_field_row(
        self, section: str, name: str, ftype: type,
        current: Any, default: Any, desc: str,
    ) -> Horizontal:
        is_overridden = current != default
        label_text = name.replace("_", " ").title()
        markup = f"[bold]{label_text}[/bold]" if is_overridden else label_text
        if desc:
            markup += f"  [dim]{desc[:60]}[/dim]"

        widget_id = f"cfg-field-{section}-{name}"
        if ftype is bool:
            control = Switch(value=bool(current), id=widget_id, classes="cfg-field-control")
        else:
            control = Input(
                value=str(current) if current is not None else "",
                id=widget_id,
                classes="cfg-field-control",
            )

        return Horizontal(
            Label(markup, classes="cfg-field-label"),
            control,
            classes="cfg-field-row" + (" cfg-overridden" if is_overridden else ""),
        )

    # ------------------------------------------------------------------
    # Collecting values
    # ------------------------------------------------------------------

    def _collect_overrides(self) -> dict:
        """Walk all field widgets, return only values that differ from defaults."""
        result: dict[str, dict] = {}
        for tab_id, _, config_class, skip in SECTIONS:
            section_defaults = config_class().model_dump()
            section: dict = {}
            for name, ftype, default, _ in _simple_fields(config_class):
                if name in skip:
                    continue
                widget_id = f"cfg-field-{tab_id}-{name}"
                try:
                    if ftype is bool:
                        val: Any = self.query_one(f"#{widget_id}", Switch).value
                    else:
                        raw = self.query_one(f"#{widget_id}", Input).value.strip()
                        if ftype is int:
                            val = int(raw)
                        elif ftype is float:
                            val = float(raw)
                        else:
                            val = raw
                    if val != section_defaults.get(name):
                        section[name] = val
                except Exception:
                    pass
            if section:
                result[tab_id] = section
        return result

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    @on(Select.Changed, "#cfg-bot-select")
    def on_bot_changed(self, event: Select.Changed):
        if event.value and event.value != Select.BLANK:
            self._load_bot(str(event.value))

    @on(Button.Pressed, "#cfg-reload-btn")
    def on_reload(self):
        if self._bot:
            self._load_bot(self._bot)

    @on(Button.Pressed, "#cfg-save-btn")
    def on_save(self):
        if not self._bot:
            self.query_one("#cfg-status", Label).update("[bold red]select a bot first[/bold red]")
            return
        overrides = self._collect_overrides()
        if save_overrides(self._bot, overrides):
            self._overrides = overrides
            count = sum(len(v) for v in overrides.values())
            msg = (
                f"[bold green]saved {count} override{'s' if count != 1 else ''}[/bold green]"
                if count else "[dim]saved (all defaults)[/dim]"
            )
            self.query_one("#cfg-status", Label).update(msg)
            self._populate_all_sections()  # refresh override highlighting
        else:
            self.query_one("#cfg-status", Label).update("[bold red]save failed[/bold red]")

    @on(Button.Pressed, "#cfg-clear-btn")
    def on_clear(self):
        if not self._bot:
            return
        if save_overrides(self._bot, {}):
            self._overrides = {}
            self._populate_all_sections()
            self.query_one("#cfg-status", Label).update("[dim]overrides cleared[/dim]")
            self.query_one("#cfg-bot-status", Label).update(f"[dim]{self._bot}[/dim]")
