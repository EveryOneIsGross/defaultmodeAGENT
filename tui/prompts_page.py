"""Prompts page for the Agent Manager TUI."""

import re
import yaml

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, ListView, Input, Button, TextArea, Select, Static

from tui.shared import (
    PATHS, discover_bots, load_yaml_file, save_yaml_file,
    validate_prompts, create_bot_stub, SelectableItem,
)


class PromptsPage(Vertical):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.current_bot = None

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Vertical(
                Select([], id="prompt-bot-select", prompt="select bot"),
                Button("Refresh", id="prompt-refresh-btn"),
                Input(placeholder="new bot name", id="new-bot-input"),
                Button("Create", variant="primary", id="prompt-create-btn"),
                Button("Save", variant="success", id="prompt-save-btn"),
                Button("Validate", id="prompt-validate-btn"),
                ListView(id="prompt-bot-list"),
                classes="prompt-sidebar",
            ),
            Vertical(
                Horizontal(
                    Vertical(Label("[bold]system_prompts.yaml[/bold]"), TextArea(id="sys-editor", language="yaml"), classes="editor-pane"),
                    Vertical(Label("[bold]prompt_formats.yaml[/bold]"), TextArea(id="fmt-editor", language="yaml"), classes="editor-pane"),
                    id="prompt-editors",
                ),
                Static("", id="validation-results"),
                classes="prompt-main",
            ),
            id="prompt-root",
        )

    def on_mount(self):
        self._refresh_bot_list()

    def _refresh_bot_list(self):
        bots = [(b, b) for b in discover_bots()]
        sel = self.query_one("#prompt-bot-select", Select)
        sel.set_options(bots)
        lv = self.query_one("#prompt-bot-list", ListView)
        lv.clear()
        for b, _ in bots:
            has_cfg = PATHS.bot_system_prompts(b).exists()
            lv.append(SelectableItem(b, b, "ready" if has_cfg else "no config", has_cfg))
        if bots and (not sel.value or sel.value == Select.BLANK):
            sel.value = bots[0][1]
            self._load_bot(sel.value)

    def _load_bot(self, bot: str):
        self.current_bot = bot
        sys_path = PATHS.bot_system_prompts(bot)
        fmt_path = PATHS.bot_prompt_formats(bot)
        sys_text = sys_path.read_text(encoding="utf-8") if sys_path.exists() else ""
        fmt_text = fmt_path.read_text(encoding="utf-8") if fmt_path.exists() else ""
        self.query_one("#sys-editor", TextArea).text = sys_text
        self.query_one("#fmt-editor", TextArea).text = fmt_text
        self.query_one("#validation-results", Static).update(f"[dim]loaded {bot}[/dim]")

    @on(Button.Pressed, "#prompt-refresh-btn")
    def on_refresh(self):
        self._refresh_bot_list()

    @on(Select.Changed, "#prompt-bot-select")
    def on_prompt_select(self, event: Select.Changed):
        if event.value and event.value != Select.BLANK:
            self._load_bot(event.value)

    @on(ListView.Selected, "#prompt-bot-list")
    def on_prompt_list(self, event: ListView.Selected):
        if isinstance(event.item, SelectableItem):
            self.query_one("#prompt-bot-select", Select).value = event.item.value
            self._load_bot(event.item.value)

    @on(Button.Pressed, "#prompt-save-btn")
    def on_save(self):
        if not self.current_bot:
            return
        try:
            sys_data = yaml.safe_load(self.query_one("#sys-editor", TextArea).text) or {}
            fmt_data = yaml.safe_load(self.query_one("#fmt-editor", TextArea).text) or {}
            save_yaml_file(PATHS.bot_system_prompts(self.current_bot), sys_data)
            save_yaml_file(PATHS.bot_prompt_formats(self.current_bot), fmt_data)
            self.query_one("#validation-results", Static).update("[bold]saved[/bold]")
        except Exception as e:
            self.query_one("#validation-results", Static).update(f"[bold]error:[/bold] {str(e).replace('[', '(').replace(']', ')')}")

    @on(Button.Pressed, "#prompt-validate-btn")
    def on_validate(self):
        try:
            sys_data = yaml.safe_load(self.query_one("#sys-editor", TextArea).text) or {}
            fmt_data = yaml.safe_load(self.query_one("#fmt-editor", TextArea).text) or {}
            r = validate_prompts(sys_data, fmt_data)
            lines = []
            for cat in ("system", "formats"):
                for k, v in r[cat].items():
                    if v["missing"]:
                        missing_str = ", ".join(sorted(v["missing"]))
                        lines.append(f"[bold]{cat}.{k} missing: {missing_str}[/bold]")
            if not lines:
                lines.append("[dim]all tokens valid[/dim]")
            self.query_one("#validation-results", Static).update("\n".join(lines))
        except Exception as e:
            self.query_one("#validation-results", Static).update(f"[bold]yaml error:[/bold] {str(e).replace('[', '(').replace(']', ')')}")

    @on(Button.Pressed, "#prompt-create-btn")
    def on_create(self):
        name = self.query_one("#new-bot-input", Input).value.strip()
        if not name or not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            self.query_one("#validation-results", Static).update("[bold]invalid name[/bold]")
            return
        if create_bot_stub(name):
            self._refresh_bot_list()
            self.query_one("#prompt-bot-select", Select).value = name
            self._load_bot(name)
            self.query_one("#validation-results", Static).update(f"[bold]created {name}[/bold]")
        else:
            self.query_one("#validation-results", Static).update(f"[dim]{name} exists[/dim]")
