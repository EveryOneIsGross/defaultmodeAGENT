"""Memory page for the Agent Manager TUI."""

import math
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Label, Input, Button, Select, Static, Checkbox, TextArea

from tui.shared import (
    STATE, get_bot_caches, load_memory_cache, save_memory_cache,
    search_memories, delete_memory, update_memory, find_replace_memories,
    delete_user_cascade, tokenize,
)


class MemoryPage(Vertical):
    BATCH_SIZE = 30

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cache = {}
        self.loaded_bot: Optional[str] = None
        self.search_results: list = []
        self.current_page: int = 1

    def _is_running(self) -> bool:
        return bool(self.loaded_bot and STATE.is_bot_running(self.loaded_bot))

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Select([], id="memory-bot-select", prompt="select bot"),
            Button("Load", id="memory-load-btn"),
            Button("Save", variant="success", id="memory-save-btn"),
            Label("", id="memory-warning"),
            id="memory-controls",
        )
        yield Horizontal(
            Input(placeholder="search query", id="memory-query"),
            Select([], id="memory-user-select", prompt="all users"),
            Button("Search", id="memory-search-btn"),
            Button("Delete User", variant="error", id="memory-delete-user-btn"),
            id="memory-search",
        )
        yield Horizontal(
            Input(placeholder="find", id="fr-find"),
            Input(placeholder="replace", id="fr-replace"),
            Checkbox("case", id="fr-case"),
            Checkbox("whole", id="fr-whole"),
            Button("Find/Replace", id="fr-btn"),
            Label("", id="fr-status"),
            id="find-replace-section",
        )
        yield Static("", id="memory-stats")
        yield ScrollableContainer(id="memory-results")
        yield Horizontal(
            Button("<", id="mem-prev-btn", disabled=True),
            Static("page 0/0", id="mem-page-info"),
            Button(">", id="mem-next-btn", disabled=True),
            id="mem-pagination",
        )

    def on_mount(self):
        self._refresh_bot_list()
        self.query_one("#memory-results").can_focus = True

    def _refresh_bot_list(self):
        bots = [(b, b) for b in get_bot_caches()]
        sel = self.query_one("#memory-bot-select", Select)
        sel.set_options(bots)
        if bots and (not sel.value or sel.value == Select.BLANK):
            sel.value = bots[0][1]

    def _refresh_user_list(self):
        users = list(self.cache.get("user_memories", {}).keys())
        self.query_one("#memory-user-select", Select).set_options([("", "all users")] + [(u, u) for u in users])

    def _update_edit_state(self):
        r = self._is_running()
        self.query_one("#memory-save-btn", Button).disabled = r
        self.query_one("#fr-btn", Button).disabled = r
        self.query_one("#memory-delete-user-btn", Button).disabled = r
        self.query_one("#memory-warning", Label).update("[bold]⚠ bot running[/bold]" if r else "")

    @on(Button.Pressed, "#memory-load-btn")
    def on_load(self):
        sel = self.query_one("#memory-bot-select", Select)
        if not sel.value or sel.value == Select.BLANK:
            return
        self.loaded_bot = sel.value
        self.cache = load_memory_cache(sel.value)
        if not self.cache:
            self.query_one("#memory-stats", Static).update("[bold]failed to load[/bold]")
            return
        active = len([m for m in self.cache.get("memories", []) if m is not None])
        users = list(self.cache.get("user_memories", {}).keys())
        self.query_one("#memory-stats", Static).update(f"memories={active} users={len(users)} terms={len(self.cache.get('inverted_index', {}))}")
        self._refresh_user_list()
        self._update_edit_state()
        self._do_search()

    @on(Button.Pressed, "#memory-save-btn")
    def on_save(self):
        if self._is_running():
            self.query_one("#memory-stats", Static).update("[bold]cannot save while running[/bold]")
            return
        self.query_one("#memory-stats", Static).update("[bold]saved[/bold]" if save_memory_cache(self.cache) else "[bold]save failed[/bold]")

    @on(Button.Pressed, "#memory-search-btn")
    def on_search(self):
        self._do_search()

    @on(Input.Submitted, "#memory-query")
    def on_search_submit(self, event: Input.Submitted):
        self._do_search()

    @on(Button.Pressed, "#memory-delete-user-btn")
    def on_delete_user(self):
        if self._is_running():
            return
        user_sel = self.query_one("#memory-user-select", Select)
        if not user_sel.value or user_sel.value == Select.BLANK:
            self.query_one("#memory-stats", Static).update("[bold]select a user first[/bold]")
            return
        result = delete_user_cascade(self.cache, user_sel.value)
        self.query_one("#memory-stats", Static).update(f"[bold]deleted {result.get('removed', 0)} memories for {result.get('user')}[/bold]")
        self._refresh_user_list()
        self._do_search()

    @on(Button.Pressed, "#fr-btn")
    def on_find_replace(self):
        if self._is_running():
            return
        find = self.query_one("#fr-find", Input).value
        replace = self.query_one("#fr-replace", Input).value
        case = self.query_one("#fr-case", Checkbox).value
        whole = self.query_one("#fr-whole", Checkbox).value
        user_sel = self.query_one("#memory-user-select", Select)
        user_id = user_sel.value if user_sel.value and user_sel.value != Select.BLANK else None
        result = find_replace_memories(self.cache, find, replace, case, whole, user_id)
        self.query_one("#fr-status", Label).update(f"[bold]{result['changes']}/{result['processed']} changed[/bold]")
        self._do_search()

    def _do_search(self):
        if not self.cache:
            return
        self._update_edit_state()
        q = self.query_one("#memory-query", Input).value.strip()
        user_sel = self.query_one("#memory-user-select", Select)
        uid = user_sel.value if user_sel.value and user_sel.value != Select.BLANK else None
        total_mems = len([m for m in self.cache.get("memories", []) if m is not None])
        r = search_memories(self.cache, q, uid, page=1, per_page=total_mems or 999999)
        self.search_results = r.get("memories", [])
        self.current_page = 1
        self._render_page()

    @property
    def _total_pages(self) -> int:
        return max(1, math.ceil(len(self.search_results) / self.BATCH_SIZE))

    def _render_page(self):
        container = self.query_one("#memory-results", ScrollableContainer)
        container.remove_children()
        total = len(self.search_results)
        tp = self._total_pages
        self.current_page = max(1, min(self.current_page, tp))
        start = (self.current_page - 1) * self.BATCH_SIZE
        batch = self.search_results[start:start + self.BATCH_SIZE]
        dis = self._is_running()
        for mem in batch:
            mid = mem["id"]
            ta = TextArea(mem["text"], id=f"edit-{mid}", soft_wrap=True, read_only=dis)
            ta.styles.height = "auto"
            ta.styles.max_height = 12
            container.mount(
                Vertical(
                    Label(f"[bold]#{mid}[/bold] user={mem.get('user_id', '?')} score={mem.get('score', 0):.2f}"),
                    ta,
                    Horizontal(
                        Button("update", id=f"upd-{mid}", variant="primary", disabled=dis),
                        Button("delete", id=f"del-{mid}", variant="error", disabled=dis),
                    ),
                    classes="memory-item",
                )
            )
        q = self.query_one("#memory-query", Input).value.strip()
        q_tokens = tokenize(q) if q else []
        if q and not q_tokens:
            mode = "substring"
        elif q_tokens:
            mode = f"index [{' '.join(q_tokens)}]"
        else:
            mode = "all"
        self.query_one("#memory-stats", Static).update(f"{total} results ({mode})")
        self.query_one("#mem-page-info", Static).update(f"page {self.current_page}/{tp}")
        self.query_one("#mem-prev-btn", Button).disabled = self.current_page <= 1
        self.query_one("#mem-next-btn", Button).disabled = self.current_page >= tp
        container.scroll_home(animate=False)

    @on(Button.Pressed, "#mem-prev-btn")
    def on_prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._render_page()

    @on(Button.Pressed, "#mem-next-btn")
    def on_next_page(self):
        if self.current_page < self._total_pages:
            self.current_page += 1
            self._render_page()

    @on(Button.Pressed)
    def on_memory_action(self, event: Button.Pressed):
        bid = event.button.id or ""
        if bid.startswith("upd-"):
            if self._is_running():
                return
            mid = int(bid.removeprefix("upd-"))
            try:
                ta = self.query_one(f"#edit-{mid}", TextArea)
            except Exception:
                return
            new_text = ta.text
            if update_memory(self.cache, mid, new_text):
                for m in self.search_results:
                    if m["id"] == mid:
                        m["text"] = new_text
                        break
                self.query_one("#memory-stats", Static).update(f"[bold]#{mid} updated (save to persist)[/bold]")
        elif bid.startswith("del-"):
            if self._is_running():
                return
            mid = int(bid.removeprefix("del-"))
            if delete_memory(self.cache, mid):
                self.search_results = [m for m in self.search_results if m["id"] != mid]
                self._render_page()
