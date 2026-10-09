"""The Textual TUI: jobs on top, and below them the timeline, run history, details and alert log."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

from cronwatch import display
from cronwatch.alerts import Alert, Alerter
from cronwatch.config import Config
from cronwatch.cronexpr import CronSyntaxError, parse
from cronwatch.crontab import (
    ConflictError,
    Crontab,
    CrontabError,
    Job,
    Session,
    has_unescaped_percent,
    suggest_job_id,
    unwrap_command,
    wrap_command,
    wrapper_timeout,
)
from cronwatch.store import Run, Store
from cronwatch.timeline import LABEL_MAX, build_timeline, render_timeline

REFRESH_SECONDS = 20
HISTORY_LIMIT = 50
PREVIEW_RUNS = 3


@dataclass(frozen=True)
class EditResult:
    schedule: str
    command: str
    description: str
    monitor: bool


def shown_command(job: Job) -> str:
    """The command as the user wrote it: wrapped jobs are shown without the ``cronwatch run`` boilerplate."""
    if not job.monitored:
        return job.command
    try:
        return unwrap_command(job.command)
    except CrontabError:
        return job.command


def preview_text(
    schedule: str, command: str, monitor: bool, now: datetime, tz: tzinfo
) -> tuple[str, bool]:
    """(Rich markup, valid) shown live while the schedule is typed: its meaning and its next runs."""
    problems = []
    lines = []
    try:
        expr = parse(schedule)
        lines.append(f"[green]{expr.describe()}[/]")
        runs = expr.next_runs(now, PREVIEW_RUNS, tz)
        if runs:
            lines.append("Next: " + ", ".join(display.local(r, tz) for r in runs))
    except CronSyntaxError as exc:
        problems.append(str(exc))
    if not command.strip():
        problems.append("the command is empty")
    elif monitor and has_unescaped_percent(command):
        problems.append(
            "unescaped % (cron turns it into a newline): write \\% or untick monitoring"
        )
    if problems:
        lines.append("[red]" + "; ".join(problems) + "[/]")
    return "\n".join(lines), not problems


class JobEditor(ModalScreen[EditResult | None]):
    """Add or edit a job; the schedule's meaning and next runs update as you type."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS: ClassVar[str] = """
    JobEditor { align: center middle; }
    #dialog { width: 90; height: auto; border: thick $primary; background: $surface; padding: 1 2; }
    #title { text-style: bold; margin-bottom: 1; }
    #preview { height: auto; min-height: 3; margin: 0 0 1 0; }
    #buttons { height: auto; align-horizontal: right; }
    #buttons Button { margin-left: 2; }
    """

    def __init__(
        self,
        title: str,
        now: Callable[[], datetime],
        tz: tzinfo,
        schedule: str = "",
        command: str = "",
        description: str = "",
        monitor: bool = True,
        ask_description: bool = True,
    ) -> None:
        super().__init__()
        self._title, self._now, self._tz = title, now, tz
        self._initial = (schedule, command, description, monitor)
        self._ask_description = ask_description

    def compose(self) -> ComposeResult:
        schedule, command, description, monitor = self._initial
        with Vertical(id="dialog"):
            yield Label(self._title, id="title")
            yield Label(
                "Schedule (minute hour day-of-month month day-of-week, or @daily)"
            )
            yield Input(schedule, id="schedule", placeholder="*/15 * * * *")
            yield Static("", id="preview")
            yield Label("Command")
            yield Input(
                command, id="command", placeholder="/usr/local/bin/backup.sh --all"
            )
            if self._ask_description:
                yield Label("Description (optional)")
                yield Input(description, id="description")
            yield Checkbox(
                "Monitor with cronwatch: record history, alert on failure",
                monitor,
                id="monitor",
            )
            with Horizontal(id="buttons"):
                yield Button("Save", variant="primary", id="save")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self._refresh_preview()
        self.query_one("#schedule", Input).focus()

    def _values(self) -> EditResult:
        description = (
            self.query_one("#description", Input).value if self._ask_description else ""
        )
        return EditResult(
            self.query_one("#schedule", Input).value.strip(), self.query_one("#command", Input).value.strip(),
            description.strip(), self.query_one("#monitor", Checkbox).value,
        )  # fmt: skip

    def _refresh_preview(self) -> bool:
        values = self._values()
        text, valid = preview_text(
            values.schedule, values.command, values.monitor, self._now(), self._tz
        )
        self.query_one("#preview", Static).update(text)
        return valid

    @on(Input.Changed)
    def _changed(self) -> None:
        self._refresh_preview()

    @on(Checkbox.Changed)
    def _toggled(self) -> None:
        self._refresh_preview()

    @on(Input.Submitted)
    def _submitted(self) -> None:
        self._save()

    @on(Button.Pressed, "#save")
    def _save_pressed(self) -> None:
        self._save()

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)

    def _save(self) -> None:
        if self._refresh_preview():
            self.dismiss(self._values())


class Confirm(ModalScreen[bool]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "yes", "Yes"),
        Binding("n,escape", "no", "No"),
    ]
    DEFAULT_CSS: ClassVar[str] = """
    Confirm { align: center middle; }
    #box { width: 70; height: auto; border: thick $error; background: $surface; padding: 1 2; }
    """

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Label(self._question)
            yield Label("[b]y[/b] yes   [b]n[/b] no")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class CronWatchApp(App[None]):
    TITLE = "cronwatch"
    CSS: ClassVar[str] = """
    #jobs { height: auto; min-height: 4; max-height: 45%; }
    #status { height: 1; padding: 0 1; }
    #output, #timeline-view, #details-view { padding: 0 1; }
    #runs { height: 40%; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("a", "add_job", "Add"),
        Binding("e", "edit_job", "Edit"),
        Binding("d", "delete_job", "Delete"),
        Binding("space", "toggle_job", "On/off"),
        Binding("w", "wrap_job", "Monitor"),
        Binding("u", "undo", "Undo"),
        Binding("r", "reload", "Reload"),
        Binding("t", "test_alert", "Test alert"),
        Binding("1", "tab('timeline')", "Timeline", show=False),
        Binding("2", "tab('history')", "History", show=False),
        Binding("3", "tab('details')", "Details", show=False),
        Binding("4", "tab('alerts')", "Alerts", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        session: Session,
        store: Store,
        config: Config,
        tz: tzinfo,
        alerter: Alerter,
        wrapper: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        super().__init__()
        self.session, self.store, self.config, self.tz = session, store, config, tz
        self.alerter, self.wrapper, self.clock = alerter, wrapper, clock
        self._selected: int | None = None  # line index of the highlighted job

    # -- layout --

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="jobs", cursor_type="row", zebra_stripes=True)
        with TabbedContent(id="tabs", initial="timeline"):
            with TabPane("Timeline", id="timeline"), VerticalScroll():
                yield Static(id="timeline-view")
            with TabPane("History", id="history"):
                yield DataTable(id="runs", cursor_type="row")
                with VerticalScroll():
                    yield Static(id="output")
            with TabPane("Details", id="details"), VerticalScroll():
                yield Static(id="details-view")
            with TabPane("Alerts", id="alerts"):
                yield DataTable(id="alert-log", cursor_type="row")
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        jobs = self.query_one("#jobs", DataTable)
        for label, width in [
            ("", 2),
            ("Job", 20),
            ("Schedule", 13),
            ("Meaning", 28),
            ("Next run", 32),
            ("Last run", 20),
            ("Command", 24),
        ]:
            jobs.add_column(label, width=width)
        self.query_one("#runs", DataTable).add_columns(
            "Started", "Status", "Exit", "Took"
        )
        self.query_one("#alert-log", DataTable).add_columns(
            "When", "Job", "Kind", "Channel", "Result"
        )
        self.refresh_view()
        self.set_interval(REFRESH_SECONDS, self.refresh_view)
        self.query_one("#jobs", DataTable).focus()

    # -- model helpers --

    def jobs(self) -> list[Job]:
        return self.session.crontab.jobs()

    def selected_job(self) -> Job | None:
        return next((j for j in self.jobs() if j.index == self._selected), None)

    def say(self, message: str, error: bool = False) -> None:
        self.query_one("#status", Static).update(
            Text(message, style="bold red" if error else "green")
        )

    # -- rendering --

    def refresh_view(self) -> None:
        self._render_jobs()
        self._render_details()
        self._render_history()
        self._render_timeline()
        self._render_alerts()

    def _render_jobs(self) -> None:
        table = self.query_one("#jobs", DataTable)
        now, last = self.clock(), self.store.last_runs()
        table.clear()
        for job in self.jobs():
            run = last.get(job.job_id or "")
            if job.error:
                glyph = Text("!", style="bold yellow")
            elif not job.enabled:
                glyph = Text("○", style="dim")
            elif run is not None:
                glyph = Text(*display.STATUS_GLYPHS.get(run.status, ("?", "")))
            else:
                glyph = Text("·", style="dim")
            last_text, style = (
                display.last_run_text(run, now, self.tz)
                if job.monitored
                else ("not monitored", "dim")
            )
            table.add_row(
                glyph, job.label, job.schedule, job.error or (job.expr.describe() if job.expr else ""),
                display.next_run_text(job, now, self.tz), Text(last_text, style=style), shown_command(job), key=str(job.index),
            )  # fmt: skip
        indexes = [j.index for j in self.jobs()]
        if indexes:
            if self._selected not in indexes:
                self._selected = indexes[0]
            table.move_cursor(row=indexes.index(self._selected))
        else:
            self._selected = None

    def _render_details(self) -> None:
        view = self.query_one("#details-view", Static)
        job = self.selected_job()
        if job is None:
            view.update("No job selected. Press [b]a[/b] to add one.")
            return
        now = self.clock()
        out = Text()
        out.append(f"{job.label}\n", style="bold")
        if job.description:
            out.append(f"{job.description}\n")
        out.append("\nSchedule  ", style="dim")
        out.append(
            f"{job.schedule}  ({'disabled' if not job.enabled else job.error or (job.expr.describe() if job.expr else '')})\n"
        )
        out.append("Command   ", style="dim")
        out.append(f"{shown_command(job)}\n")
        if job.expr and job.enabled and not job.expr.reboot:
            out.append("\nNext runs\n", style="dim")
            for run_at in job.expr.next_runs(now, 5, self.tz):
                out.append(
                    f"  {display.local(run_at, self.tz)}  ({display.humanize(run_at - now)})\n"
                )
        if job.job_id:
            state = self.store.get_state(job.job_id)
            out.append("\nMonitoring  ", style="dim")
            out.append(f"on, job id {job.job_id}")
            if state.consecutive_failures:
                out.append(
                    f", {state.consecutive_failures} failure(s) in a row", style="red"
                )
            out.append("\n")
            for run in self.store.runs(job.job_id, 5):
                glyph, style = display.STATUS_GLYPHS.get(run.status, ("?", ""))
                out.append(f"  {glyph} ", style=style)
                out.append(
                    f"{display.local(run.started_at, self.tz)}  exit {run.exit_code}  {display.duration(run.duration_s)}\n"
                )
        else:
            out.append("\nNot monitored: press ", style="dim")
            out.append("w", style="bold")
            out.append(
                " to run it under cronwatch and keep its history.\n", style="dim"
            )
        view.update(out)

    def _render_history(self) -> None:
        table, output = (
            self.query_one("#runs", DataTable),
            self.query_one("#output", Static),
        )
        table.clear()
        job = self.selected_job()
        if job is None or not job.job_id:
            output.update(
                "History appears here once the job runs under cronwatch (press w to wrap it)."
            )
            return
        runs = self.store.runs(job.job_id, HISTORY_LIMIT)
        for run in runs:
            glyph, style = display.STATUS_GLYPHS.get(run.status, ("?", ""))
            table.add_row(
                display.local(run.started_at, self.tz),
                Text(f"{glyph} {run.status}", style=style),
                "—" if run.exit_code is None else str(run.exit_code),
                display.duration(run.duration_s),
                key=str(run.id),
            )
        output.update(self._output_for(runs[0]) if runs else "No runs recorded yet.")

    @staticmethod
    def _output_for(run: Run) -> Text:
        return Text(run.output_tail.rstrip() or "(no output)")

    def _render_timeline(self) -> None:
        jobs, now = self.jobs(), self.clock()
        runs = {
            j.job_id: self.store.runs(j.job_id, 500, now - timedelta(hours=6))
            for j in jobs
            if j.job_id
        }
        since = {
            j.job_id: s.monitor_cursor
            for j in jobs
            if j.job_id and (s := self.store.get_state(j.job_id)).monitor_cursor
        }
        width = max(24, min(200, self.size.width - LABEL_MAX - 6))
        timeline = build_timeline(
            jobs,
            runs,
            now,
            self.tz,
            width,
            grace=self.config.grace,
            monitored_since=since,
        )
        self.query_one("#timeline-view", Static).update(
            render_timeline(timeline) if jobs else "Nothing scheduled yet."
        )

    def _render_alerts(self) -> None:
        table = self.query_one("#alert-log", DataTable)
        table.clear()
        for record in self.store.alerts(50):
            result = (
                Text("sent", style="green")
                if record.ok
                else Text(f"failed: {record.detail}", style="red")
            )
            table.add_row(
                display.local(record.sent_at, self.tz),
                record.job_id,
                record.kind,
                record.channel,
                result,
            )

    # -- events --

    @on(DataTable.RowHighlighted, "#jobs")
    def _job_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            self._selected = int(event.row_key.value)
            self._render_details()
            self._render_history()

    @on(DataTable.RowHighlighted, "#runs")
    def _run_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value is not None:
            run = self.store.get_run(int(event.row_key.value))
            if run is not None:
                self.query_one("#output", Static).update(self._output_for(run))

    def on_resize(self) -> None:
        if self.is_mounted if hasattr(self, "is_mounted") else True:
            self._render_timeline()

    # -- actions --

    def action_tab(self, name: str) -> None:
        self.query_one("#tabs", TabbedContent).active = name

    def _mutate(self, change: Callable[[Crontab], object], done: str) -> bool:
        """Apply an edit through the safe-write session; report failures instead of crashing."""
        try:
            self.session.apply(change)
        except ConflictError as exc:
            self.say(f"{exc}. Press r to reload.", error=True)
            return False
        except CrontabError as exc:
            self.say(str(exc), error=True)
            return False
        self.say(done)
        self.refresh_view()
        return True

    def _final_command(
        self,
        command: str,
        monitor: bool,
        existing_id: str | None,
        timeout: int | None,
        taken: set[str],
    ) -> str:
        if not monitor:
            return command
        return wrap_command(
            command,
            existing_id or suggest_job_id(command, taken),
            self.wrapper,
            timeout,
        )

    def action_add_job(self) -> None:
        def done(result: EditResult | None) -> None:
            if result is None:
                return

            def add(c: Crontab) -> None:
                taken = {j.job_id for j in c.jobs() if j.job_id}
                c.add_job(
                    result.schedule,
                    self._final_command(
                        result.command, result.monitor, None, None, taken
                    ),
                    result.description or None,
                )

            if self._mutate(add, "Job added"):
                self._select_last()

        self.push_screen(JobEditor("Add a job", self.clock, self.tz), done)

    def _select_last(self) -> None:
        jobs = self.jobs()
        if jobs:
            self._selected = jobs[-1].index
            self._render_jobs()

    def action_edit_job(self) -> None:
        job = self.selected_job()
        if job is None:
            self.say("Nothing to edit", error=True)
            return
        monitored, inner = job.monitored, job.command
        if monitored:
            try:
                inner = unwrap_command(job.command)
            except CrontabError:
                monitored = False  # a hand-edited wrapper: edit the raw line and leave monitoring to the user
        timeout = wrapper_timeout(job.command)

        def done(result: EditResult | None) -> None:
            if result is None:
                return

            def update(c: Crontab) -> None:
                taken = {
                    j.job_id for j in c.jobs() if j.job_id and j.index != job.index
                }
                final = self._final_command(
                    result.command,
                    result.monitor,
                    job.job_id if monitored else None,
                    timeout,
                    taken,
                )
                c.update_job(job.index, result.schedule, final)

            self._mutate(update, "Job updated")

        editor = JobEditor(
            f"Edit {job.label}",
            self.clock,
            self.tz,
            job.schedule,
            inner,
            "",
            monitored,
            ask_description=False,
        )
        self.push_screen(editor, done)

    def action_delete_job(self) -> None:
        job = self.selected_job()
        if job is None:
            return

        def done(confirmed: bool | None) -> None:
            if confirmed:
                self._mutate(lambda c: c.delete_job(job.index), f"Deleted {job.label}")

        self.push_screen(
            Confirm(f"Delete {job.label}?  ({job.schedule} {job.command[:40]})"), done
        )

    def action_toggle_job(self) -> None:
        job = self.selected_job()
        if job is None:
            return
        self._mutate(
            lambda c: c.set_enabled(job.index, not job.enabled),
            f"{job.label} {'disabled' if job.enabled else 'enabled'}",
        )

    def action_wrap_job(self) -> None:
        job = self.selected_job()
        if job is None:
            return
        if job.monitored:
            self._mutate(
                lambda c: c.unwrap_job(job.index), f"{job.label} is no longer monitored"
            )
            return
        taken = {j.job_id for j in self.jobs() if j.job_id}
        job_id = suggest_job_id(job.command, taken)
        self._mutate(
            lambda c: c.wrap_job(job.index, job_id, self.wrapper),
            f"{job.label} is now monitored as {job_id}",
        )

    def action_undo(self) -> None:
        try:
            self.session.undo()
        except CrontabError as exc:
            self.say(str(exc), error=True)
            return
        self.say("Undone")
        self.refresh_view()

    def action_reload(self) -> None:
        try:
            self.session.reload()
        except CrontabError as exc:
            self.say(str(exc), error=True)
            return
        self.say("Reloaded from the crontab")
        self.refresh_view()

    def action_test_alert(self) -> None:
        if not self.alerter.channels:
            self.say("No alert channels configured: see config.toml", error=True)
            return
        self.say("Sending test alert...")
        self.run_worker(self._send_test, thread=True, exclusive=True)

    def _send_test(self) -> None:
        alert = Alert(
            "test",
            "cronwatch-test",
            "this-host",
            self.clock(),
            "echo test",
            status="failed",
            exit_code=1,
        )
        results = self.alerter.dispatch(alert)
        failed = [f"{r.channel}: {r.detail}" for r in results if not r.ok]
        message = (
            "Test alert sent via " + ", ".join(r.channel for r in results)
            if not failed
            else "Test alert failed: " + "; ".join(failed)
        )
        self.call_from_thread(self.say, message, bool(failed))
        self.call_from_thread(self._render_alerts)
