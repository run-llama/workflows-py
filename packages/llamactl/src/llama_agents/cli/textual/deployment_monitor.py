"""Textual component to monitor a deployment and stream its logs."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import webbrowser
from collections.abc import AsyncGenerator
from pathlib import Path

from llama_agents.cli.client import (
    project_client_context,
)
from llama_agents.cli.log_format import parse_log_body
from llama_agents.core.iter_utils import merge_generators
from llama_agents.core.schema import LogEvent
from llama_agents.core.schema.deployments import DeploymentResponse
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, HorizontalGroup
from textual.content import Content
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, RichLog, Static
from typing_extensions import Literal

logger = logging.getLogger(__name__)

# structlog level → Rich style for Textual renderer.
_LEVEL_STYLES: dict[str, str] = {
    "debug": "dim",
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold red reverse",
}


def _trim_timestamp(ts: str) -> str:
    if "T" in ts:
        ts = ts.split("T", 1)[1]
        for suffix in ("Z", "+00:00"):
            ts = ts.removesuffix(suffix)
    return ts


def _format_log_line(line: str) -> Text:
    """Parse a structlog JSON line into styled Rich Text, or pass through as-is.

    Delegates body parsing to ``log_format.parse_log_body`` and adds Rich
    styling on top. Plain (non-structured) lines pass through unchanged.
    """
    parsed = parse_log_body(line)
    if not parsed.structured:
        return Text(parsed.event or parsed.raw)

    txt = Text()
    ts = _trim_timestamp(parsed.timestamp)
    if ts:
        txt.append(f"{ts} ", style="dim")
    if parsed.level:
        level_style = _LEVEL_STYLES.get(parsed.level, "")
        txt.append(f"{parsed.level.upper():8s}", style=level_style)
    if parsed.logger:
        txt.append(f"{parsed.logger} ", style="dim cyan")
    txt.append(parsed.event)
    if parsed.request_id:
        txt.append(f" req={parsed.request_id}", style="dim")
    if parsed.extras:
        parts = " ".join(f"{k}={v}" for k, v in parsed.extras.items())
        txt.append(f" {parts}", style="dim yellow")
    return txt


class DeploymentMonitorWidget(Widget):
    """Widget that fetches deployment details once and streams logs.

    Notes:
    - Status is polled periodically
    - Log stream is started with init container logs included on first connect
    - If the stream ends or hangs, we reconnect with duration-aware backoff
    """

    DEFAULT_CSS = """
	DeploymentMonitorWidget {
		layout: vertical;
		width: 1fr;
		height: 1fr;
	}

	.monitor-container {
		width: 1fr;
		height: 1fr;
		padding: 0;
		margin: 0;
	}

	.details-grid {
		layout: grid;
		grid-size: 2;
		grid-columns: auto 1fr;
		grid-gutter: 0 1;
		grid-rows: auto;
		height: auto;
		width: 1fr;
	}

	.log-header {
		margin-top: 1;
	}

    .status-line .status-main {
        width: auto;
    }

    .status-line .status-right {
        width: 1fr;
        text-align: right;
        min-width: 12;
    }

    .deployment-link-label {
        width: auto;
    }

    .deployment-link {
        width: 1fr;
        min-width: 16;
        height: auto;
        align: left middle;
        text-align: left;
        content-align: left middle;
    }

    .log-view-container {
        width: 1fr;
        height: 1fr;
        padding: 0;
        margin: 0;
    }
	"""

    deployment_id: str
    deployment = reactive[DeploymentResponse | None](None, recompose=False)
    error_message = reactive("", recompose=False)
    wrap_enabled = reactive(False, recompose=False)
    autoscroll_enabled = reactive(True, recompose=False)

    def __init__(self, deployment_id: str) -> None:
        super().__init__()
        self.deployment_id = deployment_id
        self._stop_stream = threading.Event()
        # Persist content written to the RichLog across recomposes
        self._log_buffer: list[Text] = []
        self._log_stream_started = False

    async def on_mount(self) -> None:
        # Kick off initial fetch and start logs stream in background
        self.run_worker(self._fetch_deployment())
        self.run_worker(self._stream_logs())
        # Start periodic polling of deployment status
        self.run_worker(self._poll_deployment_status())

    def compose(self) -> ComposeResult:
        yield Static("Deployment Status", classes="primary-message")

        with HorizontalGroup(classes=""):
            yield Static("  URL:    ", classes="deployment-link-label")
            yield Button(
                "",
                id="deployment_link_button",
                classes="deployment-link",
                compact=True,
                variant="default",
            )

        with HorizontalGroup(classes=""):
            yield Static("  Last Deployed Commit:  ", classes="deployment-link-label")
            yield Button(
                "",
                id="last_deployed_commit",
                classes="deployment-link",
                compact=True,
                variant="default",
            )
        yield Static("", classes="error-message", id="error_message")

        # Single-line status bar with colored icon and deployment ID
        with HorizontalGroup(classes="status-line"):
            yield Static(
                self._render_status_line(), classes="status-main", id="status_line"
            )
            yield Static("", classes="status-right", id="last_event_status")
        yield Static("", classes="last-event", id="last_event_details")
        yield Static("Logs", classes="secondary-message log-header")
        yield HorizontalGroup(classes="log-view-container", id="log_view_container")

        with HorizontalGroup(classes="button-row"):
            wrap_label = "Wrap: On" if self.wrap_enabled else "Wrap: Off"
            auto_label = "Scroll: Auto" if self.autoscroll_enabled else "Scroll: Off"
            yield Button(wrap_label, id="toggle_wrap", variant="default", compact=True)
            yield Button(
                auto_label, id="toggle_autoscroll", variant="default", compact=True
            )
            yield Button("Copy", id="copy_log", variant="default", compact=True)
            yield Button("Close", id="close", variant="default", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            # Signal parent app to close
            self.post_message(MonitorCloseMessage())
        elif event.button.id == "toggle_wrap":
            self.wrap_enabled = not self.wrap_enabled
        elif event.button.id == "toggle_autoscroll":
            self.autoscroll_enabled = not self.autoscroll_enabled
        elif event.button.id == "copy_log":
            txt = "\n".join([str(x) for x in self._log_buffer])
            self.app.copy_to_clipboard(txt)
        elif event.button.id == "deployment_link_button":
            self.action_open_url()
        elif event.button.id == "last_deployed_commit":
            if self.deployment is not None:
                self.action_open_url(
                    url=self.deployment.repo_url
                    + "/commit/"
                    + (self.deployment.git_sha or "")
                )

    async def _fetch_deployment(self) -> None:
        try:
            async with project_client_context() as client:
                self.deployment = await client.get_deployment(
                    self.deployment_id, include_events=True
                )
            # Clear any previous error on success
            self.error_message = ""
        except Exception as e:  # pragma: no cover - network errors
            self.error_message = f"Failed to fetch deployment: {e}"

    async def _stream_logs(self) -> None:
        """Consume the async log iterator, batch updates, and reconnect with backoff."""

        async def _flush_signal(
            frequency_seconds: float,
        ) -> AsyncGenerator[Literal["__FLUSH__"], None]:
            while not self._stop_stream.is_set():
                await asyncio.sleep(frequency_seconds)
                yield "__FLUSH__"

        failures = 0
        needs_clear = False
        while not self._stop_stream.is_set():
            async with project_client_context() as client:
                await asyncio.sleep(min(failures, 10))
                batch: list[LogEvent] = []
                try:
                    logger.info(f"Streaming logs for deployment {self.deployment_id}")
                    async for event in merge_generators(
                        client.stream_deployment_logs(
                            self.deployment_id,
                            include_init_containers=True,
                            tail_lines=10000,
                        ),
                        _flush_signal(0.2),
                        stop_on_first_completion=True,
                    ):
                        if event == "__FLUSH__" and batch:
                            self._handle_log_events(batch)
                            batch = []
                        elif isinstance(event, LogEvent):
                            if needs_clear:
                                self._clear_log_view()
                                needs_clear = False
                            batch.append(event)
                            failures = 0
                            if len(batch) >= 1000:
                                self._handle_log_events(batch)
                                batch = []
                except Exception as e:
                    if not self._stop_stream.is_set():
                        self._set_error_message(
                            f"Log stream failed: {e}. Reconnecting..."
                        )
                        failures += 1
                finally:
                    if batch:
                        self._handle_log_events(batch)
                    needs_clear = True

    def _clear_log_view(self) -> None:
        """Reset log buffer and widget so reconnect starts fresh."""
        self._log_buffer.clear()
        try:
            log_widget = self.query_one("#log_view", RichLog)
            log_widget.clear()
        except NoMatches:
            pass

    def _set_error_message(self, message: str) -> None:
        self.error_message = message

    def _handle_log_events(self, events: list[LogEvent]) -> None:
        def to_text(event: LogEvent) -> Text:
            txt = Text()
            txt.append(
                f"[{event.container}] ", style=self._container_style(event.container)
            )
            txt.append_text(_format_log_line(event.text))
            return txt

        texts = [to_text(event) for event in events]
        if not texts:
            return

        try:
            # due to bugs in the the RichLog widget, defer mounting, otherwise it won't get a "ResizeEvent" (on_resize), and be waiting indefinitely
            # before it renders (unless you manually resize the terminal window)
            log_widget = self.query_one("#log_view", RichLog)
        except NoMatches:
            log_container = self.query_one("#log_view_container", HorizontalGroup)
            log_widget = RichLog(
                id="log_view",
                classes="log-view mb-1",
                auto_scroll=self.autoscroll_enabled,
                wrap=self.wrap_enabled,
                highlight=True,
            )
            log_container.mount(log_widget)
        for text in texts:
            log_widget.write(text)
            self._log_buffer.append(text)
        log_widget.refresh()

        # One-time kick to ensure initial draw
        # Clear any previous error once we successfully receive logs
        if self.error_message:
            self.error_message = ""

    def _container_style(self, container_name: str) -> str:
        palette = [
            "bold magenta",
            "bold cyan",
            "bold blue",
            "bold green",
            "bold red",
            "bold bright_blue",
        ]
        # Stable hash to pick a color per container name
        h = int(hashlib.sha256(container_name.encode()).hexdigest(), 16)
        return palette[h % len(palette)]

    def _status_icon_and_style(self, phase: str) -> tuple[str, str]:
        # Map deployment phase to a colored icon
        phase = phase or "-"
        green = "bold green"
        yellow = "bold yellow"
        red = "bold red"
        gray = "grey50"
        if phase in {"Running"}:
            return "●", green
        if phase in {"Pending", "RollingOut"}:
            return "●", yellow
        if phase in {"Failed", "RolloutFailed"}:
            return "●", red
        return "●", gray

    def action_open_url(self, url: str | None = None) -> None:
        if not url:
            if not self.deployment or not self.deployment.apiserver_url:
                return
            logger.debug(f"Opening URL: {self.deployment.apiserver_url}")
            webbrowser.open(str(self.deployment.apiserver_url))
        else:
            logger.debug(f"Opening URL: {url}")
            webbrowser.open(str(url))

    def _render_status_line(self) -> Text:
        phase = self.deployment.status if self.deployment else "Unknown"
        icon, style = self._status_icon_and_style(phase)
        line = Text()
        line.append(icon, style=style)
        line.append(" ")
        line.append(f"Status: {phase} — Deployment ID: {self.deployment_id or '-'}")
        return line

    def _render_last_event_details(self) -> Content:
        if not self.deployment or not self.deployment.events:
            return Content()
        latest = self.deployment.events[-1]
        txt = Text(f"  {latest.message}", style="dim")
        return Content.from_rich_text(txt)

    def _render_last_event_status(self) -> Content:
        if self.deployment is None or not self.deployment.events:
            return Content()
        txt = Text()
        # Pick the most recent by last_timestamp
        latest = self.deployment.events[-1]
        ts = None
        timestamp = latest.last_timestamp or latest.first_timestamp
        if timestamp:
            ts = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts = "-"
        parts: list[str] = []
        if latest.type:
            parts.append(latest.type)
        if latest.reason:
            parts.append(latest.reason)
        kind = "/".join(parts) if parts else None
        if kind:
            txt.append(f"{kind} ", style="medium_purple3")
        txt.append(f"{ts}", style="dim")
        return Content.from_rich_text(txt)

    def on_unmount(self) -> None:
        # Attempt to stop the streaming loop
        self._stop_stream.set()

    # Reactive watchers to update widgets in place instead of recomposing
    def watch_error_message(self, message: str) -> None:
        try:
            widget = self.query_one("#error_message", Static)
        except Exception:
            return
        widget.update(message)
        widget.display = bool(message)

    def watch_deployment(self, deployment: DeploymentResponse | None) -> None:
        if deployment is None:
            return

        widget = self.query_one("#status_line", Static)
        ev_widget = self.query_one("#last_event_status", Static)
        ev_details_widget = self.query_one("#last_event_details", Static)
        deployment_link_button = self.query_one("#deployment_link_button", Button)
        last_commit_button = self.query_one("#last_deployed_commit", Button)
        widget.update(self._render_status_line())
        deployment_link_button.label = (
            f"{str(self.deployment.apiserver_url or '') if self.deployment else ''}"
        )
        if self.deployment:
            last_commit_button.label = f"{(str(self.deployment.git_sha or '-'))[:7]}"
        else:
            last_commit_button.label = "-"
        # Update last event line
        ev_widget.update(self._render_last_event_status())
        ev_details_widget.update(self._render_last_event_details())
        ev_details_widget.display = bool(self.deployment and self.deployment.events)

    def watch_wrap_enabled(self, enabled: bool) -> None:
        try:
            log_widget = self.query_one("#log_view", RichLog)
            log_widget.wrap = enabled
            # Clear existing lines; new wrap mode will apply to subsequent events
            log_widget.clear()
            for text in self._log_buffer:
                log_widget.write(text)
        except Exception:
            pass
        try:
            btn = self.query_one("#toggle_wrap", Button)
            btn.label = "Wrap: On" if enabled else "Wrap: Off"
        except Exception:
            pass

    def watch_autoscroll_enabled(self, enabled: bool) -> None:
        try:
            log_widget = self.query_one("#log_view", RichLog)
            log_widget.auto_scroll = enabled
        except Exception:
            pass
        try:
            btn = self.query_one("#toggle_autoscroll", Button)
            btn.label = "Scroll: Auto" if enabled else "Scroll: Off"
        except Exception:
            pass

    async def _poll_deployment_status(self) -> None:
        """Periodically refresh deployment status to reflect updates in the UI."""
        while not self._stop_stream.is_set():
            try:
                async with project_client_context() as client:
                    self.deployment = await client.get_deployment(
                        self.deployment_id, include_events=True
                    )
                # Clear any previous error on success
                if self.error_message:
                    self.error_message = ""
            except Exception as e:  # pragma: no cover - network errors
                # Non-fatal; will try again on next interval
                self.error_message = f"Failed to refresh status: {e}"
            await asyncio.sleep(5)


class MonitorCloseMessage(Message):
    pass


class LogBatchMessage(Message):
    def __init__(self, events: list[LogEvent]) -> None:
        super().__init__()
        self.events = events


class ErrorTextMessage(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class DeploymentMonitorApp(App[None]):
    """Standalone app wrapper around the monitor widget.

    This allows easy reuse in other flows by embedding the widget.
    """

    CSS_PATH = Path(__file__).parent / "styles.tcss"

    def __init__(self, deployment_id: str) -> None:
        super().__init__()
        self.deployment_id = deployment_id

    def on_mount(self) -> None:
        self.theme = "tokyo-night"

    def compose(self) -> ComposeResult:
        with Container():
            yield DeploymentMonitorWidget(self.deployment_id)

    def on_monitor_close_message(self, _: MonitorCloseMessage) -> None:
        self.exit(None)

    def on_key(self, event: events.Key) -> None:
        # Support Ctrl+C to exit, consistent with other screens and terminals
        if event.key == "ctrl+c":
            self.exit(None)


def monitor_deployment_screen(deployment_id: str) -> None:
    """Launch the standalone deployment monitor screen."""
    app = DeploymentMonitorApp(deployment_id)
    app.run()
