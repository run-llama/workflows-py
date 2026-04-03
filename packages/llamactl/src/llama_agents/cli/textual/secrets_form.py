"""Textual-based deployment forms for CLI interactions"""

from llama_agents.cli.env import load_env_secrets_from_string
from textual.app import ComposeResult
from textual.containers import HorizontalGroup
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Static, TextArea


class SecretsWidget(Widget):
    """Widget for managing deployment secrets with reactive updates"""

    DEFAULT_CSS = """
    SecretsWidget {
        padding-top: 1;
        border-top: wide $secondary-muted;
        layout: vertical;
        height: auto;
        overflow-y: auto;
    }


    .secrets-grid {
        layout: grid;
        grid-size: 4;
        grid-columns: auto 1fr auto auto;
        grid-rows: auto;
        margin-bottom: 1;
        grid-gutter: 0 1;
    }

    .secrets-header {
        text-style: bold;
        color: $text-accent;
        margin-bottom: 1;
    }

    .add-secrets-button-row {
        layout: horizontal;
        align: right middle;
        margin: 1 0;
    }
    #new_secrets {
        max-height: 7;
    }
    SecretsWidget .secret-label {
        color: $text-accent;
    }

    SecretsWidget .text-container {
        padding: 0;
        height: 7;
    }
    #new_secrets {
        margin: 0;
        padding: 0;
        height: 7;
    }


    """
    secrets = reactive({}, recompose=True)  # Auto-recompose when secrets change
    prior_secrets = reactive(set(), recompose=True)
    visible_secrets = reactive(set(), recompose=True)

    def __init__(
        self,
        initial_secrets: dict[str, str] | None = None,
        prior_secrets: set[str] | None = None,
        info_message: str | None = None,
    ):
        super().__init__()
        self.secrets = initial_secrets or {}
        self.prior_secrets = prior_secrets or set()
        self.visible_secrets = set()
        # Persist textarea content across recomposes triggered by other actions
        self._new_secrets_text = ""
        self.info_message = info_message

    def compose(self) -> ComposeResult:
        """Compose the secrets section - called automatically when secrets change"""
        yield Static("Secrets", classes="secrets-header")
        # Preserve deterministic order: known secrets in insertion order, then prior-only sorted
        known_secret_names = list(self.secrets.keys())
        prior_only_secret_names = sorted(
            [name for name in self.prior_secrets if name not in self.secrets]
        )
        secret_names = known_secret_names + prior_only_secret_names
        hidden = len(secret_names) == 0
        if self.info_message:
            yield Static(self.info_message, classes="secondary-message mb-1")
        with Static(
            classes="secrets-grid" + (" hidden" if hidden else ""),
            id="secrets-grid",
        ):
            for secret_name in secret_names:
                yield Label(f"{secret_name}:", classes="secret-label", shrink=True)
                is_unknown = secret_name in self.prior_secrets
                visible = secret_name in self.visible_secrets
                yield Input(
                    value=self.secrets.get(secret_name, "***"),
                    placeholder="Leave blank to delete",
                    password=not visible,
                    id=f"secret_{secret_name}",
                    compact=True,
                )
                yield Button(
                    "hide" if visible and not is_unknown else "show",
                    id=f"show_{secret_name}",
                    compact=True,
                    disabled=is_unknown,
                )
                yield Button(
                    "delete",
                    id=f"delete_{secret_name}",
                    compact=True,
                )

        # Short help text for textarea format
        yield Static(
            "Format: one per line, KEY=VALUE",
            classes="secret-label",
        )

        with HorizontalGroup(classes="text-container"):
            yield TextArea(
                self._new_secrets_text,
                id="new_secrets",
                show_line_numbers=True,
                highlight_cursor_line=True,
            )
        with HorizontalGroup(classes="add-secrets-button-row"):
            yield Button(
                "Update Secrets",
                classes="add-secret",
                id="add_secrets",
                compact=True,
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses within the secrets widget"""
        if event.button.id == "add_secrets":
            self._add_secrets_from_textarea()
        elif event.button.id and event.button.id.startswith("delete_"):
            secret_name = event.button.id.removeprefix("delete_")
            self._delete_secret(secret_name)
        elif event.button.id and event.button.id.startswith("show_"):
            secret_name = event.button.id.removeprefix("show_")
            self._toggle_secret_visibility(secret_name)

    def _add_secrets_from_textarea(self) -> None:
        """Parse and add secrets from the textarea"""
        textarea = self.query_one("#new_secrets", TextArea)
        content = textarea.text.strip()

        if not content:
            return

        # Parse .env format
        new_secrets = load_env_secrets_from_string(content)

        # Update secrets - this will trigger automatic recompose
        updated_secrets = self.secrets.copy()
        updated_secrets.update(new_secrets)
        updated_prior_secrets = self.prior_secrets.copy()
        updated_prior_secrets = updated_prior_secrets.difference(new_secrets.keys())
        self.secrets = updated_secrets

        # Clear textarea
        textarea.text = ""
        self._new_secrets_text = ""

    def _toggle_secret_visibility(self, secret_name: str) -> None:
        """Toggle the visibility of a secret"""
        visible_secrets = self.visible_secrets.copy()
        if secret_name in visible_secrets:
            visible_secrets.remove(secret_name)
        else:
            visible_secrets.add(secret_name)
        self.visible_secrets = visible_secrets

    def _delete_secret(self, secret_name: str) -> None:
        """Delete a secret from the form"""
        if secret_name in self.secrets:
            updated_secrets = self.secrets.copy()
            del updated_secrets[secret_name]
            self.secrets = updated_secrets  # Triggers automatic recompose
        if secret_name in self.prior_secrets:
            updated_prior_secrets = self.prior_secrets.copy()
            updated_prior_secrets.remove(secret_name)
            self.prior_secrets = updated_prior_secrets

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Track textarea edits so content persists across recomposes."""
        text = event.text_area.text
        self._new_secrets_text = text

    def get_updated_secrets(self) -> dict[str, str]:
        """Get current secrets with values from input fields"""
        self._add_secrets_from_textarea()
        return self.secrets

    def get_updated_prior_secrets(self) -> set[str]:
        """Get current prior secrets"""
        self._add_secrets_from_textarea()
        return self.prior_secrets
