from textwrap import dedent

from textual.app import ComposeResult
from textual.containers import HorizontalGroup
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static


class DeploymentHelpBackMessage(Message):
    pass


class DeploymentHelpWidget(Widget):
    DEFAULT_CSS = """
    DeploymentHelpWidget {
        layout: vertical;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Deploy your app to llama cloud or your own infrastructure:",
            classes="primary-message",
        )
        yield Static(
            Content.from_markup(
                dedent("""
                [b]Deployment Name[/b]
                A unique name to identify this deployment. Controls the URL where your deployment is accessible. Will have a random suffix appended if not unique.

                [b]Git Repository[/b]
                A git repository URL to pull code from. If not publicly accessible, you will be prompted to install the llama deploy github app. If code is on another platform, either provide a Personal Access Token (basic access credentials) instead.

                [b]Git Ref[/b]
                The git ref to deploy. This can be a branch, tag, or commit hash. If this is a branch, after deploying, run a `[slategrey reverse]llamactl deploy update[/]` to update the deployment to the latest git ref after you make updates.

                [b]Config File[/b]
                Path to a directory or file containing a `[slategrey reverse]pyproject.toml[/]` or `[slategrey reverse]llama_deploy.yaml[/]` containing the llama deploy configuration. Only necessary if you have the configuration not at the root of the repo, or you have an unconventional configuration file.

                [b]Personal Access Token[/b]
                A personal access token to access the git repository. Can be used instead of the github integration.

                [b]Appserver Version[/b]
                The version of the appserver to deploy. Affects features and functionality. By default this is set to the current llamactl version, and then retained until manually upgraded.

                [b]Secrets[/b]
                Secrets to add as environment variables to the deployment. e.g. to access a database or an API. Supports adding in `[slategrey reverse].env[/]` file format.

                """).strip()
            ),
        )
        with HorizontalGroup(classes="button-row"):
            yield Button("Back", variant="primary", id="help_back", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help_back":
            self.post_message(DeploymentHelpBackMessage())
