import importlib.util
import warnings

from llama_agents.cli.commands.agentcore import agentcore
from llama_agents.cli.commands.auth import auth
from llama_agents.cli.commands.deployment import deployments
from llama_agents.cli.commands.dev import dev
from llama_agents.cli.commands.env import env_group
from llama_agents.cli.commands.init import init
from llama_agents.cli.commands.pkg import pkg
from llama_agents.cli.commands.serve import serve

from .app import app

# Disable warnings in llamactl CLI, and specifically silence the Pydantic
# UnsupportedFieldAttributeWarning about `validate_default` on Field().
warnings.simplefilter("ignore")
warnings.filterwarnings(
    "ignore",
    message=r"The 'validate_default' attribute .* has no effect.*",
)


# Main entry point function (called by the script)
def main() -> None:
    app()


__all__ = [
    "app",
    "deployments",
    "auth",
    "serve",
    "init",
    "env_group",
    "pkg",
    "dev",
    "agentcore",
]

# Conditionally import agentcore command if boto3 is available
# (requires installing with: pip install llamactl[agentcore])
if importlib.util.find_spec("boto3") is not None:
    from llama_agents.cli.commands.agentcore import agentcore

    __all__ += ["agentcore"]


if __name__ == "__main__":
    app()
