# Backwards-compatibility shim: llama_deploy.cli -> llama_agents.cli
from llama_agents.core._alias import install_alias_finder

install_alias_finder()

from llama_agents.cli import *  # noqa: E402, F403
from llama_agents.cli import __all__  # noqa: E402, F401
