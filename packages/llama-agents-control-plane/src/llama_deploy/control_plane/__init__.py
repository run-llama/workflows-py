# Backwards-compatibility shim: llama_deploy.control_plane -> llama_agents.control_plane
from llama_agents.core._alias import install_alias_finder

install_alias_finder()

from llama_agents.control_plane import *  # noqa: E402, F403
from llama_agents.control_plane import __all__  # noqa: E402, F401
