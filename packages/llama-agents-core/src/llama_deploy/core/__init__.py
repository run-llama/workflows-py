# Backwards-compatibility shim: llama_deploy.core -> llama_agents.core
from llama_agents.core._alias import install_alias_finder

install_alias_finder()

from llama_agents.core import *  # noqa: E402, F403
from llama_agents.core import __all__  # noqa: E402, F401
