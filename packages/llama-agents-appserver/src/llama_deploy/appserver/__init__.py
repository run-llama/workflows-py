# Backwards-compatibility shim: llama_deploy.appserver -> llama_agents.appserver
from llama_agents.core._alias import install_alias_finder

install_alias_finder()

from llama_agents.appserver import *  # noqa: E402, F403
