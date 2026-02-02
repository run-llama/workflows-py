# SPDX-License-Identifier: MIT
# Alias: llama_agents.workflows -> workflows
#
# This module makes the entire `workflows` package available under
# `llama_agents.workflows`, including all sub-modules. It uses a
# custom meta-path finder to lazily redirect any import of
# `llama_agents.workflows.<sub>` to `workflows.<sub>`.

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Sequence

_ALIAS_PREFIX = "llama_agents.workflows"
_REAL_PREFIX = "workflows"


class _AliasLoader(Loader):
    """Loader that returns an already-imported module from sys.modules."""

    def __init__(self, real_name: str) -> None:
        self.real_name = real_name

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return importlib.import_module(self.real_name)

    def exec_module(self, module: ModuleType) -> None:
        # Module is already fully initialized by the real import.
        pass


class _AliasFinder(MetaPathFinder):
    """Meta-path finder that redirects llama_agents.workflows.* to workflows.*"""

    def find_module(
        self, fullname: str, path: Sequence[str] | None = None
    ) -> Loader | None:
        # Python 3.9 compat - some tools call find_module instead of find_spec
        spec = self.find_spec(fullname, path)
        if spec and spec.loader:
            return spec.loader  # type: ignore[return-value]
        return None

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        # Only handle llama_agents.workflows.* (not the root itself)
        if not fullname.startswith(_ALIAS_PREFIX + "."):
            return None
        suffix = fullname[len(_ALIAS_PREFIX) :]
        real_name = _REAL_PREFIX + suffix
        return ModuleSpec(fullname, _AliasLoader(real_name))


# Install the finder once
if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.append(_AliasFinder())

# Re-export everything from the real workflows package
from workflows import *  # noqa: E402, F403
from workflows import __all__  # noqa: E402, F401
