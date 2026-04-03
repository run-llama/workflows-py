# Alias: llama_deploy.* -> llama_agents.*
#
# This module makes the entire `llama_agents` namespace available under
# `llama_deploy`, including all sub-modules. It uses a custom meta-path
# finder to lazily redirect any import of `llama_deploy.<sub>` to
# `llama_agents.<sub>`.

from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import CodeType, ModuleType
from typing import Any, Sequence

_ALIAS_PREFIX = "llama_deploy"
_REAL_PREFIX = "llama_agents"


class _AliasLoader(Loader):
    """Loader that returns an already-imported module from sys.modules."""

    def __init__(self, real_name: str) -> None:
        self.real_name = real_name

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return importlib.import_module(self.real_name)

    def exec_module(self, module: ModuleType) -> None:
        # Module is already fully initialized by the real import.
        pass

    def get_code(self, fullname: str) -> CodeType | None:
        # runpy requires get_code() when using `python -m`. Delegate to the
        # real module's loader so that `python -m llama_deploy.x` works.
        real_suffix = fullname[len(_ALIAS_PREFIX) :]
        real_name = _REAL_PREFIX + real_suffix
        real_spec = importlib.util.find_spec(real_name)
        if real_spec and real_spec.loader:
            get_code: Any = getattr(real_spec.loader, "get_code", None)
            if get_code is not None:
                return get_code(real_name)
        return None


class _AliasFinder(MetaPathFinder):
    """Meta-path finder that redirects llama_deploy.* to llama_agents.*"""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        # Only handle llama_deploy.* (not the bare "llama_deploy" itself)
        if not fullname.startswith(_ALIAS_PREFIX + "."):
            return None
        suffix = fullname[len(_ALIAS_PREFIX) :]
        real_name = _REAL_PREFIX + suffix
        real_spec = importlib.util.find_spec(real_name)
        if real_spec is None:
            return None
        spec = ModuleSpec(
            fullname,
            _AliasLoader(real_name),
            origin=real_spec.origin,
            is_package=real_spec.submodule_search_locations is not None,
        )
        spec.submodule_search_locations = real_spec.submodule_search_locations
        return spec


def install_alias_finder() -> None:
    """Install the llama_deploy -> llama_agents alias finder (idempotent)."""
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.append(_AliasFinder())
