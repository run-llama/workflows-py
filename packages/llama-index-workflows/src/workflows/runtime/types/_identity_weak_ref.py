from __future__ import annotations

import weakref
from typing import Callable, Generic, TypeVar, overload

K = TypeVar("K")
V = TypeVar("V")


class _IdentityWeakRef(weakref.ref, Generic[K]):
    __slots__ = ("_hash",)

    _hash: int

    def __new__(
        cls, obj: K, callback: Callable[[_IdentityWeakRef[K]], None] | None = None
    ) -> _IdentityWeakRef[K]:
        self = super().__new__(cls, obj, callback)
        self._hash = id(
            obj
        )  # cache identity-based hash; works even if obj is unhashable
        return self

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _IdentityWeakRef):
            return NotImplemented
        return self() is other()


class IdentityWeakKeyDict(Generic[K, V]):
    _d: dict[_IdentityWeakRef[K], V]

    def __init__(self) -> None:
        self._d = {}

    def _mk(self, obj: K) -> _IdentityWeakRef[K]:
        def _cb(wr: _IdentityWeakRef[K]) -> None:
            self._d.pop(wr)

        return _IdentityWeakRef(obj, _cb)

    def __setitem__(self, obj: K, value: V) -> None:
        self._d[self._mk(obj)] = value

    def __getitem__(self, obj: K) -> V:
        return self._d[_IdentityWeakRef(obj)]

    @overload
    def get(self, obj: K) -> V | None: ...

    @overload
    def get(self, obj: K, default: V) -> V: ...

    def get(self, obj: K, default: V | None = None) -> V | None:
        return self._d.get(_IdentityWeakRef(obj), default)

    def __contains__(self, obj: K) -> bool:
        return _IdentityWeakRef(obj) in self._d

    def __delitem__(self, obj: K) -> None:
        del self._d[_IdentityWeakRef(obj)]
