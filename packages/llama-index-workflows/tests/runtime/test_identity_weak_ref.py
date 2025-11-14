from __future__ import annotations

import gc
import weakref

import pytest

from workflows.runtime.types._identity_weak_ref import IdentityWeakKeyDict


class Unhashable:
    def __hash__(self) -> int:
        raise TypeError("Unhashable")


def test_identity_weak_key_dict_removes_entry_when_object_unreferenced() -> None:
    with pytest.raises(TypeError):
        hash(Unhashable())

    d: IdentityWeakKeyDict[Unhashable, str] = IdentityWeakKeyDict()

    obj = Unhashable()
    d[obj] = "value"

    # While the object is strongly referenced, it should be present
    assert obj in d
    assert d.get(obj) == "value"

    # Keep a weak reference to verify collection happened
    w = weakref.ref(obj)

    # Drop strong reference and force collection
    del obj
    gc.collect()

    # Object should be collected and the dict entry removed via callback
    assert w() is None
    assert d._d == {}
