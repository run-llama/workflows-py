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


def test_identity_weak_key_dict_keys_returns_all_keys() -> None:
    d: IdentityWeakKeyDict[Unhashable, str] = IdentityWeakKeyDict()

    obj1 = Unhashable()
    obj2 = Unhashable()
    obj3 = Unhashable()

    d[obj1] = "one"
    d[obj2] = "two"
    d[obj3] = "three"

    keys = d.keys()
    assert len(keys) == 3
    assert obj1 in keys
    assert obj2 in keys
    assert obj3 in keys


def test_identity_weak_key_dict_items_returns_all_pairs() -> None:
    d: IdentityWeakKeyDict[Unhashable, str] = IdentityWeakKeyDict()

    obj1 = Unhashable()
    obj2 = Unhashable()

    d[obj1] = "one"
    d[obj2] = "two"

    items = d.items()
    assert len(items) == 2
    # Find values by identity
    obj1_value = next(v for k, v in items if k is obj1)
    obj2_value = next(v for k, v in items if k is obj2)
    assert obj1_value == "one"
    assert obj2_value == "two"


def test_identity_weak_key_dict_keys_empty() -> None:
    d: IdentityWeakKeyDict[Unhashable, str] = IdentityWeakKeyDict()
    assert d.keys() == []


def test_identity_weak_key_dict_items_empty() -> None:
    d: IdentityWeakKeyDict[Unhashable, str] = IdentityWeakKeyDict()
    assert d.items() == []


def test_identity_weak_key_dict_keys_excludes_collected_objects() -> None:
    d: IdentityWeakKeyDict[Unhashable, str] = IdentityWeakKeyDict()

    obj1 = Unhashable()
    obj2 = Unhashable()

    d[obj1] = "one"
    d[obj2] = "two"

    # Keep weak ref to verify collection
    w = weakref.ref(obj2)

    # Drop strong reference to obj2
    del obj2
    gc.collect()

    # obj2 should be collected
    assert w() is None

    # keys() should only return obj1
    keys = d.keys()
    assert len(keys) == 1
    assert obj1 in keys


def test_identity_weak_key_dict_items_excludes_collected_objects() -> None:
    d: IdentityWeakKeyDict[Unhashable, str] = IdentityWeakKeyDict()

    obj1 = Unhashable()
    obj2 = Unhashable()

    d[obj1] = "one"
    d[obj2] = "two"

    # Keep weak ref to verify collection
    w = weakref.ref(obj2)

    # Drop strong reference to obj2
    del obj2
    gc.collect()

    # obj2 should be collected
    assert w() is None

    # items() should only return obj1's pair
    items = d.items()
    assert len(items) == 1
    assert items[0][0] is obj1
    assert items[0][1] == "one"
