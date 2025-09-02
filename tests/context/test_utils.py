import pytest

from workflows.context.utils import (
    get_qualified_name,
    import_module_from_qualified_name,
    compare_states,
    StateModificationType,
)


def test_get_qualified_name() -> None:
    with pytest.raises(
        AttributeError,
        match="Object foo does not have required attributes: 'str' object has no attribute '__module__'",
    ):
        get_qualified_name("foo")


def test_import_module_from_qualified_name_wrong_name() -> None:
    with pytest.raises(
        ValueError, match="Qualified name must be in format 'module.attribute'"
    ):
        import_module_from_qualified_name("not containing a dot")
        import_module_from_qualified_name("")


def test_import_module_from_qualified_name_wrong_package() -> None:
    with pytest.raises(
        ImportError,
        match="Failed to import module __doesnt: No module named '__doesnt'",
    ):
        import_module_from_qualified_name("__doesnt.exist")


def test_import_module_from_qualified_name_wrong_module() -> None:
    with pytest.raises(
        AttributeError,
        match="Attribute doesntexist not found in module typing: module 'typing' has no attribute 'doesntexist'",
    ):
        import_module_from_qualified_name("typing.doesntexist")


@pytest.fixture
def state_test_cases() -> list[tuple[dict, dict, StateModificationType]]:
    return [
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}, StateModificationType.UPDATED_PROPERTY),
        (
            {"a": 1, "b": 2, "c": 3},
            {"a": 1, "b": 3},
            StateModificationType.DELETED_PROPERTY,
        ),
        (
            {"a": 1, "b": 2},
            {"a": 1, "b": 2, "c": 3},
            StateModificationType.ADDED_PROPERTY,
        ),
        ({"a": 1, "c": 2}, {"a": 1, "b": 2}, StateModificationType.UPDATED_STATE),
    ]


def test_compare_states(
    state_test_cases: list[tuple[dict, dict, StateModificationType]],
) -> None:
    for tc in state_test_cases:
        assert compare_states(tc[0], tc[1]) == tc[2]
