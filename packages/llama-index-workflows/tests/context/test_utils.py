import pytest

from workflows.context.utils import (
    get_qualified_name,
    import_module_from_qualified_name,
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
