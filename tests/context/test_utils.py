import pytest

from workflows.context.utils import get_qualified_name


def test_get_qualified_name() -> None:
    with pytest.raises(
        AttributeError,
        match="Object foo does not have required attributes: 'str' object has no attribute '__module__'",
    ):
        get_qualified_name("foo")
