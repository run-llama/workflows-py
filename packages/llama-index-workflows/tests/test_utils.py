# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

import inspect
from typing import Any, List, Optional, Union, get_type_hints

import pytest

from workflows.context import Context
from workflows.decorators import step
from workflows.errors import WorkflowValidationError
from workflows.events import StartEvent, StopEvent
from workflows.utils import (
    _get_param_types,
    _get_return_types,
    get_steps_from_class,
    get_steps_from_instance,
    inspect_signature,
    is_free_function,
    validate_step_signature,
)

from .conftest import AnotherTestEvent, OneTestEvent


def test_validate_step_signature_of_method() -> None:
    def f(self, ev: OneTestEvent) -> OneTestEvent:  # type: ignore
        return OneTestEvent()

    validate_step_signature(inspect_signature(f))


def test_validate_step_signature_of_free_function() -> None:
    def f(ev: OneTestEvent) -> OneTestEvent:
        return OneTestEvent()

    validate_step_signature(inspect_signature(f))


def test_validate_step_signature_union() -> None:
    def f(ev: Union[OneTestEvent, AnotherTestEvent]) -> OneTestEvent:
        return OneTestEvent()

    validate_step_signature(inspect_signature(f))


def test_validate_step_signature_of_free_function_with_context() -> None:
    def f(ctx: Context, ev: OneTestEvent) -> OneTestEvent:
        return OneTestEvent()

    validate_step_signature(inspect_signature(f))


def test_validate_step_signature_union_invalid() -> None:
    def f(ev: Union[OneTestEvent, str]) -> None:
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must have at least one parameter annotated as type Event",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_no_params() -> None:
    def f() -> None:
        pass

    with pytest.raises(
        WorkflowValidationError, match="Step signature must have at least one parameter"
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_no_annotations() -> None:
    def f(self, ev) -> None:  # type: ignore
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must have at least one parameter annotated as type Event",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_wrong_annotations() -> None:
    def f(self, ev: str) -> None:  # type: ignore
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must have at least one parameter annotated as type Event",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_no_return_annotations() -> None:
    def f(self, ev: OneTestEvent):  # type: ignore
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Return types of workflows step functions must be annotated with their type",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_no_events() -> None:
    def f(self, ctx: Context) -> None:  # type: ignore
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must have at least one parameter annotated as type Event",
    ):
        validate_step_signature(inspect_signature(f))


def test_validate_step_signature_too_many_params() -> None:
    def f1(self, ev: OneTestEvent, foo: OneTestEvent) -> None:  # type: ignore
        pass

    def f2(ev: OneTestEvent, foo: OneTestEvent) -> None:  # type: ignore
        pass

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must contain exactly one parameter of type Event but found 2.",
    ):
        validate_step_signature(inspect_signature(f1))

    with pytest.raises(
        WorkflowValidationError,
        match="Step signature must contain exactly one parameter of type Event but found 2.",
    ):
        validate_step_signature(inspect_signature(f2))


def test_get_steps_from() -> None:
    class Test:
        @step
        def start(self, start: StartEvent) -> OneTestEvent:
            return OneTestEvent()

        @step
        def my_method(self, event: OneTestEvent) -> StopEvent:
            return StopEvent()

        def not_a_step(self) -> None:
            pass

    steps = get_steps_from_class(Test)
    assert len(steps)
    assert "my_method" in steps

    steps = get_steps_from_instance(Test())
    assert len(steps)
    assert "my_method" in steps


def test_get_param_types() -> None:
    def f(foo: str) -> None:
        pass

    sig = inspect.signature(f)
    type_hints = get_type_hints(f)
    res = _get_param_types(sig.parameters["foo"], type_hints)
    assert len(res) == 1
    assert res[0] is str


def test_get_param_types_no_annotations() -> None:
    def f(foo) -> None:  # type: ignore
        pass

    sig = inspect.signature(f)
    type_hints = get_type_hints(f)
    res = _get_param_types(sig.parameters["foo"], type_hints)
    assert len(res) == 1
    assert res[0] is Any


def test_get_param_types_union() -> None:
    def f(foo: Union[str, int]) -> None:
        pass

    sig = inspect.signature(f)
    type_hints = get_type_hints(f)
    res = _get_param_types(sig.parameters["foo"], type_hints)
    assert len(res) == 2
    assert res == [str, int]


def test_get_return_types() -> None:
    def f(foo: int) -> str:
        return ""

    assert _get_return_types(f) == [str]


def test_get_return_types_union() -> None:
    def f(foo: int) -> Union[str, int]:
        return ""

    assert _get_return_types(f) == [str, int]


def test_get_return_types_optional() -> None:
    def f(foo: int) -> Optional[str]:
        return ""

    assert _get_return_types(f) == [str]


def test_get_return_types_list() -> None:
    def f(foo: int) -> List[str]:
        return [""]

    assert _get_return_types(f) == [List[str]]


def test_is_free_function() -> None:
    assert is_free_function("my_function") is True
    assert is_free_function("MyClass.my_method") is False
    assert is_free_function("some_function.<locals>.my_function") is True
    assert is_free_function("some_function.<locals>.MyClass.my_function") is False
    with pytest.raises(ValueError):
        is_free_function("")


def test_inspect_signature_raises_if_not_callable() -> None:
    with pytest.raises(TypeError, match="Expected a callable object, got str"):
        inspect_signature("foo")  # type: ignore
