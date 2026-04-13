# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.

from __future__ import annotations

import inspect
import types as builtin_types
import typing
from collections.abc import Callable
from typing import Protocol, Union, cast

import pytest
from tenacity import retry_all as tenacity_retry_all
from tenacity import retry_any as tenacity_retry_any
from tenacity import retry_if_exception as tenacity_retry_if_exception
from tenacity import (
    retry_if_exception_cause_type as tenacity_retry_if_exception_cause_type,
)
from tenacity import (
    retry_if_exception_message as tenacity_retry_if_exception_message,
)
from tenacity import retry_if_exception_type as tenacity_retry_if_exception_type
from tenacity import (
    retry_if_not_exception_message as tenacity_retry_if_not_exception_message,
)
from tenacity import (
    retry_if_not_exception_type as tenacity_retry_if_not_exception_type,
)
from tenacity import (
    retry_unless_exception_type as tenacity_retry_unless_exception_type,
)
from tenacity import stop_after_attempt as tenacity_stop_after_attempt
from tenacity import stop_after_delay as tenacity_stop_after_delay
from tenacity import stop_all as tenacity_stop_all
from tenacity import stop_any as tenacity_stop_any
from tenacity import stop_before_delay as tenacity_stop_before_delay
from tenacity import wait_chain as tenacity_wait_chain
from tenacity import wait_combine as tenacity_wait_combine
from tenacity import wait_exponential as tenacity_wait_exponential
from tenacity import (
    wait_exponential_jitter as tenacity_wait_exponential_jitter,
)
from tenacity import wait_fixed as tenacity_wait_fixed
from tenacity import wait_incrementing as tenacity_wait_incrementing
from tenacity import wait_none as tenacity_wait_none
from tenacity import wait_random as tenacity_wait_random
from tenacity.wait import wait_random_exponential as tenacity_wait_random_exponential
from workflows.retry_policy import (
    retry_all,
    retry_always,
    retry_any,
    retry_if_exception,
    retry_if_exception_cause_type,
    retry_if_exception_message,
    retry_if_exception_type,
    retry_if_not_exception_message,
    retry_if_not_exception_type,
    retry_never,
    retry_unless_exception_type,
    stop_after_attempt,
    stop_after_delay,
    stop_all,
    stop_any,
    stop_before_delay,
    stop_never,
    wait_chain,
    wait_combine,
    wait_exponential,
    wait_exponential_jitter,
    wait_fixed,
    wait_full_jitter,
    wait_incrementing,
    wait_none,
    wait_random,
    wait_random_exponential,
)


class NamedCallable(Protocol):
    __name__: str

    def __call__(self, *args: object, **kwargs: object) -> object: ...


CONFORMANCE_CASES: list[tuple[NamedCallable, NamedCallable | None, bool]] = cast(
    list[tuple[NamedCallable, NamedCallable | None, bool]],
    [
        (wait_fixed, tenacity_wait_fixed, True),
        (wait_none, tenacity_wait_none, True),
        (wait_exponential, tenacity_wait_exponential, True),
        (wait_incrementing, tenacity_wait_incrementing, True),
        (wait_random, tenacity_wait_random, True),
        (wait_exponential_jitter, tenacity_wait_exponential_jitter, True),
        (wait_random_exponential, tenacity_wait_random_exponential, True),
        (wait_chain, tenacity_wait_chain, True),
        (wait_combine, tenacity_wait_combine, True),
        (stop_after_attempt, tenacity_stop_after_attempt, True),
        (stop_after_delay, tenacity_stop_after_delay, True),
        (stop_any, tenacity_stop_any, True),
        (stop_all, tenacity_stop_all, True),
        (stop_before_delay, tenacity_stop_before_delay, True),
        (stop_never, None, True),
        (retry_if_exception, tenacity_retry_if_exception, True),
        (retry_if_exception_type, tenacity_retry_if_exception_type, True),
        (retry_if_not_exception_type, tenacity_retry_if_not_exception_type, True),
        (retry_if_exception_message, tenacity_retry_if_exception_message, True),
        (retry_if_not_exception_message, tenacity_retry_if_not_exception_message, True),
        (retry_if_exception_cause_type, tenacity_retry_if_exception_cause_type, True),
        (retry_any, tenacity_retry_any, True),
        (retry_all, tenacity_retry_all, True),
        (retry_always, None, True),
        (retry_never, None, True),
        (retry_unless_exception_type, tenacity_retry_unless_exception_type, True),
    ],
)


def _parameter_types(
    callable_obj: Callable[..., object],
) -> dict[str, type | object]:
    """Return {name: resolved_annotation} for all non-self parameters."""
    names = {
        name for name in inspect.signature(callable_obj).parameters if name != "self"
    }
    target = callable_obj.__init__ if isinstance(callable_obj, type) else callable_obj  # type: ignore[misc]
    try:
        hints = typing.get_type_hints(target)
    except Exception:
        hints = {}
    hints.pop("return", None)
    return {name: hints.get(name, inspect.Parameter.empty) for name in names}


def _normalize(tp: object) -> object:
    """Normalize type representations so ``X | Y`` equals ``Union[X, Y]``
    and ``type[X]`` equals ``typing.Type[X]``, etc."""
    # Callable parameter lists are represented as plain lists
    if isinstance(tp, list):
        return tuple(_normalize(a) for a in tp)
    origin = typing.get_origin(tp)
    if origin is Union or isinstance(tp, builtin_types.UnionType):
        return frozenset(_normalize(a) for a in typing.get_args(tp))
    if origin is not None:
        return (origin, tuple(_normalize(a) for a in typing.get_args(tp)))
    return tp


def _types_match(ours: object, theirs: object) -> bool:
    """Check our annotation matches tenacity's.

    Normalizes union representations so ``X | Y`` equals ``Union[X, Y]``.
    Accepts Protocol-vs-concrete-base-class mismatches for compositor *args
    (we use Protocols, tenacity uses inheritance).
    """
    if ours == theirs:
        return True
    if _normalize(ours) == _normalize(theirs):
        return True
    # Our Protocol types vs tenacity's base classes are an intentional
    # design choice (structural typing vs inheritance), not type drift.
    return (
        isinstance(ours, type)
        and isinstance(theirs, type)
        and getattr(ours, "_is_protocol", False)
    )


@pytest.mark.parametrize(
    "ours, theirs, strict",
    CONFORMANCE_CASES,
    ids=[c[0].__name__ for c in CONFORMANCE_CASES],
)
def test_retry_policy_signatures_align_with_tenacity(
    ours: NamedCallable,
    theirs: NamedCallable | None,
    strict: bool,
) -> None:
    name = ours.__name__
    ours_params = _parameter_types(ours)
    theirs_params = {} if theirs is None else _parameter_types(theirs)

    ours_names = set(ours_params)
    theirs_names = set(theirs_params)

    if strict:
        assert ours_names == theirs_names, (
            f"{name}: expected exact parameter match with tenacity; "
            f"missing={sorted(theirs_names - ours_names)}, "
            f"unexpected={sorted(ours_names - theirs_names)}"
        )
    else:
        assert ours_names <= theirs_names, (
            f"{name}: expected our parameter names to be a subset of tenacity; "
            f"unexpected={sorted(ours_names - theirs_names)}"
        )

    for param in sorted(ours_names & theirs_names):
        assert _types_match(ours_params[param], theirs_params[param]), (
            f"{name}.{param}: type mismatch — "
            f"ours={ours_params[param]!r}, theirs={theirs_params[param]!r}"
        )


def test_wait_full_jitter_alias_matches_wait_random_exponential_signature() -> None:
    assert _parameter_types(wait_full_jitter) == _parameter_types(
        wait_random_exponential
    )
