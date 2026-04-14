# Retry Policy

The retry API is built from three families of callables modeled after
[tenacity](https://tenacity.readthedocs.io/en/latest/):

- **[Retry conditions](#retry-conditions)** decide whether an exception should be retried.
- **[Wait strategies](#wait-strategies)** decide how long to sleep before the next attempt.
- **[Stop conditions](#stop-conditions)** decide when retries should stop.

You can compose them into a policy with `retry_policy(retry=..., wait=..., stop=...)`.
Retry conditions support `|` and `&`, wait strategies support `+`, and stop
conditions support `|` and `&`.

## Quick Example

```python
from workflows.retry_policy import (
    retry_policy,
    retry_if_exception_message,
    retry_if_exception_type,
    stop_after_attempt,
    stop_before_delay,
    wait_fixed,
    wait_random,
)

policy = retry_policy(
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
    | retry_if_exception_message(match="rate limit|temporarily unavailable"),
    wait=wait_fixed(1) + wait_random(0, 1),
    stop=stop_after_attempt(5) | stop_before_delay(30),
)
```

## Policy Constructor

::: workflows.retry_policy
    options:
      members:
        - retry_policy
        - RetryPolicy

## Retry Conditions

Modeled after [tenacity retry functions](https://tenacity.readthedocs.io/en/latest/api.html#retry-functions).

::: workflows.retry_policy
    options:
      members:
        - retry_if_exception
        - retry_if_exception_type
        - retry_if_not_exception_type
        - retry_unless_exception_type
        - retry_if_exception_message
        - retry_if_not_exception_message
        - retry_if_exception_cause_type
        - retry_any
        - retry_all
        - retry_always
        - retry_never

## Wait Strategies

Modeled after [tenacity wait functions](https://tenacity.readthedocs.io/en/latest/api.html#wait-functions).

::: workflows.retry_policy
    options:
      members:
        - wait_fixed
        - wait_none
        - wait_exponential
        - wait_incrementing
        - wait_random
        - wait_exponential_jitter
        - wait_random_exponential
        - wait_full_jitter
        - wait_chain
        - wait_combine

## Stop Conditions

Modeled after [tenacity stop functions](https://tenacity.readthedocs.io/en/latest/api.html#stop-functions).

::: workflows.retry_policy
    options:
      members:
        - stop_after_attempt
        - stop_after_delay
        - stop_before_delay
        - stop_any
        - stop_all
        - stop_never

## Deprecated Constructors

The following helpers predate the composable API and are kept for
backwards compatibility. Prefer `retry_policy(...)` with explicit retry,
wait, and stop arguments.

::: workflows.retry_policy
    options:
      members:
        - ConstantDelayRetryPolicy
        - ExponentialBackoffRetryPolicy
