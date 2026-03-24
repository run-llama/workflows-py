# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LlamaIndex Inc.
"""Runner wrapper that monkey-patches wait_for_next_task to pick tasks
in reversed priority order (pull before workers, last worker first).

This simulates what happens in production when ASLR or different memory
layouts cause set.pop() on asyncio.Task objects to return a different
task than the original execution. The DBOS runtime uses done.pop()
(non-deterministic) instead of pick_highest_priority (deterministic),
so this patch forces the worst-case ordering to trigger divergence.

Usage: same CLI args as runner.py.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

# Bootstrap sys.path (same as runner.py)
sys.path.insert(0, str(Path(__file__).parent))

# Must import runner_common first to set up sys.path for package imports
import runner_common  # noqa: F401, E402  # ty: ignore[unresolved-import]
from llama_agents.dbos.runtime import InternalDBOSAdapter  # noqa: E402
from workflows.runtime.types.named_task import (  # noqa: E402
    NamedTask,
    PendingStart,
    PullTask,
    get_key,
)
from workflows.runtime.types.plugin import WaitForNextTaskResult  # noqa: E402

_real_wait_for_next_task = InternalDBOSAdapter.wait_for_next_task
_shuffle_count = 0


def _pick_lowest_priority(
    named_tasks: list[NamedTask], done: set[asyncio.Task[Any]]
) -> asyncio.Task[Any] | None:
    """Pick the LOWEST priority task (opposite of pick_highest_priority).

    Iterates in reverse: pulls before workers, last worker first.
    This maximally diverges from the original execution order, which
    the non-DBOS runtime (plugin.py) gets right with pick_highest_priority
    but the DBOS runtime gets wrong with done.pop().
    """
    if not done:
        return None
    # Reverse the named_tasks list so pulls come first, last workers come before first
    for nt in reversed(named_tasks):
        if nt.task in done:
            return nt.task
    return done.pop()  # Fallback (shouldn't happen)


async def _patched_wait_for_next_task(
    self: InternalDBOSAdapter,
    running: list[NamedTask],
    pending: list[PendingStart],
    timeout: float | None = None,
) -> WaitForNextTaskResult:
    """Patched wait_for_next_task that uses reversed priority selection.

    This replaces the non-deterministic done.pop() with a deterministically
    WRONG selection (lowest priority instead of highest) to force the fid
    counter divergence that causes DBOSUnexpectedStepError in production.
    """
    global _shuffle_count  # noqa: PLW0603

    # Start pending (same as original)
    started: list[NamedTask] = []
    for p in pending:
        started.append(p.start(asyncio.create_task(p.coro)))
        await asyncio.sleep(0)

    all_named = running + started
    tasks = {nt.task for nt in all_named}
    if not tasks:
        return WaitForNextTaskResult(None, started)

    # Ensure pool is resolved before journal creation (needed for postgres)
    if self._ensure_pool is not None and self._resolved_pool is None:
        await self._resolve_pool()

    journal = self._get_or_create_journal()
    await journal.load()

    expected_key = journal.next_expected_key()
    if expected_key is not None:
        # Replay mode: wait for specific task (same as original)
        from workflows.runtime.types.named_task import find_by_key

        target_task = find_by_key(all_named, expected_key)
        if target_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(target_task), timeout=timeout)
            except (asyncio.TimeoutError, TimeoutError):
                return WaitForNextTaskResult(None, started)
            journal.advance()
            return WaitForNextTaskResult(target_task, started)

    # Fresh execution: wait for first completed
    done, _ = await asyncio.wait(
        tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
    )
    if not done:
        return WaitForNextTaskResult(None, started)

    if len(done) > 1:
        _shuffle_count += 1
        # Identify task types in done set
        pull_in_done = any(
            isinstance(nt, PullTask) and nt.task in done for nt in all_named
        )
        worker_count = sum(
            1 for nt in all_named if not isinstance(nt, PullTask) and nt.task in done
        )
        print(
            f"SHUFFLE:{_shuffle_count}:done={len(done)}"
            f":pull={'yes' if pull_in_done else 'no'}"
            f":workers={worker_count}",
            flush=True,
        )
        # Pick lowest priority (opposite of what the correct behavior would be)
        completed = _pick_lowest_priority(all_named, done)
        assert completed is not None
    else:
        completed = done.pop()

    key = get_key(all_named, completed)
    await journal.record(key)
    return WaitForNextTaskResult(completed, started)


# Apply the monkey-patch
InternalDBOSAdapter.wait_for_next_task = _patched_wait_for_next_task  # type: ignore[assignment]
print("PATCH:wait_for_next_task monkey-patched (reversed priority)", flush=True)

# Now delegate to the real runner
from runner import main  # noqa: E402  # ty: ignore[unresolved-import]

if __name__ == "__main__":
    main()
