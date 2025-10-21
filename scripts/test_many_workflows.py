from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import weakref

import tracemalloc

try:
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None  # type: ignore

from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent


class NoopWorkflow(Workflow):
    @step
    async def run_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result=True)


@dataclass
class RunStats:
    started: int
    completed: int
    failed: int


async def memory_sampler(
    *,
    stop_fn: Callable[[], bool],
    interval_s: float,
    stats_fn: Callable[[], RunStats],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    start_monotonic = time.monotonic()

    # Ensure tracemalloc tracking is on
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    # Prepare CSV
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                [
                    "ts_epoch_s",
                    "elapsed_s",
                    "started",
                    "completed",
                    "failed",
                    "rss_bytes",
                    "tracemalloc_current_bytes",
                    "tracemalloc_peak_bytes",
                ]
            )

        while not stop_fn():
            now = time.time()
            elapsed = time.monotonic() - start_monotonic

            rss_bytes = None
            if resource is not None:  # Best-effort; platform dependent
                try:
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    # On macOS, ru_maxrss is bytes; on Linux, it's kilobytes.
                    # Detect heuristically: treat values < 10**10 as kilobytes on Linux.
                    rss_val = getattr(usage, "ru_maxrss", 0)
                    if rss_val < 10**10:  # assume kilobytes
                        rss_bytes = int(rss_val * 1024)
                    else:
                        rss_bytes = int(rss_val)
                except Exception:
                    rss_bytes = None

            current, peak = tracemalloc.get_traced_memory()
            s = stats_fn()
            writer.writerow(
                [
                    f"{now:.6f}",
                    f"{elapsed:.6f}",
                    s.started,
                    s.completed,
                    s.failed,
                    rss_bytes if rss_bytes is not None else "",
                    current,
                    peak,
                ]
            )
            f.flush()
            await asyncio.sleep(interval_s)


async def leak_sampler(
    *,
    stop_fn: Callable[[], bool],
    interval_s: float,
    csv_path: Path,
    snapshot_dir: Path | None,
    tracemalloc_top_n: int,
    enable_snapshots: bool,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    start_monotonic = time.monotonic()

    if enable_snapshots and not tracemalloc.is_tracing():
        tracemalloc.start()

    new_file = not csv_path.exists()
    prev_snapshot = None
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    def count_fds() -> int | None:
        try:
            fd_dir = Path("/proc/self/fd")
            if fd_dir.exists():
                return len(list(fd_dir.iterdir()))
            dev_fd = Path("/dev/fd")
            if dev_fd.exists():
                return len(list(dev_fd.iterdir()))
        except Exception:
            return None
        return None

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                [
                    "ts_epoch_s",
                    "elapsed_s",
                    "asyncio_tasks",
                    "gc_gen0",
                    "gc_gen1",
                    "gc_gen2",
                    "gc_tracked_objects",
                    "fd_count",
                ]
            )

        while not stop_fn():
            now = time.time()
            elapsed = time.monotonic() - start_monotonic

            gen0, gen1, gen2 = gc.get_count()
            tracked = len(gc.get_objects())
            tasks = len(asyncio.all_tasks())
            fds = count_fds()

            writer.writerow(
                [
                    f"{now:.6f}",
                    f"{elapsed:.6f}",
                    tasks,
                    gen0,
                    gen1,
                    gen2,
                    tracked,
                    fds if fds is not None else "",
                ]
            )
            f.flush()

            if enable_snapshots and snapshot_dir is not None:
                try:
                    snap = tracemalloc.take_snapshot()
                    top = snap.statistics("lineno")[:tracemalloc_top_n]
                    top_path = snapshot_dir / f"top_{int(elapsed)}s.txt"
                    with top_path.open("w") as tf:
                        for stat in top:
                            tf.write(str(stat) + "\n")

                    if prev_snapshot is not None:
                        diff = snap.compare_to(prev_snapshot, "lineno")[
                            :tracemalloc_top_n
                        ]
                        diff_path = snapshot_dir / f"diff_{int(elapsed)}s.txt"
                        with diff_path.open("w") as df:
                            for stat in diff:
                                df.write(str(stat) + "\n")
                    prev_snapshot = snap
                except Exception:
                    pass

            await asyncio.sleep(interval_s)


async def run_load(
    *,
    duration_s: float,
    concurrency: int,
    csv_path: Path,
    sample_interval_s: float,
    leak_csv_path: Path | None = None,
    leak_sample_interval_s: float | None = None,
    leak_snapshot_dir: Path | None = None,
    tracemalloc_top_n: int = 20,
    enable_snapshots: bool = False,
    probe_leaks: bool = False,
    probe_examples: int = 3,
    objgraph_dir: Path | None = None,
    ref_chain_depth: int = 4,
) -> None:
    stats = RunStats(started=0, completed=0, failed=0)
    stop = False

    def stop_fn() -> bool:
        return stop

    def stats_fn() -> RunStats:
        return stats

    # Handle SIGINT for graceful shutdown
    loop = asyncio.get_running_loop()

    def _sigint_handler() -> None:
        nonlocal stop
        stop = True

    try:
        loop.add_signal_handler(signal.SIGINT, _sigint_handler)
    except NotImplementedError:
        pass

    sampler_task = asyncio.create_task(
        memory_sampler(
            stop_fn=stop_fn,
            interval_s=sample_interval_s,
            stats_fn=stats_fn,
            csv_path=csv_path,
        )
    )

    leak_task = None
    if leak_csv_path is not None:
        leak_task = asyncio.create_task(
            leak_sampler(
                stop_fn=stop_fn,
                interval_s=leak_sample_interval_s or sample_interval_s,
                csv_path=leak_csv_path,
                snapshot_dir=leak_snapshot_dir,
                tracemalloc_top_n=tracemalloc_top_n,
                enable_snapshots=enable_snapshots,
            )
        )

    end_time = time.monotonic() + duration_s

    created_workflow_refs: list[weakref.ReferenceType[NoopWorkflow]] = []

    async def one_run() -> None:
        nonlocal stats
        wf = NoopWorkflow()
        if probe_leaks:
            created_workflow_refs.append(weakref.ref(wf))
        stats.started += 1
        try:
            await wf.run()
            stats.completed += 1
        except Exception:
            stats.failed += 1

    # Maintain up to `concurrency` in-flight runs
    in_flight: set[asyncio.Task[None]] = set()

    try:
        while time.monotonic() < end_time and not stop:
            # Fill up to concurrency
            while len(in_flight) < concurrency and not stop:
                t = asyncio.create_task(one_run())
                in_flight.add(t)
                t.add_done_callback(in_flight.discard)

            # Wait briefly to allow tasks to progress; schedule next immediately
            await asyncio.sleep(0)

        # After time is up, wait for any remaining to complete
        while in_flight:
            done, _ = await asyncio.wait(
                in_flight, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
            )
            for _ in done:
                pass
    finally:
        # Stop sampler
        stop = True
        try:
            await asyncio.wait_for(sampler_task, timeout=5)
        except asyncio.TimeoutError:
            sampler_task.cancel()
        if leak_task is not None:
            try:
                await asyncio.wait_for(leak_task, timeout=5)
            except asyncio.TimeoutError:
                leak_task.cancel()

        if probe_leaks and created_workflow_refs:
            await _probe_and_report_workflow_leaks(
                created_workflow_refs=created_workflow_refs,
                max_examples=probe_examples,
                objgraph_dir=objgraph_dir,
                ref_chain_depth=ref_chain_depth,
            )


async def _probe_and_report_workflow_leaks(
    *,
    created_workflow_refs: list[weakref.ReferenceType[NoopWorkflow]],
    max_examples: int,
    objgraph_dir: Path | None,
    ref_chain_depth: int,
) -> None:
    gc.collect()
    alive = [r() for r in created_workflow_refs if r() is not None]
    if not alive:
        print("[leak-probe] No leaked workflow instances detected (all collected).")
        return
    print(
        f"[leak-probe] Detected {len(alive)} still-alive workflow instances after run."
    )

    # Limit examples
    examples = alive[:max_examples]

    def try_objgraph_dump(i: int, obj: object) -> None:
        if objgraph_dir is None:
            return
        try:
            import objgraph  # type: ignore
        except Exception:
            print(
                "[leak-probe] objgraph not installed; skip PNG backrefs. pip install objgraph graphviz"
            )
            return
        objgraph_dir.mkdir(parents=True, exist_ok=True)
        png_path = objgraph_dir / f"workflow_backrefs_{i}.png"
        try:
            objgraph.show_backrefs(
                [obj], max_depth=ref_chain_depth, filename=str(png_path)
            )
            print(f"[leak-probe] Wrote backrefs graph: {png_path}")
        except Exception as e:
            print(f"[leak-probe] Failed to write objgraph backrefs: {e}")

    for idx, obj in enumerate(examples, start=1):
        print(f"[leak-probe] Example {idx} leaked workflow: {obj!r}")
        try_objgraph_dump(idx, obj)
        # Print a lightweight textual backref chain using gc.get_referrers
        _print_backref_chains(obj, max_depth=ref_chain_depth)


def _print_backref_chains(root: object, *, max_depth: int) -> None:
    # BFS over referrers with filtering to keep output readable
    from collections import deque

    def short(o: object) -> str:
        t = type(o)
        name = getattr(t, "__name__", str(t))
        rep = repr(o)
        if len(rep) > 120:
            rep = rep[:117] + "..."
        return f"{name}: {rep}"

    def filtered_referrers(o: object) -> list[object]:
        refs = []
        for r in gc.get_referrers(o):
            # Skip frames to avoid massive noise
            if isinstance(r, dict):
                # Common case: object's __dict__ or locals; include only module/class dicts
                owner = getattr(r, "__module__", None)
                if owner is None and "__name__" not in r:
                    continue
            if isinstance(r, (list, tuple, set)):
                # Skip generic containers unless they look like module-level globals
                continue
            refs.append(r)
        return refs[:20]

    print("[leak-probe] Backref chains (up to depth", max_depth, "):")
    q = deque([(root, 0, [short(root)])])
    seen: set[int] = {id(root)}
    chains_printed = 0
    max_chains = 10
    while q and chains_printed < max_chains:
        node, depth, chain = q.popleft()
        if depth >= max_depth:
            print("  - ", " -> ".join(chain))
            chains_printed += 1
            continue
        for ref in filtered_referrers(node):
            if id(ref) in seen:
                continue
            seen.add(id(ref))
            q.append((ref, depth + 1, chain + [short(ref)]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run many simple workflows and record heap usage over time",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="Run duration in seconds"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=100,
        help="Max number of concurrent workflow runs",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="Memory sample interval in seconds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./heap_usage.csv"),
        help="Path to CSV output file",
    )
    parser.add_argument(
        "--leak-csv",
        type=Path,
        default=None,
        help="Optional CSV path for leak sampler (GC/tasks/FDs)",
    )
    parser.add_argument(
        "--leak-interval",
        type=float,
        default=None,
        help="Leak sampler interval seconds (defaults to --sample-interval)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Directory to write tracemalloc top and diff reports",
    )
    parser.add_argument(
        "--snapshot-top-n",
        type=int,
        default=20,
        help="Number of lines in top allocation reports",
    )
    parser.add_argument(
        "--enable-snapshots",
        action="store_true",
        help="Enable periodic tracemalloc snapshots",
    )
    parser.add_argument(
        "--probe-leaks",
        action="store_true",
        help="After run, probe leaked workflow instances and print backref chains",
    )
    parser.add_argument(
        "--probe-examples",
        type=int,
        default=3,
        help="Max number of leaked workflow examples to analyze",
    )
    parser.add_argument(
        "--objgraph-dir",
        type=Path,
        default=None,
        help="If set and objgraph is installed, write backref PNGs here",
    )
    parser.add_argument(
        "--ref-chain-depth",
        type=int,
        default=4,
        help="Depth for textual and PNG backref chains",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    await run_load(
        duration_s=args.duration,
        concurrency=args.concurrency,
        csv_path=args.output,
        sample_interval_s=args.sample_interval,
        leak_csv_path=args.leak_csv,
        leak_sample_interval_s=args.leak_interval,
        leak_snapshot_dir=args.snapshot_dir,
        tracemalloc_top_n=args.snapshot_top_n,
        enable_snapshots=args.enable_snapshots,
        probe_leaks=args.probe_leaks,
        probe_examples=args.probe_examples,
        objgraph_dir=args.objgraph_dir,
        ref_chain_depth=args.ref_chain_depth,
    )


if __name__ == "__main__":
    asyncio.run(main())
