# /// script
# dependencies=["matplotlib"]
# ///
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Series:
    x: List[float]
    y: List[float]
    label: str


def read_csv(
    path: Path,
) -> Tuple[Series, Series, Series]:
    ts: list[float] = []
    rss: list[float] = []
    t_current: list[float] = []
    t_peak: list[float] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                elapsed = float(row["elapsed_s"]) if row.get("elapsed_s") else None
                if elapsed is None:
                    continue
                ts.append(elapsed)

                # bytes -> MB
                def to_mb(v: str | None) -> float | None:
                    if v is None or v == "":
                        return None
                    return float(v) / (1024 * 1024)

                rss_mb = to_mb(row.get("rss_bytes"))
                cur_mb = to_mb(row.get("tracemalloc_current_bytes"))
                peak_mb = to_mb(row.get("tracemalloc_peak_bytes"))

                rss.append(rss_mb if rss_mb is not None else float("nan"))
                t_current.append(cur_mb if cur_mb is not None else float("nan"))
                t_peak.append(peak_mb if peak_mb is not None else float("nan"))
            except Exception:
                # Skip malformed rows
                continue

    return (
        Series(ts, rss, label="RSS (MB)"),
        Series(ts, t_current, label="tracemalloc current (MB)"),
        Series(ts, t_peak, label="tracemalloc peak (MB)"),
    )


def plot(
    *,
    input_csv: Path,
    output_png: Path | None,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    rss_series, cur_series, peak_series = read_csv(input_csv)

    plt.figure(figsize=(10, 6))
    plt.plot(rss_series.x, rss_series.y, label=rss_series.label, linewidth=1.5)
    plt.plot(cur_series.x, cur_series.y, label=cur_series.label, linewidth=1.5)
    plt.plot(peak_series.x, peak_series.y, label=peak_series.label, linewidth=1.5)
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Memory (MB)")
    plt.title("Heap usage over time")
    plt.legend()
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot heap usage CSV")
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV path from test script"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional PNG output path to save the chart",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the chart window (requires display)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(input_csv=args.input, output_png=args.output, show=args.show)


if __name__ == "__main__":
    main()
