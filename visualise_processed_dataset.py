#!/usr/bin/env python3
"""
Visualise the processed microservices bottleneck dataset:
- Time series of CPU, memory, network (RX/TX) over windows
- How metrics relate to each other (scatter, correlation)
Uses one CSV per run; aggregate by window_id so one row per time window.

Usage:
  pip install -r requirements.txt   # or: pip install pandas matplotlib numpy
  python visualise_processed_dataset.py [--data-dir PATH] [--csv RELATIVE_PATH] [--out-dir DIR]
  python visualise_processed_dataset.py --synthetic [--synthetic-csv PATH] [--out-dir DIR]
  python visualise_processed_dataset.py --out-dir ./figures   # save figures without opening windows
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Default path to processed dataset (override with --data-dir)
DEFAULT_DATA_DIR = Path("/Users/opeyemionikute/Downloads/archive/processed_dataset")
# Sample CSV: user service, one scenario (CPU bottleneck, 25min, 200 RPS)
DEFAULT_CSV = "user/multi-modal-data-separate/cpu_aug12_25min_200_0_graph_2.csv"
# Synthetic single-service linear dataset (2–6 h of data)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYNTHETIC_CSV = SCRIPT_DIR / "synthetic_dataset" / "single_service_linear.csv"


def _span_index(col: str) -> int:
    """Parse leading span index from column name (e.g. '2_container_cpu_...' -> 2, '0_latency' -> 0)."""
    try:
        return int(col.split("_")[0])
    except (ValueError, IndexError):
        return -1


def find_metric_columns(df: pd.DataFrame, span: int | None = None):
    """Get lists of column names for each metric type (any span index).
    Columns are always sorted by span index (0, 1, 2, ...) so plot order matches legend.
    If span is set, only return columns for that span (e.g. span=2 -> columns starting with '2_')."""
    cpu_cols = sorted([c for c in df.columns if "container_cpu_usage_seconds_total" in c], key=_span_index)
    mem_cols = sorted([c for c in df.columns if "container_memory_usage_bytes" in c], key=_span_index)
    rx_cols = sorted([c for c in df.columns if "container_network_receive_bytes_total" in c], key=_span_index)
    tx_cols = sorted([c for c in df.columns if "container_network_transmit_bytes_total" in c], key=_span_index)
    latency_cols = sorted(
        [c for c in df.columns if c.endswith("_latency") and not c.startswith("critical")],
        key=_span_index,
    )
    out = {
        "cpu": cpu_cols,
        "memory": mem_cols,
        "network_rx": rx_cols,
        "network_tx": tx_cols,
        "latency": latency_cols,
    }
    if span is not None:
        prefix = f"{span}_"
        out = {k: [c for c in v if c.startswith(prefix)] for k, v in out.items()}
    return out


def aggregate_by_window(df: pd.DataFrame, metric_cols: dict) -> pd.DataFrame:
    """One row per window_id; metrics are constant per window so take first(); latency: mean total per request.
    Also computes _time_rel_s = seconds since first window (from request start timestamps) for x-axis."""
    df = df.copy()
    df["_total_latency"] = df[metric_cols["latency"]].sum(axis=1)
    start_cols = [c for c in df.columns if c.endswith("_start") and c[0].isdigit()]
    agg = {c: "first" for c in metric_cols["cpu"] + metric_cols["memory"] + metric_cols["network_rx"] + metric_cols["network_tx"]}
    for c in metric_cols["latency"]:
        agg[c] = "mean"
    agg["_total_latency"] = "mean"
    if start_cols:
        agg[start_cols[0]] = "min"  # earliest request start in window (µs)
    out = df.groupby("window_id", sort=False).agg(agg).reset_index()
    # Sort by window_id so order is stable (parse "23_0" -> (23,0) for numeric sort)
    def window_key(w):
        parts = str(w).split("_")
        try:
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        except (ValueError, IndexError):
            return (0, 0)
    out["_order"] = out["window_id"].map(window_key)
    out = out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    # Timestamp for x-axis: min request start per window (µs) -> seconds since first window
    if start_cols and start_cols[0] in out.columns:
        out["_time_s"] = out[start_cols[0]] / 1e6  # µs -> seconds (epoch)
        out["_time_rel_s"] = out["_time_s"] - out["_time_s"].min()
        out = out.drop(columns=[start_cols[0], "_time_s"])
    else:
        out["_time_rel_s"] = np.arange(len(out))  # fallback to index
    return out


def plot_time_series(win: pd.DataFrame, metric_cols: dict, out_path: Path | None = None, single_png_path: Path | None = None, show: bool = True):
    """Plot CPU, memory, net RX, net TX, and latency per service over time."""
    x = win["_time_rel_s"].values  # seconds since first window
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Metrics over time (one line per span; aggregated by window)", fontsize=12)

    # CPU: one subplot, all spans (same graph)
    ax = axes[0, 0]
    for c in metric_cols["cpu"]:
        ax.plot(x, win[c], alpha=0.7, label=c.split("_")[0])
    ax.set_ylabel("CPU (seconds total)")
    ax.set_title("CPU usage per span")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Memory: one subplot, all spans (same graph)
    ax = axes[0, 1]
    for c in metric_cols["memory"]:
        ax.plot(x, win[c] / 1e6, alpha=0.7, label=c.split("_")[0])  # MB
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory usage per span")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Latency per service (span) over time
    ax = axes[0, 2]
    for c in metric_cols["latency"]:
        ax.plot(x, win[c] / 1e3, alpha=0.7, label=c.replace("_latency", ""))  # µs -> ms
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency per service (span)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Network RX + TX on same graph (one subplot)
    ax = axes[1, 0]
    rx_mean = win[metric_cols["network_rx"]].mean(axis=1) / 1e6  # MB
    tx_mean = win[metric_cols["network_tx"]].mean(axis=1) / 1e6
    ax.plot(x, rx_mean, label="Network RX (mean)", color="C0")
    ax.plot(x, tx_mean, label="Network TX (mean)", color="C1")
    ax.set_ylabel("Bytes (MB)")
    ax.set_title("Network RX vs TX (mean across spans)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mean total latency over time
    ax = axes[1, 1]
    total_latency = win["_total_latency"] / 1e3  # ms
    ax.plot(x, total_latency, color="C2", label="Mean total latency (ms)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Total request latency over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Network RX per span (one line per span)
    ax = axes[1, 2]
    for c in metric_cols["network_rx"]:
        ax.plot(x, win[c] / 1e6, alpha=0.7, label=c.split("_")[0])  # MB
    ax.set_ylabel("Network RX (MB)")
    ax.set_title("Network receive per span")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel("Time (seconds since start)")
    plt.tight_layout()
    if single_png_path is not None:
        single_png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(single_png_path, dpi=120, bbox_inches="tight")
    elif out_path:
        plt.savefig(out_path / "timeseries.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_relationships(win: pd.DataFrame, metric_cols: dict, out_path: Path | None = None, show: bool = True):
    """Scatter: how metrics relate to each other and to latency."""
    # Per-window aggregates (mean across spans)
    cpu_mean = win[metric_cols["cpu"]].mean(axis=1)
    mem_mean = win[metric_cols["memory"]].mean(axis=1) / 1e6  # MB
    rx_mean = win[metric_cols["network_rx"]].mean(axis=1) / 1e6
    tx_mean = win[metric_cols["network_tx"]].mean(axis=1) / 1e6
    latency_mean = win["_total_latency"] / 1e3  # ms

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle("How metrics relate to each other (one point per window)", fontsize=12)

    # CPU vs Memory (same graph)
    ax = axes[0, 0]
    sc0 = ax.scatter(cpu_mean, mem_mean, alpha=0.5, s=20, c=latency_mean, cmap="viridis")
    ax.set_xlabel("Mean CPU (s)")
    ax.set_ylabel("Mean Memory (MB)")
    ax.set_title("CPU vs Memory (coloured by latency)")
    plt.colorbar(sc0, ax=ax, label="Latency (ms)")

    # CPU vs Latency
    ax = axes[0, 1]
    ax.scatter(cpu_mean, latency_mean, alpha=0.5, s=20)
    ax.set_xlabel("Mean CPU (s)")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("CPU vs Latency")

    # Memory vs Latency
    ax = axes[1, 0]
    ax.scatter(mem_mean, latency_mean, alpha=0.5, s=20)
    ax.set_xlabel("Mean Memory (MB)")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Memory vs Latency")

    # Network RX vs TX (same graph)
    ax = axes[1, 1]
    sc1 = ax.scatter(rx_mean, tx_mean, alpha=0.5, s=20, c=latency_mean, cmap="plasma")
    ax.set_xlabel("Mean Network RX (MB)")
    ax.set_ylabel("Mean Network TX (MB)")
    ax.set_title("Network RX vs TX (coloured by latency)")
    plt.colorbar(sc1, ax=ax, label="Latency (ms)")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path / "relationships.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_correlation_heatmap(win: pd.DataFrame, metric_cols: dict, out_path: Path | None = None, show: bool = True):
    """Correlation heatmap of key metrics (mean per window) + latency."""
    cpu_mean = win[metric_cols["cpu"]].mean(axis=1)
    mem_mean = win[metric_cols["memory"]].mean(axis=1) / 1e6
    rx_mean = win[metric_cols["network_rx"]].mean(axis=1) / 1e6
    tx_mean = win[metric_cols["network_tx"]].mean(axis=1) / 1e6
    latency_mean = win["_total_latency"] / 1e3

    corr_df = pd.DataFrame({
        "CPU": cpu_mean,
        "Memory (MB)": mem_mean,
        "Net RX (MB)": rx_mean,
        "Net TX (MB)": tx_mean,
        "Latency (ms)": latency_mean,
    }).corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_df, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_df)))
    ax.set_yticks(range(len(corr_df)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.columns)
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation: metrics and latency (per window)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Synthetic single-service dataset (linear CPU / request rate / latency)
# ---------------------------------------------------------------------------


def plot_synthetic_time_series(df: pd.DataFrame, out_path: Path | None = None, show: bool = True):
    """Time series: CPU, request rate, latency, memory (single service)."""
    x = df["time_s"].values / 3600  # hours
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Synthetic single-service metrics over time (linear relationships)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(x, df["cpu_utilization"], color="C0")
    ax.set_ylabel("CPU utilization")
    ax.set_title("CPU")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, df["request_rate_rps"], color="C1")
    ax.set_ylabel("Requests / s")
    ax.set_title("Request rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, df["latency_ms"], color="C2")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, df["memory_bytes"] / 1e6, color="C3")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Time (hours)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path / "synthetic_timeseries.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_synthetic_relationships(df: pd.DataFrame, out_path: Path | None = None, show: bool = True):
    """Scatter: CPU vs latency, request rate vs latency, CPU vs request rate."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Synthetic data: linear relationships (single service)", fontsize=12)

    ax = axes[0]
    ax.scatter(df["cpu_utilization"], df["latency_ms"], alpha=0.5, s=15)
    ax.set_xlabel("CPU utilization")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("CPU vs Latency")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(df["request_rate_rps"], df["latency_ms"], alpha=0.5, s=15)
    ax.set_xlabel("Request rate (rps)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Request rate vs Latency")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(df["request_rate_rps"], df["cpu_utilization"], alpha=0.5, s=15)
    ax.set_xlabel("Request rate (rps)")
    ax.set_ylabel("CPU utilization")
    ax.set_title("Request rate vs CPU")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path / "synthetic_relationships.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_synthetic_correlation(df: pd.DataFrame, out_path: Path | None = None, show: bool = True):
    """Correlation heatmap for synthetic metrics."""
    corr_df = df[["cpu_utilization", "request_rate_rps", "memory_bytes", "latency_ms"]].corr()
    corr_df.columns = ["CPU", "Request rate", "Memory", "Latency"]
    corr_df.index = ["CPU", "Request rate", "Memory", "Latency"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_df, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_df)))
    ax.set_yticks(range(len(corr_df)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Synthetic data: correlation (linear relationships)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path / "synthetic_correlation_heatmap.png", dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualise processed bottleneck dataset metrics")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to processed_dataset root",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help="Relative path to one CSV under data-dir, e.g. user/.../cpu_aug12_25min_200_0_graph_2.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save figures (default: only display)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful when saving only)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Load and plot synthetic single-service dataset (linear CPU / request rate / latency)",
    )
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        default=DEFAULT_SYNTHETIC_CSV,
        help="Path to synthetic CSV (default: synthetic_dataset/single_service_linear.csv)",
    )
    parser.add_argument(
        "--span",
        type=int,
        default=None,
        metavar="N",
        help="Plot only metrics for span N (e.g. --span 2 for span 2 only; ignored if --synthetic)",
    )
    parser.add_argument(
        "--single-png",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save only the time-series figure to PATH (for batch: one PNG per scenario, same base name as CSV)",
    )
    args = parser.parse_args()

    out_path = args.out_dir
    if args.no_show:
        plt.ioff()

    if args.synthetic:
        csv_path = args.synthetic_csv
        if not csv_path.exists():
            raise SystemExit(f"Synthetic CSV not found: {csv_path}")
        print(f"Loading synthetic data: {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"Rows: {len(df):,} ({df['time_s'].max() / 3600:.1f} hours)")
        plot_synthetic_time_series(df, out_path, show=not args.no_show)
        plot_synthetic_relationships(df, out_path, show=not args.no_show)
        plot_synthetic_correlation(df, out_path, show=not args.no_show)
    else:
        csv_path = args.data_dir / args.csv
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        print(f"Loading {csv_path} ...")
        df = pd.read_csv(csv_path, nrows=None)
        print(f"Rows: {len(df):,}")
        metric_cols = find_metric_columns(df, span=args.span)
        if args.span is not None:
            print(f"  (filtered to span {args.span} only)")
        for k, v in metric_cols.items():
            print(f"  {k}: {len(v)} columns")
        win = aggregate_by_window(df, metric_cols)
        print(f"Windows: {len(win)}")
        plot_time_series(win, metric_cols, out_path, single_png_path=args.single_png, show=not args.no_show)
        if args.single_png is None:
            plot_relationships(win, metric_cols, out_path, show=not args.no_show)
            plot_correlation_heatmap(win, metric_cols, out_path, show=not args.no_show)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
