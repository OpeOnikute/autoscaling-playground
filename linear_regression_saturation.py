#!/usr/bin/env python3
"""
Linear regression to predict latency (saturation) from CPU and network traffic.
Uses only scenarios marked with <c-latency>- in scenarios/. Each scenario is
fitted and plotted individually (one figure per scenario).
Features: CPU (mean per window), network RX (MB), network TX (MB). Target: latency (ms).
Fits with (1) gradient descent and (2) closed-form; plots both vs actual on the same graph.

Usage:
  python linear_regression_saturation.py [--data-dir PATH] [--out-dir DIR]
  MPLBACKEND=Agg python linear_regression_saturation.py --no-show --out-dir regression/  # headless

Note: If GD predictions were worse than closed-form, that was almost certainly
convergence (too few epochs or learning rate). Both minimize the same MSE;
closed-form is exact, GD is iterative. Saturation/non-linearity would hurt
*both* equally (linear model is wrong in the flat region); it doesn't explain
GD being worse than CF. This script now uses a larger lr and early-stop so
GD converges to match closed-form.
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("./data/processed_dataset")
SCENARIOS_DIR = SCRIPT_DIR / "scenarios"
CANDIDATE_PREFIX = "<c-latency>-"
SERVICES = ("user", "home", "compose")


def _span_index(col: str) -> int:
    try:
        return int(col.split("_")[0])
    except (ValueError, IndexError):
        return -1


def find_metric_columns(df: pd.DataFrame) -> dict:
    cpu = sorted([c for c in df.columns if "container_cpu_usage_seconds_total" in c], key=_span_index)
    rx = sorted([c for c in df.columns if "container_network_receive_bytes_total" in c], key=_span_index)
    tx = sorted([c for c in df.columns if "container_network_transmit_bytes_total" in c], key=_span_index)
    lat = sorted(
        [c for c in df.columns if c.endswith("_latency") and not c.startswith("critical")],
        key=_span_index,
    )
    return {"cpu": cpu, "network_rx": rx, "network_tx": tx, "latency": lat}


def aggregate_by_window(df: pd.DataFrame, metric_cols: dict) -> pd.DataFrame:
    df = df.copy()
    df["_total_latency"] = df[metric_cols["latency"]].sum(axis=1)
    start_cols = [c for c in df.columns if c.endswith("_start") and c[0].isdigit()]
    agg = {c: "first" for c in metric_cols["cpu"] + metric_cols["network_rx"] + metric_cols["network_tx"]}
    for c in metric_cols["latency"]:
        agg[c] = "mean"
    agg["_total_latency"] = "mean"
    if start_cols:
        agg[start_cols[0]] = "min"
    out = df.groupby("window_id", sort=False).agg(agg).reset_index()

    def window_key(w):
        parts = str(w).split("_")
        try:
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        except (ValueError, IndexError):
            return (0, 0)

    out["_order"] = out["window_id"].map(window_key)
    out = out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return out


def load_scenario_data(data_dir: Path, stem: str) -> pd.DataFrame | None:
    for service in SERVICES:
        csv_path = data_dir / service / "multi-modal-data-separate" / f"{stem}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, nrows=None)
        except Exception:
            return None
        metric_cols = find_metric_columns(df)
        if not metric_cols["cpu"] or not metric_cols["latency"]:
            return None
        return aggregate_by_window(df, metric_cols)
    return None


def iter_candidate_scenarios(
    data_dir: Path, scenarios_dir: Path, use_poly: bool = False, use_log: bool = False
):
    """Yield (stem, X, y) for each <c-latency>-* scenario (X and y for that scenario only).
    use_poly: add squared CPU, net_rx, net_tx as extra features.
    use_log: add log(1+CPU), log(1+net_rx), log(1+net_tx) as extra features."""
    pngs = sorted([p for p in scenarios_dir.glob("*.png") if p.name.startswith(CANDIDATE_PREFIX)])
    if not pngs:
        raise SystemExit(f"No candidate scenarios found (no files matching {CANDIDATE_PREFIX}* in {scenarios_dir})")
    for p in pngs:
        stem = p.stem[len(CANDIDATE_PREFIX):]
        win = load_scenario_data(data_dir, stem)
        if win is None:
            continue
        metric_cols = find_metric_columns(win)
        metric_cols = {k: [c for c in v if c in win.columns] for k, v in metric_cols.items()}
        if not metric_cols["cpu"] or not metric_cols["latency"]:
            continue
        cpu = win[metric_cols["cpu"]].mean(axis=1).values
        net_rx = win[metric_cols["network_rx"]].mean(axis=1).values / 1e6  # MB
        net_tx = win[metric_cols["network_tx"]].mean(axis=1).values / 1e6
        lat_ms = (win["_total_latency"] / 1e3).values  # ms
        cols = [np.ones(len(win)), cpu, net_rx, net_tx]
        if use_poly:
            cols.extend([cpu**2, net_rx**2, net_tx**2])
        if use_log:
            cols.extend([np.log1p(cpu), np.log1p(net_rx), np.log1p(net_tx)])
        X = np.column_stack(cols)
        y = lat_ms
        if len(y) < 5:
            continue
        yield stem, X, y


def scenario_duration_minutes(stem: str) -> int | None:
    """Parse scenario stem for duration, e.g. 'cpu_aug12_25min_200_0_graph_2' -> 25. Returns None if not found."""
    m = re.search(r"(\d+)min", stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def baseline_end_index(y: np.ndarray, percentile: float = 75, min_windows: int = 5) -> int:
    """Index of first window where latency enters the high (saturation) regime.
    Baseline = y[0:baseline_end]. Uses: first t where latency[t] >= percentile of full series."""
    n = len(y)
    if n <= min_windows:
        return n
    thresh = np.percentile(y, percentile)
    for t in range(min_windows, n):
        if y[t] >= thresh:
            return t
    return n


def fit_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normal equation: theta = (X'X)^{-1} X'y (with small ridge for stability)."""
    lam = 1e-6
    theta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    return theta


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 50_000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Gradient descent: minimize MSE; theta := theta - lr * (2/n) * X'(X theta - y).
    Stops early if gradient norm < tol so we converge to the same minimum as closed-form."""
    n = X.shape[0]
    theta = np.zeros(X.shape[1])
    for _ in range(epochs):
        pred = X @ theta
        grad = (2 / n) * (X.T @ (pred - y))
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        theta -= lr * grad
    return theta


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Linear regression per <c-latency> scenario (GD vs closed-form)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for gradient descent (use larger with scaled features)")
    parser.add_argument("--epochs", type=int, default=50_000, help="Max gradient descent epochs (early stop if converged)")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "regression", help="Directory to save one PNG per scenario")
    parser.add_argument("--poly", "--polynomial", dest="poly", action="store_true", help="Add squared CPU, net_rx, net_tx as extra features")
    parser.add_argument("--log", action="store_true", help="Add log(1+CPU), log(1+net_rx), log(1+net_tx) as extra features")
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="Fit log(1+latency), then expm1 back to ms. Use when latency is right-skewed or has long tail.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Fit only on baseline (normal behaviour up until saturation spike); ignore saturation tail for fitting.",
    )
    parser.add_argument(
        "--baseline-pct",
        type=float,
        default=None,
        metavar="P",
        help="If set with --baseline-only, use first P%% of windows as baseline (e.g. 60). Else auto-detect from latency percentile.",
    )
    parser.add_argument(
        "--baseline-percentile",
        type=float,
        default=75,
        metavar="Q",
        help="With --baseline-only (no --baseline-pct): baseline ends at first window where latency >= Qth percentile (default 75).",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = args.data_dir
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")
    if not SCENARIOS_DIR.exists():
        raise SystemExit(f"Scenarios dir not found: {SCENARIOS_DIR}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing each <c-latency>-* scenario individually. Output: {out_dir}")
    feats = "intercept, CPU, net_rx, net_tx"
    if args.poly:
        feats += ", CPU², net_rx², net_tx²"
    if args.log:
        feats += ", log(1+CPU), log(1+net_rx), log(1+net_tx)"
    target = "log(1+latency) → latency (ms)" if args.log_target else "latency (ms)"
    if args.baseline_only:
        print("Baseline-only: fitting on normal behaviour segment only (pre-saturation).")
    print(f"Features: {feats} -> Target: {target}\n")

    for i, (stem, X, y) in enumerate(iter_candidate_scenarios(data_dir, SCENARIOS_DIR, use_poly=args.poly, use_log=args.log)):
        n = X.shape[0]
        if args.baseline_only:
            if args.baseline_pct is not None:
                end = max(5, int(n * args.baseline_pct / 100))
            else:
                end = baseline_end_index(y, percentile=args.baseline_percentile)
            X_fit, y_fit_full = X[:end], y[:end]
            n_fit = end
        else:
            X_fit, y_fit_full = X, y
            n_fit = n

        y_fit = np.log1p(y_fit_full) if args.log_target else y_fit_full
        # Scale for gradient descent (using baseline stats when baseline_only)
        X_gd_fit = X_fit.copy()
        X_gd_fit[:, 1:] = (X_gd_fit[:, 1:] - X_gd_fit[:, 1:].mean(axis=0)) / (X_gd_fit[:, 1:].std(axis=0) + 1e-8)
        y_mean, y_std = y_fit.mean(), y_fit.std() + 1e-8
        y_gd = (y_fit - y_mean) / y_std

        theta_cf = fit_closed_form(X_fit, y_fit)
        theta_gd = fit_gradient_descent(X_gd_fit, y_gd, lr=args.lr, epochs=args.epochs)

        # Predict on full X for plot (model was fit on baseline only when --baseline-only)
        X_gd = X.copy()
        X_gd[:, 1:] = (X_gd[:, 1:] - X_fit[:, 1:].mean(axis=0)) / (X_fit[:, 1:].std(axis=0) + 1e-8)
        pred_gd_raw = (X_gd @ theta_gd) * y_std + y_mean
        pred_cf_raw = X @ theta_cf
        if args.log_target:
            pred_gd = np.expm1(pred_gd_raw)
            pred_cf = np.expm1(pred_cf_raw)
        else:
            pred_gd = pred_gd_raw
            pred_cf = pred_cf_raw

        # MSE on baseline segment only when baseline_only (fairer comparison)
        if args.baseline_only:
            mse_cf = np.mean((y[:n_fit] - pred_cf[:n_fit]) ** 2)
            mse_gd = np.mean((y[:n_fit] - pred_gd[:n_fit]) ** 2)
            print(f"  [{i + 1}] {stem}: n={n}  baseline={n_fit}  MSE(cf)={mse_cf:.1f}  MSE(gd)={mse_gd:.1f}")
        else:
            mse_cf = np.mean((y - pred_cf) ** 2)
            mse_gd = np.mean((y - pred_gd) ** 2)
            print(f"  [{i + 1}] {stem}: n={n}  MSE(cf)={mse_cf:.1f}  MSE(gd)={mse_gd:.1f}")

        # When baseline-only: predict each feature and latency 30 min from threshold (in real minutes, not fixed 30 windows)
        pred_30min = None
        if args.baseline_only and n_fit >= 5:
            MINUTES_FROM_THRESHOLD = 30
            duration_min = scenario_duration_minutes(stem)
            if duration_min and n > 0:
                windows_per_min = n / duration_min
                num_windows_30min = int(round(MINUTES_FROM_THRESHOLD * windows_per_min))
            else:
                num_windows_30min = MINUTES_FROM_THRESHOLD  # fallback: 1 window/min
            t_baseline = np.arange(n_fit, dtype=float)
            cpu_b, net_rx_b, net_tx_b = X_fit[:, 1], X_fit[:, 2], X_fit[:, 3]
            t_30 = n_fit + num_windows_30min
            poly_cpu = np.polyfit(t_baseline, cpu_b, 1)
            cpu_30 = np.polyval(poly_cpu, t_30)
            net_rx_30 = np.polyval(np.polyfit(t_baseline, net_rx_b, 1), t_30)
            net_tx_30 = np.polyval(np.polyfit(t_baseline, net_tx_b, 1), t_30)
            t_cpu = np.arange(0, t_30 + 1, dtype=float)
            cpu_pred_curve = np.polyval(poly_cpu, t_cpu)
            row = [1.0, cpu_30, net_rx_30, net_tx_30]
            if args.poly:
                row.extend([cpu_30**2, net_rx_30**2, net_tx_30**2])
            if args.log:
                row.extend([np.log1p(cpu_30), np.log1p(net_rx_30), np.log1p(net_tx_30)])
            X_30 = np.array(row).reshape(1, -1)
            lat_30_cf = (X_30 @ theta_cf).item()
            X_30_gd = X_30.copy()
            X_30_gd[:, 1:] = (X_30[:, 1:] - X_fit[:, 1:].mean(axis=0)) / (X_fit[:, 1:].std(axis=0) + 1e-8)
            lat_30_gd_raw = (X_30_gd @ theta_gd).item() * y_std + y_mean
            lat_30_gd = np.expm1(lat_30_gd_raw) if args.log_target else lat_30_gd_raw
            lat_30_cf = np.expm1(lat_30_cf) if args.log_target else lat_30_cf
            pred_30min = {
                "t_30": t_30,
                "t_cpu": t_cpu,
                "cpu_pred": cpu_pred_curve,
                "poly_cpu": poly_cpu,
                "cpu": cpu_30,
                "net_rx_mb": net_rx_30,
                "net_tx_mb": net_tx_30,
                "latency_cf_ms": lat_30_cf,
                "latency_gd_ms": lat_30_gd,
            }
            print(f"       Predicted 30 min from threshold: CPU={cpu_30:.3f}  net_rx={net_rx_30:.2f} MB  net_tx={net_tx_30:.2f} MB  latency(cf)={lat_30_cf:.1f} ms  latency(gd)={lat_30_gd:.1f} ms")

        x_idx = np.arange(n)
        if pred_30min:
            fig, (ax, ax_cpu) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            t_30 = pred_30min["t_30"]
            duration_min = scenario_duration_minutes(stem)
            extra_windows = int(round(10 * (n / duration_min))) if duration_min and n > 0 else 10
            x_max = t_30 + extra_windows
            ax.set_xlim(0, x_max)
            t_cpu_ext = np.arange(0, x_max + 1, dtype=float)
            cpu_pred_ext = np.polyval(pred_30min["poly_cpu"], t_cpu_ext)
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_idx, y, label="Actual latency (ms)", color="black", alpha=0.7, linewidth=0.8)
        ax.plot(x_idx, pred_cf, label="Predicted (closed-form)", color="C0", alpha=0.8, linewidth=0.8)
        ax.plot(x_idx, pred_gd, label="Predicted (gradient descent)", color="C1", alpha=0.8, linewidth=0.8)
        if args.baseline_only:
            ax.axvline(n_fit - 0.5, color="gray", linestyle="--", alpha=0.7, label="Baseline end (fit region)")
        if pred_30min:
            ax.axvline(t_30 - 0.5, color="green", linestyle=":", alpha=0.8, label="30 min from threshold")
            ax.scatter([t_30], [pred_30min["latency_cf_ms"]], color="C0", s=80, zorder=5, marker="o", edgecolors="black", linewidths=0.5, label="Predicted 30 min (CF)")
            ax.scatter([t_30], [pred_30min["latency_gd_ms"]], color="C1", s=80, zorder=5, marker="s", edgecolors="black", linewidths=0.5, label="Predicted 30 min (GD)")
            label_lines = [
                f"30 min from threshold (baseline extrapolation):",
                f"  CPU = {pred_30min['cpu']:.3f}  net_rx = {pred_30min['net_rx_mb']:.2f} MB  net_tx = {pred_30min['net_tx_mb']:.2f} MB",
                f"  Predicted latency (CF) = {pred_30min['latency_cf_ms']:.1f} ms  (GD) = {pred_30min['latency_gd_ms']:.1f} ms",
            ]
            ax.text(0.02, 0.98, "\n".join(label_lines), transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
            ax_cpu.plot(t_cpu_ext, cpu_pred_ext, color="C2", alpha=0.9, linewidth=1, label="Predicted CPU (baseline trend)")
            ax_cpu.axvline(n_fit - 0.5, color="gray", linestyle="--", alpha=0.7, label="Baseline end")
            ax_cpu.axvline(t_30 - 0.5, color="green", linestyle=":", alpha=0.8, label="30 min from threshold")
            ax_cpu.set_ylabel("CPU")
            ax_cpu.set_title("Predicted CPU (linear extrapolation to 30 min from threshold)")
            ax_cpu.legend(loc="lower right")
            ax_cpu.grid(True, alpha=0.3)
        ax.set_xlabel("Window index" if not pred_30min else "")
        ax.set_ylabel("Latency (ms)")
        title = f"Linear regression: {stem}"
        if args.baseline_only:
            title += f" (fit on baseline n={n_fit})"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / f"{stem}_regression.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        if not args.no_show:
            plt.show()
        else:
            plt.close()

    print(f"\nSaved one figure per scenario in {out_dir}")


if __name__ == "__main__":
    main()
