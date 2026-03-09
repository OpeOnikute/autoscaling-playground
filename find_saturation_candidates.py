#!/usr/bin/env python3
"""
Among non-candidate scenario PNGs in scenarios/, find those whose data shows
clear latency saturation as CPU and network traffic increase.

Usage:
  python find_saturation_candidates.py                    # all non-candidates
  python find_saturation_candidates.py --limit 100       # quick test (first 100)
  python find_saturation_candidates.py -o list.txt        # write list to file

Requires: pandas, numpy. Excludes PNGs already renamed with <c-latency>- prefix.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DEFAULT_DATA_DIR = Path("./data/processed_dataset")
SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
CANDIDATE_PREFIX = "<c-latency>-"
SERVICES = ("user", "home", "compose")


def _span_index(col: str) -> int:
    try:
        return int(col.split("_")[0])
    except (ValueError, IndexError):
        return -1


def find_metric_columns(df: pd.DataFrame):
    cpu = sorted([c for c in df.columns if "container_cpu_usage_seconds_total" in c], key=_span_index)
    mem = sorted([c for c in df.columns if "container_memory_usage_bytes" in c], key=_span_index)
    rx = sorted([c for c in df.columns if "container_network_receive_bytes_total" in c], key=_span_index)
    tx = sorted([c for c in df.columns if "container_network_transmit_bytes_total" in c], key=_span_index)
    lat = sorted(
        [c for c in df.columns if c.endswith("_latency") and not c.startswith("critical")],
        key=_span_index,
    )
    return {"cpu": cpu, "memory": mem, "network_rx": rx, "network_tx": tx, "latency": lat}


def aggregate_by_window(df: pd.DataFrame, metric_cols: dict) -> pd.DataFrame:
    df = df.copy()
    df["_total_latency"] = df[metric_cols["latency"]].sum(axis=1)
    start_cols = [c for c in df.columns if c.endswith("_start") and c[0].isdigit()]
    agg = {c: "first" for c in metric_cols["cpu"] + metric_cols["memory"] + metric_cols["network_rx"] + metric_cols["network_tx"]}
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
    if start_cols and start_cols[0] in out.columns:
        out["_time_rel_s"] = out[start_cols[0]].astype(float) / 1e6
        out["_time_rel_s"] = out["_time_rel_s"] - out["_time_rel_s"].min()
        out = out.drop(columns=[start_cols[0]])
    else:
        out["_time_rel_s"] = np.arange(len(out))
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


def saturation_score(win: pd.DataFrame, metric_cols: dict) -> tuple[float, str]:
    if len(win) < 10:
        return 0.0, "too_few_windows"
    t = np.arange(len(win))
    cpu = win[metric_cols["cpu"]].mean(axis=1).values
    net_rx = win[metric_cols["network_rx"]].mean(axis=1).values / 1e6
    net_tx = win[metric_cols["network_tx"]].mean(axis=1).values / 1e6
    net = (net_rx + net_tx) / 2
    lat = (win["_total_latency"] / 1e3).values

    cpu_trend = np.corrcoef(t, cpu)[0, 1] if np.std(cpu) > 0 else 0
    net_trend = np.corrcoef(t, net)[0, 1] if np.std(net) > 0 else 0
    lat_cpu = np.corrcoef(cpu, lat)[0, 1] if np.std(lat) > 0 and np.std(cpu) > 0 else 0
    lat_net = np.corrcoef(net, lat)[0, 1] if np.std(lat) > 0 and np.std(net) > 0 else 0
    for x in (cpu_trend, net_trend, lat_cpu, lat_net):
        if np.isnan(x):
            x = 0
    mid = len(lat) // 2
    lat_step = (np.mean(lat[mid:]) - np.mean(lat[:mid])) / (np.mean(lat) + 1e-6)

    score = 0.0
    reasons = []
    if cpu_trend > 0.3:
        score += 0.25
        reasons.append("cpu_inc")
    if net_trend > 0.3:
        score += 0.25
        reasons.append("net_inc")
    if lat_cpu > 0.4:
        score += 0.25
        reasons.append("lat_cpu")
    if lat_net > 0.3:
        score += 0.15
        reasons.append("lat_net")
    if lat_step > 0.2:
        score += 0.2
        reasons.append("lat_step")
    return min(1.0, score), "+".join(reasons) if reasons else "none"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find scenarios with clear latency saturation (CPU/network increase)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None, help="Max number of scenarios to analyze (for quick test)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write list to file")
    args = parser.parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    if not SCENARIOS_DIR.exists():
        print(f"Scenarios dir not found: {SCENARIOS_DIR}", file=sys.stderr)
        sys.exit(1)

    all_pngs = list(SCENARIOS_DIR.glob("*.png"))
    candidate_stems = {p.stem[len(CANDIDATE_PREFIX):] for p in all_pngs if p.name.startswith(CANDIDATE_PREFIX)}
    non_candidate_stems = [p.stem for p in all_pngs if not p.name.startswith(CANDIDATE_PREFIX)]
    to_analyze = [s for s in non_candidate_stems if s not in candidate_stems]
    to_analyze.sort()
    if args.limit:
        to_analyze = to_analyze[: args.limit]

    print(f"Already chosen candidates: {len(candidate_stems)}")
    print(f"Non-candidate PNGs to analyze: {len(to_analyze)}")

    results = []
    for i, stem in enumerate(to_analyze):
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(to_analyze)}")
        win = load_scenario_data(data_dir, stem)
        if win is None:
            continue
        metric_cols = {k: [c for c in v if c in win.columns] for k, v in find_metric_columns(win).items()}
        if not metric_cols["cpu"] or not metric_cols["latency"]:
            continue
        score, reason = saturation_score(win, metric_cols)
        if score >= args.min_score:
            results.append((score, reason, stem))

    results.sort(key=lambda x: -x[0])
    results = results[: args.top]

    lines = [f"{stem}.png  (score={score:.2f}  {reason})" for score, reason, stem in results]
    text = "\n".join(lines)
    print()
    print(f"Scenarios with clear latency saturation (score >= {args.min_score}, top {args.top}):")
    print()
    print(text)
    if args.output:
        args.output.write_text(text)
        print(f"\nList written to {args.output}")


if __name__ == "__main__":
    main()
