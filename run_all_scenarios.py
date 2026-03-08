#!/usr/bin/env python3
"""
Run the visualisation for every scenario CSV in the processed dataset.
Saves one PNG per CSV to scenarios/, with the same base name as the CSV (e.g. foo_graph_2.csv -> scenarios/foo_graph_2.png).
Requires: pandas, matplotlib (pip install -r requirements.txt)
"""

import subprocess
import sys
from pathlib import Path

# Same default as visualise_processed_dataset.py
DEFAULT_DATA_DIR = Path("/Users/opeyemionikute/Downloads/archive/processed_dataset")
SCRIPT_DIR = Path(__file__).resolve().parent
SCENARIOS_DIR = SCRIPT_DIR / "scenarios"
VIS_SCRIPT = SCRIPT_DIR / "visualise_processed_dataset.py"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate one PNG per scenario CSV into scenarios/")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to processed_dataset root")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be run")
    args = parser.parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Data dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    # Find all CSVs under user/, home/, compose/ ... multi-modal-data-separate/*.csv
    csvs = []
    for service in ("user", "home", "compose"):
        base = data_dir / service / "multi-modal-data-separate"
        if not base.exists():
            continue
        for p in base.glob("*.csv"):
            rel = p.relative_to(data_dir)
            csvs.append(rel)
    csvs.sort()
    print(f"Found {len(csvs)} scenario CSVs. Output: {SCENARIOS_DIR}")
    if args.dry_run:
        for rel in csvs[:5]:
            stem = rel.stem
            print(f"  would: --csv {rel} --single-png scenarios/{stem}.png")
        if len(csvs) > 5:
            print(f"  ... and {len(csvs) - 5} more")
        return
    for i, rel in enumerate(csvs):
        stem = rel.stem
        out_png = SCENARIOS_DIR / f"{stem}.png"
        cmd = [
            sys.executable,
            str(VIS_SCRIPT),
            "--data-dir", str(data_dir),
            "--csv", str(rel.as_posix()),
            "--single-png", str(out_png),
            "--no-show",
        ]
        print(f"[{i + 1}/{len(csvs)}] {rel.stem}.png ...")
        r = subprocess.run(cmd, cwd=SCRIPT_DIR)
        if r.returncode != 0:
            print(f"  failed with code {r.returncode}", file=sys.stderr)
    print(f"Done. PNGs in {SCENARIOS_DIR}")


if __name__ == "__main__":
    main()
