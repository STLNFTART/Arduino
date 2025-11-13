"""Plot rolling averages for PRIMAL LOGIC simulation logs.

Usage:
    python3 scripts/plot_results.py --csv output/motor_hand_logs/simulation_log.csv --column vitality --window 10

This helper maintains IP traceability by keeping analysis local.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PRIMAL LOGIC rolling statistics")
    parser.add_argument("--csv", required=True, help="Path to simulation_log.csv")
    parser.add_argument("--column", default="vitality", help="Column to analyse (default: vitality)")
    parser.add_argument("--window", type=int, default=10, help="Rolling window size in steps")
    parser.add_argument("--output", default="plot.png", help="Output image file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logging.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not present in CSV.")

    logging.info("Computing rolling window average (window=%d)", args.window)
    df[f"{args.column}_rolling"] = df[args.column].rolling(window=args.window, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df[args.column], label=args.column)
    plt.plot(df["step"], df[f"{args.column}_rolling"], label=f"{args.column} (rolling)")
    plt.xlabel("Step")
    plt.ylabel(args.column)
    plt.title("PRIMAL LOGIC Simulation Metric")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    logging.info("Saved plot to %s", args.output)


if __name__ == "__main__":
    main()
