"""Data analysis utilities using pandas for rolling statistics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_rolling_average(
    csv_path: Path,
    column: str,
    window: int,
    output_path: Optional[Path] = None,
) -> Path:
    """Compute and plot a rolling average using pandas."""
    if window <= 0:
        raise ValueError("window must be positive")

    logger.info("Loading torque log from %s", csv_path)
    df = pd.read_csv(csv_path)

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    rolling = df[column].rolling(window=window, min_periods=1).mean()
    df[f"{column}_rolling_{window}"] = rolling

    if output_path is None:
        output_path = csv_path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[column], label=column, alpha=0.5)
    ax.plot(df.index, rolling, label=f"rolling_mean_{window}", linewidth=2.0)
    ax.set_xlabel("Sample")
    ax.set_ylabel(column)
    ax.set_title(f"Rolling average of {column} (window={window})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info("Saved rolling average plot to %s", output_path)
    return output_path
