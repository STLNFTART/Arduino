"""Data analysis utilities using pandas with optional offline fallbacks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

try:  # Prefer the real pandas library for rich analytics.
    import pandas as pd
    _USING_PANDAS_STUB = False
except ModuleNotFoundError:  # pragma: no cover - fallback for offline execution
    from vendor import pandas_stub as pd  # type: ignore
    _USING_PANDAS_STUB = True

try:  # Prefer the real matplotlib backend for proper plotting.
    import matplotlib.pyplot as plt
    _USING_MPL_STUB = False
except ModuleNotFoundError:  # pragma: no cover - fallback for offline execution
    from vendor.matplotlib_stub import pyplot as plt  # type: ignore
    _USING_MPL_STUB = True

logger = logging.getLogger(__name__)


def plot_rolling_average(
    csv_path: Path,
    column: str,
    window: int,
    output_path: Optional[Path] = None,
) -> Path:
    """Compute and plot a rolling average using pandas-style operations."""

    if window <= 0:
        raise ValueError("window must be positive")

    logger.info("Loading torque log from %s", csv_path)
    df = pd.read_csv(csv_path)

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    rolling_series = df[column].rolling(window=window, min_periods=1).mean()
    df[f"{column}_rolling_{window}"] = rolling_series

    if output_path is None:
        output_path = csv_path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(list(df.index), list(df[column]), label=column, alpha=0.5)
    ax.plot(list(df.index), list(rolling_series), label=f"rolling_mean_{window}", linewidth=2.0)
    ax.set_xlabel("Sample")
    ax.set_ylabel(column)
    ax.set_title(f"Rolling average of {column} (window={window})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    if _USING_PANDAS_STUB or _USING_MPL_STUB:
        logger.warning(
            "Using stub implementations for pandas/matplotlib; install real libraries for full fidelity."
        )

    logger.info("Saved rolling average plot to %s", output_path)
    return output_path
