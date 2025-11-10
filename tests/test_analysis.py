"""Tests for the pandas-based rolling average plot utility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from primal_logic.analysis import plot_rolling_average


def test_plot_rolling_average(tmp_path: Path) -> None:
    csv_path = tmp_path / "torque.csv"
    df = pd.DataFrame({"joint_0": np.linspace(-1.0, 1.0, 50)})
    df.to_csv(csv_path, index=False)

    plot_path = plot_rolling_average(csv_path, column="joint_0", window=5)
    assert plot_path.exists()
