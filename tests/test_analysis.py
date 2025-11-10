"""Tests for the pandas-based rolling average plot utility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from primal_logic.analysis import plot_rolling_average


def test_plot_rolling_average(tmp_path: Path) -> None:
    csv_path = tmp_path / "torque.csv"
    df = pd.DataFrame({"joint_0": [float(k) / 10.0 for k in range(50)]})
    df.to_csv(csv_path, index=False)

    plot_path = plot_rolling_average(csv_path, column="joint_0", window=5)
    assert plot_path.exists()
    with open(plot_path, "r", encoding="utf-8") as handle:
        content = handle.read()
    assert "rolling_mean_5" in content
