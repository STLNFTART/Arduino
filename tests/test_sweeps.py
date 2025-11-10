"""Regression tests for vector sweep utilities."""

from __future__ import annotations

from pathlib import Path

from primal_logic.sweeps import torque_sweep


def test_torque_sweep_generates_results(tmp_path: Path) -> None:
    output = tmp_path / "sweep.csv"
    results = torque_sweep([0.5, 1.0], steps=10, output_path=output)
    assert len(results) == 2
    assert output.exists()
    with open(output, "r", encoding="utf-8") as handle:
        content = handle.read()
    assert "mean_torque" in content
