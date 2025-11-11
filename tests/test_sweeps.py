"""Regression tests for vector sweep utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from primal_logic.sweeps import alpha_sweep, beta_sweep, tau_sweep, torque_sweep


def test_torque_sweep_generates_results(tmp_path: Path) -> None:
    output = tmp_path / "sweep.csv"
    results = torque_sweep([0.5, 1.0], steps=10, output_path=output)
    assert len(results) == 2
    assert all(len(row) == 4 for row in results)
    assert output.exists()
    with open(output, "r", encoding="utf-8") as handle:
        content = handle.read()
    assert "mean_torque" in content
    assert "saturation_ratio" in content


def test_alpha_sweep_varies_gain(tmp_path: Path) -> None:
    output = tmp_path / "alpha.csv"
    results = alpha_sweep([0.52, 0.54], steps=8, output_path=output)
    assert len(results) == 2
    assert output.exists()
    assert results[0][0] == pytest.approx(0.52)


def test_beta_sweep_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        beta_sweep([0.0])


def test_beta_sweep_records_statistics(tmp_path: Path) -> None:
    output = tmp_path / "beta.csv"
    results = beta_sweep([0.6, 1.0], steps=6, output_path=output)
    assert len(results) == 2
    assert output.exists()
    assert all(row[3] >= 0.0 for row in results)  # saturation ratio is non-negative


def test_tau_sweep_enforces_positive_limit() -> None:
    with pytest.raises(ValueError):
        tau_sweep([-0.1])


def test_tau_sweep_generates_csv(tmp_path: Path) -> None:
    output = tmp_path / "tau.csv"
    results = tau_sweep([0.6, 0.8], steps=6, output_path=output)
    assert len(results) == 2
    assert output.exists()
