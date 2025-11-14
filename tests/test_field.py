"""Tests for quantum-inspired field dynamics."""

from __future__ import annotations

import pytest

from primal_logic.field import PrimalLogicField


def test_field_initialization() -> None:
    """Test that PrimalLogicField initializes with correct dimensions."""
    field = PrimalLogicField(nx=8, ny=8)

    assert field.nx == 8
    assert field.ny == 8
    assert len(field.psi_r) == 8
    assert len(field.psi_r[0]) == 8
    assert len(field.psi_i) == 8
    assert len(field.psi_i[0]) == 8
    assert len(field.gamma_field) == 8

    # All fields should be initialized to zeros
    for i in range(8):
        for j in range(8):
            assert field.psi_r[i][j] == 0.0
            assert field.psi_i[i][j] == 0.0
            assert field.gamma_field[i][j] == 0.0


def test_step_updates_field() -> None:
    """Test that step method updates field values."""
    field = PrimalLogicField(nx=4, ny=4, alpha=0.5)

    # Initial field should be all zeros
    initial_sum = sum(sum(row) for row in field.psi_r)
    assert initial_sum == 0.0

    # After step, field should change
    coherence = field.step(theta=1.0)

    # At least some field values should be non-zero after stepping with theta=1.0
    final_sum = sum(sum(row) for row in field.psi_r)
    assert final_sum != 0.0

    # Coherence should be returned
    assert isinstance(coherence, float)


def test_coherence_bounded() -> None:
    """Test that coherence output is bounded to [0, 1]."""
    field = PrimalLogicField(nx=6, ny=6)

    # Test with various theta values
    for theta in [-1.0, 0.0, 0.5, 1.0, 2.0]:
        for _ in range(10):
            coherence = field.step(theta=theta)
            assert 0.0 <= coherence <= 1.0, f"Coherence {coherence} out of bounds for theta={theta}"


def test_field_stability() -> None:
    """Test that long simulation stays bounded (stability test)."""
    field = PrimalLogicField(nx=8, ny=8, alpha=0.3, lam=0.5)

    max_psi_r = 0.0
    max_psi_i = 0.0

    # Run for 1000 steps
    for _ in range(1000):
        field.step(theta=1.0)

        # Track maximum field values
        for i in range(field.nx):
            for j in range(field.ny):
                max_psi_r = max(max_psi_r, abs(field.psi_r[i][j]))
                max_psi_i = max(max_psi_i, abs(field.psi_i[i][j]))

    # Fields should stay bounded (not explode to infinity)
    assert max_psi_r < 100.0, f"Real field unbounded: {max_psi_r}"
    assert max_psi_i < 100.0, f"Imaginary field unbounded: {max_psi_i}"


def test_laplacian_coupling() -> None:
    """Test that coupling parameter affects field evolution."""
    # Create two fields with different coupling strengths
    field_weak = PrimalLogicField(nx=4, ny=4, coupling=0.01)
    field_strong = PrimalLogicField(nx=4, ny=4, coupling=1.0)

    # Set identical initial conditions (non-zero center)
    field_weak.psi_r[2][2] = 1.0
    field_strong.psi_r[2][2] = 1.0

    # Step both
    field_weak.step(theta=0.0)
    field_strong.step(theta=0.0)

    # Strong coupling should spread the field more
    # Check neighbor points
    weak_neighbor_sum = field_weak.psi_r[2][1] + field_weak.psi_r[2][3] + field_weak.psi_r[1][2] + field_weak.psi_r[3][2]
    strong_neighbor_sum = field_strong.psi_r[2][1] + field_strong.psi_r[2][3] + field_strong.psi_r[1][2] + field_strong.psi_r[3][2]

    # With stronger coupling, neighbors should have larger absolute values
    assert abs(strong_neighbor_sum) > abs(weak_neighbor_sum)


def test_theta_influence() -> None:
    """Test that theta parameter influences field evolution."""
    field = PrimalLogicField(nx=4, ny=4, alpha=1.0)

    # Step with theta=0 (no drive)
    coherence_zero = field.step(theta=0.0)

    # Reset and step with theta=1 (strong drive)
    field2 = PrimalLogicField(nx=4, ny=4, alpha=1.0)
    coherence_one = field2.step(theta=1.0)

    # Calculate field energy
    energy_zero = sum(field.psi_r[i][j]**2 for i in range(4) for j in range(4))
    energy_one = sum(field2.psi_r[i][j]**2 for i in range(4) for j in range(4))

    # Field with theta=1 should have more energy
    assert energy_one > energy_zero


def test_custom_parameters() -> None:
    """Test that custom parameters are properly set."""
    field = PrimalLogicField(
        nx=10,
        ny=12,
        alpha=0.7,
        lam=0.3,
        coupling=0.8,
        gamma=0.05,
        epsilon=0.02,
    )

    assert field.nx == 10
    assert field.ny == 12
    assert field.alpha == 0.7
    assert field.lam == 0.3
    assert field.coupling == 0.8
    assert field.gamma == 0.05
    assert field.epsilon == 0.02
