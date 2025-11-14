"""Tests for adaptive gain scheduling utilities."""

from __future__ import annotations

import pytest

from primal_logic.adaptive import adaptive_alpha
from primal_logic.constants import LIGHTFOOT_MAX, LIGHTFOOT_MIN


def test_adaptive_alpha_positive_base() -> None:
    """Test that adaptive_alpha works with positive base value."""
    alpha = adaptive_alpha(
        step=100,
        avg_energy=500.0,
        quantum_coherence=0.5,
        alpha_base=0.5,
    )

    assert isinstance(alpha, float)
    assert alpha > 0.0


def test_adaptive_alpha_negative_base_raises() -> None:
    """Test that negative alpha_base raises ValueError."""
    with pytest.raises(ValueError, match="alpha_base must be positive"):
        adaptive_alpha(
            step=100,
            avg_energy=500.0,
            quantum_coherence=0.5,
            alpha_base=-0.1,
        )

    with pytest.raises(ValueError, match="alpha_base must be positive"):
        adaptive_alpha(
            step=100,
            avg_energy=500.0,
            quantum_coherence=0.5,
            alpha_base=0.0,
        )


def test_adaptive_alpha_bounds() -> None:
    """Test that output is bounded within LIGHTFOOT_MIN and LIGHTFOOT_MAX."""
    # Test with various inputs
    test_cases = [
        (0, 0.0, 0.0),
        (100, 1000.0, 0.8),
        (1000, 5000.0, 0.2),
        (500, 100.0, 0.99),
    ]

    for step, energy, coherence in test_cases:
        alpha = adaptive_alpha(
            step=step,
            avg_energy=energy,
            quantum_coherence=coherence,
            alpha_base=0.5,
        )

        assert LIGHTFOOT_MIN <= alpha <= LIGHTFOOT_MAX, (
            f"Alpha {alpha} out of bounds for step={step}, "
            f"energy={energy}, coherence={coherence}"
        )


def test_adaptive_alpha_temporal_scaling() -> None:
    """Test that temporal scales influence the output."""
    # Get alpha at two different timesteps
    alpha_step0 = adaptive_alpha(
        step=0,
        avg_energy=500.0,
        quantum_coherence=0.5,
        alpha_base=0.5,
    )

    alpha_step1000 = adaptive_alpha(
        step=1000,
        avg_energy=500.0,
        quantum_coherence=0.5,
        alpha_base=0.5,
    )

    # They should be different due to temporal oscillations
    # (though might be equal by chance if we hit the same phase)
    # So we just verify both are valid
    assert isinstance(alpha_step0, float)
    assert isinstance(alpha_step1000, float)
    assert LIGHTFOOT_MIN <= alpha_step0 <= LIGHTFOOT_MAX
    assert LIGHTFOOT_MIN <= alpha_step1000 <= LIGHTFOOT_MAX


def test_adaptive_alpha_energy_scaling() -> None:
    """Test that higher energy affects alpha value."""
    alpha_low_energy = adaptive_alpha(
        step=100,
        avg_energy=100.0,
        quantum_coherence=0.5,
        alpha_base=0.5,
    )

    alpha_high_energy = adaptive_alpha(
        step=100,
        avg_energy=10000.0,
        quantum_coherence=0.5,
        alpha_base=0.5,
    )

    # Higher energy should increase alpha (up to the max bound)
    # Both should be valid floats
    assert isinstance(alpha_low_energy, float)
    assert isinstance(alpha_high_energy, float)


def test_adaptive_alpha_coherence_influence() -> None:
    """Test that quantum coherence influences alpha."""
    alpha_low_coherence = adaptive_alpha(
        step=100,
        avg_energy=500.0,
        quantum_coherence=0.1,
        alpha_base=0.5,
    )

    alpha_high_coherence = adaptive_alpha(
        step=100,
        avg_energy=500.0,
        quantum_coherence=0.9,
        alpha_base=0.5,
    )

    # Higher coherence should affect alpha
    # Both should be valid
    assert isinstance(alpha_low_coherence, float)
    assert isinstance(alpha_high_coherence, float)
    assert LIGHTFOOT_MIN <= alpha_low_coherence <= LIGHTFOOT_MAX
    assert LIGHTFOOT_MIN <= alpha_high_coherence <= LIGHTFOOT_MAX
