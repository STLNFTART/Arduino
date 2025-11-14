"""Comprehensive tests for the Multi-Heart Model."""

from __future__ import annotations

import math

import pytest

from primal_logic.constants import DT
from primal_logic.heart_model import HeartBrainState, MultiHeartModel


def test_heart_brain_initialization() -> None:
    """Test that MultiHeartModel initializes with correct default state."""
    model = MultiHeartModel()

    assert model.state.n_heart == 0.0
    assert model.state.n_brain == 0.0
    assert model.state.s_heart == 0.0
    assert model.state.s_brain == 0.0
    assert model.step_count == 0
    assert model.rpo_heart is not None
    assert model.rpo_brain is not None


def test_step_updates_state() -> None:
    """Test that step method updates all state variables."""
    model = MultiHeartModel()

    initial_n_heart = model.state.n_heart
    initial_n_brain = model.state.n_brain

    model.step(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)

    # State should change after one step
    assert model.state.n_heart != initial_n_heart or model.state.n_brain != initial_n_brain
    assert model.step_count == 1


def test_rpo_alpha_validation() -> None:
    """Test that invalid rpo_alpha raises ValueError."""
    # rpo_alpha must satisfy 0 < rpo_alpha * dt < 1
    dt = DT

    # Alpha too large (alpha * dt >= 1)
    with pytest.raises(ValueError, match="rpo_alpha must satisfy"):
        MultiHeartModel(rpo_alpha=10.0 / dt)

    # Alpha zero or negative
    with pytest.raises(ValueError, match="rpo_alpha must satisfy"):
        MultiHeartModel(rpo_alpha=0.0)

    with pytest.raises(ValueError, match="rpo_alpha must satisfy"):
        MultiHeartModel(rpo_alpha=-0.1)

    # Valid alpha should work
    model = MultiHeartModel(rpo_alpha=0.4)
    assert model.rpo_alpha == 0.4


def test_heart_brain_coupling() -> None:
    """Test that heart and brain states couple through f_heart and f_brain."""
    model = MultiHeartModel(coupling_strength=0.5)

    # Set specific brain state
    model.state.n_brain = 1.0
    model.state.s_heart = 0.5

    f_h = model.f_heart(model.state.n_brain, model.state.s_heart)
    expected = 0.5 * math.tanh(1.0 + 0.5)
    assert f_h == pytest.approx(expected)

    # Set specific heart state
    model.state.n_heart = 0.8
    model.state.s_brain = 0.3

    f_b = model.f_brain(model.state.n_heart, model.state.s_brain)
    expected = 0.5 * math.tanh(0.8 + 0.3)
    assert f_b == pytest.approx(expected)


def test_get_heart_rate_bounds() -> None:
    """Test that heart rate output is bounded to [0, 1]."""
    model = MultiHeartModel()

    # Test with various heart states
    test_states = [-10.0, -1.0, 0.0, 1.0, 10.0]
    for n_heart in test_states:
        model.state.n_heart = n_heart
        heart_rate = model.get_heart_rate()
        assert 0.0 <= heart_rate <= 1.0, f"Heart rate {heart_rate} out of bounds for n_heart={n_heart}"


def test_get_brain_activity_bounds() -> None:
    """Test that brain activity is bounded by tanh to [-1, 1]."""
    model = MultiHeartModel()

    test_states = [-100.0, -1.0, 0.0, 1.0, 100.0]
    for n_brain in test_states:
        model.state.n_brain = n_brain
        brain_activity = model.get_brain_activity()
        assert -1.0 <= brain_activity <= 1.0


def test_cardiac_output_channels() -> None:
    """Test that cardiac output returns exactly 4 channels."""
    model = MultiHeartModel()
    model.step(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)

    output = model.get_cardiac_output()

    assert len(output) == 4
    # All channels should be floats
    assert all(isinstance(val, float) for val in output)


def test_reset_clears_state() -> None:
    """Test that reset method clears all state to initial values."""
    model = MultiHeartModel()

    # Run some steps to modify state
    for _ in range(10):
        model.step(cardiac_input=0.5, brain_setpoint=0.3, theta=1.0)

    assert model.step_count == 10

    # Reset
    model.reset()

    assert model.state.n_heart == 0.0
    assert model.state.n_brain == 0.0
    assert model.state.s_heart == 0.0
    assert model.state.s_brain == 0.0
    assert model.step_count == 0


def test_step_advances_counter() -> None:
    """Test that step_count increments correctly."""
    model = MultiHeartModel()

    for i in range(1, 6):
        model.step(cardiac_input=0.1, brain_setpoint=0.1, theta=1.0)
        assert model.step_count == i


def test_f_heart_coupling_function() -> None:
    """Test the heart coupling function f_heart."""
    model = MultiHeartModel(coupling_strength=0.2)

    # Test with zero inputs
    assert model.f_heart(0.0, 0.0) == 0.0

    # Test with known values
    result = model.f_heart(1.5, 0.5)
    expected = 0.2 * math.tanh(1.5 + 0.5)
    assert result == pytest.approx(expected)


def test_f_brain_coupling_function() -> None:
    """Test the brain coupling function f_brain."""
    model = MultiHeartModel(coupling_strength=0.3)

    # Test with zero inputs
    assert model.f_brain(0.0, 0.0) == 0.0

    # Test with known values
    result = model.f_brain(0.8, 0.4)
    expected = 0.3 * math.tanh(0.8 + 0.4)
    assert result == pytest.approx(expected)


def test_sensory_feedback_update() -> None:
    """Test that sensory feedback is updated from neural potentials."""
    model = MultiHeartModel()

    # Set specific neural potentials
    model.state.n_heart = 2.0
    model.state.n_brain = 1.5

    # Step to update sensory feedback
    model.step(cardiac_input=0.0, brain_setpoint=0.0, theta=1.0)

    # After step, sensory feedback should be 0.5 * neural potential
    # (based on the update rule in step method)
    assert model.state.s_heart == pytest.approx(model.state.n_heart * 0.5)
    assert model.state.s_brain == pytest.approx(model.state.n_brain * 0.5)


def test_integration_stability() -> None:
    """Test that long simulation stays bounded (stability test)."""
    model = MultiHeartModel(rpo_alpha=0.3)

    max_n_heart = 0.0
    max_n_brain = 0.0

    # Run for 2000 steps with sinusoidal inputs
    for step in range(2000):
        cardiac_input = math.sin(0.01 * step)
        brain_setpoint = math.cos(0.01 * step)
        model.step(cardiac_input=cardiac_input, brain_setpoint=brain_setpoint, theta=1.0)

        max_n_heart = max(max_n_heart, abs(model.state.n_heart))
        max_n_brain = max(max_n_brain, abs(model.state.n_brain))

    # Values should stay bounded (not explode to infinity)
    assert max_n_heart < 20.0, f"Heart potential unbounded: {max_n_heart}"
    assert max_n_brain < 20.0, f"Brain potential unbounded: {max_n_brain}"


def test_cardiac_output_coherence_channel() -> None:
    """Test that coherence channel in cardiac output is properly calculated."""
    model = MultiHeartModel()

    model.state.n_heart = 0.5
    model.state.n_brain = 0.8

    output = model.get_cardiac_output()

    # Channel 2 should be coherence = tanh(|n_heart * n_brain|)
    expected_coherence = math.tanh(abs(0.5 * 0.8))
    assert output[2] == pytest.approx(expected_coherence)


def test_different_lambda_values() -> None:
    """Test that different decay rates work correctly."""
    model = MultiHeartModel(lambda_heart=0.5, lambda_brain=0.3)

    assert model.lambda_heart == 0.5
    assert model.lambda_brain == 0.3

    # Should initialize without errors
    model.step(cardiac_input=1.0, brain_setpoint=1.0, theta=1.0)
