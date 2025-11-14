"""Tests for memory kernel models."""

from __future__ import annotations

import pytest

from primal_logic.constants import DT, LAMBDA_DEFAULT
from primal_logic.memory import ExponentialMemoryKernel
from primal_logic.rpo import RecursivePlanckMemoryKernel


def test_exponential_kernel_update() -> None:
    """Test that ExponentialMemoryKernel updates correctly."""
    kernel = ExponentialMemoryKernel(lam=LAMBDA_DEFAULT, gain=1.0)

    # Initial state should be zero
    assert kernel.state == 0.0

    # Update with error
    result = kernel.update(theta=1.0, error=0.5, step_index=0)

    # Result should be -gain * memory
    assert isinstance(result, float)

    # Memory state should have changed
    assert kernel.state != 0.0


def test_exponential_kernel_decay() -> None:
    """Test that memory decays over time with zero input."""
    kernel = ExponentialMemoryKernel(lam=0.5, gain=1.0)

    # Apply initial error to build up memory
    kernel.update(theta=1.0, error=1.0, step_index=0)
    initial_memory = kernel.state

    # Apply zero error repeatedly and observe decay
    for _ in range(10):
        kernel.update(theta=1.0, error=0.0, step_index=0)

    # Memory should have decayed
    assert abs(kernel.state) < abs(initial_memory)


def test_exponential_kernel_state_property() -> None:
    """Test that state property returns current memory value."""
    kernel = ExponentialMemoryKernel(lam=LAMBDA_DEFAULT, gain=2.0)

    assert kernel.state == 0.0

    kernel.update(theta=1.0, error=0.5, step_index=0)
    state_after_update = kernel.state

    # State should be accessible and non-zero
    assert state_after_update != 0.0
    assert isinstance(state_after_update, float)


def test_memory_kernel_with_zero_error() -> None:
    """Test memory kernel behavior with zero error."""
    kernel = ExponentialMemoryKernel(lam=0.3, gain=1.5)

    # Build up some memory first
    kernel.update(theta=1.0, error=1.0, step_index=0)
    assert kernel.state != 0.0

    # Apply zero error
    result = kernel.update(theta=1.0, error=0.0, step_index=1)

    # Should return a value
    assert isinstance(result, float)


def test_exponential_kernel_gain_scaling() -> None:
    """Test that gain parameter scales the output correctly."""
    kernel_low = ExponentialMemoryKernel(lam=LAMBDA_DEFAULT, gain=0.5)
    kernel_high = ExponentialMemoryKernel(lam=LAMBDA_DEFAULT, gain=2.0)

    # Apply same input to both
    result_low = kernel_low.update(theta=1.0, error=1.0, step_index=0)
    result_high = kernel_high.update(theta=1.0, error=1.0, step_index=0)

    # Higher gain should produce larger magnitude result
    assert abs(result_high) > abs(result_low)


def test_recursive_planck_kernel_requires_step_index() -> None:
    """Test that RecursivePlanckMemoryKernel requires step_index (from test_rpo.py)."""
    kernel = RecursivePlanckMemoryKernel(alpha=0.35)

    # Should raise ValueError without step_index
    with pytest.raises(ValueError):
        kernel.update(theta=1.0, error=0.1)

    # Should work with step_index
    value = kernel.update(theta=1.0, error=0.1, step_index=0)
    assert isinstance(value, float)
