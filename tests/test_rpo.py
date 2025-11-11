"""Tests for the Recursive Planck Operator implementation."""

from __future__ import annotations

import math

import pytest

from primal_logic.constants import DT
from primal_logic.rpo import RecursivePlanckMemoryKernel, RecursivePlanckOperator


def test_rpo_remains_bounded() -> None:
    operator = RecursivePlanckOperator(alpha=0.4)
    theta = 1.0
    samples = []
    for step in range(2000):
        input_value = math.sin(0.01 * step)
        samples.append(operator.step(theta=theta, input_value=input_value, step_index=step))

    assert max(abs(value) for value in samples) < 10.0


def test_recursive_planck_kernel_requires_step_index() -> None:
    kernel = RecursivePlanckMemoryKernel(alpha=0.35)
    with pytest.raises(ValueError):
        kernel.update(theta=1.0, error=0.1)

    value = kernel.update(theta=1.0, error=0.1, step_index=0)
    assert isinstance(value, float)


def test_rpo_effective_planck_constant() -> None:
    operator = RecursivePlanckOperator(alpha=0.4)
    assert operator.h_eff > 0
    assert operator.beta_p == pytest.approx(operator.lightfoot / (1.0 + operator.lam))
    # ensure dt scaling matches specification
    assert operator.alpha * DT < 1.0
