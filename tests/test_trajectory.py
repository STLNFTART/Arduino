"""Tests for trajectory generation utilities."""

from __future__ import annotations

import math

import pytest

from primal_logic.trajectory import generate_grasp_trajectory


def test_grasp_trajectory_shape() -> None:
    """Test that generated trajectory has correct shape."""
    steps = 100
    n_fingers = 5
    n_joints = 3

    trajectory = generate_grasp_trajectory(steps, n_fingers, n_joints)

    assert len(trajectory) == steps
    assert len(trajectory[0]) == n_fingers
    assert len(trajectory[0][0]) == n_joints


def test_grasp_trajectory_smooth_profile() -> None:
    """Test that trajectory follows smooth cosine profile."""
    steps = 100
    n_fingers = 3
    n_joints = 2

    trajectory = generate_grasp_trajectory(steps, n_fingers, n_joints)

    # First step should be near zero
    assert trajectory[0][0][0] == pytest.approx(0.0, abs=0.1)

    # Last step should be near 1.0 (full grasp)
    assert trajectory[-1][0][0] == pytest.approx(1.0, abs=0.01)

    # Middle should be approximately 0.5
    mid_index = steps // 2
    assert 0.3 < trajectory[mid_index][0][0] < 0.7


def test_grasp_trajectory_bounds() -> None:
    """Test that all trajectory values are in [0, 1]."""
    steps = 50
    n_fingers = 4
    n_joints = 3

    trajectory = generate_grasp_trajectory(steps, n_fingers, n_joints)

    for step in trajectory:
        for finger in step:
            for joint_value in finger:
                assert 0.0 <= joint_value <= 1.0


def test_grasp_trajectory_zero_steps() -> None:
    """Test trajectory generation with zero steps."""
    trajectory = generate_grasp_trajectory(0, 3, 2)

    assert len(trajectory) == 0


def test_grasp_trajectory_monotonic() -> None:
    """Test that trajectory is monotonically increasing."""
    steps = 100
    trajectory = generate_grasp_trajectory(steps, 1, 1)

    # Extract single joint values
    values = [trajectory[k][0][0] for k in range(steps)]

    # Check monotonicity
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1], f"Trajectory not monotonic at step {i}"


def test_grasp_trajectory_uniform_across_joints() -> None:
    """Test that all joints receive the same trajectory value at each step."""
    steps = 50
    n_fingers = 3
    n_joints = 4

    trajectory = generate_grasp_trajectory(steps, n_fingers, n_joints)

    # At each timestep, all joints should have the same value
    for step in trajectory:
        reference_value = step[0][0]
        for finger in step:
            for joint_value in finger:
                assert joint_value == pytest.approx(reference_value)
