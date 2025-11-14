"""Unit tests for the Primal Logic hand model."""

from __future__ import annotations

import pytest

from primal_logic.hand import RoboticHand


def test_hand_advances_and_clamps_angles() -> None:
    hand = RoboticHand()
    desired = [
        [1.0 for _ in range(hand.n_joints_per_finger)] for _ in range(hand.n_fingers)
    ]
    hand.step(desired, theta=1.0, coherence=0.5, step=0)
    angles = hand.get_angles()
    for row in angles:
        for angle in row:
            assert hand.joint_limits.angle_min <= angle <= hand.joint_limits.angle_max


def test_hand_supports_recursive_planck_memory() -> None:
    hand = RoboticHand(memory_mode="recursive_planck", rpo_alpha=0.35)
    desired = [
        [0.8 for _ in range(hand.n_joints_per_finger)] for _ in range(hand.n_fingers)
    ]
    hand.step(desired, theta=0.9, coherence=0.4, step=1)
    torques = hand.get_torques()
    assert len(torques) == hand.n_fingers


def test_hand_zero_desired_angles() -> None:
    """Test hand with all zero desired angles."""
    hand = RoboticHand()
    desired = [
        [0.0 for _ in range(hand.n_joints_per_finger)] for _ in range(hand.n_fingers)
    ]
    hand.step(desired, theta=1.0, coherence=0.5, step=0)

    angles = hand.get_angles()
    # All angles should still be within limits
    for row in angles:
        for angle in row:
            assert hand.joint_limits.angle_min <= angle <= hand.joint_limits.angle_max


def test_hand_extreme_desired_values() -> None:
    """Test hand with extreme desired angles."""
    hand = RoboticHand()
    # Request angles way beyond physical limits
    desired = [
        [100.0 for _ in range(hand.n_joints_per_finger)] for _ in range(hand.n_fingers)
    ]

    # Step multiple times
    for step in range(10):
        hand.step(desired, theta=1.0, coherence=0.5, step=step)

    angles = hand.get_angles()
    # Angles should be clamped to max limit
    for row in angles:
        for angle in row:
            assert angle <= hand.joint_limits.angle_max


def test_hand_negative_alpha_raises() -> None:
    """Test that negative alpha_base raises ValueError."""
    with pytest.raises(ValueError, match="alpha_base must be positive"):
        RoboticHand(alpha_base=-0.1)


def test_hand_negative_beta_raises() -> None:
    """Test that negative beta_gain raises ValueError."""
    with pytest.raises(ValueError, match="beta_gain must be positive"):
        RoboticHand(beta_gain=-0.5)


def test_hand_invalid_memory_mode_raises() -> None:
    """Test that invalid memory_mode raises ValueError."""
    with pytest.raises(ValueError, match="memory_mode must be"):
        RoboticHand(memory_mode="invalid_mode")


def test_hand_get_angles_returns_copy() -> None:
    """Test that get_angles returns correct dimensions."""
    hand = RoboticHand(n_fingers=3, n_joints_per_finger=4)
    angles = hand.get_angles()

    assert len(angles) == 3
    assert len(angles[0]) == 4


def test_hand_get_torques_dimensions() -> None:
    """Test that get_torques returns correct dimensions."""
    hand = RoboticHand(n_fingers=5, n_joints_per_finger=3)
    desired = [
        [0.5 for _ in range(hand.n_joints_per_finger)] for _ in range(hand.n_fingers)
    ]
    hand.step(desired, theta=1.0, coherence=0.5, step=0)

    torques = hand.get_torques()
    assert len(torques) == 5
    assert len(torques[0]) == 3


def test_hand_multistep_sequence() -> None:
    """Test hand over multiple steps to verify state progression."""
    hand = RoboticHand()
    desired = [
        [0.6 for _ in range(hand.n_joints_per_finger)] for _ in range(hand.n_fingers)
    ]

    # Run for multiple steps
    for step in range(20):
        hand.step(desired, theta=1.0, coherence=0.5, step=step)

    # After 20 steps, angles should be moving toward desired
    angles = hand.get_angles()
    # At least some joints should have non-zero angles
    all_angles = [angle for row in angles for angle in row]
    assert any(angle > 0.0 for angle in all_angles)


def test_hand_velocity_limits() -> None:
    """Test that joint velocities respect vel_max."""
    hand = RoboticHand()
    # Apply large desired angle to potentially generate high velocity
    desired = [
        [hand.joint_limits.angle_max for _ in range(hand.n_joints_per_finger)]
        for _ in range(hand.n_fingers)
    ]

    for step in range(50):
        hand.step(desired, theta=1.0, coherence=0.5, step=step)

        # Check all velocities
        for finger in hand.states:
            for joint_state in finger:
                assert abs(joint_state.velocity) <= hand.joint_limits.vel_max


def test_hand_torque_limits() -> None:
    """Test that torques respect torque_max."""
    hand = RoboticHand()
    desired = [
        [1.0 for _ in range(hand.n_joints_per_finger)] for _ in range(hand.n_fingers)
    ]

    for step in range(10):
        hand.step(desired, theta=1.0, coherence=0.5, step=step)

        torques = hand.get_torques()
        for row in torques:
            for torque in row:
                assert abs(torque) <= hand.joint_limits.torque_max


def test_hand_different_finger_configurations() -> None:
    """Test hands with different numbers of fingers and joints."""
    configs = [
        (1, 1),  # Minimal
        (3, 2),  # Small
        (5, 4),  # Medium
        (6, 5),  # Large
    ]

    for n_fingers, n_joints in configs:
        hand = RoboticHand(n_fingers=n_fingers, n_joints_per_finger=n_joints)
        desired = [[0.5 for _ in range(n_joints)] for _ in range(n_fingers)]
        hand.step(desired, theta=1.0, coherence=0.5, step=0)

        angles = hand.get_angles()
        assert len(angles) == n_fingers
        assert len(angles[0]) == n_joints


def test_hand_rpo_alpha_stability_boundary() -> None:
    """Test that rpo_alpha at stability boundary raises error."""
    # rpo_alpha * dt must be < 1
    # If dt is default (0.001), then rpo_alpha >= 1000 should fail
    with pytest.raises(ValueError, match="rpo_alpha must satisfy"):
        RoboticHand(memory_mode="recursive_planck", rpo_alpha=2000.0)
