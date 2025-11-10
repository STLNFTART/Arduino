"""Unit tests for the Primal Logic hand model."""

from __future__ import annotations

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
