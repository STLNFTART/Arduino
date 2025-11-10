"""Unit tests for the Primal Logic hand model."""

from __future__ import annotations

import numpy as np

from primal_logic.hand import RoboticHand


def test_hand_advances_and_clamps_angles() -> None:
    hand = RoboticHand()
    desired = np.ones((hand.n_fingers, hand.n_joints_per_finger))
    hand.step(desired, theta=1.0, coherence=0.5, step=0)
    angles = hand.get_angles()
    assert np.all(angles >= hand.joint_limits.angle_min)
    assert np.all(angles <= hand.joint_limits.angle_max)
