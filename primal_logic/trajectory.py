"""Trajectory generation utilities."""

from __future__ import annotations

import numpy as np


def generate_grasp_trajectory(steps: int, n_fingers: int, n_joints: int) -> np.ndarray:
    """Generate a smooth grasp trajectory using a cosine profile."""
    trajectory = np.zeros((steps, n_fingers, n_joints))
    for k in range(steps):
        phase = min(1.0, k / (steps * 0.8))
        s = 0.5 * (1.0 - np.cos(np.pi * phase))
        trajectory[k, :, :] = s
    return trajectory
