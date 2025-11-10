"""Trajectory generation utilities."""

from __future__ import annotations

import math
from typing import List

from .utils import zeros


def generate_grasp_trajectory(steps: int, n_fingers: int, n_joints: int) -> List[List[List[float]]]:
    """Generate a smooth grasp trajectory using a cosine profile."""

    trajectory = zeros((steps, n_fingers, n_joints))
    for k in range(steps):
        phase = min(1.0, k / (steps * 0.8)) if steps else 0.0
        s = 0.5 * (1.0 - math.cos(math.pi * phase))
        for finger in range(n_fingers):
            for joint in range(n_joints):
                trajectory[k][finger][joint] = s
    return trajectory
