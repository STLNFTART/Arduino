"""Vector sweep utilities for parameter exploration."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from .field import PrimalLogicField
from .hand import RoboticHand
from .trajectory import generate_grasp_trajectory
from .utils import flatten, mean, write_csv


def torque_sweep(
    thetas: Sequence[float],
    steps: int = 200,
    output_path: Path | None = None,
) -> List[Tuple[float, float]]:
    """Run a sweep of constant ``theta`` values and summarise mean torques."""

    results: List[Tuple[float, float]] = []
    for theta in thetas:
        hand = RoboticHand()
        field = PrimalLogicField(nx=4, ny=4)
        trajectory = generate_grasp_trajectory(steps, hand.n_fingers, hand.n_joints_per_finger)
        torques: List[float] = []
        for step in range(steps):
            coherence = field.step(theta)
            hand.step(trajectory[step], theta=theta, coherence=coherence, step=step)
            torques.extend(flatten(hand.get_torques()))
        results.append((theta, mean(torques)))

    if output_path is not None:
        header = ["theta", "mean_torque"]
        write_csv(str(output_path), header, results)

    return results
